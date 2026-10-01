"""Data-quality framework: rule flags, a 0-100 score per property, and an exclusion waterfall.

Two levels of rules:
  * annual rules run on each property-year row of the LL84 disclosure;
  * monthly rules run on each property-year of the monthly electricity series and are
    merged back onto the annual row.
Every rule returns a boolean flag column named ``dq_<rule>``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import assumptions

# score penalty per rule (points off 100). Exclusion-grade issues cost the most.
PENALTY = {
    "dq_gfa_missing": 40,
    "dq_energy_missing": 40,
    "dq_eui_out_of_bounds": 35,
    "dq_scale_error": 35,
    "dq_duplicate": 15,
    "dq_annual_monthly_mismatch": 15,
    "dq_zero_months": 15,
    "dq_flatline": 10,
    "dq_spike": 10,
    "dq_missing_months": 20,
}

RULE_LABELS = {
    "dq_duplicate": "Duplicate property-year",
    "dq_gfa_missing": "Missing / zero floor area",
    "dq_energy_missing": "No energy reported",
    "dq_scale_error": "Unit / scale error (≈1,000× off)",
    "dq_eui_out_of_bounds": "EUI outside 5–1,000 kBtu/ft²",
    "dq_missing_months": "Missing months",
    "dq_zero_months": "Zero-consumption months",
    "dq_flatline": "Flatline (≥3 identical months)",
    "dq_spike": "Spike (>3× rolling median)",
    "dq_annual_monthly_mismatch": "Monthly ≠ annual total (>10%)",
}


def annual_flags(a: pd.DataFrame) -> pd.DataFrame:
    t = assumptions()["thresholds"]
    a = a.copy()
    a["dq_duplicate"] = a.duplicated(["property_id", "year"], keep="first")
    a["dq_gfa_missing"] = a["gfa_ft2"].isna() | (a["gfa_ft2"] <= 0)
    a["dq_energy_missing"] = a["site_eui"].isna() | (a["site_eui"] <= 0)
    eui = a["site_eui"]
    # unit/scale error: EUI implausibly large but plausible once divided by 1,000
    # (kBtu typed as Btu, or MWh as kWh), or a floor area under 1,000 ft² on a
    # building large enough to be LL84-covered (ft² typed as thousand-ft²)
    a["dq_scale_error"] = (
        ((eui > t["eui_max_kbtu_ft2"]) & (eui / 1000).between(t["eui_min_kbtu_ft2"], t["eui_max_kbtu_ft2"]))
        | (a["gfa_ft2"].between(1, 999))
    )
    a["dq_eui_out_of_bounds"] = eui.notna() & ((eui < t["eui_min_kbtu_ft2"]) | (eui > t["eui_max_kbtu_ft2"]))
    return a


def monthly_flags(m: pd.DataFrame, value: str = "elec_kwh") -> pd.DataFrame:
    """Per property-year flags computed from the monthly series of ``value``."""
    t = assumptions()["thresholds"]
    m = m.sort_values(["property_id", "month"]).copy()
    g = m.groupby("property_id")[value]
    roll_med = g.transform(lambda s: s.rolling(7, center=True, min_periods=4).median())
    m["_spike"] = m[value] > t["spike_ratio"] * roll_med
    m["_zero"] = m[value] == 0

    # flatline: same non-zero, non-missing value repeated >= N consecutive months
    def run_flag(s: pd.Series) -> pd.Series:
        key = (s != s.shift()).cumsum()
        size = s.groupby(key).transform("size")
        return (size >= t["flatline_months"]) & s.notna() & (s != 0)

    m["_flat"] = g.transform(run_flag).astype(bool)
    py = m.groupby(["property_id", "year"]).agg(
        n_months=(value, "size"),
        n_valid=(value, lambda s: s.notna().sum()),
        monthly_total=(value, lambda s: s.sum(min_count=1)),
        dq_zero_months=("_zero", "any"),
        dq_flatline=("_flat", "any"),
        dq_spike=("_spike", "any"),
    )
    py["dq_missing_months"] = py["n_valid"] < 12
    return py.reset_index()


def merge_flags(a: pd.DataFrame, py: pd.DataFrame) -> pd.DataFrame:
    """Attach monthly flags to annual rows.

    Calendar years absent from the monthly release entirely (2022 is not published)
    are not held against the property: missing months only counts in published years.
    """
    t = assumptions()["thresholds"]
    published = set(py["year"].unique())
    out = a.merge(py, on=["property_id", "year"], how="left")
    ratio = out["monthly_total"] / out["elec_kwh"]
    out["monthly_annual_ratio"] = ratio
    out["dq_annual_monthly_mismatch"] = (out["n_valid"] == 12) & ((ratio - 1).abs() > t["annual_monthly_mismatch"])
    out["has_monthly"] = out["n_months"].notna()
    for c in ["dq_zero_months", "dq_flatline", "dq_spike", "dq_missing_months"]:
        out[c] = out[c].astype("boolean").fillna(False).astype(bool)
    # a property-year with no monthly series at all is "missing months" too
    out.loc[~out["has_monthly"] & out["year"].isin(published), "dq_missing_months"] = True
    return out


def score(df: pd.DataFrame) -> pd.Series:
    pen = sum(df[c].astype(int) * w for c, w in PENALTY.items() if c in df)
    return (100 - pen).clip(lower=0)


def property_scores(flagged: pd.DataFrame) -> pd.DataFrame:
    """Data-quality score per property: mean of its property-year scores."""
    f = flagged.assign(dq_score=score(flagged))
    agg = f.groupby("property_id").agg(
        dq_score=("dq_score", "mean"),
        years_reported=("year", "nunique"),
        property_type=("property_type", "last"),
        cohort=("cohort", "last"),
        gfa_ft2=("gfa_ft2", "median"),
    )
    agg["dq_band"] = pd.cut(agg["dq_score"], [-1, 50, 80, 95, 100], labels=["Poor (<50)", "Fair (50–80)", "Good (80–95)", "Excellent (95+)"])
    return agg.reset_index()


EXCLUSION_ORDER = ["dq_duplicate", "dq_gfa_missing", "dq_energy_missing", "dq_scale_error", "dq_eui_out_of_bounds"]


def exclusion_waterfall(flagged: pd.DataFrame) -> pd.DataFrame:
    """Rows removed by each exclusion rule, applied in order (each row counted once)."""
    remaining = pd.Series(True, index=flagged.index)
    rows = [("All retail rows", int(remaining.sum()), 0)]
    for rule in EXCLUSION_ORDER:
        hit = remaining & flagged[rule]
        rows.append((RULE_LABELS[rule], int((remaining & ~hit).sum()), int(hit.sum())))
        remaining &= ~hit
    w = pd.DataFrame(rows, columns=["step", "rows_remaining", "rows_removed"])
    return w


def clean_mask(flagged: pd.DataFrame) -> pd.Series:
    """Rows kept for annual benchmarking (exclusion-grade rules only; soft flags kept)."""
    bad = np.zeros(len(flagged), dtype=bool)
    for rule in EXCLUSION_ORDER:
        bad |= flagged[rule].to_numpy()
    return pd.Series(~bad, index=flagged.index)
