"""Loaders: raw NYC LL84 / weather files -> tidy, harmonised tables; DuckDB setup."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
RESULTS = ROOT / "reports" / "results"
DUCKDB_PATH = ROOT / "data" / "wattwise.duckdb"

BIG_BOX_TYPES = ["Retail Store", "Wholesale Club/Supercenter"]
COMPARISON_TYPES = ["Supermarket/Grocery Store", "Enclosed Mall", "Strip Mall"]

ANNUAL_FILES = [
    "d1_ll84_annual_cy2019.csv",
    "d1_ll84_annual_cy2020.csv",
    "d1_ll84_annual_cy2021.csv",
    "d1_ll84_annual_cy2022_present.csv",
]

# harmonised name -> candidate source headers (normalised: lower-case, single spaces).
# The yearly releases rename and re-order columns, so we match on the human header.
ANNUAL_FIELDS: dict[str, list[str]] = {
    "year": ["calendar year"],
    "property_id": ["property id"],
    "property_type": ["primary property type - portfolio manager-calculated"],
    "largest_use": ["largest property use type"],
    "gfa_ft2": ["property gfa - self-reported (ft²)"],
    "site_eui": ["site eui (kbtu/ft²)"],
    "wn_site_eui": ["weather normalized site eui (kbtu/ft²)"],
    "source_eui": ["source eui (kbtu/ft²)"],
    "site_energy_kbtu": ["site energy use (kbtu)"],
    "elec_kwh": ["electricity use - grid purchase and generated from onsite renewable systems (kwh)"],
    "elec_grid_kwh": ["electricity use - grid purchase (kwh)"],
    "onsite_gen_kwh": ["electricity use – generated from onsite renewable systems (kwh)"],
    "gas_kbtu": ["natural gas use (kbtu)"],
    "steam_kbtu": ["district steam use (kbtu)"],
    "oil2_kbtu": ["fuel oil #2 use (kbtu)"],
    "water_kgal": ["water use (all water sources) (kgal)"],
    "ghg_t": ["total (location-based) ghg emissions (metric tons co2e)", "total ghg emissions (metric tons co2e)"],
    "ghg_direct_t": ["direct ghg emissions (metric tons co2e)"],
    "ghg_indirect_t": ["indirect (location-based) ghg emissions (metric tons co2e)", "indirect ghg emissions (metric tons co2e)"],
    "energy_star": ["energy star score"],
    "year_built": ["year built"],
    "borough": ["borough"],
    "latitude": ["latitude"],
    "longitude": ["longitude"],
}
TEXT_FIELDS = {"property_id", "property_type", "largest_use", "borough"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


@lru_cache(maxsize=1)
def assumptions() -> dict:
    with open(ROOT / "config" / "assumptions.yaml") as f:
        return yaml.safe_load(f)


def _harmonise(df: pd.DataFrame, fields: dict[str, list[str]]) -> pd.DataFrame:
    lookup = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=df.index)
    for name, candidates in fields.items():
        src = next((lookup[c] for c in candidates if c in lookup), None)
        if src is None:
            out[name] = np.nan
        elif name in TEXT_FIELDS:
            out[name] = df[src].astype("string").str.strip()
        else:
            out[name] = pd.to_numeric(df[src], errors="coerce")
    return out


def load_annual_raw() -> pd.DataFrame:
    """All LL84 annual retail rows, CY2019-latest, harmonised to one schema.

    Rows are kept as reported (no cleaning) so the quality framework can judge them.
    """
    frames = []
    for fname in ANNUAL_FILES:
        df = pd.read_csv(RAW / fname, dtype=str, low_memory=False)
        h = _harmonise(df, ANNUAL_FIELDS)
        h["source_file"] = fname
        frames.append(h)
    a = pd.concat(frames, ignore_index=True)
    a["year"] = a["year"].astype("Int64")
    a["cohort"] = np.where(a["property_type"].isin(BIG_BOX_TYPES), "Big-box retail", "Comparison")
    # elec_kwh is missing in some rows that report grid purchase only
    a["elec_kwh"] = a["elec_kwh"].fillna(a["elec_grid_kwh"])
    return a


def _parse_month(year: pd.Series, label: pd.Series) -> pd.Series:
    # labels look like "18-Jan"; the year prefix is redundant with Calendar Year
    mon = label.str.split("-").str[-1]
    return pd.to_datetime(year.astype(str) + "-" + mon + "-01", format="%Y-%b-%d")


def load_monthly_raw() -> pd.DataFrame:
    """LL84 monthly electricity and gas for every retail property (kBtu and kWh)."""
    m = pd.read_csv(RAW / "d2_ll84_monthly_retail.csv", dtype=str, low_memory=False)
    lookup = {_norm(c): c for c in m.columns}

    def num(header: str) -> pd.Series:
        col = lookup.get(header)
        return pd.to_numeric(m[col], errors="coerce") if col else pd.Series(np.nan, index=m.index)

    elec_total = num("electricity use (kbtu)")
    elec_grid = num("electricity use (grid) - monthly (kbtu)")
    elec_onsite = num("electricity use - onsite renewables (kbtu)")
    # 2018-2021 releases report a single total; 2023+ split grid and onsite renewables
    elec = elec_total.fillna(elec_grid + elec_onsite.fillna(0))
    out = pd.DataFrame(
        {
            "property_id": m[lookup["property id"]].astype("string").str.strip(),
            "month": _parse_month(m[lookup["calendar year"]], m[lookup["month"]]),
            "elec_kbtu": elec,
            "gas_kbtu": num("natural gas use - monthly (kbtu)"),
            "steam_kbtu": num("district steam use (kbtu)"),
        }
    )
    kbtu_per_kwh = assumptions()["units"]["kbtu_per_kwh"]
    out["elec_kwh"] = out["elec_kbtu"] / kbtu_per_kwh
    out["gas_therms"] = out["gas_kbtu"] / assumptions()["units"]["kbtu_per_therm"]
    out["year"] = out["month"].dt.year
    return out.sort_values(["property_id", "month"]).reset_index(drop=True)


def load_weather_daily() -> pd.DataFrame:
    w = pd.read_csv(RAW / "d3_weather_nyc_daily.csv", parse_dates=["date"])
    return w.rename(columns={"temperature_2m_mean": "t_mean_c", "temperature_2m_max": "t_max_c", "temperature_2m_min": "t_min_c"})


def save(df: pd.DataFrame, name: str) -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    p = PROCESSED / f"{name}.parquet"
    df.to_parquet(p, index=False)
    return p


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / f"{name}.parquet")


def duckdb_connect(refresh: bool = True):
    """Open the local analytics DB and expose every processed Parquet file as a view."""
    import duckdb

    con = duckdb.connect(str(DUCKDB_PATH))
    if refresh:
        for p in sorted(PROCESSED.glob("*.parquet")):
            con.execute(f"CREATE OR REPLACE VIEW {p.stem} AS SELECT * FROM read_parquet('{p}')")
    return con


# ---- computed-results store ---------------------------------------------------------
# Every number quoted in the executive report or README is written here by a notebook
# and read back from here — never typed by hand.
def save_result(section: str, values: dict) -> None:
    import json

    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / f"{section}.json"

    def clean(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        return v

    p.write_text(json.dumps(clean(values), indent=2, default=str))


def load_results() -> dict:
    import json

    return {p.stem: json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))}
