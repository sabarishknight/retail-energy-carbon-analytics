"""Energy KPIs: intensities, unit conversions, load-shape and PV metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import assumptions


def u() -> dict:
    return assumptions()["units"]


def kbtu_ft2_to_kwh_m2(x):
    return x * u()["kwh_per_m2_per_kbtu_per_ft2"]


def ft2_to_m2(x):
    return x * u()["m2_per_ft2"]


def add_intensities(a: pd.DataFrame) -> pd.DataFrame:
    """Per-area intensities in both US and metric units."""
    a = a.copy()
    gfa_m2 = ft2_to_m2(a["gfa_ft2"])
    a["gfa_m2"] = gfa_m2
    a["site_eui_kwh_m2"] = kbtu_ft2_to_kwh_m2(a["site_eui"])
    a["wn_site_eui_kwh_m2"] = kbtu_ft2_to_kwh_m2(a["wn_site_eui"])
    a["elec_kwh_m2"] = a["elec_kwh"] / gfa_m2
    a["water_l_m2"] = a["water_kgal"] * 1000 * u()["litres_per_gallon"] / gfa_m2
    a["water_gal_ft2"] = a["water_kgal"] * 1000 / a["gfa_ft2"]
    a["ghg_kg_m2"] = a["ghg_t"] * 1000 / gfa_m2
    elec_kbtu = a["elec_kwh"] * u()["kbtu_per_kwh"]
    tot = a["site_eui"] * a["gfa_ft2"]
    a["elec_share"] = (elec_kbtu / tot).where(tot > 0)
    a["gas_share"] = (a["gas_kbtu"] / tot).where(tot > 0)
    return a


def quartile_label(x: pd.Series) -> pd.Series:
    """Top 25% = lowest EUI (best), Bottom 25% = highest EUI."""
    q1, q3 = x.quantile([0.25, 0.75])
    return pd.Series(np.select([x <= q1, x >= q3], ["Top 25%", "Bottom 25%"], "Middle 50%"), index=x.index)


# ---- load-shape KPIs (interval data) -------------------------------------------------
def load_shape_kpis(kw: pd.Series, open_mask: pd.Series) -> dict:
    """KPIs from an hourly kW series. ``open_mask`` marks occupied/operating hours."""
    kw = kw.dropna()
    open_mask = open_mask.reindex(kw.index).fillna(False).astype(bool)
    base = float(kw.quantile(0.05))
    peak = float(kw.quantile(0.99))
    mean = float(kw.mean())
    night = kw[(kw.index.hour >= 0) & (kw.index.hour < 5)].mean()
    day = kw[(kw.index.hour >= 10) & (kw.index.hour < 16) & (kw.index.dayofweek < 5)].mean()
    after_kwh = float(kw[~open_mask].sum())
    total_kwh = float(kw.sum())
    return {
        "base_load_kw": base,
        "peak_load_kw": peak,
        "mean_load_kw": mean,
        "load_factor": mean / peak,
        "night_day_ratio": float(night / day),
        "after_hours_share": after_kwh / total_kwh,
        "after_hours_kwh": after_kwh,
        "total_kwh": total_kwh,
    }


def performance_ratio(actual_kwh, poa_irradiance_kwh_m2, kwp):
    """IEC 61724 PR = final yield / reference yield."""
    yf = np.asarray(actual_kwh, float) / kwp
    yr = np.asarray(poa_irradiance_kwh_m2, float) / 1.0  # reference irradiance 1 kW/m²
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(yr > 0, yf / yr, np.nan)
