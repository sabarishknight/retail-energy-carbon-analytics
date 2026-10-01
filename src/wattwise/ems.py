"""Loader and helpers for D4, the Dryad EMS/BMS + PV dataset (reduced/aggregated version).

Engel et al. (2025), *A real-world energy management data set from a smart company
building*, Scientific Data, doi:10.5061/dryad.73n5tb363. Offenbach am Main, Germany,
2018-01-01 to 2024-01-01, timestamps in UTC.

Reduced data = one folder per resolution with electricity_{P|W}, heating_{P|W},
cooling_{P|W} and weather CSVs. Sign convention (paper): positive = consumption /
inflow, negative = production / outflow. Power P in W, energy W in kWh.
  electricity: total (grid draw at the transformers), PV, CHP
  heating:     total, CHP_heat, CHP_elec
  cooling:     total (thermal), cool_elec (chiller electricity)
  weather:     Igm (mean global horizontal irradiance, W/m²), Ta (°C)
"""
from __future__ import annotations

import io as _io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .io import RAW

ZIP = RAW / "d4_ems" / "reduced_data.zip"
TZ = "Europe/Berlin"

# PV capacity timeline (paper, Tables 1 & 10). Groups 1-2 (136 kWp, parking lot) from
# 2019-06-29; rooftop groups commissioned 2020-06-25 bring the total to 749 kWp.
# ASSUMPTION: group 3 was commissioned with groups 4-6 (the paper lists 4-6 explicitly).
PV_CAPACITY = [("2019-06-29", 136.0), ("2020-06-25", 749.0)]
COVID_LOCKDOWN = ("2020-03-16", "2021-01-17")
HEATING_MODERNISATION = "2023-06-01"


def available() -> bool:
    return ZIP.exists() or any((RAW / "d4_ems").glob("**/electricity_P.csv*"))


def _members(res: str) -> dict[str, str]:
    """Map 'electricity_P' -> member path inside the zip for one resolution."""
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
    out = {}
    for n in names:
        parts = n.replace("\\", "/").split("/")
        folder = "/".join(parts[:-1])
        m = re.match(r"(electricity|heating|cooling)_(P|W)\.csv(\.gz)?$|(weather)\.csv(\.gz)?$", parts[-1])
        if m and re.search(rf"(^|[_/]){re.escape(res)}($|/)", folder):
            key = f"{m.group(1)}_{m.group(2)}" if m.group(1) else "weather"
            out[key] = n
    return out


def list_contents() -> pd.DataFrame:
    with zipfile.ZipFile(ZIP) as z:
        return pd.DataFrame([(i.filename, i.file_size) for i in z.infolist()], columns=["member", "bytes"])


def load(kind: str, res: str = "1h") -> pd.DataFrame:
    """Load one reduced-data table, index = local time (Europe/Berlin)."""
    members = _members(res)
    if kind not in members:
        raise FileNotFoundError(f"{kind} at {res} not found in {ZIP.name}; found {sorted(members)}")
    with zipfile.ZipFile(ZIP) as z:
        raw = z.read(members[kind])
    comp = "gzip" if members[kind].endswith(".gz") else None
    df = pd.read_csv(_io.BytesIO(raw), compression=comp)
    tcol = next(c for c in df.columns if "time" in c.lower() or "date" in c.lower() or c.startswith("Unnamed"))
    idx = pd.to_datetime(df.pop(tcol), utc=True).dt.tz_convert(TZ)
    df.index = pd.DatetimeIndex(idx, name="time_local")
    df.columns = [c.split(".")[-1] if "." in c else c for c in df.columns]
    return df.astype("float32")


def pv_capacity_kwp(index: pd.DatetimeIndex) -> pd.Series:
    cap = pd.Series(0.0, index=index)
    for start, kwp in PV_CAPACITY:
        cap[index >= pd.Timestamp(start, tz=TZ)] = kwp
    return cap


def site_frame(res: str = "1h") -> pd.DataFrame:
    """Hourly site energy balance in kW (average power over the interval).

    grid_kw  > 0 import, < 0 export (net at the transformers)
    pv_kw, chp_kw  production as positive numbers
    load_kw = grid + PV + CHP  (everything consumed on site)
    """
    e = load("electricity_P", res) / 1000.0
    w = load("weather", res)
    c = load("cooling_P", res) / 1000.0
    cols = {k.lower(): k for k in e.columns}
    grid = e[cols["total"]]
    pv = -e[cols["pv"]] if e[cols["pv"]].mean() < 0 else e[cols["pv"]]
    chp = -e[cols["chp"]] if e[cols["chp"]].mean() < 0 else e[cols["chp"]]
    out = pd.DataFrame({"grid_kw": grid, "pv_kw": pv.clip(lower=0), "chp_kw": chp.clip(lower=0)})
    out["load_kw"] = out.grid_kw + out.pv_kw + out.chp_kw
    out["import_kw"] = out.grid_kw.clip(lower=0)
    out["export_kw"] = (-out.grid_kw).clip(lower=0)
    wc = {k.lower(): k for k in w.columns}
    out["ghi_w_m2"] = w[wc["igm"]].clip(lower=0)
    out["temp_c"] = w[wc["ta"]]
    cc = {k.lower(): k for k in c.columns}
    out["cooling_th_kw"] = c[cc["total"]].abs()
    out["cooling_elec_kw"] = c[cc["cool_elec"]].abs()
    out["pv_kwp"] = pv_capacity_kwp(out.index)
    return out


def operating_mask(index: pd.DatetimeIndex, start: int = 7, end: int = 19) -> pd.Series:
    """Occupied hours: Mon–Fri, start ≤ hour < end (local). Public holidays not removed."""
    return pd.Series((index.dayofweek < 5) & (index.hour >= start) & (index.hour < end), index=index)


def christmas_floor(load_kw: pd.Series) -> pd.Series:
    """Median load per year during the Christmas shutdown (27–30 Dec): an observed,
    achievable 'empty building' floor."""
    idx = load_kw.index
    sel = load_kw[(idx.month == 12) & (idx.day >= 27) & (idx.day <= 30)]
    return sel.groupby(sel.index.year).median()
