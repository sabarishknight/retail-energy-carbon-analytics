"""Download every public dataset used by WattWise Retail into data/raw/.

    python scripts/download_data.py            # D1, D2, D3 (D4 is manual, see below)
    python scripts/download_data.py --force    # re-download even if files exist

D1  NYC LL84 annual disclosure (CY2022-latest) + the three older releases (CY2019-2021)
D2  NYC LL84 monthly data (electricity + gas per property per month)
D3  Open-Meteo historical daily temperature for New York City
D4  Dryad EMS/BMS + PV dataset (reduced_data.zip). Dryad puts file downloads behind a
    browser check, so this script only verifies the file is present and tells you
    where to put it if it is not.

Only the retail cohort is pulled from NYC Open Data (server-side filter through the
Socrata SODA API), so the raw footprint stays small. Column headers are written with the
human-readable names from each dataset's metadata, because the short API field names
are truncated differently in every yearly release.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SODA = "https://data.cityofnewyork.us"

RETAIL_TYPES = [
    "Retail Store",
    "Wholesale Club/Supercenter",
    "Supermarket/Grocery Store",
    "Enclosed Mall",
    "Strip Mall",
]

# dataset id -> (file name, calendar year if the release covers a single year)
ANNUAL = {
    "5zyy-y8am": ("d1_ll84_annual_cy2022_present.csv", None),
    "7x5e-2fxh": ("d1_ll84_annual_cy2021.csv", 2021),
    "usc3-8zwd": ("d1_ll84_annual_cy2020.csv", 2020),
    "wcm8-aq5w": ("d1_ll84_annual_cy2019.csv", 2019),
}
MONTHLY_ID = "fvp3-gcb2"

D4_FILE = RAW / "d4_ems" / "reduced_data.zip"
D4_PAGE = "https://datadryad.org/dataset/doi:10.5061/dryad.73n5tb363"

session = requests.Session()
session.headers["User-Agent"] = "wattwise-retail-portfolio/1.0 (research use)"


def _get(url: str, params: dict | None = None, tries: int = 4) -> requests.Response:
    for attempt in range(tries):
        try:
            r = session.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == tries - 1:
                raise
            wait = 2 ** attempt * 3
            print(f"   retry in {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def field_names(dataset_id: str) -> dict[str, str]:
    """Map short API field name -> human-readable column name."""
    meta = _get(f"{SODA}/api/views/{dataset_id}.json").json()
    return {c["fieldName"]: c["name"] for c in meta["columns"]}


def soda_pull(dataset_id: str, where: str, page: int = 50_000) -> pd.DataFrame:
    """Page through a SODA resource with a server-side filter."""
    frames, offset = [], 0
    while True:
        r = _get(
            f"{SODA}/resource/{dataset_id}.json",
            params={"$where": where, "$limit": page, "$offset": offset, "$order": ":id"},
        )
        rows = r.json()
        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        offset += len(rows)
        if len(rows) < page:
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def type_column(names: dict[str, str]) -> str:
    for f, n in names.items():
        if n == "Primary Property Type - Portfolio Manager-Calculated":
            return f
    raise KeyError("primary property type column not found")


def download_annual(force: bool) -> list[str]:
    ids: set[str] = set()
    types = ",".join(f"'{t}'" for t in RETAIL_TYPES)
    for ds, (fname, year) in ANNUAL.items():
        out = RAW / fname
        names = field_names(ds)
        if out.exists() and not force:
            print(f"D1  {fname}: exists, skipping")
        else:
            df = soda_pull(ds, f"{type_column(names)} in ({types})")
            df = df.rename(columns=names)
            if year is not None and "Calendar Year" not in df.columns:
                df.insert(0, "Calendar Year", year)
            df.to_csv(out, index=False)
            print(f"D1  {fname}: {len(df):,} retail rows, {df.shape[1]} columns")
        df = pd.read_csv(out, dtype=str, low_memory=False)
        id_col = next(c for c in df.columns if " ".join(c.lower().split()) == "property id")
        ids |= set(df[id_col].dropna())
    return sorted(ids)


def download_monthly(property_ids: list[str], force: bool, batch: int = 120) -> None:
    out = RAW / "d2_ll84_monthly_retail.csv"
    if out.exists() and not force:
        print("D2  monthly: exists, skipping")
        return
    names = field_names(MONTHLY_ID)
    frames = []
    for i in range(0, len(property_ids), batch):
        chunk = ",".join(f"'{p}'" for p in property_ids[i : i + batch])
        frames.append(soda_pull(MONTHLY_ID, f"property_id in ({chunk})"))
        print(f"D2  batch {i // batch + 1}/{-(-len(property_ids) // batch)}")
    df = pd.concat(frames, ignore_index=True).rename(columns=names)
    df.to_csv(out, index=False)
    print(f"D2  monthly: {len(df):,} rows for {df['Property Id'].nunique():,} properties")


def download_weather(force: bool) -> None:
    out = RAW / "d3_weather_nyc_daily.csv"
    if out.exists() and not force:
        print("D3  weather: exists, skipping")
        return
    today = dt.date.today()
    last_complete = today.replace(day=1) - dt.timedelta(days=1)
    r = _get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 40.71,
            "longitude": -74.01,
            "start_date": "2018-01-01",
            "end_date": last_complete.isoformat(),
            "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
            "timezone": "America/New_York",
        },
    )
    d = r.json()["daily"]
    df = pd.DataFrame(d).rename(columns={"time": "date"})
    df.to_csv(out, index=False)
    print(f"D3  weather: {len(df):,} days, {df['date'].min()} to {df['date'].max()}")


def check_d4() -> None:
    if D4_FILE.exists():
        print(f"D4  {D4_FILE.name}: present ({D4_FILE.stat().st_size / 1e6:.0f} MB)")
    else:
        print(
            f"D4  MISSING. Download 'reduced_data.zip' (~320 MB) by hand from\n"
            f"    {D4_PAGE}\n    and save it to {D4_FILE.parent}/"
        )


def write_manifest() -> None:
    manifest = {}
    for p in sorted(RAW.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            manifest[str(p.relative_to(RAW))] = {"sha256": h.hexdigest(), "bytes": p.stat().st_size}
    manifest["_downloaded_on"] = dt.date.today().isoformat()
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {RAW / 'MANIFEST.json'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    ids = download_annual(args.force)
    print(f"    {len(ids):,} distinct retail property IDs across all releases")
    download_monthly(ids, args.force)
    download_weather(args.force)
    check_d4()
    write_manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
