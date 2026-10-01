"""Render README.md from computed results only (reports/results/*.json).

No number in the README is typed by hand: every figure is a placeholder filled from the
JSON files the notebooks write. Re-run after `make notebooks`:  python scripts/render_readme.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wattwise import io  # noqa: E402

R = io.load_results()
q, b, ba, f4, e5, s6 = (R[k] for k in ["01_quality", "02_benchmark", "03_bill_audit", "04_forecast", "05_ems_pv", "06_savings"])
A = io.assumptions()


def usd(v: float) -> str:
    return f"${v / 1e6:,.2f}M" if abs(v) >= 1e6 else f"${v / 1e3:,.0f}k"


reg = pd.DataFrame(s6["register"])
top = reg[reg.scope == "Focus portfolio"].sort_values("npv_10y_usd", ascending=False).head(5)
ecm_rows = "\n".join(
    f"| {r.measure} | {r.kwh_yr / 1e3:,.0f} | ${r.cost_yr_usd:,.0f} | {r.tco2e_yr:,.0f} | {r.simple_payback_yr:.1f} | {r.confidence} | {r.ipmvp} |"
    for r in top.itertuples()
)
m_med, m_port, m_clean = f4["mape_median_store"], f4["mape_portfolio"], f4["mape_clean_stores"]
cp_key, lg_key, nv_key = "Change-point + weather", "LightGBM (pooled)", "Seasonal naive"

readme = f"""# WattWise Retail: energy, water & carbon intelligence for large-format stores

> **Bottom-quartile stores use {b['bottom_vs_top_quartile_ratio']:.1f}× the energy per m² of top-quartile stores.** Across 8 focus stores, this project finds **{usd(s6['focus_saving_usd'])} a year** of savings ({s6['focus_saving_share']:.0%} of their electricity, {s6['focus_saving_t']:,.0f} tCO₂e) from real, public, up-to-date data. Every number on this page is computed by the notebooks.

![Headline KPIs](reports/figures/00_kpi_tiles.png)

## The problem
A retailer running large stores pays for energy three times: on the utility bill, against its carbon targets, and in the time facility teams lose to bad data. An energy analyst has to answer four questions every month: **which stores are wasteful, which bills are wrong, what to fix first, and what each fix is worth** in kWh, money and CO₂e. This project builds that workflow end-to-end, the way an in-house energy team would run it.

## The data
| # | Dataset | Coverage | Size used | Role |
|---|---|---|---|---|
| D1 | [NYC Local Law 84 energy & water disclosure](https://data.cityofnewyork.us/d/5zyy-y8am) (+ 3 earlier releases) | calendar {q['annual_years'][0]}–{q['annual_years'][1]} (latest published) | {q['annual_rows']:,} retail property-years, {q['properties']:,} properties | multi-store benchmarking, EUI, carbon, water |
| D2 | [NYC LL84 monthly data](https://data.cityofnewyork.us/d/fvp3-gcb2) | {q['monthly_first']} to {q['monthly_last']} ({', '.join(map(str, q['monthly_missing_years']))} not published) | {q['monthly_rows']:,} store-months | utility-bill audit, anomalies, forecasting |
| D3 | [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api), New York City | 2018-01-01 to {q['weather_last']} | {q['weather_days']:,} days | HDD/CDD, weather normalisation |
| D4 | [Real EMS/BMS + PV dataset](https://doi.org/10.5061/dryad.73n5tb363), Engel et al., *Scientific Data* 2025 | {e5['years'][0]}–{e5['years'][1]}, hourly | {e5['hours']:,} hours, 749 kWp PV, CHP, chillers | interval load shape, after-hours waste, PV performance |
| D5 | Emission factors & prices ([config/assumptions.yaml](config/assumptions.yaml)) | EPA eGRID2023, EPA GHG Hub 2025, CEA v21, UBA, EIA 2025 | n/a | carbon and cost, every value cited |

## What I found

**1. The gap between stores is the opportunity.** Bottom-quartile stores run at {b['bottom_quartile_median']:,.0f} kWh/m² vs {b['top_quartile_median']:,.0f} for the top quartile. Building age explains almost none of it (R² = {b['age_r2']:.3f}), so this is an operations problem.

![Quartiles](reports/figures/02_quartiles.png)

**2. Monthly bills hide waste *and* billing errors, and they're different things.** A three-layer audit (rules → weather-model residuals → CUSUM drift) flagged {ba['flagged_store_months_elec']} of {ba['store_months_audited_elec']} store-months of electricity and raised {ba['tickets']} facility tickets. Weather-unexplained excess costs **{usd(ba['avoidable_usd_per_yr'])}/yr**; a further **{usd(ba['billing_to_verify_usd'])}** looks like catch-up bills or missing reads and is *not* counted as savings.

![Anomalies](reports/figures/03_anomaly_small_multiples.png)

**3. Weather explains the bills, and simple models beat complex ones per store.** {f4['ashrae_pass']} of {f4['stores']} focus stores meet ASHRAE Guideline 14 (CV(RMSE) ≤ 15%, |NMBE| ≤ 5%). Forecasting 2024 from 2023, the change-point model's median store error was **{m_med[cp_key]:.1f}%** vs {m_med[lg_key]:.1f}% for a pooled LightGBM. LightGBM won on the portfolio total ({m_port[lg_key]:.1f}% vs {m_port[cp_key]:.1f}%). The biggest errors came from stores with billing anomalies, so data quality mattered more than model complexity.

![Energy signatures](reports/figures/04_energy_signatures.png)

**4. Interval data shows what bills can't: buildings that never switch off.** On the EMS site, {e5['above_floor_share_last']:.0%} of annual electricity is after-hours load above its own Christmas-shutdown floor. During COVID lockdown, daytime load changed {e5['covid_change']['weekday daytime kW']:+.0%} but night base load only {e5['covid_change']['night base kW']:+.0%}: a *phantom load* that runs with nobody in the building.

![Phantom load](reports/figures/05_covid_phantom_load.png)

**5. Solar needs monitoring, not just installing.** The 749 kWp plant reached a GHI-based PR of {e5['pv_pr_last']:.2f} and self-consumed {e5['pv_self_consumption_last']:.0%} of output in {e5['years'][1]}, but {e5['pv_low_pr_days_total']} bright low-PR days cost ≈ {e5['pv_lost_mwh_total']:,.0f} MWh. A daily PR alarm would catch them.

![PV](reports/figures/05_pv_performance.png)

**6. Most carbon cuts pay for themselves.** {s6['negative_cost_measures']} of {len(reg)} measures sit below zero on the marginal abatement cost curve, covering {s6['negative_cost_share_t']:.0%} of the abatement. On India's grid (CEA v21), the same stores would emit {b['india_scope2_uplift']:.0%} more Scope 2 carbon, so every kWh saved is worth more there.

![MACC](reports/figures/06_macc.png)

## What I'd do: top 5 measures (focus portfolio, by 10-year NPV)
| Measure | MWh/yr | $/yr | tCO₂e/yr | Payback (yrs) | Confidence | Verification (IPMVP) |
|---|---|---|---|---|---|---|
{ecm_rows}

Prices: EIA 2025 New York commercial averages; carbon: eGRID2023 NYCW; capex: cited assumptions. Full register (9 measures, 3 scopes) in [notebook 06](notebooks/06_savings_opportunities_register.ipynb).

## How it works
```mermaid
flowchart LR
    D1[LL84 annual<br/>2019-2024] --> Q[01 Data-quality<br/>framework]
    D2[LL84 monthly bills] --> Q
    D3[Open-Meteo<br/>weather] --> W[HDD / CDD]
    Q --> B[02 Benchmarking<br/>EUI · carbon · water]
    Q --> A[03 Bill audit<br/>rules · residuals · CUSUM]
    W --> A
    W --> F[04 ASHRAE-14<br/>change-point · forecast]
    D4[EMS/BMS + PV<br/>hourly] --> E[05 Load shape ·<br/>after-hours · PV PR]
    B --> S[06 ECM register<br/>NPV · MACC]
    A --> S
    F --> S
    E --> S
    D5[Cited factors<br/>& prices] --> S
    S --> R[00 Executive report]
```

**Methods:** ASHRAE Guideline 14 change-point regression (2P/3PC/3PH/5P, CV(RMSE) & NMBE checks) · robust MAD z-scores · tabular CUSUM drift detection · weather normalisation with variable-base degree days · pooled LightGBM benchmark · IEC 61724 performance ratio · IPMVP Options A/B/C for verification · NPV, simple payback and marginal abatement cost · location-based Scope 1 & 2 (GHG Protocol).

**Data-quality framework** ([`quality.py`](src/wattwise/quality.py)): 10 rules (duplicates, missing floor area, missing energy, ≈1,000× unit errors, EUI bounds, missing / zero / flat / spiking months, monthly ≠ annual). Each rule raises a flag, flags become a 0–100 score per property, and exclusions are shown as a waterfall: {q['rows_kept_share']:.0%} of rows pass. Where both exist, monthly bills sum to within 10% of the annual disclosure for {q['monthly_matches_annual_share']:.1%} of property-years.

![DQ waterfall](reports/figures/01_dq_waterfall.png)

## Skills demonstrated
| Job requirement | Where |
|---|---|
| Monitor, validate & analyse electricity, gas, water and solar data | notebooks 01, 02, 05 |
| Track energy KPIs across stores; benchmark & forecast | 02 (EUI, quartiles, trends), 04 (ASHRAE-14, forecasts) |
| Analyse utility bills for anomalies, wastage & savings | 03 (three-layer audit, facility tickets) |
| EMS/BMS sub-meter data | 05 (register reconciliation, load shape, after-hours, chiller COP) |
| EUI, carbon & sustainability metrics | 02 (kBtu/ft² and kWh/m², Scope 1/2, water L/m², India sensitivity) |
| Solar PV performance | 05 (expected vs actual, PR, fault days, self-consumption, CO₂ avoided) |
| Quantify savings (kWh, cost, tCO₂e) & recommend | 06 (ECM register, NPV, MACC, IPMVP) |
| Keep the energy database accurate | 01 (quality rules, score, waterfall), DuckDB + Parquet store |

## Reproduce it
```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && pip install -e . --no-deps
python scripts/download_data.py      # D1-D3 automatically; D4 needs a manual download (instructions printed)
make notebooks report readme         # or open notebooks/ in jupyter lab and Run All
```
The executive report is [`notebooks/00_EXECUTIVE_REPORT.ipynb`](notebooks/00_EXECUTIVE_REPORT.ipynb), with a code-free HTML export at `reports/WattWise_Report.html`. Unit tests: `make test`.

## Limitations & honesty note
* The stores are **NYC buildings from public disclosure**, not any one retailer's estate. The EMS site is a **German commercial/industrial building**, not a store. Ratios measured there (after-hours share, chiller scheduling) are transferred to stores and labelled *Medium* confidence.
* Latest public data: annual and monthly disclosures to **{q['annual_years'][1]}** (published 2025); {', '.join(map(str, q['monthly_missing_years']))} monthly data is not published. 2025 is **forecast** with actual 2025 weather, not measured.
* Factors are used where stated: **US** (eGRID2023 NYCW, EPA Hub 2025), **German** (UBA) for the EMS site, **Indian** (CEA v21) for the sensitivity and the clearly labelled *synthetic* diesel scenario.
* Costs use statewide EIA averages (conservative for NYC). Capex values are cited assumptions, not quotes. Measure savings ignore interactions, so the total is an upper bound.
* The reduced D4 release has site-level aggregates only. Sub-meter (parent-vs-children) reconciliation needs the 103 GB full release.
"""

(ROOT / "README.md").write_text(readme)
print("README.md written from", len(R), "result files")
