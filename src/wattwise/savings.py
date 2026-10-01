"""Energy Conservation Measure (ECM) register: payback, NPV, abatement cost, MACC."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .io import assumptions


@dataclass
class ECM:
    measure: str
    sized_from: str           # which analysis produced the kWh figure
    kwh_yr: float
    cost_yr_usd: float
    tco2e_yr: float
    capex_usd: float
    confidence: str           # High / Medium / Low
    ipmvp: str                # verification option
    scope: str                # where it applies
    capex_basis: str = ""


def npv(annual_saving: float, capex: float, rate: float | None = None, years: int | None = None) -> float:
    fin = assumptions()["finance"]
    rate = fin["discount_rate"] if rate is None else rate
    years = fin["horizon_years"] if years is None else years
    annuity = (1 - (1 + rate) ** -years) / rate
    return annual_saving * annuity - capex


def register(ecms: list[ECM]) -> pd.DataFrame:
    fin = assumptions()["finance"]
    df = pd.DataFrame([asdict(e) for e in ecms])
    df["simple_payback_yr"] = np.where(df["cost_yr_usd"] > 0, df["capex_usd"] / df["cost_yr_usd"], np.inf)
    df["npv_10y_usd"] = [npv(s, c) for s, c in zip(df["cost_yr_usd"], df["capex_usd"])]
    annuity = (1 - (1 + fin["discount_rate"]) ** -fin["horizon_years"]) / fin["discount_rate"]
    # marginal abatement cost: levelised net cost per tonne avoided (negative = saves money)
    df["abatement_cost_usd_per_t"] = (df["capex_usd"] / annuity - df["cost_yr_usd"]) / df["tco2e_yr"]
    return df.sort_values("abatement_cost_usd_per_t").reset_index(drop=True)
