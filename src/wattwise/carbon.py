"""Scope 1 (on-site gas) and location-based Scope 2 (grid electricity) emissions."""
from __future__ import annotations

import pandas as pd

from .io import assumptions


def factors() -> dict:
    ef = assumptions()["emission_factors"]
    return {
        "us_grid": ef["us_grid_nycw_kg_per_kwh"]["value"],
        "india_grid": ef["india_grid_kg_per_kwh"]["value"],
        "gas_per_mmbtu": ef["natural_gas_kg_co2e_per_mmbtu"]["value"],
        "diesel_per_l": ef["diesel_kg_co2_per_litre"]["value"],
        "de_grid": ef["germany_grid_kg_per_kwh"]["by_year"],
    }


def scope1_t(gas_kbtu) -> pd.Series:
    """Gas combustion, tCO2e. kBtu / 1000 = MMBtu."""
    return gas_kbtu / 1000 * factors()["gas_per_mmbtu"] / 1000


def scope2_t(elec_kwh, grid: str = "us_grid") -> pd.Series:
    return elec_kwh * factors()[grid] / 1000


def emissions_table(a: pd.DataFrame) -> pd.DataFrame:
    """Recompute Scope 1/2 with our factors next to the city-reported total."""
    out = a.copy()
    out["scope1_t"] = scope1_t(out["gas_kbtu"].fillna(0))
    out["scope2_us_t"] = scope2_t(out["elec_kwh"].fillna(0), "us_grid")
    out["scope2_india_t"] = scope2_t(out["elec_kwh"].fillna(0), "india_grid")
    out["total_us_t"] = out["scope1_t"] + out["scope2_us_t"]
    out["total_india_t"] = out["scope1_t"] + out["scope2_india_t"]
    return out


def tco2e(kwh: float, grid: str = "us_grid") -> float:
    return kwh * factors()[grid] / 1000
