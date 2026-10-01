"""Degree days and ASHRAE Guideline 14 change-point regression.

Change-point models are fitted on *average daily consumption* in each month against the
month's mean outdoor temperature (the ASHRAE Inverse Modeling Toolkit formulation). Using
per-day consumption removes the month-length effect; the fitted balance points make this
equivalent to a variable-base degree-day model.

    2P : E = b0 + b1·T
    3PC: E = b0 + b1·(T − Tc)+                      cooling only
    3PH: E = b0 + b1·(Th − T)+                      heating only
    5P : E = b0 + b1·(Th − T)+ + b2·(T − Tc)+       heating + cooling, Th ≤ Tc
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .io import assumptions


def monthly_degree_days(daily: pd.DataFrame) -> pd.DataFrame:
    """Monthly HDD/CDD at base 18 °C and base 65 °F, plus mean temperature."""
    t = assumptions()["thresholds"]
    d = daily.copy()
    base_c = t["hdd_cdd_base_c"]
    tf = d["t_mean_c"] * 9 / 5 + 32
    d["hdd18"] = (base_c - d["t_mean_c"]).clip(lower=0)
    d["cdd18"] = (d["t_mean_c"] - base_c).clip(lower=0)
    d["hdd65"] = (t["hdd_cdd_base_f"] - tf).clip(lower=0)
    d["cdd65"] = (tf - t["hdd_cdd_base_f"]).clip(lower=0)
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    m = d.groupby("month").agg(
        t_mean_c=("t_mean_c", "mean"),
        hdd18=("hdd18", "sum"),
        cdd18=("cdd18", "sum"),
        hdd65=("hdd65", "sum"),
        cdd65=("cdd65", "sum"),
        days=("t_mean_c", "size"),
    )
    return m.reset_index()


def typical_year(wm: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Typical-year weather: calendar-month averages over ``years``."""
    sel = wm[wm["month"].dt.year.isin(years)]
    ty = sel.groupby(sel["month"].dt.month)[["t_mean_c", "hdd18", "cdd18", "hdd65", "cdd65"]].mean()
    ty.index.name = "cal_month"
    return ty.reset_index()


@dataclass
class CPModel:
    kind: str
    params: dict
    n: int
    p: int
    r2: float
    adj_r2: float
    cvrmse: float
    nmbe: float
    resid: np.ndarray = field(repr=False)

    def predict_daily(self, T: np.ndarray) -> np.ndarray:
        return _design(self.kind, np.asarray(T, float), self.params.get("Th"), self.params.get("Tc")) @ self.params["beta"]

    def predict(self, T: np.ndarray, days: np.ndarray) -> np.ndarray:
        return self.predict_daily(T) * np.asarray(days, float)

    @property
    def base_load_daily(self) -> float:
        return float(self.params["beta"][0])

    @property
    def cooling_slope(self) -> float:
        """kWh/day per °C above the cooling balance point (0 if no cooling term)."""
        b = self.params["beta"]
        return float(b[1]) if self.kind == "3PC" else float(b[2]) if self.kind == "5P" else 0.0

    @property
    def heating_slope(self) -> float:
        b = self.params["beta"]
        return float(b[1]) if self.kind == "3PH" else float(b[1]) if self.kind == "5P" else 0.0


def _design(kind: str, T: np.ndarray, Th: float | None, Tc: float | None) -> np.ndarray:
    one = np.ones_like(T)
    if kind == "2P":
        return np.column_stack([one, T])
    if kind == "3PC":
        return np.column_stack([one, np.clip(T - Tc, 0, None)])
    if kind == "3PH":
        return np.column_stack([one, np.clip(Th - T, 0, None)])
    if kind == "5P":
        return np.column_stack([one, np.clip(Th - T, 0, None), np.clip(T - Tc, 0, None)])
    raise ValueError(kind)


def _stats(y: np.ndarray, yhat: np.ndarray, p: int) -> tuple[float, float, float, float]:
    """R², adjusted R², ASHRAE-14 CV(RMSE) and NMBE (with n − p degrees of freedom)."""
    n = len(y)
    e = y - yhat
    ss_res = float(e @ e)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - p) if n > p else np.nan
    cvrmse = np.sqrt(ss_res / (n - p)) / y.mean()
    nmbe = e.sum() / ((n - p) * y.mean())
    return r2, adj, float(cvrmse), float(nmbe)


def fit_changepoint(T, y, days=None, kinds=("2P", "3PC", "3PH", "5P"), grid=None) -> dict[str, CPModel]:
    """Fit every model form by grid search over balance points; return valid fits.

    ``y`` is monthly consumption; ``days`` the days in each month. Fits are on per-day
    consumption; ASHRAE statistics are reported on monthly totals. Slopes must have the
    physical sign (heating/cooling slopes ≥ 0) or the form is discarded.
    """
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    days = np.ones_like(y) if days is None else np.asarray(days, float)
    yd = y / days
    grid = np.arange(np.floor(T.min()) + 1, np.ceil(T.max()) - 1, 0.5) if grid is None else grid
    fits: dict[str, CPModel] = {}
    for kind in kinds:
        best = None
        if kind == "2P":
            cands = [(None, None)]
        elif kind == "3PC":
            cands = [(None, tc) for tc in grid]
        elif kind == "3PH":
            cands = [(th, None) for th in grid]
        else:
            cands = [(th, tc) for th in grid for tc in grid if tc >= th]
        for th, tc in cands:
            X = _design(kind, T, th, tc)
            if np.linalg.matrix_rank(X) < X.shape[1]:
                continue
            beta, *_ = np.linalg.lstsq(X, yd, rcond=None)
            if kind in ("3PC", "3PH") and beta[1] < 0:
                continue
            if kind == "5P" and (beta[1] < 0 or beta[2] < 0):
                continue
            sse = float(((yd - X @ beta) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, th, tc, beta)
        if best is None:
            continue
        _, th, tc, beta = best
        # parameters: betas + each fitted change point
        p = len(beta) + (th is not None) + (tc is not None)
        yhat = (_design(kind, T, th, tc) @ beta) * days
        r2, adj, cv, nm = _stats(y, yhat, p)
        fits[kind] = CPModel(kind, {"beta": beta, "Th": th, "Tc": tc}, len(y), p, r2, adj, cv, nm, y - yhat)
    return fits


def best_model(fits: dict[str, CPModel]) -> CPModel:
    return max(fits.values(), key=lambda m: (np.nan_to_num(m.adj_r2, nan=-9)))


def ashrae_pass(m: CPModel) -> bool:
    t = assumptions()["thresholds"]
    return (m.cvrmse <= t["ashrae14_cvrmse_monthly"]) and (abs(m.nmbe) <= t["ashrae14_nmbe_monthly"])


def robust_changepoint(T, y, days, z_cut: float = 3.5, max_drop_share: float = 0.2):
    """Baseline fit with iterative outlier trimming.

    Fit, drop the single worst month if its robust z (MAD-based) exceeds ``z_cut``,
    refit, repeat, never dropping more than ``max_drop_share`` of months. One gross
    error (e.g. a catch-up bill) therefore can't drag good months out with it.
    ASHRAE-14 allows excluding documented bad data from a baseline; the dropped months
    are returned so they are reported, not hidden.
    """
    T, y, days = (np.asarray(v, float) for v in (T, y, days))
    keep = np.ones(len(y), bool)
    model = best_model(fit_changepoint(T, y, days))
    for _ in range(int(len(y) * max_drop_share)):
        resid = y - model.predict(T, days)
        r = resid[keep]
        mad = np.median(np.abs(r - np.median(r)))
        if mad <= 0:
            break
        z = np.where(keep, np.abs(resid - np.median(r)) / (1.4826 * mad), 0)
        worst = int(np.argmax(z))
        if z[worst] <= z_cut:
            break
        keep[worst] = False
        model = best_model(fit_changepoint(T[keep], y[keep], days[keep]))
    return model, keep
