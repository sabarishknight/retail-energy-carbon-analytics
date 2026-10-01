"""Monthly consumption forecasting and backtesting.

Three forecasters, all evaluated on the same hold-out year:
  * seasonal naive          — same calendar month of the most recent prior year
  * change-point + weather  — the store's ASHRAE-14 model (robust fit), driven by weather
  * LightGBM (pooled)       — one gradient-boosted model across all eligible stores,
                              features: calendar month, temperature, degree days,
                              store size and the store's own value 12 months earlier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import assumptions
from .weather import robust_changepoint


def mape(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ok = y > 0
    return float(np.mean(np.abs((y[ok] - yhat[ok]) / y[ok])) * 100)


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame, value: str) -> np.ndarray:
    last = train[train["month"].dt.year == train["month"].dt.year.max()]
    by_m = last.set_index(last["month"].dt.month)[value]
    return test["month"].dt.month.map(by_m).to_numpy(float)


def changepoint_forecast(train: pd.DataFrame, test: pd.DataFrame, value: str):
    m, _ = robust_changepoint(train["t_mean_c"], train[value], train["days"])
    return m.predict(test["t_mean_c"], test["days"]), m


FEATURES = ["cal_month", "t_mean_c", "hdd18", "cdd18", "days", "log_gfa", "lag12_per_day"]


def add_features(panel: pd.DataFrame, value: str) -> pd.DataFrame:
    p = panel.sort_values(["property_id", "month"]).copy()
    p["cal_month"] = p["month"].dt.month
    p["log_gfa"] = np.log(p["gfa_ft2"])
    # lag 12 by calendar date (not position) so the 2022 publication gap is handled
    lag = p[["property_id", "month", value, "days"]].copy()
    lag["month"] = lag["month"] + pd.DateOffset(years=1)
    lag["lag12_per_day"] = lag[value] / lag["days"]
    p = p.merge(lag[["property_id", "month", "lag12_per_day"]], on=["property_id", "month"], how="left")
    p["target_per_day"] = p[value] / p["days"]
    return p


def lightgbm_backtest(panel: pd.DataFrame, value: str, test_year: int, train_years: list[int]):
    """Train pooled LightGBM on ``train_years`` (all stores) and predict ``test_year``.

    The target is consumption per day relative to the store's lag-12 value would leak
    nothing (lag-12 is a full year before the forecast month), so the model sees only
    information that exists at forecast time.
    """
    import lightgbm as lgb

    seed = assumptions()["seed"]
    f = add_features(panel, value)
    tr = f[f["month"].dt.year.isin(train_years)].dropna(subset=["target_per_day"])
    te = f[f["month"].dt.year == test_year].dropna(subset=["target_per_day"])
    model = lgb.LGBMRegressor(
        n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1,
    )
    model.fit(tr[FEATURES], tr["target_per_day"])
    te = te.assign(pred=model.predict(te[FEATURES]) * te["days"])
    return te, model
