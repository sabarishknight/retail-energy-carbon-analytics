"""Unit tests for the core methods, on synthetic data with known answers."""
import numpy as np
import pandas as pd

from wattwise import anomalies, quality, savings
from wattwise.weather import fit_changepoint, best_model, robust_changepoint


def _signature(kind, n=36, seed=42):
    rng = np.random.default_rng(seed)
    T = rng.uniform(-2, 28, n)
    days = np.full(n, 30.0)
    base, cool, heat = 1000.0, 80.0, 50.0
    daily = base + cool * np.clip(T - 16, 0, None) + (heat * np.clip(10 - T, 0, None) if kind == "5P" else 0)
    return T, daily * days * (1 + rng.normal(0, 0.01, n)), days


def test_changepoint_recovers_cooling_model():
    T, y, d = _signature("3PC")
    m = best_model(fit_changepoint(T, y, d))
    assert m.kind in ("3PC", "5P")
    assert abs(m.params["Tc"] - 16) <= 1.0
    assert abs(m.cooling_slope - 80) / 80 < 0.1
    assert m.cvrmse < 0.05 and abs(m.nmbe) < 0.01


def test_changepoint_recovers_5p():
    T, y, d = _signature("5P")
    m = best_model(fit_changepoint(T, y, d))
    assert m.kind == "5P"
    assert m.heating_slope > 0 and m.cooling_slope > 0


def test_robust_fit_excludes_catch_up_bill():
    T, y, d = _signature("3PC")
    y = y.copy()
    y[5] *= 4                        # a catch-up bill
    _, keep = robust_changepoint(T, y, d)
    assert not keep[5] and keep.sum() == len(y) - 1


def test_cusum_detects_drift_not_noise():
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, 60)
    _, alarm_noise = anomalies.cusum(noise, 0.5, 4.0)
    drift = noise + np.r_[np.zeros(30), np.full(30, 1.5)]
    _, alarm_drift = anomalies.cusum(drift, 0.5, 4.0)
    assert alarm_noise.sum() == 0
    assert alarm_drift[30:].any() and not alarm_drift[:30].any()


def test_quality_flags_scale_error_and_bounds():
    a = pd.DataFrame({"property_id": ["1", "2", "3", "3"], "year": [2024] * 4,
                      "gfa_ft2": [100000, 100000, 500, 500], "site_eui": [80.0, 80000.0, 60.0, 60.0]})
    f = quality.annual_flags(a)
    assert f.dq_scale_error.tolist() == [False, True, True, True]
    assert f.dq_eui_out_of_bounds.tolist() == [False, True, False, False]
    assert f.dq_duplicate.tolist() == [False, False, False, True]


def test_npv_and_payback():
    assert abs(savings.npv(1000, 0, 0.08, 10) - 6710.08) < 0.1
    reg = savings.register([savings.ECM("x", "t", 1000, 500, 1.0, 1000, "High", "C", "s")])
    assert reg.simple_payback_yr.iloc[0] == 2.0
