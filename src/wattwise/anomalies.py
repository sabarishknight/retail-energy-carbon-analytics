"""Layered anomaly detection on monthly utility bills and facility-ticket generation.

Layer 1  rules      zero, spike (>3× rolling median), flatline, sudden step change
Layer 2  residuals  actual − weather-expected (robust change-point fit), robust z-score
Layer 3  CUSUM      tabular CUSUM on standardised residuals to catch slow upward drift

Tickets separate two very different things:
  * **Energy waste**: consumption above what weather explains (month or drift). This
    is the only part counted as avoidable kWh / $.
  * **Billing / meter check**: zero reads, flatlines (estimates), spikes vs neighbours
    (catch-up bills), or bills far *below* expected. Money is at stake, but it is
    a data problem rather than energy that could be saved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import assumptions
from .weather import robust_changepoint


def rule_flags(s: pd.Series, ratio: pd.Series | None = None) -> pd.DataFrame:
    """Rule-based flags on one store's monthly series (index = month).

    The step-change rule runs on ``ratio`` (actual ÷ weather-expected) when given, so a
    normal seasonal ramp is not mistaken for a level shift.
    """
    t = assumptions()["thresholds"]
    roll = s.rolling(7, center=True, min_periods=4).median()
    key = (s != s.shift()).cumsum()
    run = s.groupby(key).transform("size")
    r = s if ratio is None else ratio
    prev6 = r.shift(1).rolling(6, min_periods=4).median()
    next3 = r[::-1].rolling(3, min_periods=3).median()[::-1]
    change = (next3 / prev6 - 1) if ratio is None else (next3 - prev6)
    step = (change.abs() > t["step_change_ratio"]).fillna(False)
    step = step & ~step.shift(1, fill_value=False)        # first month of a sustained shift only
    return pd.DataFrame(
        {
            "rule_zero": s == 0,
            "rule_spike": s > t["spike_ratio"] * roll,
            "rule_flatline": (run >= t["flatline_months"]) & (s != 0),
            "rule_step": step,
        },
        index=s.index,
    )


def cusum(z: np.ndarray, k: float, h: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided upper tabular CUSUM on standardised values. Returns S+ and alarm flags."""
    sp = np.zeros(len(z))
    for i, v in enumerate(z):
        sp[i] = max(0.0, (sp[i - 1] if i else 0.0) + v - k)
    return sp, sp > h


def detect(store: pd.DataFrame, value: str, weather: pd.DataFrame):
    """Run all three layers for one store and one metric; returns (flags frame, model)."""
    t = assumptions()["thresholds"]
    d = store[["month", value]].merge(weather, on="month", how="left").dropna(subset=[value, "t_mean_c"])
    d = d.sort_values("month").reset_index(drop=True)
    model, _ = robust_changepoint(d["t_mean_c"], d[value], d["days"])
    d["expected"] = model.predict(d["t_mean_c"], d["days"])
    d["resid"] = d[value] - d["expected"]
    mad = np.median(np.abs(d["resid"] - np.median(d["resid"])))
    sigma = 1.4826 * mad if mad > 0 else d["resid"].std(ddof=1)
    d["sigma"] = sigma
    d["z"] = d["resid"] / sigma
    d["flag_resid"] = d["z"].abs() > t["anomaly_z"]
    sp, alarm = cusum(d["z"].clip(upper=t["anomaly_z"]).to_numpy(), t["cusum_k_sigma"], t["cusum_h_sigma"])
    d["cusum_pos"], d["flag_cusum"] = sp, alarm
    d["cusum_run"] = (alarm & ~np.r_[False, alarm[:-1]]).cumsum() * alarm   # 0 = no alarm, 1.. = run id
    ser = d.set_index("month")
    # residual as a share of the store's typical month (stable when summer gas ≈ 0)
    rules = rule_flags(ser[value], 1 + ser["resid"] / ser["expected"].mean()).reset_index(drop=True)
    return pd.concat([d, rules], axis=1), model


def likely_cause(month: int, metric: str, kind: str, row: pd.Series | None = None) -> str:
    if kind == "billing":
        if row is not None and row.get("rule_zero"):
            return "Zero read: meter outage or missing bill, request an actual read"
        if row is not None and row.get("rule_flatline"):
            return "Identical value repeated: estimated bills, request actual meter reads"
        if row is not None and row.get("rule_spike"):
            return "Spike vs neighbouring months: likely catch-up or mis-keyed bill, verify before paying"
        return "Far below expected: partial closure, meter fault or under-billing, verify"
    if kind == "step":
        return "Sustained level shift: confirm cause (new equipment, trading hours, meter change)"
    if kind == "drift":
        return ("Gradual upward drift: base-load creep (lighting, refrigeration, plant left on), degrading HVAC"
                if metric == "electricity" else "Gradual upward drift: boiler efficiency loss or controls override")
    if metric == "electricity":
        if month in (6, 7, 8, 9):
            return "Summer excess beyond weather: HVAC schedule, cooling setpoint or chiller efficiency"
        if month in (12, 1, 2):
            return "Winter electric excess: AHU fans / electric heat out of hours, extended holiday trading"
        return "Shoulder-season excess: base load (lighting, refrigeration, after-hours plant)"
    if month in (11, 12, 1, 2, 3):
        return "Heating excess beyond HDD: boiler scheduling, setpoint, door / air-curtain losses"
    return "Off-season gas use: boiler not isolated, DHW or reheat running"


def tickets(flags: pd.DataFrame, store_name: str, metric: str, unit: str, price: float, kwh_per_unit: float) -> pd.DataFrame:
    """Facility tickets: one per flagged month, plus one per CUSUM drift run."""
    t = assumptions()["thresholds"]
    val = flags.columns[1]
    rows = []
    month_ticketed = set()
    for i, r in flags.iterrows():
        billing = r.rule_zero or r.rule_flatline or r.rule_spike or (r.flag_resid and r.z < 0)
        waste = r.flag_resid and r.z > 0 and not r.rule_spike
        if not (billing or waste or r.rule_step):
            continue
        kind = "billing" if billing else "waste" if waste else "step"
        gap = r[val] - r.expected
        avoidable = max(gap, 0.0) if kind == "waste" else 0.0
        layers = [n for n, c in [("rule:zero", r.rule_zero), ("rule:spike", r.rule_spike), ("rule:flatline", r.rule_flatline),
                                 ("rule:step", r.rule_step), (f"residual z={r.z:+.1f}", r.flag_resid)] if c]
        rows.append({
            "store": store_name, "month": r.month.strftime("%Y-%m"), "metric": metric, "unit": unit,
            "ticket_type": {"billing": "Billing / meter check", "waste": "Energy waste", "step": "Level shift"}[kind],
            "expected": round(r.expected), "actual": round(r[val]),
            "avoidable_kwh": round(avoidable * kwh_per_unit), "avoidable_usd": round(avoidable * price),
            "at_stake_usd": round(abs(gap) * price),
            "z": round(r.z, 2), "detected_by": ", ".join(layers),
            "likely_cause": likely_cause(r.month.month, metric, kind, r),
        })
        month_ticketed.add(i)
    # one drift ticket per upward CUSUM run; avoidable = positive residual in the run's
    # months not already counted by a month-level ticket
    for run_id, g in flags[flags.cusum_run > 0].groupby("cusum_run"):
        if g.resid.sum() <= 0:
            continue
        extra = g.loc[~g.index.isin(month_ticketed), "resid"].clip(lower=0).sum()
        rows.append({
            "store": store_name, "month": f"{g.month.min():%Y-%m} → {g.month.max():%Y-%m}", "metric": metric, "unit": unit,
            "ticket_type": "Energy waste (drift)",
            "expected": round(g.expected.sum()), "actual": round(g[val].sum()),
            "avoidable_kwh": round(extra * kwh_per_unit), "avoidable_usd": round(extra * price),
            "at_stake_usd": round(g.resid.clip(lower=0).sum() * price),
            "z": round(g.z.mean(), 2), "detected_by": f"CUSUM ({len(g)} months)",
            "likely_cause": likely_cause(g.month.iloc[0].month, metric, "drift"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pr = t["ticket_priority_usd"]
    out["priority"] = np.select([out.at_stake_usd >= pr["high"], out.at_stake_usd >= pr["medium"]], ["High", "Medium"], "Low")
    return out
