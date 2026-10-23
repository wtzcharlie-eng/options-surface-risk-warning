"""Explicit competition reference protocols; never overwrite historical scores.

The supplied brief requires recall strictly before risk onset, but leaves its
maximum lead ambiguous. Both finite and unlimited recall are therefore exposed.
Precision here matches risk *starts*, a conservative declared implementation,
rather than claiming the brief specifies all interval intersection semantics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_surface.time_utils import epoch_seconds


def strict_alert_metrics(ts, levels, risk_idx, recall_window_min=120):
    t = epoch_seconds(ts)
    lv = np.asarray(levels)
    raw_idx = np.asarray(risk_idx)
    idx = raw_idx.astype(np.int64)
    if not np.array_equal(raw_idx, idx):
        raise ValueError("risk indices must be integers")
    if len(t) != len(lv) or (len(t) > 1 and np.any(np.diff(t) <= 0)):
        raise ValueError("timestamps must be unique, ordered and aligned with levels")
    if np.any((lv < 0) | (lv > 3) | (lv != np.floor(lv))):
        raise ValueError("levels must be integers in 0..3")
    if np.any((idx < 0) | (idx >= len(t))) or len(np.unique(idx)) != len(idx):
        raise ValueError("invalid risk indices")
    if recall_window_min is not None and recall_window_min <= 0:
        raise ValueError("recall window must be positive or None")
    a, r = t[lv >= 2], np.sort(t[idx])
    # (alert, alert + 120min]; same-time detection is never advance warning.
    pos = np.searchsorted(r, a, side="right")
    hits = sum(k < len(r) and r[k] <= v + 7200 for k, v in zip(pos, a))
    leads = []
    for start in r:
        k = 0 if recall_window_min is None else np.searchsorted(
            a, start - recall_window_min * 60, side="left")
        if k < len(a) and a[k] < start:
            leads.append((start - a[k]) / 60)
    return dict(n_alert=len(a), n_hit_alert=int(hits), n_risk=len(r),
                n_recalled=len(leads), precision=hits / len(a) if len(a) else 0.,
                recall=len(leads) / len(r) if len(r) else 0.,
                avg_lead_min=float(np.mean(leads)) if leads else 0.)


def combine_metrics(rows):
    out = {k: sum(r[k] for r in rows) for k in
           ("n_alert", "n_hit_alert", "n_risk", "n_recalled")}
    out.update(precision=out["n_hit_alert"] / out["n_alert"] if out["n_alert"] else 0.,
               recall=out["n_recalled"] / out["n_risk"] if out["n_risk"] else 0.,
               avg_lead_min=sum(r["avg_lead_min"] * r["n_recalled"] for r in rows)
               / out["n_recalled"] if out["n_recalled"] else 0.)
    out["three_targets_met"] = bool(out["precision"] >= .5 and out["recall"] >= .6
                                    and out["avg_lead_min"] >= 30)
    return out


def first_alert_cvar(prices: pd.DataFrame, alerts: pd.DataFrame):
    """Research simulator for the brief's first-alert, permanent half-position rule.

    prices: unique ordered trading_date (actual exchange trading day), open, close.
    alerts: decision_trading_date, level. Overnight sessions MUST be mapped upstream
    with an exchange calendar. Prices must be actual same-contract unadjusted prices
    or a documented continuous contract series. First row is the initial close.
    P&L is normalized by that fixed initial notional (one hypothetical lot).
    CVaR95 = exact probability-mass average of the worst 5% daily losses.
    No fees/slippage/margin effects. Caller must certify calendar/data provenance.
    """
    required = {"trading_date", "open", "close"}
    if not required.issubset(prices) or not {"decision_trading_date", "level"}.issubset(alerts):
        raise ValueError("missing price/alert columns")
    p = prices.copy()
    dates = pd.to_datetime(p.trading_date, errors="raise").dt.normalize()
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("trading dates must be unique and ordered")
    if len(dates) < 2 or dates.iloc[-1] < dates.iloc[0] + pd.DateOffset(months=3):
        raise ValueError("at least three continuous calendar months are required")
    values = p[["open", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("actual positive opens and closes are required")
    a = alerts.copy()
    ad = pd.to_datetime(a.decision_trading_date, errors="raise").dt.normalize()
    levels = np.asarray(a.level, dtype=float)
    if ad.isna().any() or np.any(~np.isfinite(levels) | (levels < 0) | (levels > 3)
                               | (levels != np.floor(levels))):
        raise ValueError("invalid alerts")
    if not ad.isin(dates).all():
        raise ValueError("alert trading day missing from price calendar")
    first = ad[levels >= 2].min() if (levels >= 2).any() else pd.NaT
    cut = int(dates.searchsorted(first, side="right")) if pd.notna(first) else len(p)
    base = np.diff(values[:, 1]) / values[0, 1]
    strategy = base.copy()
    if cut < len(p):
        strategy[cut - 1] = ((values[cut, 0] - values[cut - 1, 1])
                             + .5 * (values[cut, 1] - values[cut, 0])) / values[0, 1]
        strategy[cut:] *= .5

    def tail_loss(returns):
        losses = np.sort(-returns)[::-1]
        mass = .05 * len(losses)
        whole = int(np.floor(mass))
        total = losses[:whole].sum()
        if whole < len(losses):
            total += (mass - whole) * losses[whole]
        return float(total / mass)

    sides = {}
    for name, direction in (("long", 1), ("short", -1)):
        b, s = tail_loss(direction * base), tail_loss(direction * strategy)
        improvement = (b - s) / b if b > 0 else None
        if pd.isna(first):
            improvement = 0.
        sides[name] = dict(baseline_cvar95=b, alert_cvar95=s, improvement=improvement)
    final = None if any(v["improvement"] is None for v in sides.values()) else float(
        np.mean([v["improvement"] for v in sides.values()]))
    return dict(protocol="first-alert-next-trading-open-permanent-half-v1",
                n_daily_returns=len(base), first_alert_trading_date=None if pd.isna(first)
                else first.strftime("%Y-%m-%d"),
                execution_date=dates.iloc[cut].strftime("%Y-%m-%d") if cut < len(p) else None,
                sides=sides, mean_improvement=final,
                data_provenance_verified=False,
                note="算法复算不验证真实开盘价、交易日历、连续合约处理或交易成本；须另行提供来源证据。")
