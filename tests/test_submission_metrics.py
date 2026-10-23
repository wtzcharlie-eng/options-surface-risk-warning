import numpy as np
import pandas as pd
import pytest
from vol_surface.submission_metrics import strict_alert_metrics, first_alert_cvar


def test_same_time_is_not_advance_warning_and_end_is_inclusive():
    ts = ["20260101090000", "20260101110000", "20260101111500"]
    m = strict_alert_metrics(ts, [2, 2, 0], [1])
    assert m["n_hit_alert"] == 1 and m["n_recalled"] == 1
    assert m["avg_lead_min"] == 120
    m = strict_alert_metrics(ts, [0, 2, 0], [1])
    assert m["precision"] == m["recall"] == 0


def test_unlimited_recall_does_not_change_two_hour_precision():
    ts = ["20260101090000", "20260102110000"]
    finite = strict_alert_metrics(ts, [2, 0], [1])
    unlimited = strict_alert_metrics(ts, [2, 0], [1], None)
    assert finite["recall"] == 0 and unlimited["recall"] == 1
    assert unlimited["precision"] == 0 and unlimited["avg_lead_min"] == 1560


def prices():
    dates = pd.bdate_range("2025-01-01", "2025-05-01")
    close = 100 + np.sin(np.arange(len(dates))) * 10
    return pd.DataFrame(dict(trading_date=dates, open=close + 2, close=close))


def test_no_alert_zero_and_only_first_alert_changes_exposure():
    p = prices()
    empty = pd.DataFrame(columns=["decision_trading_date", "level"])
    assert first_alert_cvar(p, empty)["mean_improvement"] == 0
    a = pd.DataFrame(dict(decision_trading_date=[p.trading_date.iloc[4]], level=[2]))
    one = first_alert_cvar(p, a)
    a.loc[1] = [p.trading_date.iloc[12], 3]
    assert one == first_alert_cvar(p, a)
    assert one["execution_date"] == "2025-01-08"


def test_gap_is_full_position_until_next_open_then_half_forever():
    p = prices()
    # Every post-cut close increment and gap is half exposed, except the initial
    # execution gap. Compare to an independently constructed daily loss sequence.
    a = pd.DataFrame(dict(decision_trading_date=[p.trading_date.iloc[0]], level=[2]))
    p.loc[1, "open"] = p.close.iloc[0] - 30
    out = first_alert_cvar(p, a)
    b = np.diff(p.close) / p.close.iloc[0]
    s = b * .5
    s[0] += .5 * (p.open.iloc[1] - p.close.iloc[0]) / p.close.iloc[0]
    n = .05 * len(b)
    def independent_tail(v):
        ranked = sorted(-v, reverse=True)
        return (sum(ranked[:int(n)]) + (n-int(n))*ranked[int(n)]) / n
    assert out["sides"]["long"]["alert_cvar95"] == pytest.approx(independent_tail(s))
    assert out["sides"]["short"]["alert_cvar95"] == pytest.approx(independent_tail(-s))


def test_cvar_rejects_missing_open_short_period_or_calendar_mismatch():
    p = prices()
    a = pd.DataFrame(dict(decision_trading_date=["2025-01-04"], level=[2]))
    with pytest.raises(ValueError): first_alert_cvar(p, a)
    a = pd.DataFrame(dict(decision_trading_date=["2025-01-01"], level=[2]))
    with pytest.raises(ValueError): first_alert_cvar(p.head(5), a)
    p.loc[2, "open"] = np.nan
    with pytest.raises(ValueError): first_alert_cvar(p, a)
