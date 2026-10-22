"""Independent black-box checks of daily forecast semantics and causality."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vol_surface.predictive import (
    RAW_FEATURES, DAYTIME_SLOTS, aggregate_sessions, build_features,
    build_targets, chronological_splits, forecast_latest, probability_metrics,
)


def complete_days(n=150, symbol="ag", start="2024-01-02"):
    rows = []
    for i, day in enumerate(pd.bdate_range(start, periods=n)):
        for j, clock in enumerate(DAYTIME_SLOTS):
            at = day + pd.Timedelta(hours=int(clock[:2]), minutes=int(clock[3:]))
            row = {name: .2 + .001 * i + .00001 * j for name in RAW_FEATURES}
            row["convexity_violation"] = .1
            row.update(symbol=symbol, _timestamp=at.strftime("%Y%m%d%H%M%S"))
            rows.append(row)
    return pd.DataFrame(rows)


def test_completed_daily_features_are_prefix_invariant():
    data = complete_days()
    initial, names = build_features(data)
    changed = data.copy()
    changed.loc[100*15:, list(RAW_FEATURES)] = 10000.
    mutated, _ = build_features(changed)
    pd.testing.assert_frame_equal(initial.iloc[:100][names], mutated.iloc[:100][names])
    prefix, _ = build_features(data.iloc[:100*15])
    pd.testing.assert_frame_equal(initial.iloc[:100][names], prefix[names])


def test_pending_friday_night_belongs_to_monday_without_altering_friday():
    day = complete_days(1, start="2025-01-03")
    night = day.iloc[[-1]].copy()
    night["_timestamp"] = "20250103211500"
    night["atm_iv"] = 9.
    prefix = aggregate_sessions(pd.concat([day, night], ignore_index=True))
    assert len(prefix) == 1
    assert prefix.iloc[0].atm_iv_daily_max < 1
    monday = complete_days(1, start="2025-01-06")
    all_days = aggregate_sessions(pd.concat([day, night, monday], ignore_index=True))
    assert len(all_days) == 2
    assert all_days.iloc[1].atm_iv_daily_max == 9
    assert all_days.iloc[1].night_snapshots == 1
    pd.testing.assert_frame_equal(prefix, all_days.iloc[:1].reset_index(drop=True))


def test_sparse_day_is_excluded_and_breaks_label_window():
    a = complete_days(100)
    b = complete_days(100, symbol="cu")
    # A has only its last five day bars on date 60, matching real rb 2025-11-25.
    a = a.drop(a.index[60*15:60*15+10])
    f, _ = build_features(pd.concat([a, b], ignore_index=True))
    t = build_targets(f)
    sub = t.loc[t.symbol == "ag"].reset_index(drop=True)
    assert len(sub) == 99
    assert pd.isna(sub.loc[59, "label_1"])
    assert pd.isna(sub.loc[58, "label_3"])
    assert pd.notna(sub.loc[61, "label_1"])


def test_target_excludes_present_uses_frozen_past_threshold_and_tail_unknown():
    a = complete_days(100)
    a["atm_iv"] = .2
    a.loc[50*15:51*15-1, "atm_iv"] = 9.
    f, _ = build_features(a)
    t = build_targets(f)
    assert t.loc[49, "label_1"] == 1
    assert t.loc[48, "label_1"] == 0
    assert t.loc[48, "label_3"] == 1
    assert t.loc[50, "label_1"] == 0
    assert t.loc[50, "current_exceedance"] == 1
    assert t.loc[50, "_atm_iv_q95"] == .2
    assert t.loc[49, "label_end_1"] == t.loc[50, "_dt"]
    assert t.label_5.iloc[-5:].isna().all()
    assert pd.isna(t.label_1.iloc[-1])


def test_long_calendar_gap_cannot_become_next_trading_day():
    a = complete_days(70)
    b = complete_days(30, start="2025-01-02")
    t = build_targets(build_features(pd.concat([a, b], ignore_index=True))[0])
    assert pd.isna(t.loc[69, "label_1"])
    assert t.loc[69, "_segment"] != t.loc[70, "_segment"]


def test_train_calibration_validation_test_purge_actual_label_end():
    history = pd.concat([complete_days(300), complete_days(300, symbol="cu")], ignore_index=True)
    t = build_targets(build_features(history)[0])
    masks, cuts = chronological_splits(t, 5)
    for name, next_block in (("train", "calibration"), ("calibration", "validation"), ("validation", "test")):
        sub = t.loc[masks[name]]
        assert len(sub) > 0
        assert sub.label_end_5.max() < pd.Timestamp(cuts[next_block])
    assert sum(int(m.sum()) for m in masks.values()) <= len(t)
    assert not any((left & right).any() for i, left in enumerate(masks.values())
                   for right in list(masks.values())[i+1:])


def test_metric_ties_match_official_sklearn():
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    p = np.array([.7, .7, .5, .5, .5, .1, .1, .1])
    metrics = probability_metrics(y, p, np.full(len(y), .5))
    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert metrics["average_precision"] == pytest.approx(average_precision_score(y, p))


def baseline_artifact(history):
    f, names = build_features(history)
    return {
        "symbols": ["ag"], "target_description": "test daily proxy", "models": {
            "1": {"baseline_probability": .25, "baseline_by_symbol": {"ag": .25},
                  "mode_by_symbol": {"ag": "baseline_only"}, "coefficients": None,
                  "threshold": {"value": 1.000001}}
        }, "horizons_days": [1], "feature_names": names,
        "model_available_from": f.iloc[60]._dt.isoformat(),
    }


def test_latest_forecast_uses_daily_history_and_pending_day_is_not_origin():
    h = complete_days(100)
    artifact = baseline_artifact(h)
    # Incomplete final day does not supersede previous complete close.
    when = pd.to_datetime(h.iloc[-2]._timestamp, format="%Y%m%d%H%M%S")
    snapshot = forecast_latest(artifact, h, "ag", as_of=when)
    assert snapshot["status"] == "degraded"
    assert snapshot["data_timestamp"] == pd.to_datetime(h.iloc[-16]._timestamp, format="%Y%m%d%H%M%S").isoformat()
    assert snapshot["quality"]["past_observations"] == 60
    assert snapshot["quality"]["pending_incomplete_session_snapshots"] == 14
    assert snapshot["horizons"][0]["probability"] == .25
    assert snapshot["horizons"][0]["alert"] is False


def test_replay_cannot_use_model_before_training_and_selection_finished():
    h = complete_days(100)
    artifact = baseline_artifact(h)
    when = pd.to_datetime(h.iloc[50*15+14]._timestamp, format="%Y%m%d%H%M%S")
    snapshot = forecast_latest(artifact, h, "ag", as_of=when)
    assert snapshot["status"] == "unavailable"
    assert snapshot["horizons"] == []


def test_current_clock_does_not_masquerade_as_replay():
    h = complete_days(100)
    snapshot = forecast_latest(baseline_artifact(h), h, "ag", as_of="2026-10-22")
    assert snapshot["status"] == "stale"
    assert snapshot["horizons"] == []


def fake_daily_training_frame():
    rows = []
    days = pd.bdate_range("2023-01-03", periods=600)
    for symbol in ("ag", "cu"):
        for i, day in enumerate(days):
            state = i % 2
            row = {"symbol": symbol, "_dt": day + pd.Timedelta(hours=15),
                   "_timestamp": (day + pd.Timedelta(hours=15)).strftime("%Y%m%d%H%M%S"),
                   "atm_iv": .2 + .1 * state, "current_exceedance": float(state),
                   "_history_ready": True}
            for horizon in (1, 2, 3, 4, 5):
                row[f"label_{horizon}"] = float(state)
                row[f"label_end_{horizon}"] = (days[i+horizon] + pd.Timedelta(hours=15)
                                               if i+horizon < len(days) else pd.NaT)
                row[f"complete_window_{horizon}"] = i+horizon < len(days)
                if i+horizon >= len(days):
                    row[f"label_{horizon}"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows), ["atm_iv", "current_exceedance"]


def test_train_models_do_not_use_test_labels_for_parameters_selection_or_threshold(monkeypatch):
    import vol_surface.predictive as p

    frame, names = fake_daily_training_frame()
    monkeypatch.setattr(p, "build_features", lambda *args, **kwargs: (frame.copy(), names))
    monkeypatch.setattr(p, "build_targets", lambda x: x.copy())

    def constant_logistic(x, y, penalty=1e-4):
        return {"intercept": 0., "coefficients": np.zeros(x.shape[1]).tolist(),
                "converged": True, "iterations": 1, "optimizer_message": "synthetic"}

    monkeypatch.setattr(p, "_fit_logistic", constant_logistic)
    first, first_report, _ = p.train_models(pd.DataFrame())
    for horizon in (1, 2, 3, 4, 5):
        trained = first["models"][str(horizon)]
        assert set(trained["mode_by_symbol"].values()) == {"state_baseline"}
        masks, _ = p.chronological_splits(frame, horizon, first["split_boundaries"])
        tr = frame.loc[masks["train"]]
        for (symbol, state), sub in tr.groupby(["symbol", "current_exceedance"]):
            key = f"{symbol}:{int(state)}"
            expected = (sub[f"label_{horizon}"].sum() + 1) / (len(sub) + 2)
            assert trained["state_baseline"][key] == pytest.approx(expected)
            assert trained["state_baseline_support"][key] == len(sub)
    test_start = pd.Timestamp(first["split_boundaries"]["test"])
    for horizon in (1, 2, 3, 4, 5):
        selected = frame._dt >= test_start
        frame.loc[selected, f"label_{horizon}"] = 1 - frame.loc[selected, f"label_{horizon}"]
    second, second_report, _ = p.train_models(pd.DataFrame())
    assert first["models"] == second["models"]
    assert first["fixed_parameters"] == second["fixed_parameters"]
    assert first_report["horizons"]["1"]["test"]["brier"] != second_report["horizons"]["1"]["test"]["brier"]


def test_state_predictions_work_without_logistic_and_use_symbol_fallback():
    import vol_surface.predictive as p
    frame = pd.DataFrame({"symbol": ["ag", "ag", "cu"], "current_exceedance": [0., 1., 1.]})
    model = {"baseline_probability": .3, "baseline_by_symbol": {"ag": .2, "cu": .4},
             "state_baseline": {"ag:0": .1}, "mode_by_symbol": {"ag": "state_baseline", "cu": "baseline_only"},
             "coefficients": None, "threshold": {"value": .15}}
    probabilities, alerts = p._model_predictions(frame, [], model)
    np.testing.assert_allclose(probabilities, [.1, .2, .4])
    assert alerts.tolist() == [False, True, False]


def test_state_forecast_explanation_matches_training_conditional_rate():
    h = complete_days(100)
    artifact = baseline_artifact(h)
    model = artifact["models"]["1"]
    model.update(mode_by_symbol={"ag": "state_baseline"},
                 state_baseline={"ag:0": .1, "ag:1": .8},
                 state_baseline_support={"ag:0": 98, "ag:1": 18},
                 threshold={"value": .5})
    when = pd.to_datetime(h.iloc[-1]._timestamp, format="%Y%m%d%H%M%S")
    snapshot = forecast_latest(artifact, h, "ag", as_of=when)
    item = snapshot["horizons"][0]
    state = int(snapshot["signals"]["currently_exceeding"])
    expected = model["state_baseline"][f"ag:{state}"]
    assert item["model_status"] == "state_baseline"
    assert item["probability"] == expected
    assert item["state_condition"]["historical_conditional_probability"] == expected
    assert item["state_condition"]["training_observations_in_state"] == model["state_baseline_support"][f"ag:{state}"]
    assert item["alert"] == (expected >= .5)
    assert item["top_contributions"] == []
    assert item["intercept_log_odds"] is None
    assert item["total_log_odds"] is None
