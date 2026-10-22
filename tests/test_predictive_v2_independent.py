"""Independent checks of V2 event semantics, temporal isolation and safe inference.

Expected outcomes are calculated here from explicit toy events rather than from
the implementation's own target or prediction helpers.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from vol_surface.predictive import DAYTIME_SLOTS, RAW_FEATURES
from vol_surface import predictive_v2 as v2


def complete_history(n=140, symbols=("ag", "cu")):
    """Complete, constant-price days with no accidental threshold crossings."""
    rows = []
    for symbol in symbols:
        for day in pd.bdate_range("2023-01-02", periods=n):
            for clock in DAYTIME_SLOTS:
                at = pd.Timestamp(f"{day.date()} {clock}")
                row = {name: .2 for name in RAW_FEATURES}
                row["convexity_violation"] = .1
                row.update(symbol=symbol, _timestamp=at.strftime("%Y%m%d%H%M%S"))
                rows.append(row)
    return pd.DataFrame(rows)


def event_history():
    history = complete_history(100, ("ag",))
    for day in (50, 52):
        history.loc[day * 15:(day + 1) * 15 - 1, "atm_iv"] = .9
    return history


def test_first_future_event_excludes_present_and_distinguishes_origin_tasks():
    data = v2.build_targets(v2.build_features(event_history())[0])
    # Day 49 is normal; day 50 is already abnormal and recurs on day 52.
    assert data.loc[49, "current_exceedance"] == 0
    assert data.loc[50, "current_exceedance"] == 1
    assert data.loc[49, "prediction_task"] == "onset"
    assert data.loc[50, "prediction_task"] == "persistence"
    assert data.loc[49, "_first_event"] == 1
    assert data.loc[50, "_first_event"] == 2
    assert data.loc[50, "label_1"] == 0
    assert data.loc[50, "label_2"] == 1
    assert data.loc[53, "_first_event"] == 0
    assert [data.loc[49, f"label_{h}"] for h in range(1, 6)] == [1] * 5


def test_future_threshold_recalculation_does_not_change_frozen_origin_event():
    features, _ = v2.build_features(event_history())
    first = v2.build_targets(features)
    changed = features.copy()
    # The origin-49 event is judged against origin 49, regardless of subsequent
    # expanding/rolling threshold estimates.
    changed.loc[50:, ["_atm_iv_q95", "_convexity_violation_q95"]] = 1000.
    second = v2.build_targets(changed)
    assert first.loc[49, "_first_event"] == second.loc[49, "_first_event"] == 1
    for horizon in range(1, 6):
        assert first.loc[49, f"label_{horizon}"] == second.loc[49, f"label_{horizon}"]


def test_all_horizons_share_five_day_eligibility_including_missing_sessions():
    history = complete_history(100)
    # CU preserves the global observed date, making AG's missing date explicit.
    ag_dates = sorted(history.loc[history.symbol == "ag", "_timestamp"].str[:8].unique())
    history = history.loc[~((history.symbol == "ag") &
                           history._timestamp.str.startswith(ag_dates[60]))]
    frame = v2.build_targets(v2.build_features(history)[0])
    ag = frame.loc[frame.symbol == "ag"].reset_index(drop=True)
    assert ag.loc[58, "complete_window_1"]  # short label exists in principle
    assert not ag.loc[58, "complete_window_5"]
    assert ag.loc[58, [f"label_{h}" for h in range(1, 6)]].isna().all()
    assert ag.iloc[-5:][[f"label_{h}" for h in range(1, 6)]].isna().all().all()
    masks = [frame[f"label_{h}"].notna() for h in range(1, 6)]
    assert all(mask.equals(masks[0]) for mask in masks[1:])


def test_multiscale_inputs_are_causal_even_if_future_values_change():
    history = complete_history()
    first, names = v2.build_features(history)
    cutoff = sorted(first._dt.unique())[95]
    changed = history.copy()
    future = pd.to_datetime(changed._timestamp, format="%Y%m%d%H%M%S") > cutoff
    changed.loc[future, list(RAW_FEATURES)] = 900.
    second, second_names = v2.build_features(changed)
    assert names == second_names
    for symbol in ("ag", "cu"):
        before = first.loc[(first.symbol == symbol) & (first._dt <= cutoff), names]
        after = second.loc[(second.symbol == symbol) & (second._dt <= cutoff), names]
        pd.testing.assert_frame_equal(before, after)


def test_global_purge_removes_short_horizon_origins_with_long_label_overlap():
    from vol_surface.predictive_validation import global_purged_splits
    data = v2.build_targets(v2.build_features(complete_history())[0])
    dates = sorted(data._dt.dt.normalize().unique())
    boundaries = {name: pd.Timestamp(dates[i]).isoformat() for name, i in
                  (("calibration", 65), ("validation", 90), ("test", 115))}
    result = global_purged_splits(data, boundaries)
    masks = result[0] if isinstance(result, tuple) else result
    for part, following in (("train", "calibration"),
                            ("calibration", "validation"), ("validation", "test")):
        chosen = data.loc[masks[part]]
        assert len(chosen)
        assert (chosen.label_end_5 < pd.Timestamp(boundaries[following])).all()
    # A row two dates before calibration has a valid 1-day label, but cannot
    # enter the common training set because its 5-day outcome enters calibration.
    near = (data._dt.dt.normalize() == pd.Timestamp(dates[63]))
    assert data.loc[near, "label_1"].notna().all()
    assert not np.asarray(masks["train"])[near.to_numpy()].any()
    all_masks = list(masks.values())
    assert not any((left & right).any() for i, left in enumerate(all_masks)
                   for right in all_masks[i + 1:])


def small_daily_frame():
    """Known origin-level outcomes, with complete nonoverlapping fit blocks."""
    rng = np.random.default_rng(403)
    dates = pd.bdate_range("2021-01-04", periods=480)
    rows = []
    for symbol in ("ag", "cu"):
        for i, date in enumerate(dates):
            x = float(rng.normal())
            # Both tasks have enough events and non-events for independent
            # calibration and validation. At-risk rows cease at this first hit.
            first = int(rng.integers(1, 6)) if x > -.25 else 0
            at = date + pd.Timedelta(hours=15)
            row = {name: .2 + .02 * float(rng.normal()) for name in RAW_FEATURES}
            row.update(symbol=symbol, _dt=at, _timestamp=at.strftime("%Y%m%d%H%M%S"),
                       current_exceedance=float(i % 2),
                       prediction_task="onset" if i % 2 == 0 else "persistence",
                       _first_event=float(first), _history_ready=True,
                       _calendar_index=i, _common_eligible_5d=True,
                       atm_iv=.2 + .02 * x)
            for h in range(1, 6):
                row[f"label_{h}"] = float(0 < first <= h)
                row[f"label_end_{h}"] = at + pd.offsets.BDay(h)
            rows.append(row)
    frame = pd.DataFrame(rows)
    names = ["atm_iv", "convexity_violation", "arb_score", "liquidity_ratio"]
    return frame, names, dates


@pytest.fixture(scope="module")
def fitted_example():
    frame, names, dates = small_daily_frame()
    train = frame.loc[frame._calendar_index < 240]
    calibration = frame.loc[frame._calendar_index.between(246, 325)]
    validation = frame.loc[frame._calendar_index.between(332, 411)]
    models = v2.fit_models(train, calibration, validation, names)
    return frame, names, dates, models


def test_all_fitted_candidates_and_baseline_are_coherent_joint_curves(fitted_example):
    frame, names, _, models = fitted_example
    rows = frame.loc[frame._calendar_index >= 418]
    for mode in (None, *v2.MODES):
        result = v2.predict_curves(rows, names, models, mode=mode)
        probability = result["probabilities"]
        baseline = result["baseline_probabilities"]
        assert probability.shape == baseline.shape == (len(rows), 5)
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        assert (np.diff(probability, axis=1) >= 0).all()
        assert (np.diff(baseline, axis=1) >= 0).all()
        np.testing.assert_allclose(probability, 1 - np.cumprod(1 - result["hazards"], axis=1))
    selected = v2.predict_curves(rows, names, models)
    for task in v2.TASKS:
        for symbol in ("ag", "cu"):
            records = models["tasks"][task]
            selected_mode = records["mode_by_symbol"][symbol]
            chosen = (rows.prediction_task == task) & (rows.symbol == symbol)
            assert set(selected["modes"][chosen]) == {selected_mode}
            # Selection is one mode per full curve, rather than a horizon mix.
            candidate = v2.predict_curves(rows.loc[chosen], names, models, mode=selected_mode)
            np.testing.assert_allclose(selected["probabilities"][chosen], candidate["probabilities"])


def test_future_outcomes_are_never_prediction_inputs(fitted_example):
    frame, names, _, models = fitted_example
    rows = frame.loc[frame._calendar_index >= 418].copy()
    original = v2.predict_curves(rows, names, models)
    rows["_first_event"] = 0.
    for h in range(1, 6):
        rows[f"label_{h}"] = 1 - rows[f"label_{h}"]
        rows[f"label_end_{h}"] = pd.Timestamp("2099-01-01")
    changed = v2.predict_curves(rows, names, models)
    for key in ("probabilities", "baseline_probabilities", "hazards", "alerts", "thresholds"):
        np.testing.assert_array_equal(original[key], changed[key])


def test_fitter_refuses_censored_or_overlapping_label_blocks(fitted_example):
    frame, names, _, _ = fitted_example
    train = frame.loc[frame._calendar_index < 240].copy()
    calibration = frame.loc[frame._calendar_index.between(246, 325)].copy()
    validation = frame.loc[frame._calendar_index.between(332, 411)].copy()
    broken = train.copy()
    broken.loc[broken.index[0], "label_5"] = np.nan
    with pytest.raises(ValueError, match="censored"):
        v2.fit_models(broken, calibration, validation, names)
    broken = train.copy()
    broken.loc[broken.index[-1], "label_end_5"] = calibration._dt.min()
    with pytest.raises(ValueError, match="purge|overlap"):
        v2.fit_models(broken, calibration, validation, names)


def test_quality_gate_suppresses_whole_curve_for_ood_or_missing_core(fitted_example):
    frame, names, _, models = fitted_example
    rows = frame.loc[frame._calendar_index == 420].copy()
    rows.iloc[0, rows.columns.get_loc("atm_iv")] = np.nan
    rows.iloc[1, [rows.columns.get_loc(name) for name in names]] = 10000.
    gate = v2.operational_gates(rows, names, models)
    assert gate["allowed"].tolist() == [False, False]
    assert gate["reasons"] == ["insufficient_history_or_missing_core", "out_of_distribution"]
    assert gate["ood_fraction"][1] == 1.


def test_paired_bootstrap_preserves_exact_same_date_symbol_cancellation():
    from vol_surface.predictive_validation import paired_block_bootstrap
    rows = []
    for date in pd.bdate_range("2025-01-01", periods=27):
        for p, baseline in ((0., 1.), (1., 0.)):
            rows.append({"_dt": date, "label": 0., "probability": p,
                         "baseline_probability": baseline})
    interval = paired_block_bootstrap(pd.DataFrame(rows), n_boot=127, block_dates=7)
    assert interval["brier_delta"] == 0
    assert interval["brier_delta_ci95"] == [0., 0.]
    assert interval["skill_ci95"] == [0., 0.]
    json.dumps(interval, allow_nan=False)


def test_operational_recall_keeps_suppressed_positives_in_denominator():
    from vol_surface.predictive_validation import evaluate_predictions
    rows = []
    for h in range(1, 6):
        for i, (truth, p, allowed) in enumerate(((1, .9, False), (1, .8, True),
                                                (0, .7, True), (0, .1, True))):
            alert = p >= .5
            rows.append({"_dt": pd.Timestamp("2025-01-01") + pd.offsets.BDay(i),
                         "symbol": "ag", "horizon_days": h, "label": truth,
                         "probability": p, "baseline_probability": .5,
                         "raw_alert": alert, "gate_allowed": allowed,
                         "gate_reason": None if allowed else "out_of_distribution",
                         "operational_alert": alert and allowed, "calendar_index": i,
                         "prediction_task": "onset", "model_status": "joint_logistic"})
    metrics = evaluate_predictions(pd.DataFrame(rows), n_boot=9)
    for item in metrics.values():
        assert item["raw"]["recall_all_origins"] == 1.
        assert item["raw"]["precision"] == pytest.approx(2 / 3)
        assert item["operational"]["recall_all_origins"] == .5
        assert item["operational"]["precision"] == .5
        assert item["operational"]["probability_coverage"] == .75
        assert item["operational"]["abstentions"] == 1
        assert item["test"]["n"] == 4
    json.dumps(metrics, allow_nan=False)


def test_changing_reserved_test_features_and_labels_cannot_refit_models(monkeypatch):
    from vol_surface import predictive_validation as validation
    frame, names, dates = small_daily_frame()
    windows = [{"fold": 1, "boundaries": {
        "calibration": dates[246].isoformat(), "validation": dates[332].isoformat(),
        "test": dates[418].isoformat()}, "test_end": None}]
    monkeypatch.setattr(v2, "build_features", lambda *args, **kwargs: (frame.copy(), names))
    monkeypatch.setattr(v2, "build_targets", lambda data: data.copy())
    monkeypatch.setenv("RISK_LLM_API_KEY", "sk-independent-v2-secret-932985")
    first, first_report, _ = validation.train_models(
        pd.DataFrame(), windows=windows, v1_artifact_path="not-an-existing-v1-artifact", n_boot=3)
    reserved = frame._calendar_index >= 418
    frame.loc[reserved, names] = 900.
    frame.loc[reserved, "_first_event"] = np.where(frame.loc[reserved, "_first_event"] == 0, 5., 0.)
    for h in range(1, 6):
        frame.loc[reserved, f"label_{h}"] = (
            (frame.loc[reserved, "_first_event"] > 0) & (frame.loc[reserved, "_first_event"] <= h)).astype(float)
    second, second_report, _ = validation.train_models(
        pd.DataFrame(), windows=windows, v1_artifact_path="not-an-existing-v1-artifact", n_boot=3)
    assert first["models"] == second["models"]
    assert first["fixed_parameters"] == second["fixed_parameters"]
    assert first["fitted_until"] == second["fitted_until"]
    assert first_report["horizons"]["1"]["test"]["brier"] != second_report["horizons"]["1"]["test"]["brier"]
    assert "sk-independent-v2-secret-932985" not in json.dumps(second, allow_nan=False)


def test_numeric_artifact_rejects_nonfinite_or_wrong_dimension_arrays(fitted_example, tmp_path):
    _, names, _, models = fitted_example
    artifact = {"schema_version": 2, "feature_names": names, "models": models}
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact, allow_nan=False))
    assert v2.load_artifact(path)["models"] == models
    broken = copy.deepcopy(artifact)
    broken["models"]["preprocessing"]["median"] = [0.]
    path.write_text(json.dumps(broken, allow_nan=False))
    with pytest.raises(ValueError, match="preprocessing"):
        v2.load_artifact(path)
    broken = copy.deepcopy(artifact)
    broken["models"]["preprocessing"]["scale"][0] = float("nan")
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="Non-finite"):
        v2.load_artifact(path)


def inference_artifact(fitted_example):
    _, names, _, models = fitted_example
    return {"schema_version": 2, "feature_names": names, "models": copy.deepcopy(models),
            "symbols": ["ag", "cu"], "model_available_from": "2022-01-01", "per_symbol_test_metrics": {
                str(h): {"ag": {"brier_skill": .9, "task_metrics": {"onset": {
                    "test": {"n": 7, "brier_skill": -.1}, "operational": {"abstentions": 2}}}}}
                for h in range(1, 6)}}


def test_forecast_uses_task_specific_flat_evaluation_and_exposes_details(fitted_example):
    artifact = inference_artifact(fitted_example)
    history = complete_history(100)
    when = pd.to_datetime(history._timestamp.max(), format="%Y%m%d%H%M%S")
    snapshot = v2.forecast_latest(artifact, history, "ag", as_of=when)
    assert len(snapshot["horizons"]) == 5
    assert snapshot["prediction_task"] == "onset"
    assert snapshot["evaluation"]["1"]["brier_skill"] == -.1
    assert snapshot["evaluation_details"]["1"]["operational"]["abstentions"] == 2
    assert snapshot["horizons"][0]["heldout_skill_status"] == "nonpositive"
    json.dumps(snapshot, allow_nan=False)


def test_unknown_stale_hindsight_and_unsupported_task_refuse_probabilities(fitted_example):
    artifact = inference_artifact(fitted_example)
    history = complete_history(100)
    when = pd.to_datetime(history._timestamp.max(), format="%Y%m%d%H%M%S")
    cases = (("rb", when, "unavailable"), ("ag", when + pd.Timedelta(days=20), "stale"),
             ("ag", pd.Timestamp("2021-01-01"), "unavailable"))
    for symbol, as_of, status in cases:
        result = v2.forecast_latest(artifact, history, symbol, as_of=as_of)
        assert result["status"] == status
        assert result["horizons"] == []
    artifact["models"]["tasks"]["onset"]["training_origins"] = 0
    unsupported = v2.forecast_latest(artifact, history, "ag", as_of=when)
    assert unsupported["status"] == "unavailable"
    assert unsupported["horizons"] == []
    assert unsupported["quality"]["whole_curve_gate_reason"] == "unsupported_task"


def test_inference_filters_future_history_without_altering_persisted_model(fitted_example):
    artifact = inference_artifact(fitted_example)
    history = complete_history(100)
    when = pd.Timestamp(sorted(history._timestamp.str[:8].unique())[94]) + pd.Timedelta(hours=15)
    before_models = copy.deepcopy(artifact["models"])
    first = v2.forecast_latest(artifact, history, "ag", as_of=when)
    later = pd.to_datetime(history._timestamp, format="%Y%m%d%H%M%S") > when
    history.loc[later, list(RAW_FEATURES)] = 90000.
    second = v2.forecast_latest(artifact, history, "ag", as_of=when)
    assert first == second
    assert artifact["models"] == before_models


def test_frozen_v1_comparison_keeps_probabilities_when_incoherence_blocks_alerts(monkeypatch, tmp_path):
    from vol_surface import predictive as old
    from vol_surface.predictive_validation import _comparison_v1
    at = pd.Timestamp("2025-01-02 15:00")
    row = {name: 10. for name in RAW_FEATURES}
    row.update(symbol="ag", _dt=at, _history_ready=True, current_exceedance=0.)
    daily = pd.DataFrame([row])
    preprocessing = {"median": [0.], "scale": [1.], "low": [0.], "high": [1.], "clip": 8.}
    models = {}
    for h, p in enumerate((.8, .2, .9, .95, .99), start=1):
        models[str(h)] = {"mode_by_symbol": {"ag": "baseline_only"},
                          "baseline_probability": p, "baseline_by_symbol": {"ag": p},
                          "state_baseline": {},
                          "threshold": {"value": 1.000001}, "coefficients": None,
                          "preprocessing": preprocessing}
    artifact = {"symbols": ["ag"], "model_available_from": "2025-01-01",
                "feature_names": ["atm_iv"], "models": models}
    monkeypatch.setattr(old, "load_artifact", lambda path: artifact)
    monkeypatch.setattr(old, "build_features", lambda *args, **kwargs: (daily.copy(), ["atm_iv"]))
    frozen_path = tmp_path / "frozen-v1.json"
    frozen_path.write_text("{}")
    predictions = pd.DataFrame([{"symbol": "ag", "_dt": at, "horizon_days": h,
                                 "label": 0, "probability": .1, "raw_alert": False,
                                 "operational_alert": False, "gate_allowed": True}
                                for h in range(1, 6)])
    result = _comparison_v1(pd.DataFrame(), predictions, frozen_path, n_boot=3)
    for item in result["horizons"].values():
        # The original V1 skips OOD on baseline-only estimates and retains
        # probability estimates when its incoherence check suppresses alerts.
        assert item["v1_probability_coverage"] == 1.
        assert item["v1_alert_eligibility_coverage"] == 0.
