"""Predeclared development walk-forward evaluation for the coherent V2 model.

This module owns temporal splits and evaluation, never candidate fitting. The
entire existing history was previously inspected; no fold is an external blind
validation. All five horizons share the same complete five-session origins.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .predictive import HORIZONS, probability_metrics

TEST_STARTS = ("2024-12-19", "2025-06-20", "2025-10-17")
FIXED_SETTINGS = {
    "fold_test_starts": list(TEST_STARTS),
    "calibration_fraction_of_pretest_dates": .60,
    "validation_fraction_of_pretest_dates": .80,
    "origin_eligibility": "complete five-session label window for every horizon",
    "split_purge": "all splits use label_end_5 strictly before the next boundary",
    "candidate_selection": "past validation only; no fold/test-driven parameter changes",
    "bootstrap_repetitions": 500, "bootstrap_seed": 42,
    "bootstrap_block_observed_dates": 10,
    "bootstrap_method": "paired circular moving blocks of global observed dates; all symbols retained together",
}


def global_purged_splits(frame: pd.DataFrame, boundaries: dict,
                         test_end=None) -> dict[str, np.ndarray]:
    """Global boundaries, exact label-end purges and one eligible origin set."""
    c, v, t = (pd.Timestamp(boundaries[k]) for k in ("calibration", "validation", "test"))
    if not c < v < t:
        raise ValueError("Expected strictly increasing global split boundaries")
    dt = pd.to_datetime(frame["_dt"])
    end = pd.to_datetime(frame["label_end_5"])
    valid = frame[[f"label_{h}" for h in HORIZONS]].notna().all(axis=1) & end.notna()
    masks = {
        "train": valid & (dt < c) & (end < c),
        "calibration": valid & (dt >= c) & (dt < v) & (end < v),
        "validation": valid & (dt >= v) & (dt < t) & (end < t),
        "test": valid & (dt >= t),
    }
    if test_end is not None:
        # An evaluation fold is closed before the next model vintage. A label
        # reaching that boundary is not silently shared across test folds.
        masks["test"] &= (dt < pd.Timestamp(test_end)) & (end < pd.Timestamp(test_end))
    return {k: np.asarray(value, bool) for k, value in masks.items()}


def default_test_windows(frame: pd.DataFrame) -> list[dict]:
    """Fixed historical test boundaries; pretest observed dates define past blocks."""
    dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame._dt).dt.normalize().unique()))
    folds = []
    for i, start in enumerate(TEST_STARTS):
        earlier = dates[dates < pd.Timestamp(start)]
        if len(earlier) < 120:
            raise ValueError(f"Insufficient historical dates before fixed fold {start}")
        boundaries = {"calibration": earlier[int(len(earlier)*.60)].isoformat(),
                      "validation": earlier[int(len(earlier)*.80)].isoformat(),
                      "test": pd.Timestamp(start).isoformat()}
        folds.append({"fold": i+1, "boundaries": boundaries,
                      "test_end": TEST_STARTS[i+1] if i+1 < len(TEST_STARTS) else None})
    return folds


def paired_block_bootstrap(frame: pd.DataFrame, probability_col="probability",
                           baseline_col="baseline_probability", n_boot=500,
                           seed=42, block_dates=10) -> dict:
    """Paired Brier difference, same date blocks across symbols and horizons.

    Positive delta is improvement. Circular moving blocks retain serial
    dependence approximately, not arbitrary regime changes or model refit error.
    Bootstrap is conditional on fitted model vintages and observed sample.
    """
    if n_boot < 1 or block_dates < 1:
        raise ValueError("Bootstrap repetitions and block length must be positive")
    if frame.empty:
        return {"n": 0, "dates": 0, "brier_delta": None,
                "brier_delta_ci95": [None, None], "skill": None,
                "skill_ci95": [None, None], "repetitions": n_boot,
                "block_observed_dates": block_dates, "seed": seed}
    work = frame[["_dt", "label", probability_col, baseline_col]].copy()
    if not np.isfinite(work[["label", probability_col, baseline_col]].to_numpy(float)).all():
        raise ValueError("Bootstrap requires finite paired probabilities and labels")
    work["date"] = pd.to_datetime(work._dt).dt.normalize()
    work["model_loss"] = (work.label-work[probability_col])**2
    work["baseline_loss"] = (work.label-work[baseline_col])**2
    grouped = work.groupby("date", sort=True).agg(
        model_loss=("model_loss", "sum"), baseline_loss=("baseline_loss", "sum"),
        count=("label", "size"))
    n_dates, block = len(grouped), min(block_dates, len(grouped))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_dates, size=(n_boot, int(np.ceil(n_dates/block))))
    indices = ((starts[..., None]+np.arange(block)) % n_dates).reshape(n_boot, -1)[:, :n_dates]
    counts = grouped["count"].to_numpy()[indices].sum(axis=1)
    ml = grouped.model_loss.to_numpy()[indices].sum(axis=1)/counts
    bl = grouped.baseline_loss.to_numpy()[indices].sum(axis=1)/counts
    skill = np.divide(bl-ml, bl, out=np.full_like(bl, np.nan), where=bl > 0)
    total_model, total_base = float(work.model_loss.mean()), float(work.baseline_loss.mean())
    finite_skill = skill[np.isfinite(skill)]
    return {"n": len(frame), "dates": n_dates,
            "brier_delta": total_base-total_model,
            "brier_delta_ci95": np.quantile(bl-ml, [.025, .975]).tolist(),
            "skill": 1-total_model/total_base if total_base else None,
            "skill_ci95": np.quantile(finite_skill, [.025, .975]).tolist() if len(finite_skill) else [None, None],
            "repetitions": n_boot, "block_observed_dates": block_dates, "seed": seed,
            "method": FIXED_SETTINGS["bootstrap_method"],
            "limitations": "Conditional on fitted vintages; exploratory intervals, no multiplicity correction; not a formal claim of generalisation."}


def _episode_counts(frame: pd.DataFrame, alert_col: str) -> dict:
    """Positive-label forecast-origin runs, explicitly not realised event episodes."""
    episodes = captured = 0
    first_positions = []
    for _, sub in frame.groupby(["symbol", "horizon_days"], sort=False):
        sub = sub.sort_values("_dt")
        previous_index, active, position, first, alerted = None, False, 0, None, False
        for row in sub.itertuples():
            index = getattr(row, "calendar_index")
            positive = bool(row.label)
            if active and (not positive or previous_index is None or index != previous_index+1):
                episodes += 1; captured += int(alerted)
                if first is not None: first_positions.append(first)
                active, alerted, position, first = False, False, 0, None
            if positive:
                if not active:
                    active = True
                if bool(getattr(row, alert_col)):
                    alerted = True
                    if first is None: first = position
                position += 1
            previous_index = index
        if active:
            episodes += 1; captured += int(alerted)
            if first is not None: first_positions.append(first)
    return {"positive_origin_episodes": episodes, "episodes_with_alert": captured,
            "positive_origin_episode_recall": captured/episodes if episodes else None,
            "median_first_alert_position_within_captured_episode": float(np.median(first_positions)) if first_positions else None,
            "episode_definition": "Consecutive eligible positive-label forecast origins per symbol/horizon, interrupted by a negative label or missing observed-calendar index. A subset may further split a run. These are proxy-origin episodes, NOT deduplicated realised market events; alert position is not event lead time."}


def _decisions(frame: pd.DataFrame, alert_col: str, operational=False) -> dict:
    y = frame.label.to_numpy(int)
    a = frame[alert_col].to_numpy(bool)
    allowed = frame.gate_allowed.to_numpy(bool) if operational else np.ones(len(frame), bool)
    positives, count = int(y.sum()), int(a.sum())
    tp, fp = int(np.sum(a & (y == 1))), int(np.sum(a & (y == 0)))
    months = pd.to_datetime(frame._dt).dt.to_period("M").nunique()
    answer = {"n_total": len(frame), "n_available": int(allowed.sum()),
              "abstentions": int((~allowed).sum()),
              "probability_coverage": float(allowed.mean()) if len(frame) else None,
              "alerts": count, "true_positives": tp, "false_alarms": fp,
              "precision": tp/count if count else None,
              "recall_all_origins": tp/positives if positives else None,
              "alert_rate_all_origins": count/len(frame) if len(frame) else None,
              "false_alarms_per_observed_month": fp/months if months else None,
              "observed_months": months, "abstentions_count_as_non_alerts": True,
              "baseline_rate_origins": int((frame.model_status == "hazard_baseline").sum()),
              "predictive_signal_origins": int((frame.model_status.isin(["joint_logistic", "joint_logistic_decay"]) & allowed).sum()),
              "predictive_signal_coverage": float((frame.model_status.isin(["joint_logistic", "joint_logistic_decay"]) & allowed).mean()) if len(frame) else None}
    answer.update(_episode_counts(frame, alert_col))
    if operational:
        answer["abstention_reasons"] = dict(Counter(frame.loc[~frame.gate_allowed, "gate_reason"].fillna("unknown")))
        available = frame.loc[frame.gate_allowed]
        answer["probability_metrics_when_available"] = probability_metrics(
            available.label.to_numpy(int), available.probability.to_numpy(float),
            available.baseline_probability.to_numpy(float), available[alert_col].to_numpy(bool))
    return answer


def _metrics(frame: pd.DataFrame, n_boot=500) -> dict:
    y, p, b = (frame[col].to_numpy(float) for col in ("label", "probability", "baseline_probability"))
    raw = probability_metrics(y, p, b, frame.raw_alert.to_numpy(bool))
    raw["current_state_baseline_brier"] = raw["baseline_brier"]
    raw["skill_vs_current_state_baseline"] = raw["brier_skill"]
    raw["baseline_definition"] = "training symbol/current-state/step conditional hazard, independently calibrated on past calibration block"
    return {"test": raw, "raw": _decisions(frame, "raw_alert"),
            "operational": _decisions(frame, "operational_alert", True),
            "paired_brier_bootstrap": paired_block_bootstrap(frame, n_boot=n_boot)}


def evaluate_predictions(frame: pd.DataFrame, n_boot=500) -> dict:
    """Report raw research probability scores and all-origin deployed decisions."""
    horizons = {}
    for h in HORIZONS:
        sub = frame.loc[frame.horizon_days == h]
        result = _metrics(sub, n_boot)
        result["tasks"] = {task: _metrics(sub.loc[sub.prediction_task == task], n_boot)
                           for task in ("onset", "persistence")}
        result["new_exceedance_subset_test"] = result["tasks"]["onset"]["test"]
        result["per_symbol"] = {}
        for symbol, rows in sub.groupby("symbol", sort=True):
            item = _metrics(rows, n_boot)
            item["mode"] = ",".join(sorted(rows.model_status.unique()))
            item["task_metrics"] = {task: _metrics(rows.loc[rows.prediction_task == task], n_boot)
                                    for task in ("onset", "persistence")}
            result["per_symbol"][symbol] = item
        horizons[str(h)] = result
    return horizons


def _split_summary(split: dict) -> dict:
    return {key: {"n": len(rows), "start": rows._dt.min().isoformat() if len(rows) else None,
                  "last_origin": rows._dt.max().isoformat() if len(rows) else None,
                  "last_label_end": rows.label_end_5.max().isoformat() if len(rows) else None,
                  "onset_n": int((rows.prediction_task == "onset").sum()),
                  "persistence_n": int((rows.prediction_task == "persistence").sum())}
            for key, rows in split.items()}


def _prediction_rows(test, names, models, fold, available_from):
    from .predictive_v2 import operational_gates, predict_curves
    curves = predict_curves(test, names, models)
    gates = operational_gates(test, names, models)
    p = np.asarray(curves["probabilities"], float)
    coherence = np.all(np.diff(p, axis=1) >= -1e-12, axis=1)
    allowed = np.asarray(gates["allowed"], bool) & coherence
    reasons = np.asarray(gates["reasons"], object).copy()
    reasons[~coherence] = "cross_horizon_inconsistency"
    rows = []
    for i, h in enumerate(HORIZONS):
        part = test[["symbol", "_timestamp", "_dt", "current_exceedance", "prediction_task", "_calendar_index", f"label_end_{h}"]].copy()
        part = part.rename(columns={"_calendar_index": "calendar_index", f"label_end_{h}": "label_end"})
        part["label_end_5"] = test.label_end_5
        part["label"] = test[f"label_{h}"].to_numpy(int)
        part["horizon_days"], part["fold"] = h, fold
        part["probability"] = p[:, i]
        part["baseline_probability"] = np.asarray(curves["baseline_probabilities"])[:, i]
        part["raw_alert"] = np.asarray(curves["alerts"])[:, i]
        part["gate_allowed"] = allowed
        part["gate_reason"] = reasons
        part["operational_alert"] = part.raw_alert & allowed
        part["threshold"] = np.asarray(curves["thresholds"])[:, i]
        part["model_status"] = curves["modes"]
        part["ood_fraction"] = gates["ood_fraction"]
        part["raw_feature_missing_fraction"] = gates["raw_feature_missing_fraction"]
        part["constant_features_violated"] = [";".join(values) for values in gates.get("constant_features_violated", [[] for _ in range(len(test))])]
        part["model_available_from"] = available_from
        part["as_of"] = test._dt
        part["age_minutes"] = 0.
        part["coherent"] = coherence
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _comparison_v1(history, latest_predictions, path: Path, n_boot=500):
    """Frozen V1 can be compared only after its stored selection vintage."""
    from . import predictive as v1
    if not path.exists():
        return {"status": "unavailable", "reason": "Frozen V1 artifact not found"}
    artifact = v1.load_artifact(path)
    available = pd.Timestamp(artifact["model_available_from"])
    origins = latest_predictions[["symbol", "_dt"]].drop_duplicates()
    origins = origins.loc[origins._dt >= available]
    features, _ = v1.build_features(history, artifact["symbols"])
    features = origins.merge(features, on=["symbol", "_dt"], how="inner", validate="one_to_one")
    if features.empty:
        return {"status": "unavailable", "reason": "No common origins at/after frozen V1 vintage"}
    features = features.sort_values(["symbol", "_dt"]).reset_index(drop=True)
    curves, rawalerts, allowed = [], [], []
    names = artifact["feature_names"]
    missing = features[list(v1.RAW_FEATURES)].isna().mean(axis=1).to_numpy() > .35
    for h in HORIZONS:
        model = artifact["models"][str(h)]
        p, a = v1._model_predictions(features, names, model)
        ok = ~missing & features._history_ready.to_numpy(bool)
        if "preprocessing" in model:
            prep = model["preprocessing"]
            raw = features[names].to_numpy(float)
            low, high = np.asarray(prep["low"]), np.asarray(prep["high"])
            monitored = np.asarray([n in v1.RAW_FEATURES for n in names])[None, :] & np.isfinite(raw) & (high > low)[None, :]
            outside = monitored & ((raw < low) | (raw > high))
            ood = outside.sum(axis=1)/np.maximum(monitored.sum(axis=1), 1)
            active_mode = features.symbol.map(model["mode_by_symbol"]).isin(["logistic", "state_baseline"]).to_numpy()
            ok &= (~active_mode) | (ood <= .35)
        curves.append(p); rawalerts.append(a); allowed.append(ok)
    pmat = np.stack(curves, axis=1)
    amat = np.stack(rawalerts, axis=1)
    okmat = np.stack(allowed, axis=1)
    # V1 checks coherence over still-present probabilities, then suppresses all
    # horizons on an inconsistency, matching its runtime ordering.
    coherent = np.array([all(b >= a for a, b in zip(row[ok][:-1], row[ok][1:]))
                         for row, ok in zip(pmat, okmat)])
    comparison = []
    for i, h in enumerate(HORIZONS):
        part = features[["symbol", "_dt"]].copy()
        part["horizon_days"], part["v1_probability"] = h, pmat[:, i]
        part["v1_raw_alert"], part["v1_gate_allowed"] = amat[:, i], okmat[:, i] & coherent
        part["v1_probability_available"] = okmat[:, i]
        part["v1_operational_alert"] = amat[:, i] & okmat[:, i] & coherent
        comparison.append(part)
    joined = latest_predictions.merge(pd.concat(comparison), on=["symbol", "_dt", "horizon_days"], how="inner", validate="one_to_one")
    horizons = {}
    for h in HORIZONS:
        sub = joined.loc[joined.horizon_days == h]
        y = sub.label.to_numpy(int)
        horizons[str(h)] = {
            "n_common": len(sub), "v1_brier": float(np.mean((y-sub.v1_probability)**2)),
            "v2_brier": float(np.mean((y-sub.probability)**2)),
            "paired_v2_vs_v1": paired_block_bootstrap(sub, baseline_col="v1_probability", n_boot=n_boot),
            "v1_raw_alerts": int(sub.v1_raw_alert.sum()), "v2_raw_alerts": int(sub.raw_alert.sum()),
            "v1_operational_alerts": int(sub.v1_operational_alert.sum()),
            "v2_operational_alerts": int(sub.operational_alert.sum()),
            "v1_probability_coverage": float(sub.v1_probability_available.mean()),
            "v1_alert_eligibility_coverage": float(sub.v1_gate_allowed.mean()),
            "v2_probability_coverage": float(sub.gate_allowed.mean()),
            "v2_alert_eligibility_coverage": float(sub.gate_allowed.mean()),
            "v1_false_alarms": int((sub.v1_operational_alert & (sub.label == 0)).sum()),
            "v2_false_alarms": int((sub.operational_alert & (sub.label == 0)).sum()),
        }
    return {"status": "latest_fold_common_origins_only", "frozen_v1_available_from": available.isoformat(),
            "earlier_folds": "Not evaluated with the frozen V1 artifact: it was not available at those historical dates.",
            "same_eligible_complete_five_day_origins": True,
            "v1_operational_gate": "Historical-close age=0; missing/OOD gates then surviving-curve coherence, same as V1 runtime. V1 incoherence suppresses alerts but retains visible raw probabilities; probability coverage and alert eligibility are separate.",
            "v1_model_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "horizons": horizons}


def train_models(history: pd.DataFrame, windows=None, progress: Callable | None = None,
                 v1_artifact_path="data_out/predictive/model.json", n_boot=500):
    """Fit each past-only vintage once; return final vintage and full fold report."""
    from . import predictive_v2 as v2
    features, names = v2.build_features(history)
    data = v2.build_targets(features)
    symbols = sorted(data.symbol.unique())
    windows = default_test_windows(data) if windows is None else windows
    if not windows:
        raise ValueError("At least one predeclared fold is required")
    reports, predictions, final_models, final_split, final_window = [], [], None, None, None
    for config in windows:
        fold, boundaries = config["fold"], config["boundaries"]
        masks = global_purged_splits(data, boundaries, config.get("test_end"))
        split = {key: data.loc[mask].copy() for key, mask in masks.items()}
        if min(len(split[k]) for k in split) < 20 or len(split["train"]) < 200:
            raise ValueError(f"Insufficient sample support in predeclared fold {fold}")
        if progress: progress(f"Fold {fold}: fitting on {len(split['train']):,} past origins; selecting on {len(split['validation']):,}")
        models = v2.fit_models(split["train"], split["calibration"], split["validation"], names, symbols)
        # Latest validation label is known before the test boundary. The
        # published vintage is considered available from the fixed test boundary
        # so historical replay never claims earlier selection availability.
        available_from = pd.Timestamp(boundaries["test"]).isoformat()
        pred = _prediction_rows(split["test"], names, models, fold, available_from)
        metrics = evaluate_predictions(pred, n_boot)
        reports.append({"fold": fold, "split_boundaries": boundaries,
                        "test_end_exclusive": config.get("test_end"),
                        "model_available_from": available_from,
                        "latest_selection_label_end": split["validation"].label_end_5.max().isoformat(),
                        "splits": _split_summary(split), "horizons": metrics,
                        "selection": {task: model.get("validation_selection", {}) for task, model in models["tasks"].items()},
                        "mode_by_task_symbol": {task: model["mode_by_symbol"] for task, model in models["tasks"].items()},
                        "coherence_conflict_origins": int(pred.loc[pred.horizon_days == 1, "coherent"].eq(False).sum()),
                        "operational_abstention_origins": int(pred.loc[pred.horizon_days == 1, "gate_allowed"].eq(False).sum())})
        predictions.append(pred)
        final_models, final_split, final_window = models, split, config
        if progress:
            score = metrics["1"]["test"]
            progress(f"Fold {fold}: 1d Brier={score['brier']:.4f}; conditional-baseline skill={score['brier_skill']:.2%}; abstained origins={reports[-1]['operational_abstention_origins']}")
    all_predictions = pd.concat(predictions, ignore_index=True)
    pooled = evaluate_predictions(all_predictions, n_boot)
    final_horizons = reports[-1]["horizons"]
    per_symbol_metrics = {h: {s: {**item["test"], "mode": item["mode"],
                                "raw": item["raw"], "operational": item["operational"],
                                "task_metrics": item["task_metrics"]}
                              for s, item in value["per_symbol"].items()}
                          for h, value in final_horizons.items()}
    artifact = {"schema_version": 2, "model_type": "discrete_time_joint_hazard_json",
                "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "target_description": getattr(v2, "TARGET_DESCRIPTION", "Five-session coherent conditional hazard curves"),
                "cadence_minutes": 15, "history_window": 60, "minimum_history": 20,
                "horizons_days": list(HORIZONS), "feature_names": names, "symbols": symbols,
                "models": final_models, "source_data_start": data._dt.min().isoformat(),
                "source_data_end": data._dt.max().isoformat(),
                "split_boundaries": final_window["boundaries"],
                "model_available_from": pd.Timestamp(final_window["boundaries"]["test"]).isoformat(),
                "fitted_until": final_split["validation"].label_end_5.max().isoformat(),
                "test_metrics": {h: value["test"] for h, value in final_horizons.items()},
                "per_symbol_test_metrics": per_symbol_metrics,
                "latest_snapshot_by_symbol": {s: rows._dt.max().isoformat() for s, rows in data.groupby("symbol")},
                "fixed_parameters": {**FIXED_SETTINGS, "model": final_models.get("fixed_parameters", {})}}
    eligible = data.label_5.notna()
    report = {"schema_version": 2, "target_description": artifact["target_description"],
              "evaluation_design": "Development walk-forward on previously inspected history; not an external blind test. Fixed three historical vintages, all five horizons on the same complete-five-day origins. No test-dependent candidate, threshold, or hyperparameter choice.",
              "source_manifest": history.attrs.get("source_manifest", []),
              "legacy_labels_used": False, "future_session_features_used": False,
              "fixed_parameters": artifact["fixed_parameters"], "symbols": symbols,
              "split_boundaries": artifact["split_boundaries"], "folds": reports,
              "horizons": final_horizons, "pooled_development_horizons": pooled,
              "coverage": {"completed_origins": len(data), "eligible_complete_five_day_origins": int(eligible.sum()),
                           "excluded_incomplete_future_or_history": int((~eligible).sum()),
                           "same_origins_for_all_five_horizons": True,
                           "evaluated_unique_origins": len(all_predictions.loc[all_predictions.horizon_days == 1]),
                           "eligible_not_in_test_folds": int(eligible.sum()-len(all_predictions.loc[all_predictions.horizon_days == 1]))},
              "daily_session_quality": {"incomplete_sessions_by_symbol": features.attrs.get("incomplete_sessions_by_symbol", {}),
                                        "night_coverage": "observed source snapshots only; may be partial"},
              "operational_gate": {"historical_as_of": "each eligible completed forecast origin's close; age=0 by construction",
                                   "rules": "raw feature missingness >35%, insufficient history/core features, raw-feature training-range dynamic OOD >35%, selected raw training-constant feature departure, unsupported task, or incoherence suppresses all horizon alerts; baseline-selected modes emit no predictive alert",
                                   "staleness_limitation": "Historical-close replay does not measure real-time delivery delays, stale-feed outages or incomplete-session current forecasts; live stale checks remain necessary.",
                                   "probability_coverage_definition": "Includes visible historical-rate fallback curves; predictive_signal_coverage additionally requires a selected complex model. Baseline fallback emits no predictive alert.",
                                   "full_origin_denominator": True},
              "limitations": ["Market-surface proxy, not portfolio loss, VaR or CVaR.",
                              "Existing history and prior V1 results were inspected before V2; all intervals and walk-forward results are development evidence.",
                              "Thresholds frozen at each origin; normal-origin onset and currently-abnormal persistence are different tasks. Persistence target is any future exceedance, not uninterrupted persistence or time-to-recovery.",
                              "Positive-label origin runs are proxy episodes, not deduplicated actual event dates. No event lead-time claim is made.",
                              "Observed session calendar and >7-day gaps conservatively exclude missing data and some long holidays.",
                              "Bootstrap preserves date-level cross-symbol dependence but conditions on fitted vintages, has no multiple-comparison adjustment, and cannot prove future gains.",
                              "Candidate fallback and thresholds are selected only by preceding validation data; no test-driven fallback is applied. Validation F1 is a provisional product objective; business false-alarm and missed-event costs are not supplied.",
                              "Final artifact is exactly the latest fold vintage; no refit uses its test labels."],
              "comparison_v1": _comparison_v1(history, predictions[-1], Path(v1_artifact_path), n_boot)}
    return artifact, report, all_predictions


def readable_report(report: dict) -> str:
    def f(value, percentage=False):
        return "—" if value is None else (f"{value:.1%}" if percentage else f"{value:.4f}")
    lines = ["# V2 联合风险率模型：开发期滚动验证", "", report["target_description"], "",
             "**全部历史和 V1 结果此前已在开发中查看。本报告是开发期滚动验证，不能称为外部盲测。**", "",
             "预先固定三个历史测试窗口；每一轮仅用其之前的数据训练、校准、按验证误差选候选和设置预警阈值。"
             "五个期限统一采用未来 5 个完整观测交易日标签齐全的预测原点；所有边界按实际第 5 日标签结束时间清除重叠。"
             "最终交互系统使用第三轮原样模型，未使用第三轮测试标签重新拟合。", "",
             "候选限定为同状态分品种条件风险率基准、固定正则联合风险率、固定 126 观测日半衰期联合风险率。"
             "预测正常状态下新增异常和当前异常下未来再次超阈值分别拟合；后者不代表持续不中断或恢复时间。"
             "在过去验证段按每个品种/任务五期限平均 Brier 选择，并优先简单基准；基准只展示统计概率、不产生增量预测预警。", "",
             "## 三轮总体结果", "", "Brier 越低越好；skill=1−模型 Brier/独立校准的条件风险率基准 Brier。"
             "同日全部品种共同进行 10 个观测日期移动块重采样，500 次、固定种子 42。区间仅是条件于已拟合模型的探索性区间。", "",
             "| 轮次 | 测试起点 | 期限 | 原点数 | 事件率 | Brier | 对条件基准 skill | skill 95%区间 | 运行可用率 | 运行误报数 | 运行召回率 |", "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    for fold in report["folds"]:
        for h, item in fold["horizons"].items():
            m, op = item["test"], item["operational"]
            ci = item["paired_brier_bootstrap"]["skill_ci95"]
            lines.append(f"| {fold['fold']} | {fold['split_boundaries']['test'][:10]} | {h} | {m['n']} | {f(m['prevalence'], True)} | {f(m['brier'])} | {f(m['brier_skill'], True)} | [{f(ci[0], True)}, {f(ci[1], True)}] | {f(op['probability_coverage'], True)} | {op['false_alarms']} | {f(op['recall_all_origins'], True)} |")
    lines += ["", "## 最近一轮：新增风险与当前异常分别评价", "",
              "| 任务 | 期限 | 原点 | 事件率 | Brier | 对条件基准 skill | 原始预警 | 运行预警 | 运行误报 | 弃权 | 正标签原点片段召回 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for h, item in report["horizons"].items():
        for task, result in item["tasks"].items():
            m, op = result["test"], result["operational"]
            lines.append(f"| {task} | {h} | {m['n']} | {f(m['prevalence'], True)} | {f(m['brier'])} | {f(m['brier_skill'], True)} | {result['raw']['alerts']} | {op['alerts']} | {op['false_alarms']} | {op['abstentions']} | {f(op['positive_origin_episode_recall'], True)} |")
    lines += ["", "## 最近一轮分品种结果", "",
              "| 品种 | 期限 | 模式 | 原点 | Brier | 对条件基准 skill | 可用率 | 运行精确率 | 运行召回率 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for h, item in report["horizons"].items():
        for symbol, result in item["per_symbol"].items():
            m, op = result["test"], result["operational"]
            lines.append(f"| {symbol} | {h} | {result['mode']} | {m['n']} | {f(m['brier'])} | {f(m['brier_skill'], True)} | {f(op['probability_coverage'], True)} | {f(op['precision'], True)} | {f(op['recall_all_origins'], True)} |")
    lines += ["", "## 运行门禁与评价口径", "",
              "历史回放在每个已完成日盘收盘的时刻作预测，数据年龄为零。缺失、历史不足、严重超出训练分布、已选择的原始常量训练特征发生变化、未训练任务和跨期限不一致按实际运行规则弃权。"
              "运行召回率保留全部原点为分母，弃权当作没有预警；原始概率 Brier 与门禁之后的告警统计分别报告，不能把弃权静默删掉后宣称整体改善。"
              "门禁后概率分数只在明确列出的可用子集上计算，不代表全原点准确度。", "",
              "正标签原点片段指同品种/期限连续的正标签预测原点，不是真实事件去重。预测任务子集或缺日会切断片段；片段内首次告警位置不是实际事件提前天数。"
              "历史收盘回放无法评价实时行情延迟、断流或当前不完整交易日；实时服务仍须执行过期门禁。", "",
              "| 轮次 | 跨期限冲突原点 | 门禁弃权原点 | 最晚选模标签结束 |",
              "|---|---:|---:|---|"]
    for fold in report["folds"]:
        lines.append(f"| {fold['fold']} | {fold['coherence_conflict_origins']} | {fold['operational_abstention_origins']} | {fold['latest_selection_label_end']} |")
    comp = report.get("comparison_v1", {})
    lines += ["", "## 与冻结 V1 对照", "", "只比较第三轮且不早于 V1 模型可用时间的相同完整 5 日原点。前两轮不能使用当时尚未生成的 V1 模型。"]
    if comp.get("horizons"):
        lines += ["", "| 期限 | 共同原点 | V1 Brier | V2 Brier | V1 运行预警 | V2 运行预警 | V1 告警准入率 | V2 告警准入率 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for h, m in comp["horizons"].items():
            lines.append(f"| {h} | {m['n_common']} | {f(m['v1_brier'])} | {f(m['v2_brier'])} | {m['v1_operational_alerts']} | {m['v2_operational_alerts']} | {f(m['v1_alert_eligibility_coverage'], True)} | {f(m['v2_alert_eligibility_coverage'], True)} |")
    lines += ["", "## 使用边界", ""]
    lines += ["- "+item for item in report["limitations"]]
    lines += ["", "完整逐轮/分任务/分品种原始与运行指标、探索性区间、校验和及固定设置见 report.json。预测 CSV/Parquet 保存弃权原因和每条标签结束日期。", "",
              f"模型 SHA-256：`{report.get('model_sha256', '')}`", ""]
    return "\n".join(lines)
