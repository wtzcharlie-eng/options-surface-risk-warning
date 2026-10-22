"""Coherent five-session surface-proxy forecasts, with chronological calibration.

V1 is deliberately unchanged.  This module predicts a first *future* threshold
exceedance, using the threshold frozen at each origin.  ``persistence`` denotes
origins that are already abnormal; its event is a future recurrence/continuation,
not uninterrupted abnormal duration and not a recovery-time prediction.

All candidates, including the conditional-hazard baseline, receive the same
independent-block hazard-intercept calibration.  Selecting a complete curve per
task/symbol preserves nested cumulative probabilities without post-hoc sorting.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import predictive as v1

SCHEMA_VERSION = 2
HORIZONS = (1, 2, 3, 4, 5)
TASKS = ("onset", "persistence")
MODES = ("hazard_baseline", "joint_logistic", "joint_logistic_decay")
TASK_DEFINITIONS = {
    "onset": "当前两项曲面代理均未超阈值，预测未来窗口内首次出现超阈值。",
    "persistence": "当前至少一项代理已超阈值，预测未来窗口内再次/继续出现超阈值；不代表连续异常持续时长，也不是恢复时间预测。",
}
TARGET_DESCRIPTION = (
    "已完成日盘收盘后，联合预测未来1–5个观测交易日内首次未来超阈值的累计概率。"
    "ATM隐波或凸性违反代理任一日最大值超过预测日前最近60个完整交易日（至少20日）"
    "95%分位即为事件；阈值在预测原点冻结。当前正常与已异常状态分开建模。"
    "所有期限共享一条离散时间风险率曲线，累计概率为1−∏(1−风险率)。"
    "这是曲面异常代理，不是持仓亏损概率或VaR。观测交易日不是官方交易所日历。"
)
FIXED_PARAMETERS = {
    "ridge_penalty": .02,
    "decay_half_life_observed_sessions": 126,
    "baseline_shrinkage_at_risk_rows": 20.,
    "calibration": "same shared hazard-logit intercept for every candidate on independent calibration block; no horizon-wise calibration",
    "calibration_ridge": .01,
    "selection": "per task/symbol minimum mean Brier over the complete five-horizon curve on validation only; ties prefer baseline then uniform logistic",
    "minimum_selection_origins": 25,
    "minimum_selection_event_and_nonevent_origins": 3,
    "minimum_fit_hazard_events_and_nonevents": 10,
    "threshold": "per-task/symbol/horizon validation-only F1; baseline rates do not issue predictive alerts",
    "censoring": "exclude whole origin unless all five future sessions are complete and contiguous",
    "preprocessing": "train-only median/IQR, clipping at 8; binary indicator scale 1",
    "quality_gate": "same whole-curve missingness/dynamic RAW feature OOD fraction >.35 for all candidates; separately withhold any selected RAW feature departure from collapsed training 99% bounds (rtol 1e-6, atol 1e-8)",
}


def build_features(history: pd.DataFrame, symbols: list[str] | None = None):
    """A compact causal multiscale extension of V1's completed-session inputs."""
    out, _ = v1.build_features(history, symbols)
    if out.empty:
        return out, []
    names = ["atm_iv", "convexity_violation", "term_slope", "skew_val",
             "gamma_concentration", "vega_concentration", "arb_score",
             "liquidity_ratio", "n_contracts", "liquidity_ratio_daily_min",
             "night_snapshots", "daily_snapshots", "weekday_sin", "weekday_cos"]
    group = out.groupby("_segment", sort=False)
    for name in ("atm_iv", "convexity_violation"):
        names.extend([f"{name}_daily_max", f"{name}_daily_std",
                      f"{name}_past_z", f"{name}_q95_distance"])
        # All summaries include the current *completed* origin, never its future.
        for window in (5, 20, 60):
            column = f"{name}_mean{window}d_v2"
            out[column] = group[name].transform(
                lambda s, w=window: s.rolling(w, min_periods=min(w, 3)).mean())
            names.append(column)
        for window in (20, 60):
            column = f"{name}_std{window}d_v2"
            out[column] = group[name].transform(
                lambda s, w=window: s.rolling(w, min_periods=3).std())
            names.append(column)
        column = f"{name}_acceleration1d_v2"
        out[column] = out[f"{name}_change1d"] - group[f"{name}_change1d"].shift(1)
        names.extend([f"{name}_change1d", f"{name}_change5d", column])
    names.extend(["term_slope_change1d", "liquidity_ratio_change1d"])
    normal_duration = np.zeros(len(out), dtype=float)
    abnormal_duration = np.zeros(len(out), dtype=float)
    since_exceedance = np.zeros(len(out), dtype=float)
    for indices in group.indices.values():
        normal_run = abnormal_run = since = 0
        for i in indices:
            if not bool(out.iloc[i]._history_ready):
                normal_run = abnormal_run = since = 0
                continue
            if bool(out.iloc[i].current_exceedance):
                abnormal_run += 1
                normal_run = since = 0
            else:
                normal_run += 1
                abnormal_run = 0
                since = min(since+1, 60)
            normal_duration[i], abnormal_duration[i], since_exceedance[i] = normal_run, abnormal_run, since
    out["normal_run_days_v2"] = normal_duration
    out["abnormal_run_days_v2"] = abnormal_duration
    out["days_since_exceedance_capped60_v2"] = since_exceedance
    names.extend(["normal_run_days_v2", "abnormal_run_days_v2", "days_since_exceedance_capped60_v2"])
    syms = sorted(out.symbol.unique()) if symbols is None else list(symbols)
    names.extend(f"symbol_{symbol}" for symbol in syms)
    out[names] = out[names].replace([np.inf, -np.inf], np.nan)
    return out, names


def build_targets(features: pd.DataFrame) -> pd.DataFrame:
    """First event versus the origin's frozen threshold; conservatively censor."""
    out = v1.build_targets(features, HORIZONS)
    eligible = out.label_5.notna()
    out["prediction_task"] = np.where(out.current_exceedance == 0, "onset", "persistence")
    first = np.zeros(len(out), dtype=int)
    group = out.groupby("symbol", sort=False)
    for step in HORIZONS:
        event = ((group.atm_iv_daily_max.shift(-step) > out._atm_iv_q95) |
                 (group.convexity_violation_daily_max.shift(-step) > out._convexity_violation_q95))
        first[(first == 0) & event.to_numpy()] = step
    out["_first_event"] = np.where(eligible, first.astype(float), np.nan)
    for h in HORIZONS:
        out.loc[~eligible, f"label_{h}"] = np.nan
        out.loc[~eligible, f"label_end_{h}"] = pd.NaT
    out["_common_eligible_5d"] = eligible
    return out


def _fit_preprocessing(frame: pd.DataFrame, names: list[str]) -> dict:
    prep = v1._preprocessor_fit(frame, names)
    for i, name in enumerate(names):
        if name.startswith("symbol_"):
            prep["scale"][i] = 1.
    return prep


def _risk_rows(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand origin to person-period rows, stopping *at* its first event."""
    first = frame._first_event.to_numpy(dtype=int)
    origins, steps, outcomes = [], [], []
    for k in HORIZONS:
        at_risk = (first == 0) | (first >= k)
        index = np.flatnonzero(at_risk)
        origins.extend(index.tolist())
        steps.extend([k]*len(index))
        outcomes.extend((first[index] == k).astype(float).tolist())
    return np.asarray(origins, int), np.asarray(steps, int), np.asarray(outcomes, float)


def _design(x: np.ndarray, origins: np.ndarray, steps: np.ndarray) -> np.ndarray:
    return np.column_stack([x[origins], (steps[:, None] == np.arange(2, 6)).astype(float)])


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-8, 1-1e-8)
    return np.log(p/(1-p))


def _weighted_logistic(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> dict:
    from scipy.optimize import minimize
    weights = np.asarray(weights, float)
    weights /= weights.sum()
    initial = np.zeros(x.shape[1]+1)
    initial[0] = float(_logit((float(y.sum())+.5)/(len(y)+1)))
    penalty = FIXED_PARAMETERS["ridge_penalty"]

    def objective(w):
        z = w[0] + x @ w[1:]
        residual = weights*(v1._sigmoid(z)-y)
        loss = np.sum(weights*(np.logaddexp(0, z)-y*z)) + penalty*np.sum(w[1:]**2)/2
        gradient = np.r_[residual.sum(), x.T@residual + penalty*w[1:]]
        return float(loss), gradient

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 600, "ftol": 1e-10, "gtol": 1e-6})
    return {"intercept": float(result.x[0]), "coefficients": result.x[1:].tolist(),
            "converged": bool(result.success), "iterations": int(result.nit),
            "optimizer_message": str(result.message)}


def _fit_baseline(frame: pd.DataFrame, symbols: list[str]) -> dict:
    origins, steps, y = _risk_rows(frame)
    symbol_rows = frame.symbol.to_numpy()[origins]
    pooled, conditional, support = [], {}, {}
    for step in HORIZONS:
        hit = steps == step
        pooled.append(float((y[hit].sum()+1)/(hit.sum()+2)))
    shrink = FIXED_PARAMETERS["baseline_shrinkage_at_risk_rows"]
    for symbol in symbols:
        conditional[symbol], support[symbol] = [], []
        for step in HORIZONS:
            hit = (steps == step) & (symbol_rows == symbol)
            conditional[symbol].append(float((y[hit].sum()+shrink*pooled[step-1])/(hit.sum()+shrink)))
            support[symbol].append(int(hit.sum()))
    return {"mode": "hazard_baseline", "pooled_hazards": pooled,
            "hazards_by_symbol": conditional, "support_by_symbol": support,
            "calibration": {"intercept": 0., "applied": False}}


def _candidate_logits(frame: pd.DataFrame, names: list[str], prep: dict, candidate: dict) -> np.ndarray:
    if candidate["mode"] == "hazard_baseline":
        p = np.array([candidate["hazards_by_symbol"].get(s, candidate["pooled_hazards"])
                      for s in frame.symbol], dtype=float).reshape(len(frame), 5)
        return _logit(p)
    x = v1._transform(frame, names, prep)
    coefficients = np.asarray(candidate["coefficients"], float)
    base = candidate["intercept"] + x@coefficients[:len(names)]
    step = np.r_[0., coefficients[len(names):]]
    return base[:, None]+step[None, :]


def _fit_calibration(frame: pd.DataFrame, names: list[str], prep: dict, candidate: dict) -> dict:
    """One intercept recalibrates hazards jointly; cumulative nesting survives."""
    from scipy.optimize import minimize_scalar
    if not len(frame):
        return {"intercept": 0., "applied": False, "reason": "empty calibration task", "at_risk_rows": 0}
    origins, steps, y = _risk_rows(frame)
    z = _candidate_logits(frame, names, prep, candidate)[origins, steps-1]
    if y.sum() < 5 or (1-y).sum() < 5:
        return {"intercept": 0., "applied": False, "reason": "fewer than five hazard events or nonevents", "at_risk_rows": len(y)}
    penalty = FIXED_PARAMETERS["calibration_ridge"]
    result = minimize_scalar(lambda offset: float(np.mean(np.logaddexp(0, z+offset)-y*(z+offset)) + penalty*offset**2/2),
                             method="bounded", bounds=(-4., 4.), options={"xatol": 1e-8})
    return {"intercept": float(result.x) if result.success else 0.,
            "applied": bool(result.success), "at_risk_rows": len(y),
            "events": int(y.sum()), "method": "shared hazard-logit intercept; ridge .01; bounded [-4,4]"}


def _hazards(frame: pd.DataFrame, names: list[str], prep: dict, candidate: dict):
    return v1._sigmoid(_candidate_logits(frame, names, prep, candidate) + candidate["calibration"]["intercept"])


def _cumulative(hazards: np.ndarray):
    return 1-np.cumprod(1-hazards, axis=1)


def _validate_training_blocks(train, calibration, validation):
    if not len(train):
        raise ValueError("Training block is empty")
    for name, frame in (("train", train), ("calibration", calibration), ("validation", validation)):
        if frame.label_5.isna().any() or frame._first_event.isna().any():
            raise ValueError(f"{name} includes censored five-session origins")
        first = frame._first_event.to_numpy(float)
        if not np.isin(first, np.arange(6)).all():
            raise ValueError(f"{name} has invalid first-event steps")
        if not np.isin(frame.prediction_task.to_numpy(object), TASKS).all():
            raise ValueError(f"{name} has an unknown prediction task")
        expected = ((first[:, None] > 0) & (first[:, None] <= np.arange(1, 6))).astype(float)
        if not np.array_equal(frame[[f"label_{h}" for h in HORIZONS]].to_numpy(float), expected):
            raise ValueError(f"{name} labels disagree with first-event step")
        if frame.label_end_5.isna().any() or not (frame.label_end_5 > frame._dt).all():
            raise ValueError(f"{name} lacks complete future label-end timestamps")
    for left, right in ((train, calibration), (calibration, validation), (train, validation)):
        if len(left) and len(right) and not left.label_end_5.max() < right._dt.min():
            raise ValueError("Chronological blocks overlap future label windows; purge using label_end_5")


def fit_models(train: pd.DataFrame, calibration: pd.DataFrame, validation: pd.DataFrame,
               names: list[str], symbols: list[str] | None = None) -> dict:
    """Fit/calibrate/select solely on supplied purged blocks, never test data."""
    _validate_training_blocks(train, calibration, validation)
    symbols = sorted(train.symbol.unique()) if symbols is None else list(symbols)
    preprocessing = _fit_preprocessing(train, names)
    bundle = {"tasks": {}, "preprocessing": preprocessing, "symbols": symbols,
              "feature_names": list(names), "fixed_parameters": dict(FIXED_PARAMETERS)}
    for task in TASKS:
        tr = train.loc[train.prediction_task == task].reset_index(drop=True)
        ca = calibration.loc[calibration.prediction_task == task].reset_index(drop=True)
        va = validation.loc[validation.prediction_task == task].reset_index(drop=True)
        baseline = _fit_baseline(tr, symbols)
        candidates = {"hazard_baseline": baseline}
        origins, steps, y = _risk_rows(tr)
        if y.sum() >= 10 and (1-y).sum() >= 10:
            x = _design(v1._transform(tr, names, preprocessing), origins, steps)
            unique_dates = np.array(sorted(train._dt.dt.normalize().unique()))
            date_rank = np.searchsorted(unique_dates, tr._dt.dt.normalize().to_numpy())
            age = (len(unique_dates)-1)-date_rank
            for mode in MODES[1:]:
                weights = (np.exp2(-age[origins]/126.) if mode.endswith("_decay") else np.ones(len(y)))
                candidate = _weighted_logistic(x, y, weights)
                candidate["mode"] = mode
                candidate["calibration"] = {"intercept": 0., "applied": False}
                if candidate["converged"]:
                    candidates[mode] = candidate
        for candidate in candidates.values():
            candidate["calibration"] = _fit_calibration(ca, names, preprocessing, candidate)
        task_model = {"definition": TASK_DEFINITIONS[task], "candidate_models": candidates,
                      "mode_by_symbol": {}, "validation_selection": {}, "thresholds_by_symbol": {},
                      "training_origins": len(tr), "calibration_origins": len(ca), "validation_origins": len(va)}
        for symbol in symbols:
            sub = va.loc[va.symbol == symbol]
            support = len(sub)
            event_support = int((sub._first_event > 0).sum())
            supported = support >= 25 and event_support >= 3 and support-event_support >= 3
            scores, curve_by_mode = {}, {}
            y_curve = sub[[f"label_{h}" for h in HORIZONS]].to_numpy(float)
            for mode, candidate in candidates.items():
                curve = _cumulative(_hazards(sub, names, preprocessing, candidate))
                curve_by_mode[mode] = curve
                scores[mode] = float(np.mean((y_curve-curve)**2)) if support else None
            mode = (min((scores[m], MODES.index(m), m) for m in candidates)[2]
                    if supported else "hazard_baseline")
            task_model["mode_by_symbol"][symbol] = mode
            task_model["validation_selection"][symbol] = {
                "n": support, "event_origins_5d": event_support, "candidate_mean_brier": scores,
                "selected_mode": mode, "supported": supported,
                "reason": ("minimum validation mean Brier over whole five-day curve" if supported else
                           "insufficient validation origins/classes: conditional hazard baseline fallback")}
            chosen = curve_by_mode[mode]
            task_model["thresholds_by_symbol"][symbol] = [
                v1._threshold(y_curve[:, h-1].astype(int), chosen[:, h-1]) if supported and mode != "hazard_baseline" else
                {"value": 1.000001, "objective": "historical baseline or insufficient validation support: predictive alerts disabled", "f1": 0.}
                for h in HORIZONS]
        bundle["tasks"][task] = task_model
    return bundle


def predict_curves(frame: pd.DataFrame, names: list[str], model_bundle: dict,
                   mode: str | None = None) -> dict:
    """Predict one whole coherent curve, or a specified fitted candidate."""
    if mode is not None and mode not in MODES:
        raise ValueError(f"Unknown V2 mode: {mode}")
    n = len(frame)
    probabilities = np.zeros((n, 5)); baseline_p = np.zeros((n, 5))
    hazards = np.zeros((n, 5)); thresholds = np.full((n, 5), 1.000001)
    modes = np.empty(n, object)
    tasks = (frame.prediction_task.to_numpy(object) if "prediction_task" in frame else
             np.where(frame.current_exceedance.to_numpy() == 0, "onset", "persistence"))
    if not np.isin(tasks, TASKS).all():
        raise ValueError("Unknown prediction task; expected onset or persistence")
    for task in TASKS:
        task_model = model_bundle["tasks"][task]
        task_indices = np.flatnonzero(tasks == task)
        sub = frame.iloc[task_indices]
        if not len(sub):
            continue
        baseline = task_model["candidate_models"]["hazard_baseline"]
        baseline_p[task_indices] = _cumulative(_hazards(sub, names, model_bundle["preprocessing"], baseline))
        requested = np.array([mode or task_model["mode_by_symbol"].get(s, "hazard_baseline") for s in sub.symbol], object)
        for selected_mode in set(requested):
            local = np.flatnonzero(requested == selected_mode)
            candidate = task_model["candidate_models"].get(selected_mode, baseline)
            actual_mode = candidate["mode"]
            if task_model.get("training_origins", 0) == 0:
                actual_mode = "unsupported_task"
            target_indices = task_indices[local]
            candidate_rows = sub.iloc[local]
            h = _hazards(candidate_rows, names, model_bundle["preprocessing"], candidate)
            hazards[target_indices], probabilities[target_indices], modes[target_indices] = h, _cumulative(h), actual_mode
            for index, symbol in zip(target_indices, candidate_rows.symbol):
                records = task_model["thresholds_by_symbol"].get(symbol, [{"value": 1.000001}]*5)
                thresholds[index] = [record["value"] for record in records]
    alerts = (probabilities >= thresholds) & np.isin(modes[:, None], MODES[1:])
    return {"probabilities": probabilities, "baseline_probabilities": baseline_p,
            "hazards": hazards, "alerts": alerts, "modes": modes,
            "tasks": tasks, "thresholds": thresholds}


def operational_gates(frame: pd.DataFrame, names: list[str], models: dict) -> dict:
    """Same whole-curve quality gate in retrospective evaluation and inference."""
    n = len(frame)
    missing = frame[list(v1.RAW_FEATURES)].isna().mean(axis=1).to_numpy(float)
    raw = frame[names].to_numpy(float)
    low, high = np.asarray(models["preprocessing"]["low"]), np.asarray(models["preprocessing"]["high"])
    monitored = np.array([name in v1.RAW_FEATURES for name in names])[None, :] & np.isfinite(raw) & (high > low)[None, :]
    outside = monitored & ((raw < low) | (raw > high))
    fractions = outside.sum(axis=1)/np.maximum(monitored.sum(axis=1), 1)
    # A collapsed training range carries no useful IQR scale.  Keep the dynamic
    # feature fraction unchanged and abstain separately when a formerly constant
    # raw input departs beyond numerical tolerance.  It must not dilute the
    # dynamic-feature denominator or silently pass through transformation clips.
    constant_monitored = (np.array([name in v1.RAW_FEATURES for name in names])[None, :]
                          & np.isfinite(raw) & (high == low)[None, :])
    constant_outside = constant_monitored & ~np.isclose(raw, low[None, :], rtol=1e-6, atol=1e-8)
    constant_violation = constant_outside.any(axis=1)
    core_finite = np.isfinite(frame[["atm_iv", "convexity_violation"]].to_numpy(float)).all(axis=1)
    history_ready = frame._history_ready.fillna(False).to_numpy(bool)
    tasks = (frame.prediction_task.to_numpy(object) if "prediction_task" in frame else
             np.where(frame.current_exceedance.to_numpy() == 0, "onset", "persistence"))
    if not np.isin(tasks, TASKS).all():
        raise ValueError("Unknown prediction task; expected onset or persistence")
    supported_task = np.array([models["tasks"][task].get("training_origins", 0) > 0 for task in tasks])
    allowed = ((missing <= .35) & (fractions <= .35) & core_finite & history_ready
               & supported_task & ~constant_violation)
    reasons = []
    for i in range(n):
        reasons.append("unsupported_task" if not supported_task[i] else
                       "insufficient_history_or_missing_core" if not history_ready[i] or not core_finite[i] else
                       "raw_feature_missing" if missing[i] > .35 else
                       "constant_feature_violation" if constant_violation[i] else
                       "out_of_distribution" if fractions[i] > .35 else None)
    return {"allowed": allowed, "reasons": reasons, "ood_fraction": fractions,
            "ood_features": [[name for name, flag in zip(names, row) if flag] for row in outside],
            "constant_features_violated": [[name for name, flag in zip(names, row) if flag] for row in constant_outside],
            "raw_feature_missing_fraction": missing}


def load_artifact(path: str | Path = "data_out/predictive_v2/model.json") -> dict:
    """Numeric JSON only, with checked dimensions and finite persisted arrays."""
    artifact = json.loads(Path(path).read_text(encoding="utf-8"),
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite JSON number {value}")))
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported predictive V2 schema")
    names, bundle = artifact.get("feature_names"), artifact.get("models")
    if not isinstance(names, list) or not names or not isinstance(bundle, dict):
        raise ValueError("Invalid predictive V2 artifact")
    width = len(names)
    prep = bundle.get("preprocessing", {})
    for key in ("median", "scale", "low", "high"):
        array = np.asarray(prep.get(key, []), dtype=float)
        if array.shape != (width,) or not np.isfinite(array).all():
            raise ValueError(f"Invalid V2 preprocessing {key}")
        if key == "scale" and (array <= 0).any():
            raise ValueError("Non-positive V2 scale")
    if not np.isfinite(float(prep.get("clip", 0))) or float(prep.get("clip", 0)) <= 0:
        raise ValueError("Invalid V2 preprocessing clip")
    for task in TASKS:
        task_model = bundle.get("tasks", {}).get(task)
        if not isinstance(task_model, dict) or "hazard_baseline" not in task_model.get("candidate_models", {}):
            raise ValueError(f"Missing V2 {task} baseline")
        for mode, candidate in task_model["candidate_models"].items():
            if mode not in MODES or candidate.get("mode") != mode:
                raise ValueError("Unknown V2 candidate mode")
            if not np.isfinite(float(candidate["calibration"]["intercept"])):
                raise ValueError("Non-finite V2 calibration")
            if mode == "hazard_baseline":
                for values in [candidate["pooled_hazards"], *candidate["hazards_by_symbol"].values()]:
                    array = np.asarray(values, float)
                    if array.shape != (5,) or not np.isfinite(array).all() or ((array <= 0)|(array >= 1)).any():
                        raise ValueError("Invalid baseline hazard curve")
            else:
                array = np.asarray(candidate["coefficients"], float)
                if array.shape != (width+4,) or not np.isfinite(array).all() or not np.isfinite(float(candidate["intercept"])):
                    raise ValueError("Invalid V2 coefficients")
        for records in task_model.get("thresholds_by_symbol", {}).values():
            if len(records) != 5 or not np.isfinite([r["value"] for r in records]).all():
                raise ValueError("Invalid V2 threshold curve")
    return artifact


def forecast_latest(artifact: dict, history: pd.DataFrame, symbol: str, as_of: Any = None,
                    max_staleness_minutes: float = 4*24*60) -> dict:
    """Latest completed-session forecast with shared quality/freshness gates."""
    symbol = symbol.lower()
    when = v1._local_timestamp(as_of)
    if max_staleness_minutes < 0:
        raise ValueError("max_staleness_minutes must be non-negative")
    result = {"status": "unavailable", "symbol": symbol, "as_of": when.isoformat(),
              "model_version": "v2", "target_description": artifact.get("target_description", TARGET_DESCRIPTION),
              "data_timestamp": None, "age_minutes": None, "horizons": [], "signals": {},
              "warnings": [], "quality": {}, "evaluation": {str(h): {} for h in HORIZONS},
              "prediction_task": None, "prediction_task_definition": None}
    if symbol not in artifact["symbols"]:
        result["warnings"].append("该品种没有经过训练和验证的联合风险模型。")
        return result
    available_from = artifact.get("model_available_from")
    if available_from and when < pd.Timestamp(available_from):
        result["quality"]["model_available_from"] = available_from
        result["warnings"].append("该时点早于模型训练、校准和选择完成时间，暂停预测。")
        return result
    normal = v1._normalise(history)
    normal = normal.loc[normal._dt <= when]
    if normal.empty or not (normal.symbol == symbol).any():
        result["warnings"].append("指定时点之前该品种没有可用观测。")
        return result
    features, _ = build_features(normal, artifact["symbols"])
    if features.empty or not (features.symbol == symbol).any():
        result["warnings"].append("尚无包含全部15个日盘截面的完整交易日，暂停日级预测。")
        return result
    calendar = sorted(features._trading_date.unique())
    row = features.loc[features.symbol == symbol].iloc[[-1]].copy()
    latest = row.iloc[0]
    age = float((when-latest._dt).total_seconds()/60)
    task = "onset" if latest.current_exceedance == 0 else "persistence"
    row["prediction_task"] = task
    result.update({"data_timestamp": latest._dt.isoformat(), "age_minutes": age,
                   "age_calendar_days": age/(24*60), "trading_date": latest._trading_date.date().isoformat(),
                   "prediction_task": task, "prediction_task_definition": TASK_DEFINITIONS[task]})
    result["evaluation_details"] = {}
    for h in HORIZONS:
        overall = artifact.get("per_symbol_test_metrics", {}).get(str(h), {}).get(symbol, {})
        details = overall.get("task_metrics", {}).get(task, overall)
        result["evaluation"][str(h)] = details.get("test", details)
        result["evaluation_details"][str(h)] = details
    missing_dates = [pd.Timestamp(d).date().isoformat() for d in calendar if pd.Timestamp(d) > latest._trading_date]
    pending = normal.loc[(normal.symbol == symbol) & (normal._dt > latest._dt)]
    result["quality"].update({"last_completed_session_snapshots": int(latest.daily_snapshots),
        "past_observations": int(latest._history_count), "newer_completed_dates_without_symbol": missing_dates,
        "pending_incomplete_session_snapshots": len(pending), "horizon_probabilities_consistent": True})
    result["signals"] = {name: float(latest[name]) if pd.notna(latest[name]) else None for name in v1.RAW_FEATURES}
    result["signals"].update({"atm_iv_past_q95": float(latest._atm_iv_q95) if pd.notna(latest._atm_iv_q95) else None,
        "convexity_violation_past_q95": float(latest._convexity_violation_q95) if pd.notna(latest._convexity_violation_q95) else None,
        "currently_exceeding": bool(latest.current_exceedance), "atm_iv_daily_max": float(latest.atm_iv_daily_max),
        "convexity_violation_daily_max": float(latest.convexity_violation_daily_max),
        "signal_description": "曲面观测与严格历史阈值；特征贡献只解释标注交易日的条件风险率log-odds，不解释累计概率log-odds，也不代表因果。"})
    if len(pending):
        result["warnings"].append("最新交易时段尚未完成，预测基于上一完整日盘收盘。")
    if age > max_staleness_minutes or missing_dates:
        result["status"] = "stale"
        result["warnings"].append("数据已过期或该品种缺少更新完整交易日，暂停当前预测。")
        return result
    names, models = artifact["feature_names"], artifact["models"]
    if models["tasks"][task].get("training_origins", 0) == 0:
        result["quality"]["whole_curve_gate_reason"] = "unsupported_task"
        result["warnings"].append("训练期没有当前状态任务的观测，无法给出经过验证的条件风险率，暂停整条曲线。")
        return result
    gates = operational_gates(row, names, models)
    result["quality"].update({"raw_feature_missing_fraction": float(gates["raw_feature_missing_fraction"][0]),
        "maximum_ood_fraction": float(gates["ood_fraction"][0]), "ood_features": gates["ood_features"][0],
        "constant_features_violated": gates["constant_features_violated"][0],
        "whole_curve_gate_reason": gates["reasons"][0]})
    if not gates["allowed"][0]:
        reason = gates["reasons"][0]
        result["status"] = "out_of_distribution" if reason in ("out_of_distribution", "constant_feature_violation") else "unavailable"
        result["warnings"].append("核心特征超出训练范围、历史不足或特征缺失过多；整条期限曲线暂停。")
        return result
    pred = predict_curves(row, names, models)
    mode = str(pred["modes"][0])
    task_model = models["tasks"][task]
    candidate = task_model["candidate_models"][mode]
    result["status"] = "degraded" if mode == "hazard_baseline" or gates["ood_fraction"][0] > 0 else "ready"
    if mode == "hazard_baseline":
        result["warnings"].append("当前任务使用经过独立校准的历史条件风险率基准；未验证出特征模型增量，预测型预警已关闭。")
    if gates["ood_fraction"][0] > 0:
        result["warnings"].append("部分观测超出训练期99%范围，请结合数据质量解读。")
    raw, x = row[names].to_numpy(float)[0], v1._transform(row, names, models["preprocessing"])[0]
    contribution = None if mode == "hazard_baseline" else np.asarray(candidate["coefficients"][:len(names)])*x
    for h in HORIZONS:
        evaluation = result["evaluation"][str(h)]
        skill = evaluation.get("brier_skill")
        item = {"days": h, "probability": float(pred["probabilities"][0, h-1]),
            "hazard_probability": float(pred["hazards"][0, h-1]), "threshold": float(pred["thresholds"][0, h-1]),
            "alert": bool(pred["alerts"][0, h-1]), "model_status": mode,
            "baseline_probability": float(pred["baseline_probabilities"][0, h-1]),
            "prediction_task": task, "prediction_task_definition": TASK_DEFINITIONS[task], "evaluation": evaluation,
            "heldout_skill_status": "unavailable" if skill is None else "positive" if skill > 0 else "nonpositive",
            "incremental_skill_vs_state_status": "unavailable" if skill is None else "positive" if skill > 0 else "nonpositive",
            "forecast_origin": latest._dt.isoformat(), "horizon_basis": "之后的观测交易日；未来确切日期未知",
            "ood_fraction": float(gates["ood_fraction"][0]), "ood_features": gates["ood_features"][0],
            "interpretation": ("当前任务训练期条件风险率经独立校准后的历史统计曲线；不宣称特征增量" if mode == "hazard_baseline" else
                               "未来窗口首次超阈值累计概率；由各日条件风险率联合合成"),
            "top_contributions": [], "intercept_log_odds": None, "total_log_odds": None,
            "contribution_scale": "conditional_hazard_log_odds", "contribution_hazard_step": h}
        if contribution is not None:
            coefficients = np.asarray(candidate["coefficients"])
            step_intercept = 0. if h == 1 else float(coefficients[len(names)+h-2])
            intercept = float(candidate["intercept"] + candidate["calibration"]["intercept"] + step_intercept)
            item.update({"hazard_intercept_log_odds": intercept,
                         "hazard_total_log_odds": float(intercept+contribution.sum()),
                         "hazard_contribution_sum": float(contribution.sum()),
                         "top_contributions": [{"feature": names[i], "value": float(raw[i]) if np.isfinite(raw[i]) else None,
                            "imputed": bool(not np.isfinite(raw[i])), "log_odds_contribution": float(contribution[i]),
                            "hazard_step": h, "scale": "hazard_log_odds", "contribution_scale": "conditional_hazard_log_odds"}
                            for i in np.argsort(-np.abs(contribution))[:6]]})
        result["horizons"].append(item)
    if any((item["evaluation"].get("brier_skill") or 0) < 0 for item in result["horizons"]):
        result["warnings"].append("部分期限在保留测试段未超过同任务条件风险率基准；测试结果没有用于回头选择模型。")
    return result
