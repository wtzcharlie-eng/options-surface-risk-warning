"""V3: independently calibrated, baseline-anchored coherent hazard curves.

V1/V2 remain frozen. Fixed experiments are documented in docs/V3实验设计.md.
No test argument exists in fitting; the selection unit is a whole task curve.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import predictive as v1
from . import predictive_v2 as v2

SCHEMA_VERSION = 3
HORIZONS, TASKS = v2.HORIZONS, v2.TASKS
TASK_DEFINITIONS, TARGET_DESCRIPTION = v2.TASK_DEFINITIONS, v2.TARGET_DESCRIPTION
build_features, build_targets, operational_gates = v2.build_features, v2.build_targets, v2.operational_gates
WEIGHTS = (.25, .50)
MODES = ("hazard_baseline", "joint_logistic_shrunk", "joint_logistic_decay_shrunk")
FIXED_PARAMETERS = {
    "base_training": copy.deepcopy(v2.FIXED_PARAMETERS),
    "calibration": "independent task affine hazard-logit plus shrunk symbol offsets; all candidates equally treated",
    "calibration_slope_bounds": [0., 1.], "calibration_offset_bounds": [-4., 4.],
    "calibration_intercept_ridge": .01, "calibration_slope_ridge": .05,
    "symbol_offset_prior_at_risk_rows": 20.,
    "feature_blend_weights": list(WEIGHTS), "blend_space": "conditional hazard probabilities before survival product",
    "selection": "pooled per task; maximize paired validation mean Brier gain minus one nonoverlapping 10-date-block standard error; positive margin required; ties prefer baseline/smaller weight/uniform model",
    "selection_minimum_origins": 60, "selection_minimum_event_nonevent": 10,
    "selection_block_dates": 10, "selection_minimum_blocks": 4,
    "threshold": "task-pooled validation F1; baseline does not issue predictive alerts",
}


def _fit_calibration(frame, names, prep, candidate, symbols):
    """Constrained, regularized affine logit + symbol intercepts; calibration only."""
    from scipy.optimize import minimize
    origins, steps, y = v2._risk_rows(frame)
    identity = {"intercept": 0., "slope": 1., "symbol_offsets": {s: 0. for s in symbols},
                "applied": False, "at_risk_rows": len(y), "events": int(y.sum())}
    if y.sum() < 5 or (1-y).sum() < 5:
        return {**identity, "reason": "fewer than five hazard events or nonevents"}
    z = v2._candidate_logits(frame, names, prep, candidate)[origins, steps-1]
    symbol_index = pd.Index(symbols).get_indexer(frame.symbol.to_numpy()[origins])
    if (symbol_index < 0).any():
        raise ValueError("Calibration includes unknown symbol")
    n = len(y)
    initial = np.r_[0., 1., np.zeros(len(symbols))]

    def objective(w):
        eta = w[0] + w[1]*z + w[2:][symbol_index]
        residual = (v1._sigmoid(eta)-y)/n
        loss = (np.mean(np.logaddexp(0, eta)-y*eta) + .01*w[0]**2/2
                + .05*(w[1]-1)**2/2 + 20./n*np.sum(w[2:]**2)/2)
        gradient = np.r_[residual.sum()+.01*w[0], (residual*z).sum()+.05*(w[1]-1),
                         np.bincount(symbol_index, weights=residual, minlength=len(symbols))+20./n*w[2:]]
        return float(loss), gradient

    fitted = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(-4., 4.), (0., 1.)]+[(-4., 4.)]*len(symbols),
                      options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-7})
    if not fitted.success or not np.isfinite(fitted.x).all():
        return {**identity, "reason": "calibration solver failed; identity fallback"}
    return {**identity, "intercept": float(fitted.x[0]), "slope": float(fitted.x[1]),
            "symbol_offsets": dict(zip(symbols, fitted.x[2:].tolist())), "applied": True,
            "iterations": int(fitted.nit), "optimizer_message": str(fitted.message)}


def _hazards(frame, names, prep, candidate):
    z = v2._candidate_logits(frame, names, prep, candidate)
    calibration = candidate["calibration"]
    offsets = frame.symbol.map(calibration["symbol_offsets"]).fillna(0.).to_numpy(float)
    return v1._sigmoid(calibration["intercept"] + calibration["slope"]*z + offsets[:, None])


def _selection(frame, curves, market_dates=None):
    """Conservative paired block margin; the test set never enters this function."""
    y = frame[[f"label_{h}" for h in HORIZONS]].to_numpy(float)
    loss = {key: np.mean((y-p)**2, axis=1) for key, p in curves.items()}
    dates = pd.to_datetime(frame._dt).dt.normalize()
    unique = pd.Index(sorted(dates.unique() if market_dates is None else pd.to_datetime(market_dates)))
    if (unique.get_indexer(dates) < 0).any():
        raise ValueError("Selection calendar omits an origin")
    blocks = unique.get_indexer(dates)//10
    events = int((frame._first_event > 0).sum())
    supported = len(frame) >= 60 and min(events, len(frame)-events) >= 10 and len(set(blocks)) >= 4
    records = {}
    baseline = ("hazard_baseline", 0.)
    for key, values in loss.items():
        gains = loss[baseline]-values
        block_gains = pd.Series(gains).groupby(blocks).mean().to_numpy()
        se = float(block_gains.std(ddof=1)/np.sqrt(len(block_gains))) if len(block_gains) > 1 else None
        gain = float(gains.mean()) if len(gains) else None
        records[f"{key[0]}@{key[1]:.2f}"] = {
            "candidate": key[0], "weight": key[1], "mean_brier": float(values.mean()) if len(values) else None,
            "gain_vs_baseline": gain, "block_standard_error": se,
            "selection_margin": gain-se if gain is not None and se is not None else None}
    eligible = [key for key in loss if key != baseline and supported
                and (records[f"{key[0]}@{key[1]:.2f}"]["selection_margin"] or 0) > 1e-12]
    selected = min(eligible, key=lambda key: (-records[f"{key[0]}@{key[1]:.2f}"]["selection_margin"], key[1], v2.MODES.index(key[0]))) if eligible else baseline
    return selected, {"n": len(frame), "events_5d": events, "nonoverlapping_date_blocks": len(set(blocks)),
                      "supported": supported, "candidates": records,
                      "selected_candidate": selected[0], "selected_weight": selected[1],
                      "reason": "positive paired validation gain minus block SE" if eligible else "insufficient support or no positive conservative margin: baseline"}


def fit_models(train, calibration, validation, names, symbols=None):
    """Reuse frozen training candidates; replace only calibration/curve selection."""
    bundle = v2.fit_models(train, calibration, validation, names, symbols)
    bundle["fixed_parameters"] = copy.deepcopy(FIXED_PARAMETERS)
    prep, symbols = bundle["preprocessing"], bundle["symbols"]
    for task in TASKS:
        task_model = bundle["tasks"][task]
        ca = calibration.loc[calibration.prediction_task == task].reset_index(drop=True)
        va = validation.loc[validation.prediction_task == task].reset_index(drop=True)
        for candidate in task_model["candidate_models"].values():
            candidate["calibration"] = _fit_calibration(ca, names, prep, candidate, symbols)
        baseline_h = _hazards(va, names, prep, task_model["candidate_models"]["hazard_baseline"])
        curves = {("hazard_baseline", 0.): v2._cumulative(baseline_h)}
        for mode, candidate in task_model["candidate_models"].items():
            if mode == "hazard_baseline":
                continue
            hz = _hazards(va, names, prep, candidate)
            for weight in WEIGHTS:
                curves[(mode, weight)] = v2._cumulative((1-weight)*baseline_h+weight*hz)
        (mode, weight), record = _selection(va, curves, validation._dt.dt.normalize().unique())
        chosen = curves[(mode, weight)]
        task_model["selected_candidate"], task_model["blend_weight"] = mode, weight
        task_model["validation_selection"] = record
        actual = mode+"_shrunk" if weight else "hazard_baseline"
        task_model["mode_by_symbol"] = {s: actual for s in symbols}
        threshold_records = [v1._threshold(va[f"label_{h}"].to_numpy(int), chosen[:, h-1])
                             if weight else {"value": 1.000001, "objective": "statistical reference: alerts disabled", "f1": 0.}
                             for h in HORIZONS]
        task_model["thresholds_by_symbol"] = {s: copy.deepcopy(threshold_records) for s in symbols}
    return bundle


def predict_curves(frame, names, model_bundle, mode=None, weight=None):
    if mode is not None and mode not in v2.MODES:
        raise ValueError("Unknown V3 source candidate")
    if weight is not None and weight not in (0., *WEIGHTS):
        raise ValueError("Unknown V3 blend weight")
    tasks = (frame.prediction_task.to_numpy(object) if "prediction_task" in frame else
             np.where(frame.current_exceedance.to_numpy() == 0, "onset", "persistence"))
    if not np.isin(tasks, TASKS).all():
        raise ValueError("Unknown prediction task")
    n = len(frame)
    probabilities, baseline_p, hazards = [np.zeros((n, 5)) for _ in range(3)]
    thresholds, modes, weights = np.full((n, 5), 1.000001), np.empty(n, object), np.zeros(n)
    for task in TASKS:
        indices = np.flatnonzero(tasks == task)
        if not len(indices):
            continue
        model = model_bundle["tasks"][task]
        sub = frame.iloc[indices]
        candidate_name = mode or model["selected_candidate"]
        if candidate_name not in model["candidate_models"]:
            raise ValueError("Requested source candidate was not fitted")
        alpha = model["blend_weight"] if weight is None else weight
        if candidate_name == "hazard_baseline":
            alpha = 0.
        base = _hazards(sub, names, model_bundle["preprocessing"], model["candidate_models"]["hazard_baseline"])
        feature = _hazards(sub, names, model_bundle["preprocessing"], model["candidate_models"][candidate_name])
        mixed = (1-alpha)*base+alpha*feature
        hazards[indices], probabilities[indices], baseline_p[indices] = mixed, v2._cumulative(mixed), v2._cumulative(base)
        status = candidate_name+"_shrunk" if alpha else "hazard_baseline"
        modes[indices] = status if model["training_origins"] else "unsupported_task"
        weights[indices] = alpha
        for index, symbol in zip(indices, sub.symbol):
            thresholds[index] = [r["value"] for r in model["thresholds_by_symbol"].get(symbol, [{"value": 1.000001}]*5)]
    return {"probabilities": probabilities, "baseline_probabilities": baseline_p, "hazards": hazards,
            "thresholds": thresholds, "modes": modes, "tasks": tasks, "blend_weights": weights,
            "alerts": (probabilities >= thresholds) & np.isin(modes[:, None], MODES[1:])}


def _load_artifact(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"),
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Non-finite JSON number")))
    if not isinstance(artifact, dict) or type(artifact.get("schema_version")) is not int or artifact["schema_version"] != 3:
        raise ValueError("Unsupported V3 artifact")
    names, models = artifact.get("feature_names"), artifact.get("models")
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or not isinstance(models, dict):
        raise ValueError("Invalid V3 feature/model schema")
    if models.get("feature_names") != names or artifact.get("symbols") != models.get("symbols"):
        raise ValueError("Inconsistent V3 metadata")
    prep = models.get("preprocessing", {})
    for key in ("median", "scale", "low", "high"):
        arr = np.asarray(prep.get(key, []), float)
        if arr.shape != (len(names),) or not np.isfinite(arr).all() or (key == "scale" and (arr <= 0).any()):
            raise ValueError("Invalid V3 preprocessing")
    if not np.isfinite(float(prep.get("clip", 0))) or prep.get("clip", 0) <= 0 or (np.asarray(prep['low']) > np.asarray(prep['high'])).any():
        raise ValueError("Invalid V3 preprocessing bounds")
    for task in TASKS:
        m = models.get("tasks", {}).get(task, {})
        candidates = m.get("candidate_models", {})
        if type(m.get('training_origins')) is not int or m['training_origins'] < 0:
            raise ValueError('Invalid V3 task support')
        if "hazard_baseline" not in candidates or m.get("selected_candidate") not in candidates or m.get("blend_weight") not in (0., *WEIGHTS):
            raise ValueError("Invalid V3 selected curve")
        if (m["selected_candidate"] == "hazard_baseline") != (m["blend_weight"] == 0):
            raise ValueError("Inconsistent V3 baseline weight")
        actual = m['selected_candidate']+'_shrunk' if m['blend_weight'] else 'hazard_baseline'
        if m.get('mode_by_symbol') != {s: actual for s in artifact['symbols']}:
            raise ValueError('Inconsistent V3 task mode metadata')
        if set(m.get('thresholds_by_symbol', {})) != set(artifact['symbols']):
            raise ValueError('Missing V3 symbol threshold')
        for name, c in candidates.items():
            if name not in v2.MODES or c.get("mode") != name:
                raise ValueError("Unknown V3 source candidate")
            cal = c["calibration"]
            values = [cal["intercept"], cal["slope"], *cal["symbol_offsets"].values()]
            if not np.isfinite(values).all() or not 0 <= cal["slope"] <= 1 or abs(cal["intercept"]) > 4 or any(abs(x) > 4 for x in cal['symbol_offsets'].values()):
                raise ValueError("Invalid V3 calibration")
            if set(cal['symbol_offsets']) != set(artifact['symbols']):
                raise ValueError("Missing V3 symbol calibration")
            if name == "hazard_baseline":
                if set(c['hazards_by_symbol']) != set(artifact['symbols']):
                    raise ValueError('Missing V3 symbol baseline')
                for p in [c["pooled_hazards"], *c["hazards_by_symbol"].values()]:
                    arr = np.asarray(p, float)
                    if arr.shape != (5,) or not np.isfinite(arr).all() or ((arr <= 0)|(arr >= 1)).any():
                        raise ValueError("Invalid V3 baseline hazards")
            elif np.asarray(c["coefficients"]).shape != (len(names)+4,) or not np.isfinite([c["intercept"], *c["coefficients"]]).all():
                raise ValueError("Invalid V3 coefficients")
        for records in m["thresholds_by_symbol"].values():
            values = [r['value'] for r in records]
            if len(records) != 5 or not np.isfinite(values).all() or any(v < 0 or v > 1.000001 for v in values):
                raise ValueError("Invalid V3 thresholds")
    return artifact


def load_artifact(path: str | Path = 'data_out/predictive_v3/model.json'):
    """Reject malformed numeric payloads consistently at the shared UI boundary."""
    try:
        return _load_artifact(path)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as exc:
        raise ValueError('Malformed V3 numeric artifact') from exc


def forecast_latest(artifact: dict, history: pd.DataFrame, symbol: str, as_of = None,
                    max_staleness_minutes: float = 4*24*60) -> dict:
    """Latest completed-session forecast with shared quality/freshness gates."""
    symbol = symbol.lower()
    when = v1._local_timestamp(as_of)
    if max_staleness_minutes < 0:
        raise ValueError("max_staleness_minutes must be non-negative")
    result = {"status": "unavailable", "symbol": symbol, "as_of": when.isoformat(),
              "model_version": "v3", "target_description": artifact.get("target_description", TARGET_DESCRIPTION),
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
        "signal_description": "曲面观测与严格历史阈值；本版融合风险率后生成累计概率，未提供可相加的线性特征贡献。"})
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
    candidate = task_model["candidate_models"][task_model["selected_candidate"]]
    result["status"] = "degraded" if mode == "hazard_baseline" or gates["ood_fraction"][0] > 0 else "ready"
    if mode == "hazard_baseline":
        result["warnings"].append("当前任务使用经过独立校准的历史条件风险率基准；未验证出特征模型增量，预测型预警已关闭。")
    if gates["ood_fraction"][0] > 0:
        result["warnings"].append("部分观测超出训练期99%范围，请结合数据质量解读。")
    result["model_context"] = {
        "selected_candidate": task_model["selected_candidate"],
        "feature_hazard_weight": task_model["blend_weight"],
        "statistical_hazard_weight": 1-task_model["blend_weight"],
        "calibration_slope": candidate["calibration"]["slope"],
        "symbol_calibration_offset": candidate["calibration"]["symbol_offsets"].get(symbol, 0.),
        "selection_scope": "pooled per task; prior validation only",
        "explanation": "风险率层融合后合成累计概率，权重不是单项特征贡献。未提供累计概率可加的线性解释。",
    }
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
                               "特征与统计风险率保守融合后合成的未来窗口累计概率"),
            "top_contributions": [], "intercept_log_odds": None, "total_log_odds": None,
            "contribution_scale": "conditional_hazard_log_odds", "contribution_hazard_step": h}
        item["feature_hazard_weight"] = task_model["blend_weight"]
        item["contribution_scale"] = "not_additive_after_hazard_mixture"
        result["horizons"].append(item)
    if any((item["evaluation"].get("brier_skill") or 0) < 0 for item in result["horizons"]):
        result["warnings"].append("部分期限在保留测试段未超过同任务条件风险率基准；测试结果没有用于回头选择模型。")
    return result
