"""Fixed V3 walk-forward study, with exact frozen-V2 pairing and reliability tables."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Callable
import hashlib
import json
import numpy as np
import pandas as pd
from .predictive import HORIZONS, probability_metrics
from .predictive_validation import (global_purged_splits, default_test_windows,
    paired_block_bootstrap, _episode_counts, _split_summary, FIXED_SETTINGS)


def reliability(frame):
    """Fixed bins; descriptive calibration diagnostics, not a significance test."""
    y, p = frame.label.to_numpy(float), frame.probability.to_numpy(float)
    bins = []
    for i in range(5):
        hit = (p >= i/5) & ((p < (i+1)/5) if i < 4 else (p <= 1))
        bins.append({'lower':i/5, 'upper':(i+1)/5, 'n':int(hit.sum()),
                     'mean_probability':float(p[hit].mean()) if hit.any() else None,
                     'event_rate':float(y[hit].mean()) if hit.any() else None})
    clipped = np.clip(p, 1e-12, 1-1e-12)
    return {'mean_probability':float(p.mean()) if len(p) else None,
            'log_loss':float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped))) if len(p) else None,
            'calibration_bias':float((p-y).mean()) if len(p) else None,
            'ece_5_fixed_bins':sum(b['n']*abs(b['mean_probability']-b['event_rate']) for b in bins if b['n'])/len(p) if len(p) else None,
            'reliability_bins':bins}


def compare_frozen_v2(predictions, path, n_boot=500):
    """Exact pairing; never silently compare a convenient intersecting subset."""
    if not path.exists():
        raise ValueError('Frozen V2 predictions required')
    previous = pd.read_parquet(path)
    keys = ['fold','symbol','_dt','horizon_days']
    for f in (previous,predictions):
        if f.duplicated(keys).any():
            raise ValueError('Duplicate comparison origins')
    old = previous[keys+['label','probability','baseline_probability','operational_alert','gate_allowed']].rename(columns={
        'label':'v2_label','probability':'v2_probability','baseline_probability':'v2_baseline_probability',
        'operational_alert':'v2_operational_alert','gate_allowed':'v2_gate_allowed'})
    paired = predictions.merge(old,on=keys,how='outer',validate='one_to_one',indicator=True)
    if not paired['_merge'].eq('both').all() or not np.array_equal(paired.label,paired.v2_label):
        raise ValueError('V2/V3 origin sets or labels disagree; cannot claim paired comparison')
    def compare(sub):
        y = sub.label.to_numpy(float)
        return {'n_common':len(sub), 'v2_brier':float(np.mean((y-sub.v2_probability)**2)),
                'v3_brier':float(np.mean((y-sub.probability)**2)),
                'v3_reference_brier':float(np.mean((y-sub.baseline_probability)**2)),
                'paired_v3_vs_v2':paired_block_bootstrap(sub,baseline_col='v2_probability',n_boot=n_boot),
                'v2_operational_alerts':int(sub.v2_operational_alert.sum()),
                'v3_operational_alerts':int(sub.operational_alert.sum()),
                'v2_probability_coverage':float(sub.v2_gate_allowed.mean()),
                'v3_probability_coverage':float(sub.gate_allowed.mean())}
    return {'status':'paired_all_origins','v2_predictions_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'folds':{str(fold):{str(h):compare(rows.loc[rows.horizon_days==h]) for h in HORIZONS}
                     for fold,rows in paired.groupby('fold')},
            'pooled_horizons':{str(h):compare(paired.loc[paired.horizon_days==h]) for h in HORIZONS}}

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
              "predictive_signal_origins": int((frame.model_status.isin(["joint_logistic_shrunk", "joint_logistic_decay_shrunk"]) & allowed).sum()),
              "predictive_signal_coverage": float((frame.model_status.isin(["joint_logistic_shrunk", "joint_logistic_decay_shrunk"]) & allowed).mean()) if len(frame) else None}
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
    raw.update(reliability(frame))
    raw["current_state_baseline_brier"] = raw["baseline_brier"]
    raw["skill_vs_current_state_baseline"] = raw["brier_skill"]
    raw["baseline_definition"] = "same-task symbol/step conditional hazard with identical independent affine and shrunk symbol calibration"
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

def _prediction_rows(test, names, models, fold, available_from):
    from .predictive_v3 import operational_gates, predict_curves
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
        part["feature_hazard_weight"] = curves["blend_weights"]
        part["ood_fraction"] = gates["ood_fraction"]
        part["raw_feature_missing_fraction"] = gates["raw_feature_missing_fraction"]
        part["constant_features_violated"] = [";".join(values) for values in gates.get("constant_features_violated", [[] for _ in range(len(test))])]
        part["model_available_from"] = available_from
        part["as_of"] = test._dt
        part["age_minutes"] = 0.
        part["coherent"] = coherence
        rows.append(part)
    return pd.concat(rows, ignore_index=True)

def train_models(history: pd.DataFrame, windows=None, progress: Callable | None = None,
                 v2_predictions_path="data_out/predictive_v2/test_predictions.parquet", n_boot=500):
    """Fit each past-only vintage once; return final vintage and full fold report."""
    from . import predictive_v3 as v3
    features, names = v3.build_features(history)
    data = v3.build_targets(features)
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
        models = v3.fit_models(split["train"], split["calibration"], split["validation"], names, symbols)
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
    artifact = {"schema_version": 3, "model_type": "calibrated_hazard_mixture_json",
                "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "target_description": getattr(v3, "TARGET_DESCRIPTION", "Five-session coherent conditional hazard curves"),
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
    report = {"schema_version": 3, "target_description": artifact["target_description"],
              "evaluation_design": "Development walk-forward on previously inspected history; not an external blind test. Fixed three historical vintages, all five horizons on the same complete-five-day origins. Five predeclared shrunk candidates; no test-dependent candidate, threshold, or hyperparameter choice.",
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
                              "Existing history and all V2 results were inspected before V3; all intervals and walk-forward results are development evidence.",
                              "Thresholds frozen at each origin; normal-origin onset and currently-abnormal persistence are different tasks. Persistence target is any future exceedance, not uninterrupted persistence or time-to-recovery.",
                              "Positive-label origin runs are proxy episodes, not deduplicated actual event dates. No event lead-time claim is made.",
                              "Observed session calendar and >7-day gaps conservatively exclude missing data and some long holidays.",
                              "Bootstrap preserves date-level cross-symbol dependence but conditions on fitted vintages, has no multiple-comparison adjustment, and cannot prove future gains.",
                              "Candidate fallback and thresholds are selected only by preceding validation data; no test-driven fallback is applied. Validation F1 is a provisional product objective; business false-alarm and missed-event costs are not supplied.",
                              "Final artifact is exactly the latest fold vintage; no refit uses its test labels."],
              "comparison_v2": compare_frozen_v2(all_predictions, Path(v2_predictions_path), n_boot)}
    return artifact, report, all_predictions


def readable_report(report):
    def f(x,p=False):
        return '—' if x is None else (f'{x:.1%}' if p else f'{x:.4f}')
    lines=['# V3 分层校准与保守风险率融合：开发期验证','',report['target_description'],'',
           '**此前已查看全部历史和 V2 结果；本报告不是外部盲测。所有实验规则在 V3 首次评分前固定，未按测试结果追加候选。**','',
           'V3 沿用 V2 的训练候选、因果特征和标签，新增独立校准段的受约束斜率与品种收缩偏置。基准接受完全相同的校准。'
           '特征候选只以 25% 或 50% 风险率权重与统计基准融合；每个任务按过去验证段的配对误差改善减十日期块标准误选择，支持不足或改善不足则保留统计基准。'
           '完整曲线共用融合权重，不对累计概率排序修饰。统计基准关闭预测预警。','',
           '三轮时间窗口、完整五日原点及门禁与 V2 一致。最终模型保持第三轮原样，不以最后测试标签重训。'
           '与冻结 V2 逐条原点严格配对；不能把校准基准的改善算作特征模型增量。','',
           '## 三轮概率误差与冻结 V2 对照','',
           '| 轮次 | 期限 | 原点 | V2 Brier | V3 Brier | V3 对 V2 改善 | 改善探索区间 | V3 对自身强基准 skill |',
           '|---|---:|---:|---:|---:|---:|---|---:|']
    for fold in report['folds']:
        for h,item in fold['horizons'].items():
            comp=report['comparison_v2']['folds'][str(fold['fold'])][h]
            paired=comp['paired_v3_vs_v2'];ci=paired['skill_ci95']
            lines.append(f"| {fold['fold']} | {h} | {comp['n_common']} | {f(comp['v2_brier'])} | {f(comp['v3_brier'])} | {f(paired['skill'],True)} | [{f(ci[0],True)}, {f(ci[1],True)}] | {f(item['test']['brier_skill'],True)} |")
    lines+=['','## 验证段选择（从未使用当轮测试标签）','',
            '| 轮次 | 任务 | 来源候选 | 特征风险率权重 | 验证原点 | 日期块 | 选择原因 |',
            '|---|---|---|---:|---:|---:|---|']
    for fold in report['folds']:
        for task,record in fold['selection'].items():
            lines.append(f"| {fold['fold']} | {task} | {record['selected_candidate']} | {f(record['selected_weight'],True)} | {record['n']} | {record['nonoverlapping_date_blocks']} | {record['reason']} |")
    lines+=['','## 最近一轮分任务可靠性与运行结果','',
            '| 任务 | 期限 | 原点 | 事件率 | 平均概率 | Brier | 对强基准 skill | 运行预警 | 误报 | 全原点召回 | 特征信号覆盖 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for h,item in report['horizons'].items():
        for task,result in item['tasks'].items():
            m,op=result['test'],result['operational']
            lines.append(f"| {task} | {h} | {m['n']} | {f(m['prevalence'],True)} | {f(m['mean_probability'],True)} | {f(m['brier'])} | {f(m['brier_skill'],True)} | {op['alerts']} | {op['false_alarms']} | {f(op['recall_all_origins'],True)} | {f(op['predictive_signal_coverage'],True)} |")
    lines+=['','## 原始概率与运行口径','',
            'Brier/log-loss 评价全部测试原点上的原始校准概率。报告另存固定五档可靠性表、分箱描述性 ECE、平均概率和事件率。'
            'ECE 依赖分箱且小样本易波动，不能单独证明校准改善。运行召回保留全部原点，弃权按未预警计入；'
            '概率可用率包含统计基准，特征信号覆盖率只包含已选特征融合且通过门禁的原点。','',
            '| 轮次 | 原点 | 期限冲突 | 弃权 | 概率可用率 | 特征信号覆盖率 |',
            '|---|---:|---:|---:|---:|---:|']
    for fold in report['folds']:
        m=fold['horizons']['1'];op=m['operational']
        lines.append(f"| {fold['fold']} | {m['test']['n']} | {fold['coherence_conflict_origins']} | {op['abstentions']} | {f(op['probability_coverage'],True)} | {f(op['predictive_signal_coverage'],True)} |")
    lines+=['','## 边界与复现','',
            '- 验证选择的块标准误是预设保守启发式，不是正式显著性检验；候选可能在未来窗口失效。',
            '- 表中 95% 区间来自同日跨品种共同十观测日期块、500 次重采样，条件于已拟合模型，未做多重比较校正。',
            '- 当前为历史曲面代理回放，无实时行情、官方日历、账户持仓或损益；不能称为 VaR/亏损概率。',
            '- persistence 为当前异常后未来窗口至少一次超阈值，不是连续异常时长或恢复时间。',
            '- 历史收盘回放年龄为零，不能检验真实断流、传输延迟；实时调用仍检查数据过期。',
            '- 正标签连续原点片段不等于真实去重事件，不宣称实际事件提前天数。','',
            '固定设计见 docs/V3实验设计.md；完整分品种、分任务、运行指标和可靠性表见 report.json。'
            'test_predictions.csv/Parquet 保留所有原点、标签末端、融合权重和弃权原因。','',
            f"模型 SHA-256：`{report.get('model_sha256','')}`",'']
    return '\n'.join(lines)
