"""Train reproducible future surface-risk forecasts with chronological holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vol_surface.predictive import load_anchor_data, train_models


def readable_report(report: dict) -> str:
    def number(value, percent=False):
        if value is None:
            return "—"
        return f"{value:.1%}" if percent else f"{value:.3f}"

    lines = ["# 未来 1–5 个观测交易日曲面风险预测：开发期时间留出评估", "",
             report["target_description"], "",
             "## 验证设计", "",
             "训练、概率校准、模型及报警阈值选择、保留测试按所有品种共用的自然时间边界严格分开。"
             "每个边界前剔除未来标签延伸到下一段的样本。测试段不参与拟合、校准、选模或阈值选择。", "",
             "逐品种按验证集 Brier 最小值，在训练期品种基础率、当前超阈值状态条件基础率、校准逻辑回归三者中选择，"
             "并列时偏向简单候选。constant只展示基础率而不预警；state是简单统计基准，明确解释当前状态对应的历史条件率。"
             "state和logistic报警阈值在同一独立验证段上按 F1 选取。", "",
             "**透明说明：本测试段指标已在开发期间查看过，因此属于开发期时间留出评估，不是外部盲测。**"
             "开发中将数值优化迭代上限从400提高至2000以修复未收敛；随后修复选模逻辑，保留验证获胜的状态基准，"
             "不再在逻辑回归落败时一律回退常数。上述修复后各候选仍只使用验证段做选择。真正部署前仍需新增前瞻样本验证。", "",
             "| 用途 | 起始边界 |", "|---|---|",
             f"| 校准段 | {report['split_boundaries']['calibration']} |",
             f"| 阈值/选模验证段 | {report['split_boundaries']['validation']} |",
             f"| 保留测试段（模型可用时间） | {report['split_boundaries']['test']} |", "",
             "## 保留测试结果", "",
             "Brier 越小越好；Brier skill=1−模型误差/基准误差，正数表示超过基准。"
             "AP 应与事件基础率比较。精确率在没有预警时无定义，显示“—”。"
             "以下是原始模型/验证阈值的研究评估，未把跨期限运行时一致性检查作为指标后处理；"
             "实际交互系统遇到累计概率非单调时保留原始概率，但暂停全部预警，因此实际预警数量可能更少。", "",
             "| 未来交易日 | 样本 | 事件率 | Brier | 基础率 Brier | skill | 状态基准 Brier | 状态 skill | AUC | AP | 精确率 | 召回率 | 预警比例 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon, item in report["horizons"].items():
        m = item["test"]
        lines.append(f"| {horizon} | {m['n']} | {number(m['prevalence'], True)} | {number(m['brier'])} | "
                     f"{number(m['baseline_brier'])} | {number(m['brier_skill'])} | "
                     f"{number(m['current_state_baseline_brier'])} | {number(m['skill_vs_current_state_baseline'])} | "
                     f"{number(m['roc_auc'])} | {number(m['average_precision'])} | {number(m['precision'], True)} | "
                     f"{number(m['recall'], True)} | {number(m['alert_rate'], True)} |")
    weaker_than_state = [h for h, item in report["horizons"].items()
                         if item["test"]["skill_vs_current_state_baseline"] <= 0]
    lines += ["", f"**预测增量边界：{','.join(weaker_than_state) or '无'} 日期限在此测试段未超过简单状态基准。**"
              "不能仅凭高于常数基础率的 skill，就宣称优于简单状态延续判断。", "",
              "训练到测试的事件率变化很大，概率存在明显市场状态漂移。", "",
              "| 未来日 | 训练事件率 | 校准事件率 | 验证事件率 | 测试事件率 |",
              "|---|---:|---:|---:|---:|"]
    for horizon, item in report["horizons"].items():
        prevalences = [number(item["splits"][name]["prevalence"], True)
                       for name in ("train", "calibration", "validation", "test")]
        lines.append(f"| {horizon} | " + " | ".join(prevalences) + " |")
    lines += ["", "## 分品种结果", "",
              "模式由验证集决定。即使测试段表现不佳，也不利用测试结果回头更换模式；界面会披露非正测试 skill。", "",
              "| 品种 | 未来日 | 模式 | 测试样本 | 事件率 | skill | 对状态基准 skill | AUC | AP | 精确率 | 召回率 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon, item in report["horizons"].items():
        for symbol, m in item["per_symbol"].items():
            lines.append(f"| {symbol} | {horizon} | {m['mode']} | {m['n']} | {number(m['prevalence'], True)} | "
                         f"{number(m['brier_skill'])} | {number(m.get('skill_vs_current_state_baseline'))} | "
                         f"{number(m['roc_auc'])} | {number(m['average_precision'])} | {number(m['precision'], True)} | "
                         f"{number(m['recall'], True)} |")
    lines += ["", "## 原点尚未超阈值的子样本", "",
              "只纳入预测时点当日两个代理最大值均未超阈值的原点，观察新增超阈值能力。"
              "这是原点子集结果，不能当作独立事件计数或事件起点去重后的召回。", "",
              "| 未来日 | 样本 | 后续事件率 | skill | AUC | AP | 精确率 | 召回率 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon, item in report["horizons"].items():
        m = item["new_exceedance_subset_test"]
        lines.append(f"| {horizon} | {m['n']} | {number(m['prevalence'], True)} | {number(m['brier_skill'])} | "
                     f"{number(m['roc_auc'])} | {number(m['average_precision'])} | {number(m['precision'], True)} | "
                     f"{number(m['recall'], True)} |")
    lines += ["", "## 标签覆盖与剔除", "",
              "日盘必须包含全部 15 个预设收盘截面且核心特征有限。夜盘归下一观测日盘日期，"
              "但原始夜盘覆盖可能不全，因此标签只代表已观测日内最大值。"
              "全品种观测日盘日期并集用于识别缺日；相邻日期相隔超过 7 个自然日也中断窗口，"
              "春节等长假可能被保守排除。没有将缺失月份直接算成下一个交易日。", "",
              "| 品种 | 未来日 | 已完成日盘 | 标签有效日 | 覆盖率 | 不完整未来窗口 | 历史不足 | 边界剔除 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for horizon, item in report["horizons"].items():
        for symbol, m in item["coverage"].items():
            lines.append(f"| {symbol} | {horizon} | {m['completed_daily_origins']} | {m['eligible_daily_origins']} | "
                         f"{number(m['eligible_fraction'], True)} | {m['incomplete_or_missing_trading_day_windows']} | "
                         f"{m['insufficient_past_history']} | {m['purged_at_split_boundaries']} |")
    lines += ["", "## 各时间段的实际范围", "",
              "| 未来日 | 段 | 样本 | 首个原点 | 最后原点 | 最晚标签结束 |",
              "|---|---|---:|---|---|---|"]
    for horizon, item in report["horizons"].items():
        for name, m in item["splits"].items():
            lines.append(f"| {horizon} | {name} | {m['n']} | {m['start']} | {m['last_origin']} | {m['last_label_end']} |")
    lines += ["", "## 使用边界", "",
              "- 预测输出是市场曲面代理超阈值概率，无法替代持仓、损益或组合 VaR/CVaR。",
              "- 重叠预测窗口有相关性；当前指标是每日预测原点口径。",
              "- 模型只允许在训练、校准及阈值选择结束后回放。更早日期不会输出预测。",
              "- 默认超过 4 个自然日的观测视为过期；当前项目未接入实时行情。",
              "- 特征缺失过多或严重超出训练分布时暂停对应预测；贡献值是校准后的 log-odds 分解，不代表因果。",
              "- 各期限独立估计，累计概率偶尔可能非单调；保留原始估计，并通过运行时逻辑一致性检查暂停全部预警。该检查不重新拟合模型，也不改变研究评估概率或原始指标。",
              "- 测试 skill 不为正的结果不能宣称拥有可用增量预测能力。", "",
              f"模型 SHA-256：`{report.get('model_sha256', '')}`", "",
              "完整数值、每个数据分片的校验和及选模依据见 report.json；逐条测试预测含真实标签结束日期。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data_out/anchor"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_out/predictive"))
    parser.add_argument("--symbols", help="Optional comma-separated subset")
    parser.add_argument("--calibration-start", help="Global calendar boundary, ISO date")
    parser.add_argument("--validation-start", help="Global calendar boundary, ISO date")
    parser.add_argument("--test-start", help="Global calendar boundary, ISO date")
    args = parser.parse_args()
    values = [args.calibration_start, args.validation_start, args.test_start]
    if any(values) and not all(values):
        parser.error("All three explicit chronological boundaries must be supplied together")
    boundaries = dict(zip(("calibration", "validation", "test"), values)) if all(values) else None
    symbols = args.symbols.split(",") if args.symbols else None
    print("Loading raw snapshots; legacy labels/models are not loaded", flush=True)
    history = load_anchor_data(args.data_dir, symbols)
    print(f"Loaded {len(history):,} snapshots across {history.symbol.nunique()} symbols", flush=True)
    artifact, report, predictions = train_models(history, boundaries, progress=lambda s: print(s, flush=True))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.json"
    model_text = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False)
    model_path.write_text(model_text, encoding="utf-8")
    report["model_sha256"] = hashlib.sha256(model_text.encode()).hexdigest()
    report["training_command"] = " ".join(sys.argv)
    report["artifact_format"] = "JSON numeric coefficients; no executable deserialization"
    import numpy, pandas, scipy
    report["runtime_versions"] = {"python": sys.version.split()[0], "numpy": numpy.__version__,
                                  "pandas": pandas.__version__, "scipy": scipy.__version__}
    source = Path(__file__).resolve().parents[1] / "vol_surface" / "predictive.py"
    report["algorithm_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "report.md").write_text(readable_report(report), encoding="utf-8")
    predictions.to_parquet(args.output_dir / "test_predictions.parquet", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(f"Saved model, report and chronological-holdout predictions in {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
