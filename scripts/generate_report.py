"""generate_report — 离线回测报告：整合赛题全部量化指标。

输出三部分：
1. 量化指标（赛题指标1）：精确率/召回率/平均提前时间 —— 连续截面口径
2. CVaR 改善率（赛题指标2，核心业务指标）：预警减仓 vs 不减仓
3. 事件表回测（自定辅助）：对照 extreme_event.csv 的召回

报告写为 Markdown（data_out/backtest_report.md）+ 控制台打印。
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.backtest import backtest as event_backtest, format_result, load_events
from vol_surface.cvar_backtest import backtest_cvar, format_cvar, get_futures_price
from vol_surface.quant_metrics import format_quant, run_quant_evaluation

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_out")

# 评测窗口（覆盖各品种事件期）：连续截面口径
QUANT_WINDOWS = [
    ("ag", "2026-01 贵金属", "20260110", "20260208"),
    ("ag", "2024-04 贵金属", "20240401", "20240420"),
    ("sc", "2026-03 原油", "20260220", "20260330"),
    ("sc", "2023-06 能化", "20230520", "20230610"),
]

# CVaR 测试品种与测试期（≥3个月连续数据）
CVAR_CASES = [
    ("ag", "2025-11-01", "2026-03-31", "含2026-01白银极端波动"),
    ("sc", "2025-12-01", "2026-04-01", "含2026-03原油极端波动"),
]


def main(use_akshare: bool = True):
    lines = ["# 离线回测报告\n", "对照赛题量化技术指标（问题1主指标 + 核心业务指标）。\n"]

    # ---- 1. 量化指标 ----
    lines.append("## 1. 预警准确性与时效性（赛题主指标）\n")
    lines.append("口径：未来 5 个 15min 点内 ATM IV 或凸性违反 > 过去 20 交易日 95 分位 → 之后 2h 为"
                 "风险区间。预警 level≥2。目标：精确率≥50%、召回≥60%、平均提前≥30min。\n")
    lines.append("| 品种 | 事件期 | 风险区间 | 预警(≥2) | 精确率 | 召回率 | 平均提前 | 达标 |")
    lines.append("|------|--------|---------|---------|--------|--------|---------|------|")
    for sym, label, sd, ed in QUANT_WINDOWS:
        _, a, rs, res = run_quant_evaluation(sym, sd, ed, use_state=False)
        ok = "✓" if (res.recall >= 0.6 and res.avg_lead_minutes >= 30 and res.precision >= 0.5) else (
            "部分" if res.recall >= 0.6 and res.avg_lead_minutes >= 30 else "✗")
        lines.append(f"| {sym} | {label} | {res.n_risk_windows} | {res.n_alerts} | "
                     f"{res.precision:.1%} | {res.recall:.1%} | {res.avg_lead_minutes:.0f}min | {ok} |")
        print(f"[量化] {sym} {label}: P={res.precision:.1%} R={res.recall:.1%} lead={res.avg_lead_minutes:.0f}min")

    # ---- 2. CVaR 改善率 ----
    lines.append("\n## 2. CVaR 改善率（赛题核心业务指标）\n")
    lines.append("口径：预警≥2 级时次日开盘持仓减半（多头平一半/空头回补一半），首次减仓后持有至期末。"
                 "对比无预警策略日收益 CVaR(95%)。目标：平均改善率>10% 视为显著有效。\n")
    lines.append("| 品种 | 测试期 | 首次预警 | 多头改善 | 空头改善 | 平均改善 | 显著 |")
    lines.append("|------|--------|---------|---------|---------|---------|------|")
    for sym, sd, ed, note in CVAR_CASES:
        try:
            px = get_futures_price(sym, use_akshare=use_akshare)
            px["date"] = pd.to_datetime(px["date"]).dt.normalize()
            px = px[(px["date"] >= sd) & (px["date"] <= ed)]
            alerts = pd.read_parquet(os.path.join(DATA_OUT, f"alerts_{sym}.parquet"))
            alerts["dt"] = pd.to_datetime(alerts["timestamp"], format="%Y%m%d%H%M%S")
            alerts = alerts[(alerts["dt"] >= sd) & (alerts["dt"] <= ed)]
            r = backtest_cvar(alerts, px)
            sig = "✓" if r.avg_improve > 0.1 else "✗"
            lines.append(f"| {sym} | {sd}~{ed} | {r.first_alert_date} | {r.long_improve:.1%} | "
                         f"{r.short_improve:.1%} | {r.avg_improve:.1%} | {sig} |")
            print(f"[CVaR] {sym}: 平均改善={r.avg_improve:.1%} (首次预警={r.first_alert_date})")
        except Exception as e:
            lines.append(f"| {sym} | {sd}~{ed} | - | - | - | - | 失败:{e} |")
            print(f"[CVaR] {sym} 失败: {e}")

    # ---- 3. 事件表回测 ----
    lines.append("\n## 3. 事件表回测（辅助验证）\n")
    lines.append("口径：对照 extreme_event.csv，按品种相关事件算召回与提前时间。\n")
    events = load_events()
    for sym in ["ag", "si", "au", "sc"]:
        ap = os.path.join(DATA_OUT, f"alerts_{sym}.parquet")
        if not os.path.exists(ap):
            continue
        a = pd.read_parquet(ap)
        a["symbol"] = sym
        bt = event_backtest(a[["timestamp", "date", "level", "symbol"]], events, symbol=sym, min_level=2)
        lines.append(f"\n**{sym}**: 召回={bt.recall:.0%}({bt.n_hit}/{bt.n_events}) 提前={bt.avg_lead_minutes:.0f}min FPR={bt.fpr:.0%}")

    # 自足的出处声明。**必须写在文件自己头上**，不能只写在 README 里——
    # 评委可能直接打开这个文件，那时他看到的是一张宣称达标的表和一节全是 +50.0%
    # 的 CVaR，却没有任何「已被推翻」的提示。
    # （本项目已有同族教训：解读层不继承证据层的限定，限定等于白写。）
    _hdr = [
        "> ## ⚠️ 本文件是**历史快照**，不是交付结论",
        ">",
        "> 生成配置：**旧风险起点口径**（无预热）、**原始 7 条规则**"
        "（R2/R4/R6 尚未移除）、仅 **ag/sc/au/si 四个已校准品种**，"
        "且 §1 的评测**启用了 ML 层**（`use_model` 默认 True）。",
        ">",
        "> **已知问题**：当时的 `AlertModel.score_samples` 存在批内归一化缺陷，"
        "逐截面调用时异常分恒为 1.0，会把预警等级整体抬升（详见 README §7.25）。"
        "该缺陷已修复，但**本文件未按修复后的代码重跑**——保留原样是为了留存历史记录。",
        ">",
        "> **交付结论以下列为准**，不要引用本文件的数字：",
        "> - 问题1 事件窗口：README §7.11 / §7.18，落盘 `data_out/q1_7sym.json`",
        "> - 规则引擎的改造：README §7.13–§7.15（R2/R4/R6 的处置）",
        "> - **§2 的 CVaR +50.0% 是算术产物，不是业务成果**——"
        "原口径在实测预警密度下退化为「全程半仓」，恒预警/随机预警都能拿到同样的数；"
        "已被 README §7.11 推翻，替代口径见 §7.20。",
        "",
    ]
    report = "\n".join(_hdr + lines) + "\n"
    out = os.path.join(DATA_OUT, "backtest_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n报告已写入 {out}")
    print("\n" + "=" * 50)
    print(report)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-akshare", action="store_true", help="用反推标的价格代替 AKShare")
    args = ap.parse_args()
    main(use_akshare=not args.no_akshare)
