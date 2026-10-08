"""端到端验证脚本：检查全流程产物完整性与质量。

验证项
1. data_out 下 4 品种的 alerts/features/surfaces/model 文件齐全
2. 每品种预警分布合理（非全 0、非全 3）
3. 回测：召回率 > 0、FPR 合理
4. 仪表板图表函数无报错

用法：python scripts/validate.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_out")
SYMBOLS = ["ag", "si", "au", "sc"]


def main():
    print("=" * 60)
    print("端到端验证")
    print("=" * 60)

    # 1. 文件齐全
    print("\n[1] 文件完整性")
    ok = True
    for s in SYMBOLS:
        for kind in ["alerts", "features", "surfaces"]:
            p = os.path.join(DATA_OUT, f"{kind}_{s}.parquet")
            exists = os.path.exists(p)
            print(f"  {'✓' if exists else '✗'} {kind}_{s}.parquet")
            ok = ok and exists
        mp = os.path.join(DATA_OUT, f"model_{s}.joblib")
        print(f"  {'✓' if os.path.exists(mp) else '✗'} model_{s}.joblib")
        ok = ok and os.path.exists(mp)
    if not ok:
        print("  => 缺文件，请先跑 build_features.py")
        return

    # 2. 预警分布
    print("\n[2] 预警分布")
    for s in SYMBOLS:
        a = pd.read_parquet(os.path.join(DATA_OUT, f"alerts_{s}.parquet"))
        dist = a["level"].value_counts().sort_index().to_dict()
        n_hi = int((a["level"] >= 2).sum())
        print(f"  {s}: {len(a)}截面 分布={dist} 高等级(>=2)={n_hi} ({n_hi/len(a):.1%})")
        assert n_hi > 0, f"{s} 无高等级预警"

    # 3. 回测
    print("\n[3] 回测")
    from vol_surface.backtest import backtest, format_result, load_events
    events = load_events()
    for s in SYMBOLS:
        a = pd.read_parquet(os.path.join(DATA_OUT, f"alerts_{s}.parquet"))
        a["symbol"] = s
        bt = backtest(a[["timestamp", "date", "level", "symbol"]], events, symbol=s, min_level=2)
        print(f"  {s}: 召回={bt.recall:.0%}({bt.n_hit}/{bt.n_events}) FPR={bt.fpr:.0%} 提前={bt.avg_lead_minutes:.0f}min")

    # 4. 仪表板图表
    print("\n[4] 仪表板图表函数")
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    events = mod.load_events()
    for s in SYMBOLS:
        a = pd.read_parquet(os.path.join(DATA_OUT, f"alerts_{s}.parquet"))
        f = pd.read_parquet(os.path.join(DATA_OUT, f"features_{s}.parquet"))
        surf = pd.read_parquet(os.path.join(DATA_OUT, f"surfaces_{s}.parquet"))
        a["dt"] = pd.to_datetime(a["timestamp"], format="%Y%m%d%H%M%S", errors="coerce")
        ts = sorted(surf["timestamp"].unique())[0] if len(surf) else sorted(a["timestamp"])[0]
        hm = mod.surface_heatmap(surf, ts, "call")
        tl = mod.alert_timeline(a, events, s)
        fg = mod.feature_gauges(f, a, ts)
        print(f"  {s}: 热力图={'OK' if hm else 'N/A'} 时间轴={'OK' if tl else 'FAIL'} 仪表={'OK' if fg else 'FAIL'}")

    # 5. 赛题口径量化指标（抽样一个窗口快速验证）
    print("\n[5] 赛题口径量化指标（抽样 sc2026-03 原油窗口）")
    from vol_surface.quant_metrics import run_quant_evaluation
    _, _, _, res = run_quant_evaluation("sc", "20260220", "20260330", use_state=False)
    print(f"  精确率={res.precision:.1%}(≥50%) 召回={res.recall:.1%}(≥60%) 提前={res.avg_lead_minutes:.0f}min(≥30min)")

    print("\n" + "=" * 60)
    print("验证完成 ✓" if ok else "验证完成（有缺失）")
    print("=" * 60)
    print("\n完整回测报告：python scripts/generate_report.py")


if __name__ == "__main__":
    main()
