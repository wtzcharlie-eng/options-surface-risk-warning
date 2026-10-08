"""grid_search — 在事件期窗口网格搜索规则阈值，达成赛题口径指标。

赛题主指标（问题1）：
  精确率 ≥ 50%、召回率 ≥ 60%、平均提前时间 ≥ 30 分钟。

策略：
  连续截面特征计算是最贵的部分（清洗+PCHIP+SVI）。先在每个评测窗口把连续特征
  序列算一次并缓存，再对缓存的特征序列反复应用不同阈值参数（廉价），搜索让指标
  达标且误报低的参数组合。最后选出最优参数写回 alert_rules.DEFAULT_PARAMS。
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.alert_engine import AlertState
from vol_surface.alert_rules import DEFAULT_PARAMS, evaluate_rules
from vol_surface.cleaning import clean_slice
from vol_surface.features import FeatureHistory, compute_features
from vol_surface.io_loader import slice_at_timestamp, timestamps_in
from vol_surface.quant_metrics import _load_window, identify_risk_windows, evaluate_quant

# 评测窗口：(label, symbol, start, end) —— 覆盖各品种的极端事件期
WINDOWS = [
    ("ag2026-01贵金属", "ag", "20260110", "20260208"),
    ("ag2024-04贵金属", "ag", "20240401", "20240420"),
    ("sc2026-03原油", "sc", "20260220", "20260330"),
    ("sc2023-06能化", "sc", "20230520", "20230610"),
]


def precompute_window(symbol: str, start: str, end: str, prefill_days: int = 25) -> dict:
    """算一个窗口的连续特征序列 + 风险区间起始（只算一次）。"""
    hist = FeatureHistory(window=500)
    sd0 = (pd.to_datetime(start) - pd.Timedelta(days=prefill_days)).strftime("%Y%m%d")
    if sd0 < "20221227":
        sd0 = "20221227"
    # 预跑历史
    from vol_surface.quant_metrics import build_continuous_features
    build_continuous_features(symbol, sd0, start, history=hist)
    # 评测期
    df = _load_window(symbol, start, end)
    tss = [t for t in timestamps_in(df) if start <= t[:8] <= end]
    feats_list, prev, used_ts = [], None, []
    for ts in tss:
        sl = slice_at_timestamp(df, ts)
        if len(sl) < 15:
            continue
        clean, _ = clean_slice(sl)
        if len(clean) < 10:
            continue
        feats = compute_features(clean, prev_features=prev, history=hist, lightweight=True)
        feats["_timestamp"] = ts
        feats_list.append(feats)
        used_ts.append(ts)
        prev = feats
    cont = pd.DataFrame({"timestamp": used_ts,
                         "atm_iv": [f["atm_iv"] for f in feats_list],
                         "convexity_violation": [f["convexity_violation"] for f in feats_list]})
    risk_starts = identify_risk_windows(cont)
    return {"feats": feats_list, "risk_starts": risk_starts, "n_slices": len(feats_list)}


def eval_window_with_params(window: dict, params: dict, state_params: dict | None,
                             use_state: bool = True) -> dict:
    """对缓存特征应用参数，返回指标。廉价：只跑规则+状态机。"""
    levels = []
    state = AlertState(**(state_params or {})) if use_state else None
    for feats in window["feats"]:
        rr = evaluate_rules(feats, params=params)
        triggers = rr["triggers"]
        if state is not None:
            state.update(triggers)
            adj, mx = [], 0
            for t in triggers:
                lv = state.rule_level_boost(t["rule"], t["level"])
                if lv >= 1:
                    t2 = dict(t); t2["level"] = lv; adj.append(t2); mx = max(mx, lv)
            level = mx
        else:
            level = rr["level"]
        levels.append(level)
    alerts = pd.DataFrame({"timestamp": [f["_timestamp"] for f in window["feats"]],
                           "level": levels})
    res = evaluate_quant(alerts, window["risk_starts"], min_level=2)
    return {"precision": res.precision, "recall": res.recall, "lead": res.avg_lead_minutes,
            "n_alert": res.n_alerts, "n_risk": res.n_risk_windows}


# 搜索空间：聚焦影响最大的几组阈值
PARAM_GRID = {
    "r1_warn_z": [2.0, 2.5, 3.0],
    "r1_ser_z": [3.0, 3.5, 4.0],
    "r2_warn": [0.3, 0.4, 0.5],
    "r3_warn": [2.5, 3.0, 3.5],
    "r5_warn_z": [2.5, 3.0, 3.5],
    "r7_warn": [0.05, 0.06, 0.07],
    "r8_warn": [2.0, 2.5, 3.0],
    "r8_ser": [3.0, 3.5, 4.0],
}
STATE_GRID = [
    {"persistence": 8, "confirm": 2, "escalate": 5, "cooldown": 0},
    {"persistence": 8, "confirm": 2, "escalate": 5, "cooldown": 4},
    {"persistence": 8, "confirm": 2, "escalate": 5, "cooldown": 6},
    {"persistence": 10, "confirm": 3, "escalate": 8, "cooldown": 0},
    {"persistence": 10, "confirm": 3, "escalate": 8, "cooldown": 8},
    None,  # 不用状态机
]


def score_windows(windows_cache: list, params: dict, state_params: dict | None, use_state: bool) -> dict:
    """对全部窗口算指标，返回汇总。"""
    pr, rc, ld, na, nr = [], [], [], [], []
    for w in windows_cache:
        r = eval_window_with_params(w, params, state_params, use_state)
        pr.append(r["precision"]); rc.append(r["recall"]); ld.append(r["lead"])
        na.append(r["n_alert"]); nr.append(r["n_risk"])
    return {"precision": np.mean(pr), "recall": np.mean(rc), "lead": np.mean(ld),
            "n_alert": sum(na), "n_risk": sum(nr),
            # 达标分：召回和提前达标的奖励，精确率越高越好
            "pass_recall": np.mean([r >= 0.6 for r in rc]),
            "pass_lead": np.mean([l >= 30 for l in ld])}


def main():
    print("预计算各窗口连续特征...")
    caches = []
    for label, sym, sd, ed in WINDOWS:
        print(f"  {label}...", flush=True)
        caches.append(precompute_window(sym, sd, ed))
    print("预计算完成\n")

    keys = list(PARAM_GRID.keys())
    vals = list(PARAM_GRID.values())
    best = None
    n_combos = 1
    for v in vals:
        n_combos *= len(v)
    print(f"搜索 {n_combos} 组阈值 × {len(STATE_GRID)} 种状态机...")
    tried = 0
    for combo in itertools.product(*vals):
        params = {**DEFAULT_PARAMS, **dict(zip(keys, combo))}
        for sp in STATE_GRID:
            use_state = sp is not None
            s = score_windows(caches, params, sp, use_state)
            tried += 1
            # 综合目标：召回≥0.6 且 提前≥30 是硬约束；满足后精确率最大化
            feasible = s["recall"] >= 0.6 and s["lead"] >= 30
            obj = s["precision"] + 0.3 * s["pass_recall"] + 0.3 * s["pass_lead"]
            if feasible:
                obj += 1.0  # 可行解优先
            if best is None or obj > best["obj"]:
                best = {"obj": obj, "params": params, "state": sp, "use_state": use_state,
                        "feasible": feasible, **s}
            if tried % 50 == 0:
                print(f"  {tried}/{n_combos*len(STATE_GRID)} 当前最优: "
                      f"P={best['precision']:.2%} R={best['recall']:.2%} lead={best['lead']:.0f}min "
                      f"{'✓可行' if best['feasible'] else '✗'}", flush=True)

    print("\n=== 最优参数 ===")
    print(f"精确率={best['precision']:.2%} (目标≥50%)")
    print(f"召回率={best['recall']:.2%} (目标≥60%)")
    print(f"平均提前={best['lead']:.0f}min (目标≥30min)")
    print(f"可行性: {'达标' if best['feasible'] else '未完全达标'}")
    print(f"状态机: {best['state']}")
    print("参数:", {k: best["params"][k] for k in keys})
    # 保存最优参数
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data_out", "best_params.json")
    import json
    with open(out, "w") as f:
        json.dump({"params": {k: float(best["params"][k]) for k in PARAM_GRID},
                   "state": best["state"],
                   "metrics": {"precision": float(best["precision"]),
                               "recall": float(best["recall"]),
                               "lead": float(best["lead"]),
                               "feasible": bool(best["feasible"])}},
                  f, indent=2, ensure_ascii=False)
    print(f"\n已保存到 {out}")


if __name__ == "__main__":
    main()
