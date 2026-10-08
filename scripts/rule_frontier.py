"""rule_frontier — 问题1 规则引擎在 anchor 连续口径上的达标研究。

**判据写于跑数之前**，见 `data_out/rule_frontier_preregistration.md`。

问题
----
死区过滤开启后，规则引擎在 anchor 连续测试集上最接近目标的点是
`zmult=0.70`：召回 60.50% 刚过线、**精确率 49.05%，差 0.95pp**。
即召回只剩 0.50pp 余量，却需要 0.95pp 精确率。

关键区分（本脚本的全部意义）
----------------------------
待测的两个杠杆——冷却去抖 `cooldown` 与流动性门 `liquidity_ratio ≥ 0.55`——
都是「过滤预警」类操作，**必然用召回换精确率**。而只调 `zmult` 也能换。
所以「精确率涨了」本身毫无信息量，必须问：

    **在同一召回水平上，精确率是否更高？**

只有 `P_lever(R=60%) > P_base(R=60%) = 49.05%` 才叫**移动前沿**；
若约等于，说明杠杆只是把工作点**沿同一条前沿**挪了位置，对达标毫无帮助。

协议
----
- 完整 2×2×网格：`cooldown ∈ {0,4,8}` × `qgate ∈ {关,开}` × `zmult ∈ 6 档`
- **只在验证集上选**；**若验证集上无可行点，则不评测试集**
  （没有可行点就是结论，不许去测试集碰运气）
- 必须报出各组合的**完整验证集前沿**，不只报选中的点

用法::

    python scripts/rule_frontier.py --underlying
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.api import AlertAgentAPI
from drl.dataset import (Normalizer, apply_continuous_risk, feature_names,
                         load_episodes, split_episodes, suppress_dead_zone)
from drl.dqn import DQNAgent
from drl.env import AlertEnv, RewardSpec
from drl.metrics import aggregate, evaluate_alerts
from vol_surface.alert_engine import evaluate as engine_eval
from vol_surface.alert_rules import DEFAULT_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 rule_v2_search.py 完全一致的一组 z 类阈值键
_Z = ("r1_watch_z", "r1_warn_z", "r1_ser_z", "r3_watch", "r3_warn", "r3_ser",
      "r5_watch_z", "r5_warn_z", "r5_ser_z", "r8_watch", "r8_warn", "r8_ser")
LIQ_GATE = 0.55
COOLDOWNS = [0, 4, 8]
QGATES = [False, True]
ZMULTS = [0.40, 0.55, 0.70, 0.85, 1.00, 1.30]
TARGET = {"precision": 0.50, "recall": 0.60, "avg_lead_min": 30.0}


def levels_for(e, FN, sp, zmult, cooldown, qgate):
    """按给定配置算一幕的等级序列——直接走 alert_engine（交付本体），不另写判定。"""
    from vol_surface.alert_engine import AlertState
    p = dict(sp.get(e.symbol, {}))
    for k in _Z:
        p[k] = DEFAULT_PARAMS[k] * zmult
    st = AlertState(cooldown=cooldown) if cooldown else None
    j_liq = FN.index("liquidity_ratio") if "liquidity_ratio" in FN else None
    out = np.zeros(len(e), dtype=np.int64)
    for i in range(len(e)):
        feats = {k: float(e.X[i, j]) for j, k in enumerate(FN)}
        r = engine_eval(feats, params=p, state=st, model=None,
                        composite=False, composite_min_rules=2)
        lv = int(r["level"] if isinstance(r, dict) else r)
        if qgate and j_liq is not None and float(e.X[i, j_liq]) < LIQ_GATE:
            lv = 0
        out[i] = lv
    return out


def measure(eps, FN, sp, zmult, cooldown, qgate) -> dict:
    """跑一个配置，返回 micro 指标（死区过滤开，与交付一致）。"""
    per = []
    n_alert = 0
    for e in eps:
        lv = suppress_dead_zone(e.ts, levels_for(e, FN, sp, zmult, cooldown, qgate))
        n_alert += int((lv >= 2).sum())
        per.append({"symbol": e.symbol, "ym": e.ym, "ts": e.ts,
                    "actions": lv, "risk_idx": e.risk_idx, "total_reward": 0.0,
                    "n_alert": int((lv >= 2).sum())})
    m = aggregate(per)["micro"]
    return {"precision": m["precision"], "recall": m["recall"],
            "avg_lead_min": m["avg_lead_min"], "n_alert": n_alert}


def _pass3(m) -> bool:
    return bool(m["precision"] >= TARGET["precision"]
                and m["recall"] >= TARGET["recall"]
                and m["avg_lead_min"] >= TARGET["avg_lead_min"])


def _p_at_recall(rows, target_r=0.60):
    """同召回处的精确率：取召回 ≥target 中召回最小的那个点（最贴近目标）。"""
    ok = [r for r in rows if r["recall"] >= target_r]
    return min(ok, key=lambda r: r["recall"]) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out", "rule_frontier.json"))
    ap.add_argument("--underlying", action="store_true")
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    FN = feature_names(a.underlying)
    sp = json.load(open(os.path.join(ROOT, "data_out", "symbol_params.json"),
                        encoding="utf-8"))
    print(f"验证 {len(va)} 幕 / 测试 {len(te)} 幕；特征 {len(FN)} 维")

    res = {"preregistration": "data_out/rule_frontier_preregistration.md",
           "target": TARGET, "val_frontiers": {}, "frontier_moved": {}}

    # ---------------- 验证集：完整前沿 ----------------
    print("\n=== 验证集前沿（每格一条 P-R 曲线）===")
    base_key = "cd0_q0"
    for cd in COOLDOWNS:
        for qg in QGATES:
            key = f"cd{cd}_q{int(qg)}"
            rows = []
            for z in ZMULTS:
                m = measure(va, FN, sp, z, cd, qg)
                rows.append({"zmult": z, **m, "pass3": _pass3(m)})
            res["val_frontiers"][key] = rows
            at60 = _p_at_recall(rows)
            res["frontier_moved"][key] = {
                "p_at_recall60": at60["precision"] if at60 else None,
                "recall_there": at60["recall"] if at60 else None,
                "zmult_there": at60["zmult"] if at60 else None,
                "n_alert_there": at60["n_alert"] if at60 else None,
            }
            tag = (f"P@R60={at60['precision']:.2%} (R={at60['recall']:.1%},"
                   f" z={at60['zmult']}, n={at60['n_alert']})" if at60
                   else "召回够不到 60%")
            n_ok = sum(1 for r in rows if r["pass3"])
            print(f"  冷却={cd} 流动性门={'开' if qg else '关'} | {tag} | 可行点 {n_ok}")

    # 是否移动了前沿：与基线（无冷却、无流动性门）在 R≈60% 处比
    b = res["frontier_moved"][base_key]["p_at_recall60"]
    res["baseline_p_at_recall60"] = b
    if b is not None:
        print(f"\n=== 是否移动前沿（基线 P@R60 = {b:.2%}）===")
        for k, v in res["frontier_moved"].items():
            if v["p_at_recall60"] is None:
                print(f"  {k:<10} 召回够不到 60%")
                continue
            d = (v["p_at_recall60"] - b) * 100
            v["delta_vs_base_pp"] = d
            print(f"  {k:<10} P@R60={v['p_at_recall60']:.2%}  {d:+.2f}pp"
                  f"  {'← 移动了前沿' if d > 0.5 else '（≈沿前沿滑动）'}")

    # ---------------- 工作点选择：只用验证集 ----------------
    feas = [(k, r) for k, rows in res["val_frontiers"].items()
            for r in rows if r["pass3"]]
    res["val_feasible"] = [{"cfg": k, **r} for k, r in feas]
    if not feas:
        res["picked"] = None
        res["verdict"] = {
            "passed": False,
            "reason": "验证集上没有任何配置能同时满足三项门槛",
            "text": ("**未通过**：在 3×2×6 = 36 个配置里，验证集上**没有任何一个**"
                     "同时满足 P≥50%、R≥60%、lead≥30min。按预登记协议，"
                     "**不进行测试集评估**——没有可行点就是结论。"),
        }
        print("\n" + res["verdict"]["text"])
    else:
        k, r = max(feas, key=lambda x: x[1]["precision"])
        cd = int(k.split("_")[0][2:]); qg = bool(int(k.split("_")[1][1:]))
        res["picked"] = {"cfg": k, "cooldown": cd, "qgate": qg, **r}
        print(f"\n=== 验证集选中 {k} zmult={r['zmult']}"
              f"（验证集 P={r['precision']:.2%} R={r['recall']:.2%}）===")
        tm = measure(te, FN, sp, r["zmult"], cd, qg)
        res["test"] = {**tm, "pass3": _pass3(tm)}
        print(f"  测试集 P={tm['precision']:.2%} R={tm['recall']:.2%} "
              f"lead={tm['avg_lead_min']:.1f}min → "
              f"{'✓ 三项全达标' if _pass3(tm) else '✗ 未全达标'}")
        res["verdict"] = {
            "passed": bool(_pass3(tm)),
            "text": (("**通过**：规则引擎在 anchor 连续口径上三项全达标。"
                      "但这是规则层的工程改进，不改变 DRL 与规则的判别力对比。")
                     if _pass3(tm) else
                     "**未通过**：验证集上有可行点，测试集未达标——"
                     "计入本项目已有的「验证集→测试集选择误差」记录，不重选工作点。"),
        }
        print("  " + res["verdict"]["text"])

    # ---------------- 并列 DRL 同口径成绩（防止只报规则） ----------------
    wp = json.load(open(os.path.join(a.drl_dir, "workpoint.json"), encoding="utf-8"))
    res["drl_same_caliber"] = {"lean": wp["lean"], "precision": wp["precision"],
                               "recall": wp["recall"],
                               "avg_lead_min": wp["avg_lead_min"]}
    print(f"\n  同口径 DRL 对照：P={wp['precision']:.2%} R={wp['recall']:.2%} "
          f"lead={wp['avg_lead_min']:.1f}min（lean={wp['lean']}）")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
