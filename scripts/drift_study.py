"""drift_study — 「本窗选阈值 → 下窗实际表现」的漂移是否可估？

要回答的问题
------------
交付工作点在验证集上判别力 29.75%、测试集只有 23.19%，**系统性乐观 6.19pp**
（14 个 lean 值全为负、标准差仅 0.47pp，是稳定的水平位移而非噪声）。
若这个 margin 能在选点时就估出来，选点协议加上它即可；估不出，就说明漂移来自
真实的 regime 变化，任何只用历史数据的协议都救不了——那时诚实的做法是披露
「前沿上有达标点但样本外选不中」，而不是回头去测试集上挑。

设计：不碰真测试集
------------------
用**规则引擎**（阈值是 z 分倍数，无需训练，可在任意时间窗上瞬时评估）跑
滚动起点：对相邻窗口对 (W_i, W_{i+1})，在 W_i 上按赛题约束选阈值，
再看它在 W_{i+1} 上的实际表现，记录漂移。全部窗口都取自
**训练期 + 验证期**（≤2025-06），真测试集（2025-07 起）一次都不读。

若各窗口对的漂移量级相近 → margin 可估；若正负横跳或量级差数倍 → 不可估。

**这是规则引擎上的结论，不能直接外推到 DRL**——两者的过拟合机制不同。
它只用于决定「值不值得为 DRL 付重训成本」。

用法::

    python scripts/drift_study.py --underlying
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.baseline import RulePolicy
from drl.dataset import (Normalizer, apply_continuous_risk, load_episodes,
                         split_episodes)
from drl.env import AlertEnv, RewardSpec
from drl.metrics import aggregate
from drl.train import hit_base_rate
from vol_surface.alert_rules import DEFAULT_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 alert_rules.RECALL_FIRST 同一组 z 阈键；倍数越小越松（召回↑ 精确率↓）
_Z_KEYS = ("r1_watch_z", "r1_warn_z", "r1_ser_z", "r3_watch", "r3_warn", "r3_ser",
           "r5_watch_z", "r5_warn_z", "r5_ser_z", "r8_watch", "r8_warn", "r8_ser")
GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50]

# 滚动起点的窗口边界（半年一段，全部 ≤2025-06）
WINDOWS = [("2023-07", "2023-12"), ("2024-01", "2024-06"),
           ("2024-07", "2024-12"), ("2025-01", "2025-06")]


def _params(mult: float) -> dict:
    return {k: DEFAULT_PARAMS[k] * mult for k in _Z_KEYS}


def _eval(eps, norm, spec, mult: float, base: float) -> dict:
    ros = [AlertEnv(e, norm, spec).rollout(RulePolicy(e, params=_params(mult)))
           for e in eps]
    m = aggregate(ros)["micro"]
    return {"mult": mult, "P": m["precision"], "R": m["recall"],
            "lead": m["avg_lead_min"], "DP": m["precision"] - base,
            "n_alert": m["n_alert"]}


def _select(curve: list, min_recall: float, min_lead: float) -> dict | None:
    """与 select_workpoint.py 同一条选点规则：可行域内精确率最大。"""
    ok = [r for r in curve if r["R"] >= min_recall and r["lead"] >= min_lead]
    return max(ok, key=lambda r: r["P"]) if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out", "drift_study.json"))
    ap.add_argument("--underlying", action="store_true")
    ap.add_argument("--min-recall", type=float, default=0.60)
    ap.add_argument("--min-lead", type=float, default=30.0)
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()   # 归一化只用训练集，口径不变

    # 真测试集在此之后不再出现——只用它报个数确认没混进来
    print(f"载入 {len(eps)} 幕；真测试集 {len(te)} 幕**本实验完全不用**\n")
    pool = tr + va
    by_win = {}
    for lo, hi in WINDOWS:
        g = [e for e in pool if lo <= e.ym <= hi]
        if g:
            by_win[(lo, hi)] = g
    for (lo, hi), g in by_win.items():
        print(f"  窗口 {lo}~{hi}: {len(g)} 幕  基础率 {hit_base_rate(g):.2%}")

    curves, bases = {}, {}
    for k, g in by_win.items():
        bases[k] = hit_base_rate(g)
        curves[k] = [_eval(g, norm, spec, m, bases[k]) for m in GRID]
        print(f"  {k[0]}~{k[1]} 曲线算完")

    keys = list(by_win)
    rows = []
    print(f"\n{'选点窗':16s}{'→应用窗':16s}{'选中倍数':>8s}"
          f"{'选点DP':>8s}{'应用DP':>8s}{'漂移':>8s}{'精确率漂移':>10s}")
    for i in range(len(keys) - 1):
        ks, ka = keys[i], keys[i + 1]
        pick = _select(curves[ks], a.min_recall, a.min_lead)
        if pick is None:
            print(f"  {ks[0]}~{ks[1]}: 可行域为空，跳过")
            continue
        applied = next(r for r in curves[ka] if r["mult"] == pick["mult"])
        rows.append({"sel_win": list(ks), "app_win": list(ka),
                     "mult": pick["mult"],
                     "sel_DP": pick["DP"], "app_DP": applied["DP"],
                     "drift_DP": applied["DP"] - pick["DP"],
                     "sel_P": pick["P"], "app_P": applied["P"],
                     "drift_P": applied["P"] - pick["P"]})
        r = rows[-1]
        print(f"  {ks[0]}~{ks[1]:9s}{ka[0]}~{ka[1]:9s}{pick['mult']:8.2f}"
              f"{pick['DP']*100:7.2f}%{applied['DP']*100:8.2f}%"
              f"{r['drift_DP']*100:+8.2f}{r['drift_P']*100:+10.2f}")

    if len(rows) >= 2:
        d = np.array([r["drift_DP"] for r in rows]) * 100
        print(f"\n判别力漂移: 均值 {d.mean():+.2f}pp  标准差 {d.std(ddof=1):.2f}pp  "
              f"范围 [{d.min():+.2f}, {d.max():+.2f}]")
        same_sign = bool(np.all(d < 0) or np.all(d > 0))
        print(f"符号是否一致: {'是' if same_sign else '否——方向都不稳定'}")
        print(f"\n对照：真 val→test 的判别力漂移为 −6.19pp（本实验未使用该数据）")
        verdict = ("margin 看起来可估：漂移同号且离散度小"
                   if same_sign and d.std(ddof=1) < abs(d.mean()) * 0.6
                   else "margin 不可估：漂移方向或量级不稳定，"
                        "说明它来自 regime 变化而非可外推的系统偏差")
        print(f"判定：{verdict}")
    else:
        print("\n有效窗口对不足 2 对，无法判定")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"windows": [list(k) for k in keys], "grid": GRID,
                   "curves": {f"{k[0]}~{k[1]}": v for k, v in curves.items()},
                   "bases": {f"{k[0]}~{k[1]}": v for k, v in bases.items()},
                   "pairs": rows,
                   "constraint": {"min_recall": a.min_recall,
                                  "min_lead": a.min_lead}},
                  f, ensure_ascii=False, indent=2, default=float)
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
