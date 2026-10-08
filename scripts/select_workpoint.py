"""select_workpoint — 沿 P-R 前沿选定交付工作点，并落盘供报告引用。

为什么需要单独一步
------------------
贪心（argmax，等价于 `lean=0`）只是前沿上的**一个点**，不是最优工作点。
赛题的约束是「精确率≥50% 且 召回≥60% 且 平均提前≥30min」，
在前沿上换一个点往往更接近达标。

**选择只能用验证集。** 这一条是本项目反复踩过的坑（§7.8 的逐品种阈值贪心版
在验证集上「过线」51.08%，测试集只有 44.66%）。故本脚本：
  1. 在**验证集**上扫 `lean` 网格，取满足召回/提前约束下精确率最大的那个；
  2. 用该阈值在**测试集**上评估一次，作为交付数字；
  3. 另外报出「若允许在测试集上挑阈值」的前沿值，作为对照——
     两者之差即阈值选择误差，是诚实披露的一部分，不作为成绩。

用法::

    python scripts/select_workpoint.py --drl-dir data_out/drl --underlying
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
from drl.dataset import (Normalizer, apply_continuous_risk, load_episodes,
                         split_episodes, suppress_dead_zone)
from drl.dqn import DQNAgent
from drl.env import AlertEnv, RewardSpec
from drl.metrics import aggregate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = [0.0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .8, 1.0, 1.3]


def _eval(api, grp, norm, spec, lean, dead_zone: bool = True):
    api.set_confidence_threshold(lean)
    ros = [AlertEnv(e, norm, spec).rollout(api.policy()) for e in grp]
    if dead_zone:
        # 结构性死区过滤：收盘前发的预警承诺的 120min 里几乎没有可交易时间，
        # 结构上不可能命中（实测 15:00 命中率 0.65%，全样本 24.4%）。
        # **该过滤对规则引擎同样成立且增益同等**，故它不是模型改进——
        # 判别力实测仅变 −0.26pp，见 data_out/dead_zone_study.json。
        ros = [{**r, "actions": suppress_dead_zone(r["ts"], r["actions"])}
               for r in ros]
    m = aggregate(ros)["micro"]
    return {"lean": lean, "precision": m["precision"], "recall": m["recall"],
            "avg_lead_min": m["avg_lead_min"], "n_alert": m["n_alert"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--underlying", action="store_true")
    ap.add_argument("--monthly-risk", action="store_true")
    ap.add_argument("--min-recall", type=float, default=0.60)
    ap.add_argument("--min-lead", type=float, default=30.0)
    ap.add_argument("--no-dead-zone", action="store_true",
                    help="关闭结构性死区过滤（默认开启，见 drl.dataset.session_dead_zone）")
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    if not a.monthly_risk:
        apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()

    files = sorted(f for f in glob.glob(os.path.join(a.drl_dir, "agent_seed*.npz"))
                   if "shuf_" not in os.path.basename(f))
    if not files:
        raise SystemExit(f"{a.drl_dir} 下没有 checkpoint")
    agents, extra = [], None
    for f in files:
        ag, ex = DQNAgent.load(f)
        agents.append(ag)
        extra = extra or ex
    api = AlertAgentAPI(agents, Normalizer.from_dict(extra["normalizer"]), extra)

    dz = not a.no_dead_zone
    val = [_eval(api, va, norm, spec, t, dz) for t in GRID]
    ok = [r for r in val if r["recall"] >= a.min_recall
          and r["avg_lead_min"] >= a.min_lead]
    if not ok:
        raise SystemExit(f"验证集上没有满足 召回≥{a.min_recall:.0%} 且 "
                         f"提前≥{a.min_lead:.0f}min 的工作点")
    pick = max(ok, key=lambda r: r["precision"])

    test = [_eval(api, te, norm, spec, t, dz) for t in GRID]
    chosen = next(r for r in test if r["lean"] == pick["lean"])

    # 对照：测试集前沿在目标召回处的精确率（**不作为成绩**，只用于量化选择误差）
    #
    # 若目标召回落在测试集曲线的召回range之外，则该量**无法插值**，如实记 NaN 并
    # 给出原因，不做外推。15-seed 集成就撞上了这个边界：测试集召回下界 60.4% > 60%，
    # 整条曲线都在目标之上。
    # **不为此扩大 lean 网格**——扩网格本身无害（选点仍只用验证集），但那一步发生在
    # 已经看过测试集曲线之后，与"看了测试集再改搜索空间"无法区分，故不做。
    pts = sorted((r["recall"], r["precision"]) for r in test)
    R = np.array([p[0] for p in pts])
    P = np.array([p[1] for p in pts])
    if R.min() <= a.min_recall <= R.max():
        front = float(np.interp(a.min_recall, R, P))
        front_note = ""
    else:
        front = float("nan")
        front_note = (f"目标召回 {a.min_recall:.0%} 落在测试集曲线召回范围 "
                      f"[{R.min():.1%}, {R.max():.1%}] 之外，无法插值（不外推）")

    out = {"lean": pick["lean"], "n_models": len(files),
           "val_precision": pick["precision"], "val_recall": pick["recall"],
           "val_lead_min": pick["avg_lead_min"],
           "precision": chosen["precision"], "recall": chosen["recall"],
           "avg_lead_min": chosen["avg_lead_min"], "n_alert": chosen["n_alert"],
           "frontier_p_at_r60": front, "frontier_note": front_note,
           "n_seeds": len(files),
           "constraint": {"min_recall": a.min_recall, "min_lead": a.min_lead},
           "grid": GRID, "val_curve": val, "test_curve": test}
    p = os.path.join(a.drl_dir, "workpoint.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    print(f"验证集选出 lean={pick['lean']:.2f}（验证精确率 {pick['precision']:.2%}，"
          f"召回 {pick['recall']:.2%}，提前 {pick['avg_lead_min']:.1f}min）")
    print(f"测试集      精确率 {chosen['precision']:.2%} "
          f"{'✓' if chosen['precision'] >= .5 else '✗'}  "
          f"召回 {chosen['recall']:.2%} {'✓' if chosen['recall'] >= .6 else '✗'}  "
          f"提前 {chosen['avg_lead_min']:.1f}min "
          f"{'✓' if chosen['avg_lead_min'] >= 30 else '✗'}")
    if front == front:
        print(f"对照（不作为成绩）：测试集前沿在召回 {a.min_recall:.0%} 处 {front:.2%}，"
              f"阈值选择误差 {(front - chosen['precision']) * 100:+.2f}pp")
    else:
        print(f"对照（不作为成绩）：不可用——{front_note}")
    print(f"→ {p}")


if __name__ == "__main__":
    main()
