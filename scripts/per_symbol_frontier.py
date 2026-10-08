"""per_symbol_frontier — 逐品种 P-R-lead 前沿：未达标品种能否靠逐品种选点修好。

结论先说：**不能**。si / lc / cu / rb **四个**品种在**验证集**上都没有可行点，
且都卡在**平均提前时间**——在精确率与召回都达标的子区间内，
最高 lead 分别只有 21.4 / 20.1 / 27.2 / 29.7 分钟，全部够不到 30 分钟门槛。

为什么要有这个脚本
------------------
`per_symbol.json` 只给出交付工作点（全局 `lean`）下的逐品种成绩，
回答不了「换成逐品种工作点能不能达标」。本脚本扫每个品种自己的 `lean` 曲线，
把可行域是否为空这件事落盘，供文档引用——**不手写数字**。

一个必须避开的陷阱
------------------
在**测试集**上扫曲线时，si 看起来能过（`lean=0` 处 lead 恰好 30.0min）。
但那是**看了测试集才发现的**，且 si 只有 4 幕测试样本、lead 正好压在门槛上。
按本项目规矩，工作点只能在**验证集**上选——而 si 在验证集上无任何可行点。
**cu 更能说明问题**：它在测试集上有 7 个可行点、在全局交付工作点下也达标，
但**验证集上一个都没有**——即它的「达标」并非验证集所能支持的。
故本脚本**同时输出验证集与测试集两条曲线**，并明确标注：
**可行性判断只看验证集，测试集曲线仅供诊断。**

关键机制
--------
`lean` 只能在**精确率与召回之间**换，**换不出提前时间**——
提前时间由「信号相对风险起点何时出现」决定，是特征与标签的性质，不是阈值能调的。
这与 README §7.12 记的「规则是探测器不是预测器」是同一个机制，
只是这次出现在 DRL 上、且只出现在部分品种上。

用法::

    python scripts/per_symbol_frontier.py --underlying
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
GRID = [0.0, .1, .2, .3, .5, .8, 1.2, 1.8, 2.5, 3.5, 5.0]
SYMS = ["ag", "au", "sc", "si", "lc", "cu", "rb"]
TARGET = {"precision": 0.50, "recall": 0.60, "avg_lead_min": 30.0}


def _pass3(m) -> bool:
    return bool(m["precision"] >= TARGET["precision"]
                and m["recall"] >= TARGET["recall"]
                and m["avg_lead_min"] >= TARGET["avg_lead_min"])


def _binding(m) -> list:
    """列出未满足的门槛——用于说明「卡在哪一项」。"""
    return [k for k, v in (("精确率", m["precision"] >= TARGET["precision"]),
                           ("召回", m["recall"] >= TARGET["recall"]),
                           ("提前时间", m["avg_lead_min"] >= TARGET["avg_lead_min"]))
            if not v]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out",
                                                  "per_symbol_frontier.json"))
    ap.add_argument("--underlying", action="store_true")
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()

    files = sorted(f for f in glob.glob(os.path.join(a.drl_dir, "agent_seed*.npz"))
                   if "shuf_" not in os.path.basename(f))
    agents, extra = [], None
    for f in files:
        ag, ex = DQNAgent.load(f)
        agents.append(ag); extra = extra or ex
    api = AlertAgentAPI(agents, Normalizer.from_dict(extra["normalizer"]), extra)

    def curve(group):
        rows = []
        for lv in GRID:
            api.set_confidence_threshold(lv)
            ros = []
            for e in group:
                ro = AlertEnv(e, norm, spec).rollout(api.policy())
                ros.append({**ro, "actions": suppress_dead_zone(ro["ts"], ro["actions"])})
            m = aggregate(ros)["micro"]
            rows.append({"lean": lv, "precision": m["precision"], "recall": m["recall"],
                         "avg_lead_min": m["avg_lead_min"], "n_alert": m["n_alert"],
                         "pass3": _pass3(m), "binding": _binding(m)})
        return rows

    res = {"target": TARGET,
           "caliber": "anchor 连续测试集；死区过滤开；micro 口径",
           "protocol": ("**可行性判断只看验证集**——工作点只能在验证集上选。"
                        "测试集曲线仅供诊断，不得据以选点。"),
           "per_symbol": {}}
    print(f"验证 {len(va)} 幕 / 测试 {len(te)} 幕\n")
    for sym in SYMS:
        gv = [e for e in va if e.symbol == sym]
        gt = [e for e in te if e.symbol == sym]
        if not gv or not gt:
            continue
        cv, ct = curve(gv), curve(gt)
        fv = [r for r in cv if r["pass3"]]
        ft = [r for r in ct if r["pass3"]]
        res["per_symbol"][sym] = {
            "n_val_ep": len(gv), "n_test_ep": len(gt),
            "val_curve": cv, "test_curve": ct,
            "val_feasible": [r["lean"] for r in fv],
            "test_feasible": [r["lean"] for r in ft],
            # ⚠ 「整条曲线的最高 lead」是个**会骗人的量**：cu 在 lean=5.0 处 lead=36.7min
            # 达标，但那里召回只剩 14.4%。据此说「cu 不受提前时间约束」每个字都真、
            # 合起来是假。故同时给出**在另两项已达标的子区间内**的最高 lead——
            # 那才是「还差多少」的正确度量。
            "val_max_lead_anywhere": max(r["avg_lead_min"] for r in cv),
            "val_max_precision_anywhere": max(r["precision"] for r in cv),
            "val_max_lead_where_pr_ok": (
                max([r["avg_lead_min"] for r in cv
                     if r["precision"] >= TARGET["precision"]
                     and r["recall"] >= TARGET["recall"]], default=None)),
            "val_max_precision_where_rl_ok": (
                max([r["precision"] for r in cv
                     if r["recall"] >= TARGET["recall"]
                     and r["avg_lead_min"] >= TARGET["avg_lead_min"]], default=None)),
            "val_binding_at_best": min(
                cv, key=lambda r: len(r["binding"]))["binding"],
        }
        v = res["per_symbol"][sym]
        _mw = v["val_max_lead_where_pr_ok"]
        print(f"  {sym}: 验证集可行点 {len(fv):>1} 个 | 测试集 {len(ft):>1} 个"
              f" | P、R 均达标处的最高 lead "
              f"{('%.1fmin' % _mw) if _mw is not None else '该区间为空'}"
              f" | 卡 {'/'.join(v['val_binding_at_best']) or '（无）'}")

    unfix = [s for s, v in res["per_symbol"].items() if not v["val_feasible"]]
    # 「因提前时间无解」的正确判据：**在精确率与召回都达标的子区间内**，
    # 最高 lead 仍够不到门槛（含该子区间为空的情形——那说明另两项就先冲突了）。
    def _lead_bound(v):
        m = v["val_max_lead_where_pr_ok"]
        return m is None or m < TARGET["avg_lead_min"]
    lead_bound = [s for s in unfix if _lead_bound(res["per_symbol"][s])]
    res["summary"] = {
        "no_feasible_on_val": unfix,
        "lead_bound": lead_bound,
        "text": (f"验证集上无可行点的品种：{unfix or '（无）'}；"
                 f"其中因**平均提前时间**够不到门槛而无解的：{lead_bound or '（无）'}。"
                 "`lean` 只能在精确率与召回之间换，**换不出提前时间**——"
                 "提前时间由信号相对风险起点何时出现决定，是特征与标签的性质。"
                 "故逐品种选工作点**修不了**这些品种。"),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n" + res["summary"]["text"])
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
