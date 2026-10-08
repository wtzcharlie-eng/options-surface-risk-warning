"""dead_zone_study — 结构性死区过滤的 2×2 对照（预登记判据见文末）。

缺陷
----
精确率的命中窗口是**墙钟分钟制**的 `[a, a+120min]`（`drl/metrics.py`），
而风险起点是**根数制**的（未来 `RISK_HORIZON` 根，`drl/dataset.py`）。
两者口径不一致，后果是**收盘前发出的预警承诺的那 120 分钟里几乎没有可交易时间，
结构上不可能命中**。实测（anchor 测试集 29,696 截面，全样本命中率 24.39%）：

    15:00 → 0.65%   14:45 → 1.58%   14:30 → 2.88%   02:30 → 1.89%

按「(t, t+120min] 内可交易根数」分组总体上升：0 根 8.8% → 8 根 36.6%
（**并非逐档单调**，5 根 32.6% → 6 根 30.8% 有一处回落）。

为什么必须做 2×2 而不是「开/关」两格
------------------------------------
死区过滤会改变精确率的**绑定位置**，工作点是在验证集上按「可行域内精确率最大」
选的，所以过滤一开，选出来的 lean 可能就变了。若只比「过滤前的工作点 vs
过滤后的同一个工作点」，等于把两个变量绑在一起测。本项目已经因为
「改动与另一个组件耦合、单独测给出反向结论」栽过一次（README §7.13），
故这里四格全给：{过滤关, 过滤开} × {旧工作点, 各自重选的工作点}。

**规则引擎同样施加过滤**——只给 DRL 加就不是公平对照。预期两者都会涨，
若判别力（DRL 精确率 − 规则精确率）基本不变，就必须如实说「这是全员抬升，
不是模型变好」。

预登记判据（写于跑数之前）
--------------------------
1. 过滤**只作用于休市前 `DEAD_ZONE_BARS` 根**，休市阈值 `DEAD_ZONE_BREAK_MIN=255min`（空隙 [135,375] 的中点）
   （必须 > **实测日内最长休市**；本数据集为 **135min**——早市末根 11:30 →
   下午首根 13:45，**不是想当然的 120min**。凭「午休 120」把阈值写成 130 时，
   11:00/11:15/11:30 会被 100% 误标，而它们的实测命中率 22~23%、接近基础率。
   G12 已改为从真实数据现算该值，不把 120/135 写死进断言）。
   **不过滤幕尾**：那是按月切幕的评测产物，不是市场结构（实测仅占 0.60% 截面）。
2. 工作点一律在**验证集**上选（可行域：召回≥60% 且平均提前≥30min，
   取精确率最大者），测试集只报一次。
3. 判据：**若 DRL 相对规则的精确率优势（判别力）变化不超过 ±1pp，
   即判定「该过滤是全员抬升、不构成模型改进」**，并在所有引用处如实写明。

用法::

    python scripts/dead_zone_study.py --underlying
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
from drl.baseline import RulePolicy
from drl.dataset import (DEAD_ZONE_BARS, DEAD_ZONE_BREAK_MIN, Normalizer,
                         apply_continuous_risk, load_episodes, session_dead_zone,
                         split_episodes, suppress_dead_zone)
from drl.dqn import DQNAgent
from drl.env import AlertEnv, RewardSpec
from drl.metrics import aggregate
from vol_surface.alert_rules import RECALL_FIRST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = [0.0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .8, 1.0, 1.3]
MIN_RECALL, MIN_LEAD = 0.60, 30.0


def _apply_dz(rollouts, on: bool):
    """对 rollout 施加死区过滤（不改原对象）。"""
    if not on:
        return rollouts
    out = []
    for ro in rollouts:
        acts = suppress_dead_zone(ro["ts"], ro["actions"])
        out.append({**ro, "actions": acts,
                    "n_alert": int((acts >= 2).sum())})
    return out


def _m(rollouts, dz: bool) -> dict:
    m = aggregate(_apply_dz(rollouts, dz))["micro"]
    return {k: m[k] for k in ("precision", "recall", "avg_lead_min",
                              "n_alert", "n_risk", "alert_rate")}


def _online_policy(base_policy, ep):
    """**在线抑制**版策略：死区内直接返回 0，让环境上下文按真实决策演进。

    与「事后掩码」的区别
    --------------------
    交付路径用的是事后掩码：先跑完 rollout 再把死区内的动作置 0。
    那等价于「智能体照常决策，上线时由下游网关拦掉收盘前的预警」——
    智能体的内部上下文（`since_alert` 距上次预警的分钟数、`last_level` 上一步动作）
    仍然按**未被拦截**的动作演进。

    在线抑制则等价于「智能体自己知道现在不许报」，上下文按真实发出的动作演进，
    于是它在休市后第一根上看到的「距上次预警」会更长、`last_level` 会是 0。

    两者是**不同的系统**，数字可以不同。本函数用于量化这个差别到底多大——
    若可忽略，则交付路径用哪种都行（但必须说明用的是哪种）；
    若不可忽略，交付口径必须明确，且不能拿一种的数字去支持另一种的结论。
    """
    dead = session_dead_zone(ep.ts)

    def _p(state, i):
        if dead[i]:
            return 0
        return base_policy(state, i)
    return _p


def _pass3(m) -> bool:
    return bool(m["precision"] >= .5 and m["recall"] >= MIN_RECALL
                and m["avg_lead_min"] >= MIN_LEAD)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out",
                                                  "dead_zone_study.json"))
    ap.add_argument("--underlying", action="store_true")
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()
    print(f"训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)} 幕")

    files = sorted(f for f in glob.glob(os.path.join(a.drl_dir, "agent_seed*.npz"))
                   if "shuf_" not in os.path.basename(f))
    agents, extra = [], None
    for f in files:
        ag, ex = DQNAgent.load(f)
        agents.append(ag)
        extra = extra or ex
    api = AlertAgentAPI(agents, Normalizer.from_dict(extra["normalizer"]), extra)
    print(f"集成 {len(agents)} 个 seed")

    # ---- 规则引擎（逐品种标定阈值），VAL / TEST 各跑一次即可
    sp = {}
    spp = os.path.join(ROOT, "data_out", "symbol_params.json")
    if os.path.exists(spp):
        sp = json.load(open(spp, encoding="utf-8"))

    def rule_ro(group):
        return [AlertEnv(e, norm, spec).rollout(
                RulePolicy(e, params=sp.get(e.symbol) or None)) for e in group]

    ro_rule_va, ro_rule_te = rule_ro(va), rule_ro(te)

    # ---- DRL：每个 lean 在 VAL/TEST 各跑一次，之后两种过滤都复用同一批 rollout
    ro_drl_va, ro_drl_te = {}, {}
    for t in GRID:
        api.set_confidence_threshold(t)
        ro_drl_va[t] = [AlertEnv(e, norm, spec).rollout(api.policy()) for e in va]
        ro_drl_te[t] = [AlertEnv(e, norm, spec).rollout(api.policy()) for e in te]
        print(f"  lean={t:.2f} 跑完", flush=True)

    res = {"caliber": "anchor 连续测试集；精确率/召回/提前均为 micro 口径",
           "dead_zone": {"bars": DEAD_ZONE_BARS,
                         "break_min": DEAD_ZONE_BREAK_MIN,
                         "note": "只过滤休市前 N 根，不过滤幕尾"},
           "constraint": {"min_recall": MIN_RECALL, "min_lead": MIN_LEAD},
           "arms": {}}

    picks = {}
    for dz in (False, True):
        tag = "dz_on" if dz else "dz_off"
        val = [{"lean": t, **_m(ro_drl_va[t], dz)} for t in GRID]
        ok = [r for r in val if r["recall"] >= MIN_RECALL
              and r["avg_lead_min"] >= MIN_LEAD]
        pick = max(ok, key=lambda r: r["precision"]) if ok else None
        picks[tag] = pick
        res["arms"][tag] = {
            "val_curve": val,
            "picked_lean": pick["lean"] if pick else None,
            "val_at_pick": pick,
            "rule_val": _m(ro_rule_va, dz),
            "rule_test": _m(ro_rule_te, dz),
        }
        print(f"[{tag}] 验证集选出 lean="
              f"{pick['lean'] if pick else '—'}")

    # ---- 2×2：{旧工作点, 新工作点} × {过滤关, 过滤开}
    lean_old = picks["dz_off"]["lean"]
    lean_new = picks["dz_on"]["lean"]
    cells = {}
    for wp_tag, lean in (("wp_old", lean_old), ("wp_new", lean_new)):
        for dz in (False, True):
            dz_tag = "dz_on" if dz else "dz_off"
            m = _m(ro_drl_te[lean], dz)
            r = _m(ro_rule_te, dz)
            cells[f"{wp_tag}|{dz_tag}"] = {
                "lean": lean, "drl": m, "rule": r,
                "drl_pass3": _pass3(m),
                # 判别力：同一口径下 DRL 与规则的精确率之差
                "discrimination_pp": (m["precision"] - r["precision"]) * 100,
            }
    res["grid_2x2"] = cells
    res["lean_old"], res["lean_new"] = lean_old, lean_new

    # ---- 预登记判据的裁决
    #
    # ⚠ 这里的 before/after 取的是 **2×2 的对角线**（`wp_old|dz_off` → `wp_new|dz_on`），
    # 即「换工作点」与「开死区过滤」两个效应叠在一起。判据是预登记的，**不改**；
    # 但只报对角线就成了本项目反复点名的「跨口径相减」——2×2 存在的唯一理由
    # 就是把这两个效应拆开。故**同时**落盘同工作点（纯过滤）的那一组，
    # 并在文档里以它为主。两组结论一致（都判为全员抬升），故不影响任何结论。
    d_off = cells["wp_old|dz_off"]["discrimination_pp"]
    d_on = cells["wp_new|dz_on"]["discrimination_pp"]
    # 纯过滤：固定在交付工作点 wp_new 上，只切换死区开关
    p_off = cells["wp_new|dz_off"]["discrimination_pp"]
    p_on = cells["wp_new|dz_on"]["discrimination_pp"]
    _gain = lambda a, b, who: (cells[b][who]["precision"]
                               - cells[a][who]["precision"]) * 100
    res["verdict"] = {
        "discrimination_before_pp": d_off,
        "discrimination_after_pp": d_on,
        "delta_pp": d_on - d_off,
        "is_uniform_lift": abs(d_on - d_off) <= 1.0,
        # ---- 同工作点（纯过滤）的拆解，**推荐引用这一组**
        "same_wp": {
            "note": ("固定在交付工作点 wp_new，只切换死区开关——"
                     "对角线那组混进了『换工作点』的效应"),
            "discrimination_before_pp": p_off,
            "discrimination_after_pp": p_on,
            "delta_pp": p_on - p_off,
            "is_uniform_lift": abs(p_on - p_off) <= 1.0,
            "drl_gain_pp": _gain("wp_new|dz_off", "wp_new|dz_on", "drl"),
            "rule_gain_pp": _gain("wp_new|dz_off", "wp_new|dz_on", "rule"),
        },
        "diagonal_drl_gain_pp": _gain("wp_old|dz_off", "wp_new|dz_on", "drl"),
        "text": ("判别力变化在 ±1pp 内 → **全员抬升，不构成模型改进**，"
                 "任何引用都必须并列规则引擎的同等增益"
                 if abs(d_on - d_off) <= 1.0 else
                 "判别力变化超过 ±1pp → 该过滤对两方不对称，需单独解释"),
    }

    # ---- 事后掩码 vs 在线抑制：两者是不同的系统，差别必须量化后再选口径
    online = {}
    for _tag, _lean in (("wp_new", lean_new),):
        api.set_confidence_threshold(_lean)
        _bp = api.policy()
        ro_on = [AlertEnv(e, norm, spec).rollout(_online_policy(_bp, e)) for e in te]
        m_on = _m(ro_on, False)              # 策略已自行抑制，无需再掩码
        m_mask = _m(ro_drl_te[_lean], True)  # 事后掩码（交付路径）
        online[_tag] = {
            "lean": _lean, "online": m_on, "posthoc_mask": m_mask,
            "delta_precision_pp": (m_on["precision"] - m_mask["precision"]) * 100,
            "delta_recall_pp": (m_on["recall"] - m_mask["recall"]) * 100,
            "delta_lead_min": m_on["avg_lead_min"] - m_mask["avg_lead_min"],
            "delta_n_alert": m_on["n_alert"] - m_mask["n_alert"],
        }
    res["online_vs_posthoc"] = {
        "note": ("交付路径用**事后掩码**（先跑完 rollout 再把死区动作置 0），"
                 "等价于「智能体照常决策、上线时由下游网关拦截」；"
                 "在线抑制则让智能体上下文按真实发出的动作演进。"
                 "规则引擎无自身状态，两种做法对它完全等价，故只测 DRL。"),
        "cells": online,
    }
    _o = online["wp_new"]
    res["online_vs_posthoc"]["verdict"] = (
        f"精确率差 {_o['delta_precision_pp']:+.2f}pp、召回差 {_o['delta_recall_pp']:+.2f}pp、"
        f"预警数差 {_o['delta_n_alert']:+d}。"
        + ("差异可忽略（<0.5pp），交付沿用事后掩码并在文档说明即可。"
           if abs(_o["delta_precision_pp"]) < 0.5 and abs(_o["delta_recall_pp"]) < 0.5
           else "**差异不可忽略**，交付口径必须明确写出用的是哪一种，"
                "且不得用一种的数字支持另一种的结论。"))

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("\n=== 2×2（测试集）===")
    print(f"{'格':<20}{'DRL P':>9}{'DRL R':>9}{'lead':>8}{'三项':>6}"
          f"{'规则 P':>10}{'判别力':>9}")
    for k, v in cells.items():
        print(f"{k:<20}{v['drl']['precision']:>9.2%}{v['drl']['recall']:>9.2%}"
              f"{v['drl']['avg_lead_min']:>7.1f}m{'✓' if v['drl_pass3'] else '✗':>6}"
              f"{v['rule']['precision']:>10.2%}{v['discrimination_pp']:>8.2f}pp")
    print(f"\n判别力 {d_off:+.2f}pp → {d_on:+.2f}pp（变化 {d_on-d_off:+.2f}pp）")
    print("\n=== 事后掩码 vs 在线抑制（DRL @ 新工作点）===")
    print(f"  事后掩码: P={_o['posthoc_mask']['precision']:.2%} "
          f"R={_o['posthoc_mask']['recall']:.2%} n={_o['posthoc_mask']['n_alert']}")
    print(f"  在线抑制: P={_o['online']['precision']:.2%} "
          f"R={_o['online']['recall']:.2%} n={_o['online']['n_alert']}")
    print(f"  {res['online_vs_posthoc']['verdict']}")
    print(res["verdict"]["text"])
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
