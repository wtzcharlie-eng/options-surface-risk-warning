"""eval_v2 — 按命题方 2026-08 答复的新口径做全量评测，落盘 `data_out/metrics_v2.json`。

命题方澄清的口径（与旧口径的差别）
----------------------------------
| 项 | 旧口径 | 新口径（本脚本） |
|---|---|---|
| 精确率分母 | 每个触发的 15min 截面 | **合并连续预警后的预警事件** |
| 召回匹配 | 预警须落在风险起点前 120min 内 | **不设时间上限** |
| 平均提前时间 | 同上 120min 上限 | **不设上限**（本脚本取 nearest：紧邻风险起点之前的那次预警） |
| 分区报告 | 无 | **分高波 / 低波区** |

**新口径下三条门槛实际塌缩成一条。** 召回不设上限 ⇒ 幕内任意一次早期预警即覆盖其后
全部风险起点，实测所有平凡策略召回均为 98.9%、平均提前虚高至 2 万分钟。
故本脚本**强制**同时输出平凡策略成绩，并把「DRL 精确率 − 最强平凡策略精确率」
作为唯一有意义的判别力指标。任何只报三项达标而不报这个差值的结论都不成立。

高低波分区的定义
----------------
按**训练期**每品种 `atm_iv` 的中位数划分，**不使用测试期任何信息**——
否则分区本身就带入了未来信息。分区是为回应命题方
「固定阈值在低波动市况下频繁误报」这一自陈痛点。

选点协议
--------
与旧口径完全一致：**只在验证集上选**（可行域内精确率最大），测试集只评一次。
旧口径的结果保留在 `data_out/drl/workpoint.json`，两个口径都要报、都要标明。

用法::

    python scripts/eval_v2.py --underlying
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.api import AlertAgentAPI
from drl.baseline import RulePolicy
from drl.dataset import (Normalizer, apply_continuous_risk, feature_names,
                         load_episodes, split_episodes)
from drl.dqn import DQNAgent
from drl.env import AlertEnv, RewardSpec
from drl.metrics_v2 import (HIT_WINDOW_MIN, aggregate_v2, evaluate_v2,
                            merge_alerts, trivial_policies, _epoch)
from vol_surface.alert_rules import DEFAULT_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMS = ["ag", "au", "sc", "si", "lc", "cu", "rb"]
LEAN_GRID = [0.0, .35, .6, 1.0, 1.3, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0]
ZMULT_GRID = [0.55, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.8, 4.5]
_Z = ("r1_watch_z", "r1_warn_z", "r1_ser_z", "r3_watch", "r3_warn", "r3_ser",
      "r5_watch_z", "r5_warn_z", "r5_ser_z", "r8_watch", "r8_warn", "r8_ser")

_ok = lambda m: (m["precision"] >= .5 and m["recall"] >= .6
                 and m["avg_lead_min"] >= 30)


def _levels_rule(e, sp, mult):
    p = dict(sp.get(e.symbol, {}))
    for k in _Z:
        p[k] = DEFAULT_PARAMS[k] * mult
    return RulePolicy(e, params=p).levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out", "metrics_v2.json"))
    ap.add_argument("--underlying", action="store_true")
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()
    FN = feature_names(underlying=a.underlying)
    j_iv = FN.index("atm_iv")

    # 高/低波阈值：仅用训练期，避免分区定义泄漏未来信息
    thr = {s: float(np.median(np.concatenate(
        [e.X[:, j_iv] for e in tr if e.symbol == s])))
        for s in sorted({e.symbol for e in tr})}

    files = sorted(f for f in glob.glob(os.path.join(a.drl_dir, "agent_seed*.npz"))
                   if "shuf_" not in os.path.basename(f))
    agents, extra = [], None
    for f in files:
        ag, ex = DQNAgent.load(f)
        agents.append(ag)
        extra = extra or ex
    api = AlertAgentAPI(agents, Normalizer.from_dict(extra["normalizer"]), extra)
    with open(os.path.join(ROOT, "data_out", "symbol_params.json"),
              encoding="utf-8") as f:
        sp = json.load(f)

    # ---- 通用：给定「幕 → 等级序列」的函数，算新口径指标
    def run(grp, fn, per_symbol=False, vol_split=False):
        ms, by_sym, reg_hit, reg_tot = [], {}, {"高波": 0, "低波": 0}, {"高波": 0, "低波": 0}
        for e in grp:
            ro = AlertEnv(e, norm, spec).rollout(lambda s, i: 0)
            lv = np.asarray(fn(e, ro))
            m = evaluate_v2(ro["ts"], lv, ro["risk_idx"])
            ms.append(m)
            if per_symbol:
                by_sym.setdefault(e.symbol, []).append(m)
            if vol_split:
                ep = _epoch(ro["ts"])
                evs = merge_alerts(np.sort(ep[lv >= 2]))
                rl = (np.sort(ep[np.asarray(ro["risk_idx"], dtype=np.int64)]).tolist()
                      if len(ro["risk_idx"]) else [])
                pos = {int(t): k for k, t in enumerate(ep)}
                for t in evs:
                    k = pos[int(t)]
                    reg = "高波" if e.X[k, j_iv] >= thr.get(e.symbol, 0) else "低波"
                    reg_tot[reg] += 1
                    q = bisect.bisect_left(rl, t)
                    if q < len(rl) and rl[q] <= t + HIT_WINDOW_MIN * 60:
                        reg_hit[reg] += 1
        out = aggregate_v2(ms)
        if per_symbol:
            out["per_symbol"] = {s: aggregate_v2(v) for s, v in by_sym.items()}
        if vol_split:
            out["vol_split"] = {r: {"n_event": reg_tot[r],
                                    "precision": (reg_hit[r] / reg_tot[r]
                                                  if reg_tot[r] else 0.0)}
                                for r in reg_tot}
        return out

    def drl_fn(t):
        api.set_confidence_threshold(t)
        pol = api.policy()
        def f(e, ro):
            return AlertEnv(e, norm, spec).rollout(pol)["actions"]
        return f

    # ---- DRL：验证集选点 → 测试集评一次
    print("DRL 验证集扫描 ...")
    val = [(t, run(va, drl_fn(t))) for t in LEAN_GRID]
    feas = [(t, m) for t, m in val if m["recall"] >= .6 and m["avg_lead_min"] >= 30]
    if not feas:
        raise SystemExit("验证集可行域为空")
    lean, mv = max(feas, key=lambda x: x[1]["precision"])
    mt = run(te, drl_fn(lean), per_symbol=True, vol_split=True)
    print(f"  选出 lean={lean:.2f}（验证 P={mv['precision']:.1%} R={mv['recall']:.1%}）")
    print(f"  测试 P={mt['precision']:.2%} R={mt['recall']:.2%} "
          f"lead={mt['avg_lead_min']:.0f}m → {'✓ 三项全达标' if _ok(mt) else '✗'}")

    # ---- 规则引擎：同协议
    print("规则引擎 验证集扫描 ...")
    rval = [(m_, run(va, lambda e, ro, m_=m_: _levels_rule(e, sp, m_)))
            for m_ in ZMULT_GRID]
    rfeas = [(m_, x) for m_, x in rval
             if x["recall"] >= .6 and x["avg_lead_min"] >= 30]
    zmult, rmv = max(rfeas, key=lambda x: x[1]["precision"])
    rmt = run(te, lambda e, ro: _levels_rule(e, sp, zmult),
              per_symbol=True, vol_split=True)
    print(f"  选出 系数={zmult:.2f}（验证 P={rmv['precision']:.1%}）")
    print(f"  测试 P={rmt['precision']:.2%} R={rmt['recall']:.2%} "
          f"→ {'✓' if _ok(rmt) else '✗'}")

    # ---- 平凡策略对照（**必须报**，见模块 docstring）
    print("平凡策略对照 ...")
    triv = {}
    names = list(trivial_policies(10))
    for nm in names:
        triv[nm] = run(te, lambda e, ro, nm=nm:
                       trivial_policies(len(ro["ts"]))[nm])
    best_triv = max(triv.values(), key=lambda m: m["precision"])
    best_nm = [k for k, v in triv.items() if v is best_triv][0]
    disc = mt["precision"] - best_triv["precision"]
    print(f"  最强平凡策略「{best_nm}」精确率 {best_triv['precision']:.1%}"
          f"（召回 {best_triv['recall']:.1%}）")
    print(f"  → DRL 判别力 = {disc * 100:+.1f}pp")

    out = {
        "caliber": {
            "precision_unit": "合并连续预警后的预警事件",
            "merge_gap_min": 120.0, "hit_window_min": HIT_WINDOW_MIN,
            "recall_window_min": None, "lead_mode": "nearest",
            "vol_split_rule": "训练期每品种 atm_iv 中位数（不使用测试期信息）",
            "source": "命题方 2026-08 答复",
        },
        "drl": {"lean": lean, "val": mv, "test": mt, "pass3": bool(_ok(mt))},
        "rule": {"zmult": zmult, "val": rmv, "test": rmt, "pass3": bool(_ok(rmt))},
        "trivial": triv,
        "best_trivial": {"name": best_nm, "precision": best_triv["precision"]},
        "discrimination_pp": disc * 100,
        "vol_threshold_train": thr,
        "degeneracy_note": (
            "召回不设时间上限 ⇒ 幕内任一次早期预警即覆盖其后全部风险起点；"
            "实测全部平凡策略召回 98.9%、平均提前最高 20651min。"
            "三条门槛实际塌缩为只剩精确率，故「判别力＝精确率−最强平凡策略精确率」"
            "是本口径下唯一有意义的成绩指标。"),
        "lead_note": (
            f"平均提前 {mt['avg_lead_min']:.0f}min 是「不设上限」的产物"
            "（预警稀疏，紧邻风险起点之前的那次预警自然很远），"
            "满足门槛但**不应按业务意义上的提前预警时间解读**。"),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n高低波：DRL 高 {mt['vol_split']['高波']['precision']:.1%} / "
          f"低 {mt['vol_split']['低波']['precision']:.1%}；"
          f"规则 高 {rmt['vol_split']['高波']['precision']:.1%} / "
          f"低 {rmt['vol_split']['低波']['precision']:.1%}")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
