"""eval_per_symbol — 在 anchor 连续测试集上逐品种评测，落盘供平台与报告引用。

为什么要有这个脚本
------------------
`data_out/per_symbol.json` 与 `per_symbol_workpoints.json` 承载着「逐品种达标状态」
这一交付结论，但它们此前是用临时代码跑出来的、**仓库里没有任何脚本能重新生成**——
全链路复现时才暴露：三份落盘只被 `scripts/build_platform.py` 读取，无人写入。
本脚本补回这条链路。

两个口径必须分清（这是本项目反复踩过的坑）
------------------------------------------
- **本脚本 = anchor 连续测试集**：59 幕、按品种切分，命中基础率 24.4%。
- 事件窗口口径见 `scripts/eval_q1_windows.py`，基础率 28.3%，门槛更容易过。
同一套系统在两个口径下能报出差很多的结论，**报数字必须标口径**。

三个配置
--------
- `def` 规则默认工作点：`symbol_params.json` 的逐品种分位阈值 + R2/R4/R6 关闭
- `rf`  保召回工作点：在上面再叠 `RECALL_FIRST`（z 阈 ×0.55，系数在验证集上选定）
- `drl` DRL 集成在验证集选出的 `lean`（默认从 `workpoint.json` 读，不写死）

用法::

    python scripts/eval_per_symbol.py --underlying
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
from drl.dataset import (Normalizer, apply_continuous_risk, load_episodes,
                         split_episodes, suppress_dead_zone)
from drl.dqn import DQNAgent
from drl.env import AlertEnv, RewardSpec
from drl.metrics import aggregate
from vol_surface.alert_rules import RECALL_FIRST

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMS = ["ag", "au", "sc", "si", "lc", "cu", "rb"]


def _rollout_rule(eps, norm, spec, params):
    out = []
    for e in eps:
        pol = RulePolicy(e, params=params)
        out.append(AlertEnv(e, norm, spec).rollout(pol))
    return out


def _dz(rollouts, on: bool):
    """结构性死区过滤。**规则与 DRL 必须同时施加**，只给一方加就不是公平对照。

    实测（scripts/dead_zone_study.py，预登记 2×2）：两方各涨约 7pp，
    判别力仅变 −0.26pp —— 属**全员抬升，不构成模型改进**，引用时必须一并说明。
    """
    if not on:
        return rollouts
    out = []
    for ro in rollouts:
        acts = suppress_dead_zone(ro["ts"], ro["actions"])
        out.append({**ro, "actions": acts, "n_alert": int((acts >= 2).sum())})
    return out


def _base_rate(eps) -> float:
    """命中基础率：直接复用 `drl.train.hit_base_rate`，**不要自己重算**。

    初版这里自己写了个「[i, i+8) 内有风险起点」的近似，结果逐品种基础率比归档值
    高出 6~9pp（ag 37.1% vs 28.7%），判别力随之全错。根因是**索引距离不等于时间距离**：
    权威口径是 `lead_min <= LEAD_WINDOW_MIN`（120 分钟），而 8 个截面在跨越
    夜盘/日盘休市时远超 120 分钟，按格数数就会把大量实际够不着的截面算成命中。
    这类「自己重写一遍已有口径」是本项目反复踩过的坑，一律改为复用本体。
    """
    from drl.train import hit_base_rate
    return hit_base_rate(eps)


# 与 scripts/eval_v2.py 保持同一套网格——逐品种「同召回」对照要在同一组工作点上找。
LEAN_GRID = [0.0, .35, .6, 1.0, 1.3, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out"))
    ap.add_argument("--underlying", action="store_true")
    ap.add_argument("--monthly-risk", action="store_true")
    ap.add_argument("--lean", type=float, default=None,
                    help="不传则从 workpoint.json 读（推荐：避免与交付工作点脱钩）")
    ap.add_argument("--no-dead-zone", action="store_true",
                    help="关闭结构性死区过滤（默认开启，见 drl.dataset.session_dead_zone）")
    a = ap.parse_args()

    sp_path = os.path.join(ROOT, "data_out", "symbol_params.json")
    symbol_params = {}
    if os.path.exists(sp_path):
        with open(sp_path, encoding="utf-8") as f:
            symbol_params = json.load(f)
    else:
        print(f"警告：缺 {sp_path}，规则将退回全局 DEFAULT_PARAMS")

    lean = a.lean
    if lean is None:
        wp = os.path.join(a.drl_dir, "workpoint.json")
        if not os.path.exists(wp):
            raise SystemExit(f"缺 {wp}，请先跑 scripts/select_workpoint.py，"
                             f"或用 --lean 显式指定")
        with open(wp, encoding="utf-8") as f:
            lean = float(json.load(f)["lean"])
    print(f"DRL 工作点 lean={lean:.2f}（来源：{'命令行' if a.lean else 'workpoint.json'}）")

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    if not a.monthly_risk:
        apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()
    print(f"测试集 {len(te)} 幕")

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
    api.set_confidence_threshold(lean)

    per_symbol, workpoints = {}, {}
    for sym in SYMS:
        grp = [e for e in te if e.symbol == sym]
        if not grp:
            print(f"  {sym}: 测试集无该品种，跳过")
            continue
        base_p = symbol_params.get(sym, {})

        dz_on = not a.no_dead_zone
        ro_def = _dz(_rollout_rule(grp, norm, spec, base_p or None), dz_on)
        ro_rf = _dz(_rollout_rule(grp, norm, spec, {**base_p, **RECALL_FIRST}), dz_on)
        ro_drl = _dz([AlertEnv(e, norm, spec).rollout(api.policy()) for e in grp], dz_on)

        m_def = aggregate(ro_def)["micro"]
        m_rf = aggregate(ro_rf)["micro"]
        m_drl = aggregate(ro_drl)["micro"]
        base = _base_rate(grp)

        # 逐品种「同召回」对照。
        # 缺陷史：本文件此前只落 `drl_R` 而**不落 `rule_R`**，于是 per_symbol.json 里
        # rule_P 与 drl_P 被并排展示、被当成可比。实际两者工作点完全不同——
        # lc 是规则 P=45.9%@R=34.4% 对 DRL P=38.2%@R=87.2%，
        # 拿低召回的高精确率去打高召回的低精确率，任何模型都会「输」。
        # 据此曾得出「lc/cu 两品种 DRL 不如规则」的**反向结论**并写进了交付文档。
        # 现在做两件事：① 把 rule_R / rule_lead 一并落盘（根因）；
        # ② 扫 lean 找到与规则同召回的那个 DRL 工作点，给出真正可比的精确率差。
        def _drl_at_recall(target_r):
            """返回 DRL 在召回最接近 target_r 处的 (lean, P, R)。"""
            best = None
            for lv in LEAN_GRID:
                api.set_confidence_threshold(lv)
                mm = aggregate(_dz([AlertEnv(e, norm, spec).rollout(api.policy())
                                    for e in grp], dz_on))["micro"]
                d = abs(mm["recall"] - target_r)
                if best is None or d < best[0]:
                    best = (d, lv, mm["precision"], mm["recall"])
            api.set_confidence_threshold(lean)      # 复位到交付工作点
            return {"lean": best[1], "P": best[2], "R": best[3], "dR": best[0]}

        matched = {k: _drl_at_recall(v["recall"])
                   for k, v in (("vs_def", m_def), ("vs_rf", m_rf))}

        workpoints[sym] = {
            k: {"P": v["precision"], "R": v["recall"], "lead": v["avg_lead_min"],
                "pass3": bool(v["precision"] >= .5 and v["recall"] >= .6
                              and v["avg_lead_min"] >= 30)}
            for k, v in (("def", m_def), ("rf", m_rf), ("drl", m_drl))}
        per_symbol[sym] = {
            "n_ep": len(grp), "base": base,
            "rule_P": m_def["precision"], "rule_DP": m_def["precision"] - base,
            "rule_R": m_def["recall"], "rule_lead": m_def["avg_lead_min"],
            "rule_n": m_def["n_alert"],
            "drl_P": m_drl["precision"], "drl_DP": m_drl["precision"] - base,
            "drl_R": m_drl["recall"], "drl_lead": m_drl["avg_lead_min"],
            "drl_n": m_drl["n_alert"],
            # 唯一可以直接相减的一组：DRL 在与规则同召回处的精确率优势
            "matched": matched,
            "matched_gain_vs_def_pp": (matched["vs_def"]["P"] - m_def["precision"]) * 100,
            "matched_gain_vs_rf_pp": (matched["vs_rf"]["P"] - m_rf["precision"]) * 100,
            "pass3": bool(m_drl["precision"] >= .5 and m_drl["recall"] >= .6
                          and m_drl["avg_lead_min"] >= 30),
        }
        print(f"  {sym}: 基础率 {base:.1%} | 规则 P={m_def['precision']:.1%}"
              f"@R={m_def['recall']:.1%} | DRL P={m_drl['precision']:.1%}"
              f"@R={m_drl['recall']:.1%} | **同召回** DRL {matched['vs_def']['P']:.1%}"
              f"@R={matched['vs_def']['R']:.1%}（lean={matched['vs_def']['lean']:.2f}）"
              f" → {per_symbol[sym]['matched_gain_vs_def_pp']:+.1f}pp")

    per_symbol["_meta"] = {
        "rule_config": "symbol_params.json + R2/R4/R6 默认关闭",
        "drl_lean": lean,
        "caliber": "anchor 连续测试集（非事件窗口）——报数字务必标明口径",
        "dead_zone_filter": not a.no_dead_zone,
        "dead_zone_note": ("休市前 3 根不预警（休市阈值 255min，取观测间隔空隙 [135,375] 的中点；"
                           "本数据集为 135min，非想当然的 120min）。"
                           "规则与 DRL 同时施加。两方各涨约 7pp、判别力仅变 −0.26pp，"
                           "属全员抬升而非模型改进——见 data_out/dead_zone_study.json"),
    }
    # psw 也必须带 _meta。归档版没有，结果它的 def 档用的是**全局 DEFAULT_PARAMS**，
    # 而同时展示的 per_symbol.rule_P 用的是**逐品种标定阈值**——两份文件对同名的
    # 「默认工作点」用了两套配置（7/7 品种可精确区分，lc 差 9.4pp），
    # 在平台上并排展示却无从分辨。现统一为逐品种标定并写明。
    workpoints["_meta"] = dict(per_symbol["_meta"],
                               configs={"def": "symbol_params.json 逐品种分位阈值",
                                        "rf": "def + RECALL_FIRST（z 阈 ×0.55）",
                                        "drl": f"DRL 集成 @ lean={lean}"})
    for name, obj in (("per_symbol.json", per_symbol),
                      ("per_symbol_workpoints.json", workpoints)):
        p = os.path.join(a.out, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=float)
        print(f"→ {p}")

    n_pass = sum(1 for s in SYMS if per_symbol.get(s, {}).get("pass3"))
    n_sym = len([s for s in SYMS if s in per_symbol])
    print(f"\n三项全达标（精确率≥50% 且 召回≥60% 且 提前≥30min）：")
    for cfg, label in (("def", "规则默认"), ("rf", "规则保召回"), ("drl", "DRL")):
        k = sum(1 for s in SYMS if workpoints.get(s, {}).get(cfg, {}).get("pass3"))
        print(f"  {label:10s} {k}/{n_sym}")
    print(f"（per_symbol.pass3 口径下 DRL {n_pass}/{n_sym}）")


if __name__ == "__main__":
    main()
