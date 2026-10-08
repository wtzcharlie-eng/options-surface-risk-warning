"""kg.reason — 传导路径推理与证据链。

输入一个截面的特征，输出「根因 → 传导环节 → 可能后果」的结构化推理结果，
每一步都带可回溯的证据。

推理只走**实测层**（AMPLIFIES / PROPAGATES_TO）。诠释层（MECH / CONSTRAINT）
只作为注解挂在路径上，不参与打分——避免用编出来的权重推出结论。

拒答原则
--------
若某个异常在图里没有任何通过检验的出边，就**如实说「未观测到稳定的下游关联」**，
而不是退化到默认路径硬凑一条解释。宁可少说，不可编造。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vol_surface.alert_rules import evaluate_rules

# ---------------------------------------------------------------- 异常检测口径
# 知识图谱**显式启用全部 8 类异常**，包括在问题1 里已被默认关闭的 R2/R4/R6。
#
# 为什么不跟随规则引擎的开关
# --------------------------
# 两者目标不同：
#   - 规则引擎（问题1）的任务是**预测**未来 120min 内的风险区间。R2/R4/R6 在这个
#     任务上判别力为负或被支配，故默认关闭（README §7.13–§7.15）。
#   - 知识图谱（问题2）的任务是**描述**「当前发生了什么异常、历史上它之后常发生什么」。
#     这是条件概率统计，不要求该异常本身是好的预警触发器。
#
# 这三类异常在训练期都有通过检验的边，删掉会让图谱丢掉 8 类异常里的 3 类：
#   ANOM-R6_liq  → CONSEQ-LiquidityDrop  lift 2.80（训练 p=0.005；测试期严格复现）
#   ANOM-R4_conc → CONSEQ-ArbPersist     lift 1.95（训练 p=0.005；测试期严格复现）
#   ANOM-R2_term → CONSEQ-ConvexityBreak lift 1.37（训练 p=0.030；**测试期 p=0.154
#                  不显著，仅满足宽松判据**——这条的样本外证据偏弱，文本已就地标注）
#
# **写成显式常量而不是依赖默认值**：否则问题1 那边一改开关，图谱会在无人察觉的情况下
# 少掉三类节点，而所有落盘的标定结果仍是旧的——这正是本项目反复踩过的那类静默失配。
KG_RULE_PARAMS = {"r2_enabled": True, "r4_enabled": True, "r6_enabled": True}


from .graph import DiGraph
from .schema import (ANOM_BY_ID, ANOM_BY_RULE, CONSEQ_BY_ID, CONSTRAINT_FEATURE,
                     CONSTRAINT_MAP, no_mech_reason, weak_lift_threshold,
                     CONSTRAINTS, HORIZON, MECH_BY_ID, SYMS, TRIGGER_EXPR)


@dataclass
class Evidence:
    """一条可回溯的证据。"""
    kind: str            # observation | statistic
    text: str
    detail: dict = field(default_factory=dict)


@dataclass
class Path:
    anom: str
    conseq: str
    mech: str | None
    weight: float        # lift 的 95% 保守下界
    lift: float
    p_cond: float
    p_conseq: float
    n_anom: int
    p_value: float
    self_feature: bool
    replicated: bool
    test_lift: float = float("nan")     # 测试期 lift，用于披露样本外衰减
    test_p: float = float("nan")        # 测试期 p 值：replicated 判据**不含**它，须单独披露
    test_ci_low: float = float("nan")   # 测试期 lift 下界
    replicated_loose: bool = False      # 仅满足宽松判据（下界>1.0）但未达严格判据
    # 无机制时的**结构性原因**（弱效应 / 跨量族）。有机制则为空串。
    # 见 schema.no_mech_reason：**不编造机制，而是解释缺失**。
    no_mech_why: str = ""


def _z_threshold_text(rule: str, level: int, zval) -> str:
    """给 R1/R5 这类**双重条件**规则补出「滚动 z 分」那一侧的实际阈值与读数。

    R1/R5 的判定是 `原值 > 下限 且 滚动z > z阈`——早前解释文本只写
    「并同时要求其滚动 z 分超阈」，不给阈值也不给读数，读者无从验证第二个合取条件
    （评审实测 14 处）。

    **阈值从 `DEFAULT_PARAMS` 动态读，不写死**：否则问题1 那边一改参数，
    这段文案就变成假陈述——本项目已在别处栽过同一类跟头。

    等级与阈值的对应遵循 `alert_rules.evaluate_rules` 的分支：
    level 3 → ser_z，level 2 → warn_z，level 1 → watch_z。
    """
    from vol_surface.alert_rules import DEFAULT_PARAMS
    pre = {"R1_convexity": "r1", "R5_arb": "r5"}.get(rule)
    if not pre:
        return ""
    key = {3: f"{pre}_ser_z", 2: f"{pre}_warn_z"}.get(int(level), f"{pre}_watch_z")
    z = DEFAULT_PARAMS.get(key)
    if z is None:
        return ""
    txt = f"，本级要求 z > {float(z):.1f}"
    if zval is not None:
        try:
            txt += f"，实测 z = {float(zval):.2f}"
        except (TypeError, ValueError):
            pass
    return txt


def active_anomalies(feats: dict, min_level: int = 1) -> list:
    """当前截面上被触发的异常（复用问题1 的规则引擎本体）。"""
    out = []
    for t in evaluate_rules(feats, params=KG_RULE_PARAMS)["triggers"]:
        if t["level"] < min_level:
            continue
        a = ANOM_BY_RULE.get(t["rule"])
        if a:
            out.append({"anom": a.id, "rule": t["rule"], "level": int(t["level"]),
                        "reason": t["reason"], "value": float(t["value"]),
                        "threshold": float(t["threshold"])})
    # 等级高的排前面，同级按规则名稳定排序，保证输出确定性
    out.sort(key=lambda d: (-d["level"], d["rule"]))
    return out


def reason(g: DiGraph, feats: dict, symbol: str | None = None,
           timestamp: str | None = None, min_level: int = 1,
           top_paths: int = 3) -> dict:
    """对单截面做传导路径推理，返回结构化结果（含证据链）。"""
    acts = active_anomalies(feats, min_level=min_level)
    results, unresolved = [], []
    # 「弱」的判据从**当前图**现算（有机制边的最小 lift），不写死——
    # 否则重标定后这条判据会变成错的。
    _weak_thr = weak_lift_threshold(g.edges("AMPLIFIES"))

    for act in acts:
        aid = act["anom"]
        spec = ANOM_BY_ID[aid]

        # ---- 根因证据：当前实测读数
        ev = [Evidence("observation",
                       f"{spec.name}：{act['reason']}",
                       {"rule": act["rule"], "level": act["level"],
                        "value": act["value"], "threshold": act["threshold"]})]
        # 触发量本身。`spec.feature` 只是该异常的代表性读数，与规则实际比较的标量
        # 往往不是一回事（R3 比的是 z 分、R4/R8 比的是两特征的较大者、R6 是下侧触发）。
        # 早前笼统写成「取多个特征中的较大者」，产生了「触发量 = −5.33（较大者）」
        # 这类自相矛盾的表述。此处改为按 TRIGGER_EXPR 逐规则给出准确口径。
        tv = float(act["value"])
        expr, note, op, comps, use_abs = TRIGGER_EXPR.get(
            act["rule"], (spec.feature, "", ">", (spec.feature,), False))
        # 缺陷1：max() 型规则早前只打印**未触发**的那个分量当「参考读数」，
        #        出现过「参考读数 atm_iv_vel_z = −0.0255」挂在「加速度异常」根因下、
        #        以及参考读数低于阈值的自相矛盾（评审实测 5 条）。改为打印胜出分量。
        # 缺陷2：R3/R2 比较的是绝对值，早前印成「atm_iv_z = −5.3294，判定为 > 3.5000」
        #        字面为假。改为直接印比较式左端的实际标量。
        win, win_v = None, None
        for c in comps:
            if c in feats:
                v = abs(float(feats[c])) if use_abs else float(feats[c])
                if win_v is None or (v > win_v if op == ">" else v < win_v):
                    win, win_v = c, v
        thr = float(act["threshold"])
        # 绝对值型规则（R2/R3）比较的是 |x|，而 act["value"] 是带符号原值。
        # 直接印原值会得到「= −5.3294，判定为 > 3.5000」这种字面为假的不等式。
        shown = abs(tv) if use_abs else tv
        head = f"触发量 `{expr}` = {shown:.4f}，判定为 {op} {thr:.4f}"
        if len(comps) > 1 and win is not None:
            head += f"；本截面由 `{win}` = {float(feats[win]):.4f} 触发"
        elif use_abs:
            head += f"（原始值 = {tv:+.4f}，取绝对值后参与比较）"
        if note:
            # R1/R5 是**双重条件**（原值超阈 且 滚动 z 超阈），早前只印「并同时要求其
            # 滚动 z 分超阈」而不给阈值，读者无法验证第二个合取条件（评审实测 14 处）。
            # 阈值**从 DEFAULT_PARAMS 动态取，不写死**——否则改了参数文案就成假的。
            n2 = note
            if "滚动 z" in note:
                z = _z_threshold_text(act["rule"], act["level"],
                                      feats.get(spec.z_feature))
                if z:
                    n2 = note + z
            head += f"（{n2}）"
        ev.append(Evidence("observation", head,
                           {"trigger_expr": expr, "trigger_value": shown,
                            "raw_value": tv,
                            "threshold": thr, "op": op,
                            "winning_component": win,
                            # 胜出分量的读数也要进 detail，否则 K2（数字可回溯门）
                            # 会把它判为"编造的数字"——门抓得对，是 detail 漏了
                            "winning_value": (float(feats[win]) if win in feats else None)}))
        # 只在「代表性读数既非触发量、也不是胜出分量」时才作为补充信息打印
        if (spec.feature and spec.feature in feats
                and spec.feature != expr and spec.feature != win):
            ev.append(Evidence("observation",
                               f"参考读数 `{spec.feature}` = {float(feats[spec.feature]):.4f}"
                               f"（非触发量，仅供了解当前水平）",
                               {"feature": spec.feature,
                                "value": float(feats[spec.feature]),
                                "is_reference_only": True}))
        if (spec.z_feature and spec.z_feature in feats
                and spec.z_feature != expr and spec.z_feature != win):
            ev.append(Evidence("observation",
                               f"其滚动 z 分 `{spec.z_feature}` = {float(feats[spec.z_feature]):.2f}",
                               {"feature": spec.z_feature,
                                "value": float(feats[spec.z_feature])}))

        # ---- 下游路径：只走实测层
        paths = []
        for v, a in g.successors(aid, kind="AMPLIFIES"):
            paths.append(Path(aid, v, a.get("mech"), float(a["weight"]),
                              float(a["lift"]), float(a["p_cond"]),
                              float(a["p_conseq"]), int(a["n_anom"]),
                              float(a["p_value"]), bool(a.get("self_feature")),
                              bool(a.get("replicated")),
                              float(a.get("test_lift", float("nan"))),
                              float(a.get("test_p", float("nan"))),
                              float(a.get("test_ci_low", float("nan"))),
                              bool(a.get("replicated_loose", False)),
                              ("" if a.get("mech") else
                               no_mech_reason(aid, v, float(a["lift"]), _weak_thr))))
        paths.sort(key=lambda p: (-p.weight, p.conseq))
        # 只印前 top_paths 条。**这个截断此前是静默的**，而被截掉的**永远是提升度
        # 最低的那几条**（实测 R1/R3/R8 各有 4 条边、各截掉 lift 1.28/1.27/1.24），
        # 方向上一律让材料显得更强——第 10 位评审据此指出 30 条样本静默隐去 15%。
        # 「按强度排序后取前 N」本身合理，但**不说就等于选择性呈现**。
        # 故把被截掉的条数与其提升度上限记下来，由解释文本如实交代。
        n_all = len(paths)
        dropped = paths[top_paths:]
        paths = paths[:top_paths]
        # `min_shown_lift` 供解释文本**现算比较**用：排序键是 `weight`(=lift 下界)，
        # 而印给读者看的是 `lift`，两者口径不同——「被略去的都更弱」原理上可以反转。
        # 跨品种侧已因此改过一次，主干侧漏了（第 12 位评审指出，属侥幸成立）。
        trunc = ({"n_hidden": len(dropped),
                  "max_hidden_lift": max(p.lift for p in dropped),
                  "min_shown_lift": min(p.lift for p in paths)}
                 if dropped and paths else None)

        if not paths:
            unresolved.append({"anom": aid, "name": spec.name,
                               "note": "图谱中没有该异常通过检验的下游关联"})

        # ---- 约束注解（诠释层）
        #
        # `CONSTRAINT_MAP` 是**规则→约束的静态映射**，不含任何本截面校验。
        # 第 9 位评审指出：R5_arb→买卖权平价 与 R2_term→日历价差 都**没有逻辑蕴含**
        # （`arb_score` 是 calendar/butterfly/parity 三项复合，综合分越阈不蕴含
        # parity 分量非零；`term_slope_roc` 是斜率一阶差分，与 w(k,T) 沿 T 单调性无关），
        # 而文本却以事实语气陈述「该形态同时违反了 X」——且这是全文**唯一**
        # 不挂「业务先验」标签的诠释层断言。
        #
        # 实测 400 条含异常截面：169 次标注里 168 次该类违反数确实 >0（99.4%），
        # 但仍有 1 次为 0，即字面为假。**「几乎总是真」不等于「已核验」。**
        # 特征里本就有逐截面的分项违反数，故改为**就地校验并印出计数**：
        # 校验不过就不标注。这样断言由构造保证为真，且顺带多给了读者一个数。
        cst = CONSTRAINT_MAP.get(aid)
        constraint = None
        if cst and cst in CONSTRAINTS:
            n_viol = feats.get(CONSTRAINT_FEATURE.get(cst, ""), float("nan"))
            if n_viol == n_viol and n_viol > 0:
                title, stmt = CONSTRAINTS[cst]
                constraint = {"id": cst, "title": title, "statement": stmt,
                              "n_violation": float(n_viol)}

        results.append({"anom": aid, "name": spec.name, "desc": spec.desc,
                        "trigger": act, "evidence": [e.__dict__ for e in ev],
                        "paths": [p.__dict__ for p in paths],
                        "n_paths_all": n_all, "truncated": trunc,
                        "constraint": constraint})

    # ---- 跨品种同步共现（实测层，仅当给定 symbol 时）
    cross = []
    if symbol:
        for act in acts:
            u = f"XSYM-{act['rule']}-{symbol}"
            if u not in g:
                continue
            for v, a in g.successors(u, kind="PROPAGATES_TO"):
                cross.append({"rule": act["rule"], "src": symbol, "dst": a["dst"],
                              "dst_title": SYMS.get(a["dst"], {}).get("title", a["dst"]),
                              "lift": float(a["lift"]),
                              "lift_naive": float(a.get("lift_naive", a["lift"])),
                              "p_cond": float(a["p_cond"]),
                              "n_src": int(a["n_src"]), "p_value": float(a["p_value"])})
        cross.sort(key=lambda d: (-d["lift"], d["dst"]))

    return {
        "timestamp": timestamp, "symbol": symbol,
        "symbol_title": SYMS.get(symbol, {}).get("title", symbol) if symbol else None,
        "horizon": HORIZON,
        "n_active": len(acts),
        "anomalies": results,
        "cross_symbol": cross,
        "unresolved": unresolved,
    }


def rank_root_causes(res: dict) -> list:
    """按「触发等级 × 最强下游权重」给根因排序，用于解释文本的主次安排。"""
    out = []
    for r in res["anomalies"]:
        w = max((p["weight"] for p in r["paths"]), default=0.0)
        out.append((r["trigger"]["level"] * 10 + w, r))
    out.sort(key=lambda t: -t[0])
    return [r for _, r in out]
