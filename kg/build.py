"""kg.build — 由标定结果装配知识图谱。

装配原则
--------
1. **只有通过检验的边才进入推理图**。`strong`（支撑度足够 + 置换检验显著 + lift 置信
   下界 > 1.1）的 ANOM→CONSEQ 边才被加入；未通过的候选边全部保留在
   `kg_edges.json` 里供审计，但不参与推理，也不会出现在解释文本中。

2. **机制是边的注解，不是边本身**。若把 ANOM→MECH→CONSEQ 拆成两条边，就得给
   ANOM→MECH 和 MECH→CONSEQ 各编一个权重——那正是 v1 的毛病。这里把机制作为
   已实测的 ANOM→CONSEQ 边上的 `mech` / `how` 属性，路径展示为
   「异常 →(经由 机制)→ 后果」，权重始终只有一个，且来自数据。

3. **诠释层单独成边**。MECH、CONSTRAINT 与 ANOM/CONSEQ 之间的连线 kind 分别为
   EXPLAINS / VIOLATES，`layer="interpretive"`，仅供可视化与文本生成，
   推理搜索不会沿它们计算权重。
"""

from __future__ import annotations

import numpy as np

from .graph import DiGraph
from .schema import (ANOM_BY_ID, ANOMS, CONSEQ_BY_ID, CONSEQS, CONSTRAINT_MAP,
                     CONSTRAINTS, HORIZON, MECH_BY_ID, MECH_MAP, MECHS,
                     SECTORS, SYMS)


def build(calib: dict, include_weak: bool = False) -> DiGraph:
    """把 kg.calibrate.calibrate() 的输出装配成图。"""
    g = DiGraph()

    # ---- 节点：异常（根因）
    for a in ANOMS:
        g.add_node(a.id, type="ANOM", layer="measured", title=a.name,
                   rule=a.rule, feature=a.feature, z_feature=a.z_feature,
                   desc=a.desc)
    # ---- 节点：后果（可测）
    for c in CONSEQS:
        g.add_node(c.id, type="CONSEQ", layer="measured", title=c.name,
                   feature=c.feature, detect=c.rule_text(), desc=c.desc)
    # ---- 节点：机制（诠释）
    for m in MECHS:
        g.add_node(m.id, type="MECH", layer="interpretive", title=m.name, how=m.how)
    # ---- 节点：无套利约束（诠释）
    for cid, (title, stmt) in CONSTRAINTS.items():
        g.add_node(cid, type="CONSTRAINT", layer="interpretive",
                   title=title, statement=stmt)
    # ---- 节点：品种 / 板块
    for sym, meta in SYMS.items():
        g.add_node(f"SYM-{sym}", type="SYM", layer="meta",
                   title=meta["title"], exchange=meta["exchange"])
        g.add_edge(f"SYM-{sym}", meta["sector"], kind="MEMBER_OF",
                   layer="meta", weight=1.0)
    for sid, title in SECTORS.items():
        g.add_node(sid, type="SECTOR", layer="meta", title=title)

    # ---- 边：ANOM → CONSEQ（实测层，推理主干）
    n_added = 0
    for e in calib["edges"]:
        if not (e["strong"] or include_weak):
            continue
        mech = MECH_MAP.get((e["anom"], e["conseq"]))
        attrs = {
            "kind": "AMPLIFIES", "layer": "measured",
            "weight": float(e["lift_ci_low"]),   # 用保守下界当权重，不用点估计
            "lift": float(e["lift"]),
            "lift_ci_low": float(e["lift_ci_low"]),
            "p_cond": float(e["p_cond"]), "p_conseq": float(e["p_conseq"]),
            "n_anom": int(e["n_anom"]), "p_value": float(e["p_value"]),
            "self_feature": bool(e["self_feature"]),
            "strong": bool(e["strong"]),
            "replicated": bool(e.get("replicated", False)),
            "replicated_loose": bool(e.get("replicated_loose", False)),
            "test_lift": float(e.get("test", {}).get("lift", float("nan"))),
            "test_p": float(e.get("test", {}).get("p_value", float("nan"))),
            "test_ci_low": float(e.get("test", {}).get("lift_ci_low", float("nan"))),
        }
        if mech:
            attrs["mech"] = mech
            attrs["how"] = MECH_BY_ID[mech].how
        g.add_edge(e["anom"], e["conseq"], **attrs)
        n_added += 1
        # 诠释层：机制节点与两端相连，仅供可视化
        if mech:
            g.add_edge(e["anom"], mech, kind="EXPLAINS", layer="interpretive", weight=0.0)
            g.add_edge(mech, e["conseq"], kind="EXPLAINS", layer="interpretive", weight=0.0)

    # ---- 边：ANOM → CONSTRAINT（诠释层）
    for aid, cid in CONSTRAINT_MAP.items():
        if aid in g and cid in g:
            g.add_edge(aid, cid, kind="VIOLATES", layer="interpretive", weight=0.0)

    # ---- 边：跨品种传导（实测层，可选）
    for rec in calib.get("cross_symbol", []):
        if not rec.get("strong"):
            continue
        u, v = f"XSYM-{rec['rule']}-{rec['src']}", f"XSYM-{rec['rule']}-{rec['dst']}"
        for nid, s in ((u, rec["src"]), (v, rec["dst"])):
            if nid not in g:
                a = next((x for x in ANOMS if x.rule == rec["rule"]), None)
                g.add_node(nid, type="XSYM", layer="measured", sym=s,
                           rule=rec["rule"],
                           title=f"{SYMS[s]['title']}·{a.name if a else rec['rule']}")
        g.add_edge(u, v, kind="PROPAGATES_TO", layer="measured",
                   weight=float(rec["lift_ci_low"]),
                   lift_naive=float(rec["lift"]),
                   lift=float(rec.get("lift_matched", rec["lift"])),
                   p_cond=float(rec["p_cond"]), p_base=float(rec["p_base"]),
                   p_base_matched=float(rec.get("p_base_matched", rec["p_base"])),
                   n_src=int(rec["n_src"]), p_value=float(rec["p_value"]),
                   rule=rec["rule"], src=rec["src"], dst=rec["dst"])

    g.add_node("_meta", type="META", layer="meta",
               horizon=HORIZON, n_measured_edges=n_added,
               n_candidate_edges=len(calib["edges"]),
               calib_meta=calib.get("meta", {}))
    return g


def summary(g: DiGraph) -> str:
    from collections import Counter
    nt = Counter(a.get("type") for _, a in
                 ((n, g.node(n)) for n in g.nodes()) if a.get("type") != "META")
    et = Counter(a.get("kind") for _, _, a in g.edges())
    meas = [a for _, _, a in g.edges("AMPLIFIES")]
    rep = sum(1 for a in meas if a.get("replicated"))
    selfp = sum(1 for a in meas if a.get("self_feature"))
    lines = [f"节点 {len(g) - 1}  边 {len(g.edges())}",
             f"  节点类型: {dict(nt)}",
             f"  边类型:   {dict(et)}",
             f"  实测主干边 {len(meas)} 条，其中测试期可复现 {rep} 条、"
             f"同源特征 {selfp} 条"]
    if meas:
        lines.append(f"  lift 范围 {min(a['lift'] for a in meas):.2f} ~ "
                     f"{max(a['lift'] for a in meas):.2f}")
    return "\n".join(lines)
