"""kg.reasoner — 知识图谱推理与中文解释生成。

输入:
  - g: build_graph 构出的 NetworkX DiGraph
  - ts: 触发预警的时间戳 (YYYYMMDDHHMMSS)
  - symbol: 品种代码 (ag/au/sc/si)
  - triggered: 所触发的规则 id 列表 (如 ['R1_convexity', 'R4_concentration'])
  - feats: 当前截面特征 dict(含 z 分), 用于证据链

输出:
  {
    "summary": "根因 → 传导路径 → 可能后果",
    "cause": "根因描述",
    "path": ["节点标题", ...],
    "consequence": "可能后果",
    "evidence": [ (feat, z, rule, value) ... ],
    "how_chain": ["ANOM→MECH 的 how", "MECH→CONSEQ 的 how", ...],
    "activation": {node: weight},
  }
"""

from __future__ import annotations

import argparse
import collections
import networkx as nx
import numpy as np
import pandas as pd

from .schema import ANOMS, MECHS, ORDERS, SYMS


# -------------------------- 工具 --------------------------

def _anom_rule_map():
    return {a.id: a for a in ANOMS}


def _title(node_data: dict, default: str = "") -> str:
    return node_data.get("title", default)


def _token_name(nid: str) -> str:
    # ANOM-R1_convexity → R1 凸性违反
    if nid.startswith("ANOM-"):
        suffix = nid.split("-", 1)[1]
        for a in ANOMS:
            if a.id == f"ANOM-{suffix.split('-', 1)[0]}":
                return a.name
    if nid.startswith("MECH-"):
        return {
            "MakerGamma": "做市商对冲压力",
            "VolFeedback": "波动率反馈环",
            "Liquidity": "流动性螺旋",
            "ExpiryCluster": "临到期集中移仓",
            "TermInversion": "期限结构倒挂",
            "ConvexitySpread": "凸性传播",
        }.get(nid.split("-", 1)[1], nid)
    if nid.startswith("CONSEQ-"):
        return {
            "VolSpike": "波动剧烈上扬",
            "ExtremeSkew": "偏度极端",
            "LiquidityDrop": "流动性减低",
            "ArbPersist": "无套利违例持续",
            "MarginStress": "保证金压力",
        }.get(nid.split("-", 1)[1], nid)
    if nid.startswith("ORDER-"):
        return ORDERS.get(nid, nid)
    if nid.startswith("CATALOG-"):
        return nid.split("-", 1)[1]
    if nid.startswith("RULE-"):
        return nid
    return nid


def _bfs_paths_to_conseq(g: nx.DiGraph, start: str,
                         max_depth: int = 4) -> list:
    """从 start 出发, 只沿 AMPLIFIES 边 BFS, 收集能走到 CONSEQ-*/CATALOG-* 的所有路径。
    (不走 MEMBER_OF, 避免品种节点把传导路径拉偏。)
    返回 [{"path": [node,...], "edge": [edge_data,...]}...]。
    """
    paths = []
    queue = collections.deque([(start, [start], [])])
    while queue:
        node, path, edge_data = queue.popleft()
        if len(path) > max_depth + 1:
            continue
        for _, nxt, d in g.out_edges(node, data=True):
            if d.get("kind") != "AMPLIFIES":
                continue
            if nxt in path:
                continue
            d2 = dict(d)
            d2["source"] = node
            d2["target"] = nxt
            new_path = path + [nxt]
            new_edge = edge_data + [d2]
            if nxt.startswith(("CONSEQ-", "CATALOG-")):
                paths.append({"path": new_path, "edge": new_edge})
            else:
                queue.append((nxt, new_path, new_edge))
    return paths


def _score_path(path: list, edge_data: list) -> float:
    """按平均边权来打分,遍历时有权重用权重,无权重默认 0.5."""
    ws = [d.get("weight", 0.5) for d in edge_data]
    return float(np.mean(ws)) if ws else 0.0


# -------------------------- 主解释 --------------------------

def _feats_to_evidence(rule: str, feats: dict) -> list:
    """从特征 dict 找到此规则的证据(feature, z, value)."""
    for a in ANOMS:
        if a.rule != rule:
            continue
        items = []
        if a.feature and a.feature in feats:
            v = feats[a.feature]
            z = feats.get(a.z_feature, np.nan) if a.z_feature else np.nan
            items.append((a.feature, float(z) if np.isfinite(z) else np.nan,
                          rule, float(v) if np.isfinite(v) else v))
        return items
    return []


def _rule_to_anom(rule: str) -> str | None:
    """规则 id → ANOM 节点 id(如 R4_concentration → ANOM-R4_conc)。"""
    for a in ANOMS:
        if a.rule == rule:
            return a.id
    return None


def explain_alert(g: nx.DiGraph, ts: str, symbol: str,
                  triggered: list, feats: dict) -> dict:
    causes, path_nodes, consequences, hows, evidence = [], [], [], [], []

    # 激活集合(用于 feature 提取及可视化)
    activation = collections.Counter()

    # 1. 根因: 触发规则对应的 ANOM 从激活集开始
    for rule in triggered:
        anom_id = _rule_to_anom(rule)
        if anom_id is None:
            continue
        # 优先取品种级节点(能走机制链), 没有则取通用节点
        candidate = f"{anom_id}-{symbol}"
        if candidate not in g:
            candidate = anom_id
        if candidate not in g:
            continue

        activation[candidate] += 1.0
        causes.append(f"规则 {rule}: {g.nodes[candidate].get('title', candidate)}")
        evidence.extend(_feats_to_evidence(rule, feats))

        # 找所有到 CONSEQ/CATALOG 的路径, 选评分最高的
        paths = _bfs_paths_to_conseq(g, candidate, max_depth=4)
        if paths:
            best = max(paths, key=lambda p: _score_path(p["path"], p["edge"]))
            # 只保留 AMPLIFIES 链(跳过 MEMBER_OF 到 SYM 的)
            amp_edges = [e for e in best["edge"] if e["kind"] == "AMPLIFIES"]
            if amp_edges:
                for e in amp_edges:
                    activation[e["target"]] += e.get("weight", 0.5)
                    src_name = _token_name(e["source"])
                    tgt_name = _token_name(e["target"])
                    how = e.get("how", "")
                    hows.append(f"{src_name}→{tgt_name}: {how}")
                consequences.append(_token_name(amp_edges[-1]["target"]))
            # 完整路径节点
            path_nodes.extend([_token_name(n) for n in best["path"]])

    # 2. 跨品种传导(CORRELATES): 查同板块/事件
    cross = []
    for rule in triggered:
        aid = f"ANOM-{rule}"
        for n in (f"{aid}-{symbol}", aid):
            if n not in g:
                continue
            for _, v, d in g.out_edges(n, data=True):
                if d.get("kind") != "CORRELATES":
                    continue
                tgt = d.get("to_sym", "?")
                w = d.get("weight", 0)
                if tgt != symbol:
                    cross.append(f"{_token_name(n)}→{_token_name(v)} (传导率 {w:.2f})")
    if cross:
        hows.append(" | ".join(cross))

    # 3. 汇总
    cause_txt = "; ".join(causes)
    uniq_path = list(dict.fromkeys(path_nodes))
    conseq_txt = "; ".join(dict.fromkeys(consequences))
    summary = f"{cause_txt} ⇒ {conseq_txt}"

    return {
        "summary": summary,
        "cause": cause_txt,
        "path": uniq_path,
        "consequence": conseq_txt,
        "how_chain": hows,
        "evidence": evidence,
        "activation": dict(activation),
    }


# -------------------------- smoke test --------------------------

def smoke_test():
    from .build_graph import load_graph
    g = load_graph("data_out/kg_graph.pkl")
    print(f"图: {len(g)} 节点")

    # 拿 2026-01-14 的 anchor(预警日)看解释
    import os, glob
    d = glob.glob("data_out/anchor/ag/2026-01/features.parquet")
    if not d:
        print("❌ 未找到 anchor 数据，跳 smoke test")
        return
    df = pd.read_parquet(d[0])
    # 找一个 lv>=2 的截面作样例
    lv = df["convexity_violation"] if "convexity_violation" in df else None
    if lv is None:
        print("❌ anchor 数据无 convexity_violation")
        return

    # 手动指定一个 ese 触发规则组合,检查解释是否会生成
    test_feats = {"gamma_concentration": 0.12, "iv_spike": 0.09,
                   "convexity_violation": 3.5, "convexity_violation_z": 4.0,
                   "term_slope": 0.08}
    triggered = ["R4_concentration", "R7_iv_spike"]
    info = explain_alert(g, ts="20260114103000", symbol="ag",
                         triggered=triggered, feats=test_feats)
    print("\n=== 中文解释 ===")
    print("summary:", info["summary"])
    print("cause:", info["cause"])
    print("path:", info["path"])
    print("consequence:", info["consequence"])
    print("how:", info["how_chain"])
    print("evidence:", info["evidence"])
    print("activation:", info["activation"])


if __name__ == "__main__":
    smoke_test()
