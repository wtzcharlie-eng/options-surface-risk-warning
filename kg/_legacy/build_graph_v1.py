"""kg.build_graph — 构建知识图谱的 NetworkX 实例。

静态层：从 schema.static_edges() 复制机制图/板块归属/约束链。
动态层：用 data_out/anchor 的截面级特征，量化 CORRELATES 边权。

输出：data_out/kg_graph.pkl（包含 NetworkX DiGraph 与节点/边属性）

用法：python3 -m kg.build_graph          # 构建并落盘
      python3 -m kg.build_graph --show   # 打印边分布统计
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from vol_surface.alert_rules import evaluate_rules

from .schema import (ANOMS, Edge, MECHS, ORDERS, SYMS,
                     static_edges)


def _ts_to_min(ts_str: str) -> int:
    """把 YYYYMMDDHHMMSS 字符串转为分钟整数(自 epoch 起)。"""
    t = pd.Timestamp(year=int(ts_str[:4]), month=int(ts_str[4:6]),
                     day=int(ts_str[6:8]), hour=int(ts_str[8:10]),
                     minute=int(ts_str[10:12]))
    return int(t.timestamp() // 60)


def _coalert_rate(anchor_root: str = "data_out/anchor",
                  lookback_min: int = 120,
                  limit_months: int | None = None) -> dict:
    """实测品种对的 (rule → 同 rule 跨品种) 传导率。

    流程:
      1) 对每个 (symbol, rule) 在 anchor 数据集里收集 lv>=2 且该 rule 在 triggers
         中的时间戳集合 (去重, 精确到分钟)
      2) 对每个有序品种对 (s1,s2), 计算 P(s2 在 2h 内也触发 | s1 触发) =
         hits / len(s1_ts), hits = #{t in s1_ts : s2 在 [t, t+lookback] 内有 ts}
    返回 dict[(rule, s1, s2)] = rate
    """
    import glob

    rules_a = {a.rule: a.id for a in ANOMS}
    # 预聚合每个品种、每个 rule 的触发时间戳(分钟整)
    sym_rule_alerts = defaultdict(lambda: defaultdict(set))

    paths = sorted(glob.glob(os.path.join(anchor_root, "*/*/features.parquet")))
    if limit_months is not None:
        paths = paths[:max(limit_months, 1)]
    print(f"[coalert] 扫描 {len(paths)} 月份 parquet (limit={
        'All' if limit_months is None else limit_months})...")
    for path in paths:
        sym = os.path.normpath(path).split(os.sep)[-3]
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty or "_timestamp" not in df.columns:
            continue

        ts_list = df["_timestamp"].astype(str).tolist()
        feats_list = df.to_dict("records")
        for ts, fdict in zip(ts_list, feats_list):
            # 只保留数值型有限值的特征（anchor 含 y、_timestamp、NaN 等非特征列）
            clean_feats = {k: float(v) for k, v in fdict.items()
                           if isinstance(v, (int, float, np.integer, np.floating))
                           and np.isfinite(v)}
            try:
                res = evaluate_rules(clean_feats)
            except Exception:
                continue
            if res["level"] < 2:
                continue
            tmin = _ts_to_min(ts)
            for trig in res["triggers"]:
                if trig["level"] >= 2:
                    sym_rule_alerts[sym][trig["rule"]].add(tmin)

    rate = {}
    for rule in rules_a:
        for s1 in list(sym_rule_alerts):
            for s2 in list(sym_rule_alerts):
                if s1 == s2:
                    continue
                src = sorted(sym_rule_alerts[s1][rule])
                tgt = sorted(sym_rule_alerts[s2][rule])
                if not src or not tgt:
                    continue
                tgt_a = np.array(tgt)
                hits = 0
                for t in src:
                    i = np.searchsorted(tgt_a, t)
                    if i < len(tgt_a) and tgt_a[i] <= t + lookback_min:
                        hits += 1
                rate[(rule, s1, s2)] = hits / len(src)
    print(f"[coalert] 得 {len(rate)} 条 (rule,sym,sym) 组合")
    return rate


def build_graph_dynamic(quick: bool = False,
                        coalert: dict | None = None) -> nx.DiGraph:
    """构建 NetworkX 图,包含静态 schema 边与动态 CORRELATES 边。

    quick=True: 不读 anchor 数据,只落 schema(用于unittest/dev).
    coalert: 外部预计算的 dict[(rule,s1,s2)] -> rate, 避免重复跑 90s 的 coalert。
    """
    g = nx.DiGraph()
    # 静态边
    for e in static_edges():
        g.add_edge(e.u, e.v, weight=e.weight, kind=e.kind, **e.attrs)

    # 添加核心节点的类型属性(对 schema 定义但静态边未连通的节点也补上)
    def _ensure_node(nid: str, **attrs):
        if nid not in g:
            g.add_node(nid, **attrs)
        else:
            g.nodes[nid].update(**attrs)

    for a in ANOMS:
        _ensure_node(a.id, type="ANOM", title=a.name)
        for sym in SYMS:
            _ensure_node(f"{a.id}-{sym}", type="ANOM", title=a.name, sym=sym)
    for m in MECHS:
        _ensure_node(m, type="MECH", title=m)
    for sym, meta in SYMS.items():
        _ensure_node(f"SYM-{sym}", type="SYM", **meta)
    for o, t in ORDERS.items():
        _ensure_node(o, type="ORDER", title=t)
    for greek in ("gamma_concentration", "vega_concentration", "atm_iv",
                  "atm_iv_vel", "convexity_vel", "skew_val",
                  "convexity_violation", "term_slope"):
        _ensure_node(f"GREEK-{greek}", type="GREEK", title=greek)
    for rule_map in (("R1_convexity", "R1_convexity"),
                     ("R2_term_slope_roc", "R2_term_slope_roc"),
                     ("R3_atm_iv_z", "R3_atm_iv_z"),
                     ("R4_concentration", "R4_concentration"),
                     ("R5_arb", "R5_arb"),
                     ("R6_liquidity", "R6_liquidity"),
                     ("R7_iv_spike", "R7_iv_spike"),
                     ("R8_acceleration", "R8_acceleration")):
        _ensure_node(f"RULE-{rule_map[0]}", type="RULE", title=rule_map[0])
    for c in ("Calendar", "Butterfly", "Parity", "TermStruct"):
        _ensure_node(f"CONSTRAINS-{c}", type="CONSTRAINS", title=c)
    for cid, t in {"MetalsSurge": "贵金属冲高风险组合",
                    "OilShock": "原油/能源冲击风险组合",
                    "BulkReversal": "大宗行情逆转风险组合"}.items():
        _ensure_node(f"CATALOG-{cid}", type="CATALOG", title=t)
    for c, t in {"VolSpike": "月末波动剧烈上扬",
                  "ExtremeSkew": "偏度极端(单向看多/看空)",
                  "LiquidityDrop": "买卖盘减低深度骤减",
                  "ArbPersist": "无套利违例持续对做市价可争性显著造成偏移",
                  "MarginStress": "保证金压力+临近交割压仓"}.items():
        _ensure_node(f"CONSEQ-{c}", type="CONSEQ", title=t)

    if not quick:
        order_of = {sym: meta["order"] for sym, meta in SYMS.items()}
        syms = list(SYMS.keys())
        # 用外部传入或本地实测（约 90 s）；失败退化为板块近似
        rate = coalert
        if rate is None:
            try:
                rate = _coalert_rate()
            except Exception:
                rate = {}
        for x in ANOMS:
            for i, s1 in enumerate(syms):
                for j, s2 in enumerate(syms):
                    if i == j:
                        continue
                    r = rate.get((x.rule, s1, s2))
                    if r is not None:
                        w = float(r)
                        src_kind = "anchor_coalert"
                    else:
                        w = 1.0 if order_of[s1] == order_of[s2] else 0.2
                        src_kind = "order_fallback"
                    g.add_edge(f"{x.id}-{s1}", f"{x.id}-{s2}",
                               weight=w, kind="CORRELATES",
                               rule=x.rule, from_sym=s1, to_sym=s2,
                               src=src_kind)
    return g


def save_graph(g: nx.DiGraph, path: str = "data_out/kg_graph.pkl") -> str:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pickle.dump(g, open(path, "wb"))
    return path


def load_graph(path: str = "data_out/kg_graph.pkl") -> nx.DiGraph:
    return pickle.load(open(path, "rb"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="跳过 anchor 动态权重")
    ap.add_argument("--show", action="store_true", help="输出边分布统计")
    ap.add_argument("--coalert-json", default=None,
                    help="把 (rule,sym1,sym2)→rate 保存为 json 以便复现/审查")
    ap.add_argument("--smoke-months", type=int, default=None,
                    help="冒烟: 只跑此数量的 anchor 月份计算 coalert")
    ap.add_argument("--out", default="data_out/kg_graph.pkl")
    args = ap.parse_args()

    coalerts = None
    if not args.quick:
        coalerts = _coalert_rate(limit_months=args.smoke_months)
        if args.coalert_json:
            import json
            ks = [f"{r}|{s1}|{s2}" for (r, s1, s2) in coalerts.keys()]
            json.dump(dict(zip(ks, coalerts.values())),
                      open(args.coalert_json, "w"),
                      indent=2, ensure_ascii=False)
            print(f"coalert json 已写 {args.coalert_json}")

    g = build_graph_dynamic(quick=args.quick, coalert=coalerts)
    n_e = g.number_of_edges(); n_n = g.number_of_nodes()
    print(f"图: {n_n} 节点, {n_e} 边 (quick={args.quick})")
    if args.show:
        from collections import Counter
        kinds = Counter(d.get("kind") for _, _, d in g.edges(data=True))
        print("  边类型统计:", dict(kinds))
        print("  节点类型:", Counter(d.get("type") for _, d in g.nodes(data=True)))
        for k, v in list(kinds.items())[:6]:
            print(f"  {k}: {v} 边")
    path = save_graph(g, args.out)
    print(f"已写 {path}")


if __name__ == "__main__":
    main()
