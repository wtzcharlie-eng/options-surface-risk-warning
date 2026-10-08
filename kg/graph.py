"""kg.graph — 极简有向图核（纯 Python，无第三方依赖）。

为什么不直接用 NetworkX
-----------------------
赛题写的是「可使用 NetworkX 或 Neo4j」，并非强制。本项目此前的 `kg/build_graph.py`
依赖 NetworkX，但项目 requirements 里并没有它；与 `drl/` 一样，这里选择零新增依赖：
本图规模只有百量级节点，需要的算子也只有「按边类型做有界深度路径搜索」。

同时提供 `to_networkx()`，需要用 NetworkX 生态（布局算法、Gephi 导出等）时一行转换。

序列化用 JSON 而非 pickle：图谱是要给人看、给评审查的产物，纯文本可 diff、可审计，
也避免 pickle 的版本/安全问题。
"""

from __future__ import annotations

import json
from collections import defaultdict


class DiGraph:
    """有向多属性图。节点/边都可挂任意属性字典。"""

    def __init__(self):
        self._nodes: dict = {}
        self._out: dict = defaultdict(dict)   # u -> {v: attrs}
        self._in: dict = defaultdict(dict)    # v -> {u: attrs}

    # ---------------------------------------------------------------- 增删查

    def add_node(self, nid: str, **attrs) -> None:
        self._nodes.setdefault(nid, {}).update(attrs)

    def add_edge(self, u: str, v: str, **attrs) -> None:
        self.add_node(u)
        self.add_node(v)
        self._out[u][v] = attrs
        self._in[v][u] = attrs

    def has_node(self, nid: str) -> bool:
        return nid in self._nodes

    def has_edge(self, u: str, v: str) -> bool:
        return v in self._out.get(u, {})

    def node(self, nid: str) -> dict:
        return self._nodes.get(nid, {})

    def edge(self, u: str, v: str) -> dict:
        return self._out.get(u, {}).get(v, {})

    def nodes(self, type: str | None = None) -> list:
        if type is None:
            return list(self._nodes)
        return [n for n, a in self._nodes.items() if a.get("type") == type]

    def edges(self, kind: str | None = None) -> list:
        out = []
        for u, tgts in self._out.items():
            for v, a in tgts.items():
                if kind is None or a.get("kind") == kind:
                    out.append((u, v, a))
        return out

    def successors(self, u: str, kind: str | None = None) -> list:
        return [(v, a) for v, a in self._out.get(u, {}).items()
                if kind is None or a.get("kind") == kind]

    def predecessors(self, v: str, kind: str | None = None) -> list:
        return [(u, a) for u, a in self._in.get(v, {}).items()
                if kind is None or a.get("kind") == kind]

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, nid: str) -> bool:
        return nid in self._nodes

    # ---------------------------------------------------------------- 路径

    def paths(self, start: str, stop_types: tuple, kinds: tuple | None = None,
              max_depth: int = 4) -> list:
        """从 start 出发做有界深度 DFS，收集所有终点类型在 stop_types 的简单路径。

        返回 [{"nodes": [...], "edges": [attrs,...]}, ...]。不走重复节点（无环）。
        """
        out = []

        def _walk(cur, nodes, edges):
            if len(nodes) > max_depth + 1:
                return
            for v, a in self._out.get(cur, {}).items():
                if kinds is not None and a.get("kind") not in kinds:
                    continue
                if v in nodes:                    # 简单路径，天然无环
                    continue
                nn, ne = nodes + [v], edges + [dict(a, _u=cur, _v=v)]
                if self._nodes.get(v, {}).get("type") in stop_types:
                    out.append({"nodes": nn, "edges": ne})
                else:
                    _walk(v, nn, ne)

        _walk(start, [start], [])
        return out

    # ---------------------------------------------------------------- 序列化

    def to_dict(self) -> dict:
        return {
            "nodes": [{"id": n, **a} for n, a in sorted(self._nodes.items())],
            "edges": [{"source": u, "target": v, **a}
                      for u, v, a in sorted(self.edges(), key=lambda e: (e[0], e[1]))],
        }

    def save(self, path: str) -> str:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=float)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "DiGraph":
        g = cls()
        for n in d.get("nodes", []):
            n = dict(n)
            g.add_node(n.pop("id"), **n)
        for e in d.get("edges", []):
            e = dict(e)
            g.add_edge(e.pop("source"), e.pop("target"), **e)
        return g

    @classmethod
    def load(cls, path: str) -> "DiGraph":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_networkx(self):
        """转成 NetworkX DiGraph（仅在需要其生态时调用，非必需依赖）。"""
        import networkx as nx
        g = nx.DiGraph()
        for n, a in self._nodes.items():
            g.add_node(n, **a)
        for u, v, a in self.edges():
            g.add_edge(u, v, **a)
        return g
