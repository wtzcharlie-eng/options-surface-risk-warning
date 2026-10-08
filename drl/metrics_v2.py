"""metrics_v2 — 按命题方答复的新评测口径，**并自带退化检验**。

命题方答复（2026-08）
--------------------
1. **精确率**：按**合并连续预警后的预警事件**计，不再按每个 15 分钟截面计；
   并建议**分高波/低波区**分别报告。
2. **召回**：匹配**不设时间上限**——只要存在早于风险区间起始时刻的预警即算召回。
3. **平均提前时间**：口径不设限（由参赛队选定并说明）。
4. CVaR 改善率公式：`½[(CVaR_无,多−CVaR_预,多)/CVaR_无,多 + (空同理)]×100%`。
5. 指标3 采用自评形式。

为什么本文件必须自带退化检验
----------------------------
「召回不设时间上限」有一个必须点破的后果：**在测试期最开头发一次预警，
就早于其后所有风险起点，召回直接 100%**。若同时精确率按事件计，
那么"只报一次且恰好命中"的平凡策略可以拿到 P=100%/R=100%/提前=数月。

这不是钻空子的假设，是该口径的字面推论。故本模块把
`trivial_controls()` 与指标实现放在同一文件里，**任何引用本口径的结论
都必须同时报出平凡策略在同口径下的成绩**——若平凡策略也达标，
该口径就不携带区分信息，模型优化无从谈起。

这条纪律在本项目已被验证过一次：CVaR 指标「仅首次预警减仓」使得改善率
恒为 +50%，与预警质量无关，当时正是靠平凡策略对照才发现。

提前时间的口径选择
------------------
「不设限」下有两种取法：
  - `nearest`（本模块默认）：取**紧邻该风险起点之前**的那次预警，提前时间有业务含义；
  - `first`：取全期第一次预警，提前时间可达数月，数值虚高且无意义。
默认 `nearest` 并在报告中写明；`first` 仅用于展示口径敏感性。
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

ALERT_LEVEL = 2
HIT_WINDOW_MIN = 120.0     # 精确率：预警后 2 小时内出现风险起点即命中（原文明确）
MERGE_GAP_MIN = 120.0      # 合并连续预警的间隔阈值（间隔 ≤ 此值即视为同一次预警）


def _epoch(ts) -> np.ndarray:
    return (pd.to_datetime(pd.Series(list(ts)), format="%Y%m%d%H%M%S")
            .astype("int64").to_numpy() // 10 ** 9)


def merge_alerts(a_ep: np.ndarray, gap_min: float = MERGE_GAP_MIN) -> list:
    """把连续预警合并成预警事件，返回每个事件的**起始**时刻（秒）。

    「预警时间」取事件起点——即该次风险被首次示警的时刻，
    这也是命题方所说「合并连续预警的预警时间」的自然读法。
    """
    if len(a_ep) == 0:
        return []
    gap = gap_min * 60.0
    out, start, prev = [], a_ep[0], a_ep[0]
    for t in a_ep[1:]:
        if t - prev > gap:
            out.append(start)
            start = t
        prev = t
    out.append(start)
    return out


@dataclass
class MetricsV2:
    n_risk: int
    n_alert_slice: int      # 原始截面数（保留，便于与旧口径对照）
    n_alert_event: int      # 合并后的预警事件数
    n_hit_event: int
    n_recalled: int
    precision: float        # 事件级
    recall: float           # 不设时间上限
    avg_lead_min: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_v2(ts, levels, risk_idx, min_level: int = ALERT_LEVEL,
                merge_gap_min: float = MERGE_GAP_MIN,
                hit_window_min: float = HIT_WINDOW_MIN,
                lead_mode: str = "nearest",
                recall_window_min: float | None = None) -> MetricsV2:
    """新口径单幕指标。

    `recall_window_min=None` 即命题方答复的「不设上限」；传入数值可退回旧口径做对照。
    """
    ep = _epoch(ts)
    levels = np.asarray(levels)
    a_ep = np.sort(ep[levels >= min_level])
    r_ep = (np.sort(ep[np.asarray(risk_idx, dtype=np.int64)])
            if len(risk_idx) else np.array([], dtype=np.int64))
    events = merge_alerts(a_ep, merge_gap_min)
    r_list = r_ep.tolist()

    # 精确率：每个**预警事件**，其起始时刻后 hit_window 内是否出现风险起点
    hw = hit_window_min * 60.0
    n_hit = 0
    for a in events:
        k = bisect.bisect_left(r_list, a)
        if k < len(r_list) and r_list[k] <= a + hw:
            n_hit += 1

    # 召回：每个风险起点，其**之前**是否存在预警事件（默认不设上限）
    n_rec, leads = 0, []
    lo_bound = None if recall_window_min is None else recall_window_min * 60.0
    for rs in r_list:
        k = bisect.bisect_left(events, rs)      # events[:k] 全部早于 rs
        if k == 0:
            continue
        cand = events[k - 1] if lead_mode == "nearest" else events[0]
        if lo_bound is not None and rs - cand > lo_bound:
            continue
        n_rec += 1
        leads.append((rs - cand) / 60.0)

    return MetricsV2(
        n_risk=len(r_list), n_alert_slice=int(len(a_ep)),
        n_alert_event=len(events), n_hit_event=n_hit, n_recalled=n_rec,
        precision=n_hit / len(events) if events else 0.0,
        recall=n_rec / len(r_list) if r_list else 0.0,
        avg_lead_min=float(np.mean(leads)) if leads else 0.0)


def aggregate_v2(per_episode: list) -> dict:
    """多幕汇总（micro：按计数合并，与旧口径同）。"""
    if not per_episode:
        return {}
    ne = sum(m.n_alert_event for m in per_episode)
    nr = sum(m.n_risk for m in per_episode)
    nh = sum(m.n_hit_event for m in per_episode)
    nc = sum(m.n_recalled for m in per_episode)
    w = np.array([m.n_recalled for m in per_episode], dtype=float)
    lead = float((np.array([m.avg_lead_min for m in per_episode]) * w).sum()
                 / w.sum()) if w.sum() else 0.0
    return {"n_episodes": len(per_episode), "n_alert_event": ne, "n_risk": nr,
            "n_hit_event": nh, "n_recalled": nc,
            "n_alert_slice": sum(m.n_alert_slice for m in per_episode),
            "precision": nh / ne if ne else 0.0,
            "recall": nc / nr if nr else 0.0, "avg_lead_min": lead}


# ---------------------------------------------------------------- 退化检验

def trivial_policies(n: int) -> dict:
    """构造若干**不看任何特征**的平凡预警序列，用于检验口径是否携带区分信息。

    `once_at_start` 是针对「召回不设上限」量身构造的：
    只在最开头报一次，它早于其后**全部**风险起点，故召回应为 100%。
    若它同时也过了精确率线，就说明该口径无法区分好坏系统。
    """
    out = {}
    z = np.zeros(n, dtype=np.int64)
    a = z.copy(); a[0] = 3
    out["只在开头报一次"] = a
    a = z.copy(); a[:max(1, n // 100)] = 3
    out["只在前1%截面报"] = a
    for k in (3, 8, 20):
        a = z.copy(); a[::k] = 3
        out[f"每{k}步机械报警"] = a
    out["恒报警"] = np.full(n, 3, dtype=np.int64)
    return out
