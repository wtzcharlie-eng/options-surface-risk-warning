"""drl.metrics — 赛题口径的精确率 / 召回率 / 平均提前时间。

本模块是 `vol_surface/quant_metrics.evaluate_quant` 的独立实现（纯 numpy/pandas，
不引入 scipy 依赖链），判定逻辑逐条对齐：

  - 预警集合 = {t : level(t) ≥ 2}
  - **召回**：对每个风险起点 rs，若存在预警落在 [rs−120min, rs]，则该风险起点被
    召回；提前时间取该窗口内**最早**的那条预警
  - **精确率**：对每条预警 a，若存在风险起点落在 [a, a+120min]，则该预警命中
  - 两者共用同一个 120 分钟对称窗口——这正是 README 里论证 P-R 前沿的结构性根源

`drl/tests.py::test_metrics_matches_quant_metrics` 会在 scipy 可用时导入真实的
`evaluate_quant` 做逐窗口比对，确保两份实现不漂移。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

LEAD_WINDOW_MIN = 120.0
ALERT_LEVEL = 2


def _epoch(ts: list) -> np.ndarray:
    return (pd.to_datetime(pd.Series(list(ts)), format="%Y%m%d%H%M%S")
            .astype("int64").to_numpy() // 10**9)


@dataclass
class QuantMetrics:
    n_risk: int
    n_alert: int
    n_hit_alert: int
    n_recalled: int
    precision: float
    recall: float
    avg_lead_min: float

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_alerts(ts: list, levels: np.ndarray, risk_idx,
                    lead_window_min: float = LEAD_WINDOW_MIN,
                    min_level: int = ALERT_LEVEL) -> QuantMetrics:
    """算单幕的赛题指标。`risk_idx` 是风险起点在 ts 中的索引。"""
    ep = _epoch(ts)
    levels = np.asarray(levels)
    a_ep = np.sort(ep[levels >= min_level])
    r_ep = np.sort(ep[np.asarray(risk_idx, dtype=np.int64)]) if len(risk_idx) else np.array([], dtype=np.int64)
    win = lead_window_min * 60.0

    a_list = a_ep.tolist()
    r_list = r_ep.tolist()

    # 召回：每个风险起点找 [rs-win, rs] 内最早的预警
    n_recalled, leads = 0, []
    for rs in r_list:
        k = bisect.bisect_left(a_list, rs - win)
        if k < len(a_list) and a_list[k] <= rs:
            n_recalled += 1
            leads.append((rs - a_list[k]) / 60.0)

    # 精确率：每条预警找其后 [a, a+win] 内是否有风险起点
    n_hit = 0
    for a in a_list:
        k = bisect.bisect_left(r_list, a)
        if k < len(r_list) and r_list[k] <= a + win:
            n_hit += 1

    return QuantMetrics(
        n_risk=len(r_list), n_alert=len(a_list), n_hit_alert=n_hit,
        n_recalled=n_recalled,
        precision=n_hit / len(a_list) if a_list else 0.0,
        recall=n_recalled / len(r_list) if r_list else 0.0,
        avg_lead_min=float(np.mean(leads)) if leads else 0.0,
    )


def aggregate(rollouts: list, **kw) -> dict:
    """把多幕 rollout 汇总成 micro（按计数合并）与 macro（按幕平均）两套口径。

    micro 是主口径：它等价于把所有幕拼成一条长序列后算指标，不会被小样本幕放大。
    """
    per = []
    for ro in rollouts:
        m = evaluate_alerts(ro["ts"], ro["actions"], ro["risk_idx"], **kw)
        per.append({"symbol": ro["symbol"], "ym": ro["ym"],
                    "total_reward": ro["total_reward"], **m.as_dict()})
    df = pd.DataFrame(per)
    if df.empty:
        return {"micro": {}, "per_episode": df}

    n_alert, n_risk = int(df.n_alert.sum()), int(df.n_risk.sum())
    n_hit, n_rec = int(df.n_hit_alert.sum()), int(df.n_recalled.sum())
    # 平均提前时间按被召回的风险起点数加权
    w = df.n_recalled.to_numpy()
    lead = float((df.avg_lead_min.to_numpy() * w).sum() / w.sum()) if w.sum() else 0.0

    micro = {
        "n_episodes": len(df),
        "total_reward": float(df.total_reward.sum()),
        "mean_reward_per_episode": float(df.total_reward.mean()),
        "n_alert": n_alert, "n_risk": n_risk,
        "precision": n_hit / n_alert if n_alert else 0.0,
        "recall": n_rec / n_risk if n_risk else 0.0,
        "avg_lead_min": lead,
        "alert_rate": float(n_alert / sum(len(r["ts"]) for r in rollouts)),
    }
    return {"micro": micro, "per_episode": df}


def format_metrics(name: str, m: dict) -> str:
    def mark(v, thr, fmt="{:.1%}"):
        return f"{fmt.format(v)}{'✓' if v >= thr else '✗'}"
    return (f"{name:22s} 累计奖励={m['total_reward']:9.1f}  "
            f"精确率={mark(m['precision'], .50)}  "
            f"召回率={mark(m['recall'], .60)}  "
            f"提前={m['avg_lead_min']:5.1f}min{'✓' if m['avg_lead_min'] >= 30 else '✗'}  "
            f"预警数={m['n_alert']:5d}  预警率={m['alert_rate']:.1%}")
