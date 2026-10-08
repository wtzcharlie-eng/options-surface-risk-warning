"""quant_metrics — 赛题口径的量化指标评测。

赛题定义（量化技术指标1）：
  风险区间：若未来 5 个时间点（75 分钟）内，ATM IV 或曲面凸性违反程度超过
  历史滚动窗口（过去 20 个交易日）的 95% 分位数，则当前时刻之后的 2 小时为
  "风险区间"。

  精确率 precision = 预警中、预警后 2h 内确有风险区间的比例（≥50%）
  召回率 recall    = 风险区间中、被提前预警（预警早于风险区间起始）的比例（≥60%）
  平均提前时间    = 预警发出到风险区间起始的时间差（≥30 分钟）

实现要点：
  - "风险区间"基于连续 15min 时间序列判定，故需在评测窗口内跑连续截面（不能
    用采样后的 features，因为采样间隔不固定）。
  - 预警来自已落盘的 alerts（采样），与连续风险区间按时间对齐。
  - 评测窗口默认取极端事件期 ± 5 天，以控制计算量；也可全量跑。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from .cleaning import clean_slice
from .features import compute_features, FeatureHistory
from .io_loader import (_files_for_root, load_symbol_family, slice_at_timestamp,
                        timestamps_in)


def _rolling_q95(series: pd.Series, window: int) -> pd.Series:
    """过去 window 个交易日（约 window*16 个 15min 截面）的 95 分位。"""
    return series.rolling(window=window, min_periods=max(window // 4, 20)).quantile(0.95)


@dataclass
class QuantResult:
    n_risk_windows: int
    n_alerts: int
    precision: float
    recall: float
    avg_lead_minutes: float
    risk_starts: list  # 风险区间起始时刻
    detail: list


def _iter_slices(symbol: str, start_date: str, end_date: str):
    """按月流式产出 [start_date, end_date] 内的 (timestamp, 截面 DataFrame)。

    为什么不一次性 concat 再切
    -------------------------
    2026 年的月度文件单个就有 21 万行 / 86MB，一个跨 3 个月的事件窗口 concat 下来
    连同中间拷贝要 1.5GB+，在 4GB 内存的环境里直接 OOM（实测 SIGKILL）。
    按月加载、处理完即释放，峰值内存降到单月量级。

    同时用 `groupby` 一次切好所有截面，而不是对每个时间戳扫一遍全表——
    后者是 O(n²)，在 21 万行上会慢到不可用（这个坑在 `underlying.front_month_series`
    里也踩过一次）。
    """
    sd, ed = pd.to_datetime(start_date), pd.to_datetime(end_date)
    cur = sd.replace(day=1)
    while cur <= ed:
        files = sorted(set(_files_for_root(symbol, year=cur.year, month=cur.month)))
        if files:
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            df = df[df["timestamp"].astype(str).str[:8].between(start_date, end_date)]
            if not df.empty:
                for ts, sl in df.groupby(df["timestamp"].astype(str), sort=True):
                    yield ts, sl
            del df
        cur = (cur.replace(day=1) + pd.Timedelta(days=32)).replace(day=1)


def build_continuous_features(
    symbol: str,
    start_date: str,
    end_date: str,
    history: FeatureHistory | None = None,
) -> pd.DataFrame:
    """在 [start_date, end_date] 内跑连续 15min 截面，返回含 atm_iv/convexity_violation
    的连续时间序列。按月流式加载以控内存；history 可预填以保证 95 分位滚动窗口有意义。
    """
    rows, prev = [], None
    for ts, sl in _iter_slices(symbol, start_date, end_date):
        if len(sl) < 15:
            continue
        clean, _ = clean_slice(sl)
        if len(clean) < 10:
            continue
        feats = compute_features(clean, prev_features=prev, history=history, lightweight=True)
        rows.append({"timestamp": ts, "atm_iv": feats["atm_iv"],
                     "convexity_violation": feats["convexity_violation"]})
        prev = feats
    return pd.DataFrame(rows)


def identify_risk_windows(cont: pd.DataFrame, lookback_days: int = 20,
                           horizon: int = 5, q: float = 0.95) -> list:
    """按赛题定义识别风险区间起始时刻。

    若 t 之后 horizon 个时间点内，atm_iv 或 convexity_violation 超过过去
    lookback_days*~16 个截面的 q 分位，则 t 之后的 2 小时为风险区间，t 计为起始。
    返回风险区间起始的 timestamp 列表（去重相邻）。
    """
    if cont.empty:
        return []
    cont = cont.sort_values("timestamp").reset_index(drop=True)
    n = len(cont)
    # 过去 20 交易日 ≈ 20*16=320 个 15min 截面（交易日约 16 个 15min 柱）
    win = lookback_days * 16
    iv_thr = _rolling_q95(cont["atm_iv"], win)
    cx_thr = _rolling_q95(cont["convexity_violation"], win)
    starts = []
    for i in range(n - horizon):
        future = cont.iloc[i + 1: i + 1 + horizon]
        if (future["atm_iv"] > iv_thr.iloc[i]).any() or (future["convexity_violation"] > cx_thr.iloc[i]).any():
            starts.append(cont["timestamp"].iloc[i])
    # 去重相邻（同一风险区间的连续起始合并为首点）
    dedup = []
    for t in starts:
        if not dedup or _ts_diff_min(dedup[-1], t) > 120:
            dedup.append(t)
        else:
            # 保留更早的，风险区间由首点定义
            pass
    return dedup


def _ts_to_dt(ts: str) -> pd.Timestamp:
    return pd.to_datetime(ts, format="%Y%m%d%H%M%S")


def _ts_diff_min(t1: str, t2: str) -> float:
    return abs((_ts_to_dt(t2) - _ts_to_dt(t1)).total_seconds()) / 60.0


def evaluate_quant(
    alerts: pd.DataFrame,
    risk_starts: list,
    lead_window_min: float = 120.0,
    min_level: int = 2,
    ahead_only: bool = True,
) -> QuantResult:
    """算精确率/召回率/平均提前时间（向量化，O(n log n)）。

    alerts 需含 timestamp、level。
    风险区间 = [risk_start, risk_start + 2h]。预警（level >= min_level）在风险区间
    内或之前 lead_window 内计命中。min_level 默认 2，对应赛题"等级≥2"减仓规则。
    """
    import bisect

    al = alerts.sort_values("timestamp").reset_index(drop=True)
    # 预警时间戳转 int（YYYYMMDDHHMMSS），便于二分
    al_ts_int = [int(t) for t in al["timestamp"].tolist()]
    al_lv = al["level"].tolist()
    alert_ints = [al_ts_int[i] for i in range(len(al)) if al_lv[i] >= min_level]
    alert_ints.sort()
    n_alert = len(alert_ints)

    # 风险区间起始转 int；lead 窗口 = lead_window_min 分钟，用整数分钟偏移
    risk_ints = sorted([int(rs) for rs in risk_starts])
    n_risk = len(risk_ints)

    # 时间戳按 YYYYMMDDHHMMSS 整数，分钟差不能直接减；转 epoch 秒
    def _to_epoch(t_int: int) -> int:
        s = str(t_int)
        return int(pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=int(s[6:8]),
                                hour=int(s[8:10]), minute=int(s[10:12])).timestamp())
    alert_ep = [_to_epoch(t) for t in alert_ints]
    risk_ep = [_to_epoch(t) for t in risk_ints]

    # 召回：每个风险区间找 [rs-lead, rs] 内最早预警
    hits = 0
    leads = []
    for rs_e in risk_ep:
        lo, hi = rs_e - lead_window_min * 60, rs_e
        # 在 alert_ep 找落在 [lo, hi] 的、最早（最小 epoch）的
        i = bisect.bisect_left(alert_ep, lo)
        if i < len(alert_ep) and alert_ep[i] <= hi:
            hits += 1
            leads.append((rs_e - alert_ep[i]) / 60.0)

    # 精确率：每个预警找其后 lead_window 内是否有风险区间
    alerted_hit = 0
    for a_e in alert_ep:
        lo, hi = a_e, a_e + lead_window_min * 60
        i = bisect.bisect_left(risk_ep, lo)
        if i < len(risk_ep) and risk_ep[i] <= hi:
            alerted_hit += 1

    precision = alerted_hit / max(n_alert, 1)
    recall = hits / max(n_risk, 1)
    avg_lead = float(np.mean(leads)) if leads else 0.0

    return QuantResult(
        n_risk_windows=n_risk, n_alerts=n_alert,
        precision=float(precision), recall=float(recall),
        avg_lead_minutes=float(avg_lead),
        risk_starts=risk_starts,
        detail=[(rs, True) for rs in risk_starts[:hits]] + [(rs, False) for rs in risk_starts[hits:]],
    )


def run_quant_evaluation(
    symbol: str,
    start_date: str,
    end_date: str,
    min_level: int = 2,
    lead_window_min: float = 120.0,
    prefill_history_days: int = 25,
    params: dict | None = None,
    state_params: dict | None = None,
    use_state: bool = True,
    use_model: bool = True,
    composite: bool = False,
    composite_min_rules: int = 2,
    warmup_risk: bool = True,
    lookback_days: int = 20,
) -> tuple:
    """一体化赛题口径评测：在事件期跑连续截面，同时识别风险区间与预警。

    流程：
      1. 从 start_date 往前 prefill_history_days 预跑（仅喂 history，保证 95 分位
         滚动窗与 z-score 有历史基础），不参与评测
      2. [start_date, end_date] 连续跑：每截面算特征（lightweight）+ 预警（规则为主，
         若有训练好的 IF 模型则融合），同时记录 atm_iv/convexity 供风险区间判定
      3. identify_risk_windows 识别风险区间起始（`warmup_risk=True` 时把预跑段
         一并喂进滚动阈值，消除窗口开头的结构性死区，见下方实现处的注释）
      4. evaluate_quant 算精确率/召回率/提前时间

    `warmup_risk=False` 复现修正前的口径（README §7.5 原表用的就是它）。

    **一处保守偏差需知晓**：预跑段只算特征、不产生预警记录，因此落在评测窗最开头
    的风险起点无法被窗口之前的预警召回。这会**低估**召回率，方向对本文结论从严。

    返回 (cont_features_df, alerts_cont_df, risk_starts, QuantResult)
    """
    import os
    from .alert_engine import evaluate, AlertState
    from .alert_model import AlertModel

    # 预跑历史填充滚动窗
    hist = FeatureHistory(window=500)
    sd0 = (pd.to_datetime(start_date) - pd.Timedelta(days=prefill_history_days)).strftime("%Y%m%d")
    if sd0 < "20221227":
        sd0 = "20221227"
    cont0 = build_continuous_features(symbol, sd0, start_date, history=hist)
    # 评测期连续跑，同时出预警（按月流式，见 _iter_slices）
    # 加载 IF 模型（若有）
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data_out", f"model_{symbol}.joblib")
    model = (AlertModel.load(model_path) if use_model and os.path.exists(model_path) else None)

    rows, prev = [], None
    state = AlertState(**(state_params or {"persistence": 10, "confirm": 3, "escalate": 8}))
    for ts, sl in _iter_slices(symbol, start_date, end_date):
        if len(sl) < 15:
            continue
        clean, _ = clean_slice(sl)
        if len(clean) < 10:
            continue
        feats = compute_features(clean, prev_features=prev, history=hist, lightweight=True)
        res = evaluate(feats, model=model, state=state if use_state else None,
                       params=params, composite=composite, composite_min_rules=composite_min_rules)
        rows.append({"timestamp": ts, "atm_iv": feats["atm_iv"],
                     "convexity_violation": feats["convexity_violation"],
                     "level": res["level"]})
        prev = feats
    cont = pd.DataFrame(rows)
    if cont.empty:
        return cont, cont, [], QuantResult(0, 0, 0.0, 0.0, 0.0, [], [])

    if warmup_risk and not cont0.empty:
        # 用预跑段一起算滚动阈值，再只取评测窗内的风险起点。
        #
        # 为什么必须这样：`_rolling_q95` 的 min_periods=80，只喂评测窗的话，
        # **窗口开头 80 根截面的阈值是 NaN、判不出风险起点**，经 horizon 反推后
        # 前 72 根在结构上不可能有正标签——在那里发的预警必然是误报，与模型无关。
        # 这与 README §7.10 在问题3 上查出的是同一个缺陷；事件窗只有 20~40 天，
        # 死区占比比整月口径更高，影响更大。
        #
        # 赛题定义是「过去 20 个交易日」，线上系统的回看窗本就跨越窗口起点，
        # 故这是修正而非放宽。预跑段只提供阈值所需的历史，不参与任何指标计数。
        pre = cont0[["timestamp", "atm_iv", "convexity_violation"]]
        pre = pre[pre["timestamp"] < cont["timestamp"].iloc[0]]
        merged = pd.concat([pre, cont[["timestamp", "atm_iv", "convexity_violation"]]],
                           ignore_index=True).sort_values("timestamp")
        all_starts = identify_risk_windows(merged, lookback_days=lookback_days)
        t0 = cont["timestamp"].iloc[0]
        risk_starts = [t for t in all_starts if t >= t0]
    else:
        risk_starts = identify_risk_windows(cont, lookback_days=lookback_days)

    res = evaluate_quant(cont[["timestamp", "level"]], risk_starts, lead_window_min=lead_window_min)
    return cont, cont[["timestamp", "level"]], risk_starts, res


def _load_window(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载事件期附近的数据月文件。"""
    from .io_loader import _files_for_root
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)
    files = []
    cur = sd.replace(day=1) - pd.Timedelta(days=1)
    while cur <= ed:
        files.extend(_files_for_root(symbol, year=cur.year, month=cur.month))
        cur = (cur.replace(day=1) + pd.Timedelta(days=32)).replace(day=1)
    files = sorted(set(files))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def format_quant(r: QuantResult) -> str:
    lines = [
        f"量化指标（赛题口径）: 风险区间={r.n_risk_windows} 预警={r.n_alerts}",
        f"  精确率={r.precision:.2%} (目标≥50%)",
        f"  召回率={r.recall:.2%} (目标≥60%)",
        f"  平均提前={r.avg_lead_minutes:.0f}min (目标≥30min)",
        f"  命中风险区间 {sum(1 for _,h in r.detail if h)}/{r.n_risk_windows}",
    ]
    return "\n".join(lines)


    lines = [
        f"量化指标（赛题口径）: 风险区间={r.n_risk_windows} 预警={r.n_alerts}",
        f"  精确率={r.precision:.2%} (目标≥50%)",
        f"  召回率={r.recall:.2%} (目标≥60%)",
        f"  平均提前={r.avg_lead_minutes:.0f}min (目标≥30min)",
        f"  命中风险区间 {sum(1 for _,h in r.detail if h)}/{r.n_risk_windows}",
    ]
    return "\n".join(lines)
