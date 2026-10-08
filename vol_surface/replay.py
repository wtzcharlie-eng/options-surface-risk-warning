"""replay — 唯一历史重放管线（raw → clean → compute_features → FH push）。

职责(统一入口,避免各模块重复重放):
  1. 按时间顺序逐截面调用 compute_features,维护唯一一份 FeatureHistory。
  2. 所有 z-score / velocity / prev_features 语义与 compute_features 完全一致
     (含当前登录与否的口径见 FeatureHistory 文档)。
  3. 构造跨截面调用的 prev_features(上一截面的输出,用于 term_slope_roc /
     iv_change 等一阶差分特征)。
  4. 可选同时调 alert_rules / alert_engine,输出最终等级(与 anchor 数据集
     所需的 raw_level/final_level 对齐)。

硬约束:
- quant_metrics / anchor 构建 / kg.coalert 都应调用本模块,严禁各自重放。
- 只要调用同一 raw 输入,输出必须逐点相等(任何回放差异都是 bug,按差异排查)。

用法例子:
    cp = replay_full_history(symbol="ag", end="2026-02", prefill_days=25)
    # cp.features_df 含所有特征+ timestamp; cp.alerts_df 含触发等级(+rule:level)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .alert_engine import evaluate as _engine_evaluate
from .cleaning import clean_slice
from .features import FeatureHistory, compute_features
from .io_loader import _files_for_root, slice_at_timestamp, timestamps_in


# ---------------------- 数据类型 ----------------------

@dataclass
class ReplayColumn:
    """回放输出的一张表的的数据结构。"""
    features: pd.DataFrame        # 特征 + _timestamp
    rules: pd.DataFrame | None    # 触发的 rule / level / 指标明细(可选)
    final: pd.DataFrame | None    # level 后的 raw_level/final_level
    meta: dict


@dataclass
class ReplayInput:
    """准备 replay 必要的输入."""
    symbol: str
    start: str         # YYYYMMDD, 回放的起点
    end: str           # YYYYMMDD, 回放的终点
    prefill_days: int = 25   # raw 从这个天之前向前回回撤的天数


@dataclass
class ReplayResult:
    features: pd.DataFrame          # _timestamp + 所有特征
    rule_hits: pd.DataFrame | None  # _timestamp, rule, level, value, threshold
    raw_level: pd.Series | None     # evaluate_rules 的 level (无状态机)
    final_level: pd.Series | None   # 含状态机 + model 融合的最高 level


# ---------------------- 辅助: IO ----------------------

def _ymd(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y%m%d")


def _year_month(d: str) -> tuple[int, int]:
    return int(d[:4]), int(d[4:6])


def load_symbol_month_slice(symbol: str, ymd: str) -> pd.DataFrame:
    """加载该品种当日/近月全部截面(为不断重放提供最小单位)。"""
    y, m = _year_month(ymd)
    files = sorted(_files_for_root(symbol, year=y, month=m))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# ---------------------- 核心重放器 ----------------------

def run_replay(symbol: str,
               raw: pd.DataFrame,
               start: str,
               prefill_days: int = 25,
               do_rules: bool = False,
               do_dual_alerts: bool = False,
               fh: FeatureHistory | None = None,
               state=None,
               ) -> ReplayResult:
    """对 raw 做一次完整 replay.

    参数:
        symbol   : 品种代码
        raw      : 原始截面 parquet(按 timestamp 排)
        start    : 首个需截面的 yyyymmdd(在此之前的 slice 仅用于 prefill)
        prefill_days : history 向前推的天数(从 start 往前)
        do_rules : 是否调用 evaluate_rules(记录每截面触发的规则)
        do_dual_alerts : 是否同时调用 alert_engine,输出 final_level
                       (依赖 do_rules=True)
        fh       : 可选外部 FeatureHistory(会原地 push);默认新建
        state    : 可选 AlertState(会原地更新);默认不维护跨截面状态

    返回: ReplayResult(含 features_df / rule_hits / raw_level / final_level)
    """
    if raw.empty:
        return ReplayResult(pd.DataFrame(), None, None)

    raw = raw.sort_values("timestamp").reset_index(drop=True)
    tss = sorted(set(raw["timestamp"].astype(str)))
    # 按时间顺序保证 prev_features 的正确性(prev 只在连续截面上成立)
    start_ts = pd.to_datetime(start, format="%Y%m%d")
    prefill_from = (start_ts - pd.Timedelta(days=prefill_days)).strftime("%Y%m%d")

    if fh is None:
        fh = FeatureHistory(window=500)

    # 准备输出容器
    feats_rows = []        # 每个截面 compute_features 输出
    if do_rules:
        rule_rows = []         # evaluate_rules 的 triggers 扁平化
        raw_level_rows = []
        final_level_rows = []
    else:
        rule_rows = None
        raw_level_rows = None
        final_level_rows = None

    prev_feats = None
    for ts in tss:
        # 当前截面
        sl = raw[raw["timestamp"].astype(str) == ts]
        if len(sl) < 15:
            continue
        clean, _ = clean_slice(sl)
        if len(clean) < 10:
            continue

        # 特征(此处会 push fh,并生成 z/velocity)
        feats = compute_features(clean, prev_features=prev_feats,
                                 history=fh, lightweight=True)
        feats["_timestamp"] = ts

        # 过滤: 预填段不记录,只填 history
        if ts >= prefill_from + "000000":
            feats_rows.append({k: feats.get(k, np.nan) for k in feats})

            if do_rules:
                res = _engine_evaluate(feats, model=None, state=state)
                # 触发的 rules(每条取最后一次 hit)
                for t in res["triggers"]:
                    rule_rows.append({"_timestamp": ts, "rule": t["rule"],
                                      "level": t["level"], "value": t["value"],
                                      "threshold": t["threshold"]})
                raw_level_rows.append({"_timestamp": ts, "raw_level": res["level"]})
                if do_dual_alerts:
                    final_level_rows.append({"_timestamp": ts,
                                             "final_level": res["level"]})
            prev_feats = feats
        else:
            # 预填期: 即使不记录也要更新 prev
            prev_feats = feats

    feats_df = (pd.DataFrame(feats_rows)
                  .sort_values("_timestamp").reset_index(drop=True))
    if do_rules and rule_rows:
        hits = pd.DataFrame(rule_rows)
    else:
        hits = None
    raw_lv = (pd.DataFrame(raw_level_rows).set_index("_timestamp")["raw_level"]
              if raw_level_rows else None)
    final_lv = (pd.DataFrame(final_level_rows).set_index("_timestamp")["final_level"]
                if final_level_rows else None)
    if final_lv is None and raw_lv is not None:
        final_lv = raw_lv
    return ReplayResult(feats_df, hits, raw_lv, final_lv)


# ---------------------- 摘要: 全历史重放 ----------------------

def replay_full_history(symbol: str, start: str, end: str,
                        prefill_days: int = 25,
                        do_rules: bool = False,
                        do_dual_alerts: bool = False) -> ReplayColumn:
    """加载 [start-prefill, end] 的 raw, 调用 run_replay 一次拿到一致输出.

    该方法汇总输出为 ReplayColumn 便于下游统一引用(quant/anchor/kg 开关)。
    """
    start_ts = pd.to_datetime(start, format="%Y%m%d")
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    prefill = (start_ts - pd.Timedelta(days=prefill_days)).strftime("%Y%m%d")

    # 所需月份序列(重点这里只能逐月装, 一次顺序推入以保证 prev 与 history)
    cur = pd.to_datetime(prefill, format="%Y%m%d").replace(day=1)
    months = []
    while cur <= end_ts:
        months.append((cur.year, cur.month))
        cur = (cur.replace(day=1) + pd.Timedelta(days=32)).replace(day=1)

    # 逐个月拼装/重放
    feats_pool, rules_pool, raw_pool, final_pool = [], [], [], []
    fh = FeatureHistory(window=500)
    state = None
    for y, m in months:
        files = sorted(_files_for_root(symbol, year=y, month=m))
        if not files:
            continue
        raw = pd.concat([pd.read_parquet(f) for f in files],
                        ignore_index=True)
        month_start = f"{y}{m:02d}"
        r = run_replay(symbol, raw, start=month_start,
                       prefill_days=prefill_days, do_rules=do_rules,
                       do_dual_alerts=do_dual_alerts, fh=fh, state=state)
        if r.features.empty:
            continue
        feats_pool.append(r.features)
        if r.rule_hits is not None:
            rules_pool.append(r.rule_hits)
        if r.raw_level is not None:
            raw_pool.append(r.raw_level)
        if final_pool is not None and r.final_level is not None:
            final_pool.append(r.final_level)

    feats_df = (pd.concat(feats_pool, ignore_index=True)
                if feats_pool else pd.DataFrame())
    rules_df = (pd.concat(rules_pool, ignore_index=True)
                if rules_pool else None)
    raw_lv = (pd.concat(raw_pool) if raw_pool else None)
    final_lv = (pd.concat(final_pool) if final_pool else None)

    # 仅保留 [start, end] 区间
    keep = (feats_df["_timestamp"] >= start + "000000") & \
           (feats_df["_timestamp"] <= end + "235959")
    feats_df = feats_df[keep].reset_index(drop=True)
    if rules_df is not None:
        rules_df = rules_df[rules_df["_timestamp"].isin(feats_df["_timestamp"])]
    if raw_lv is not None:
        raw_lv = raw_lv[raw_lv.index.isin(feats_df["_timestamp"])]
    if final_lv is not None:
        final_lv = final_lv[final_lv.index.isin(feats_df["_timestamp"])]

    return ReplayColumn(features=feats_df, rules=rules_df,
                        raw_level=raw_lv, final_level=final_lv,
                        meta={"symbol": symbol, "start": start,
                              "end": end, "prefill_days": prefill_days})
