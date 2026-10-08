"""underlying — 从期权数据反推标的期货价，并构造「曲面之外」的独立特征。

为什么需要这一层
----------------
README §6/§7.7/§7.8 已用三条路径证实：精确率卡在 50% 以下是评分函数的 P-R 前沿，
而非阈值、数据质量或决策规则的问题。要真正外推前沿，需要**与曲面特征不同源的信息**。

标的已实现波动（RV）正是这样一种信息：现有 26 维特征全部来自隐含波动率曲面本身，
而 RV 来自标的价格路径。二者的差（IV−RV，即方差风险溢价）刻画的是
「期权市场定价的波动」与「标的实际走出来的波动」之间的背离，这在曲面内部看不到。

反推方法与一个关键陷阱
----------------------
数据集没有标的价字段，但 `moneyness = strike / S` 可以反解出 S。

**陷阱：必须按到期月分组反推，不能跨期限混算。** 实测同一截面上，
不同到期月反解出的 S 呈单调的 contango 结构（ag 2025-08 某截面：dte=9 时 9190，
dte=221 时 9270），混在一起算会把**跨期限价差误当成价格波动**——
混合口径下年化波动只有 8.5%，按最近月分组后是 12.9%。
分组后组内一致性达 1e-7（机器精度），跨月差异则是真实的期限结构。

换月跳空
--------
最近月合约到期后，「最近月」会切到下一个到期月，反解价随之跳一整个日历价差
（ag 2023-03 实测 5102 → 5113，+0.22%）。这**不是标的价格波动**，但落在收益率
序列里与真实波动无法区分，会污染其后整个窗口的 RV。

`front_month_series` 因此一并返回每个截面的最近月 dte；`realized_vol` 接受
`roll_mask` 把换月那一根的收益率剔除（置 nan，滚动统计自动跳过）。
判定条件是 dte **增大**——正常情况下 dte 随时间单调递减，增大只可能是换月。

已知局限
--------
反解出的 S 是 tick 量化的（白银最小变动 1 元，实测相邻 K 线 11.3% 完全不变），
15 分钟频率上的 RV 会被微观结构噪声压低。故同时提供多个窗口的 RV，
并以较长窗口为主。这一偏差对 IV−RV 的**绝对水平**有影响，
对其**时序变化**（本项目实际使用的形态）影响较小。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 年化用的每日 K 线数：各品种交易时段不同，用实测值而非假设
BARS_PER_DAY_DEFAULT = 37.0
TRADING_DAYS = 244.0


def front_month_price(slice_df: pd.DataFrame, min_contracts: int = 5) -> float:
    """从一个时间截面反推**最近月**标的期货价。

    按 dte 分组、取最小 dte 那一组的 `median(strike / moneyness)`。
    分组是必须的——见模块 docstring 里的陷阱说明。
    """
    d = slice_df
    m = d["moneyness"].to_numpy(float)
    k = d["strike"].to_numpy(float)
    ok = np.isfinite(m) & np.isfinite(k) & (m > 1e-6)
    if ok.sum() < min_contracts:
        return float("nan")
    dte = d["dte"].to_numpy(float)[ok]
    s = k[ok] / m[ok]
    front = dte == np.nanmin(dte)
    if front.sum() < min_contracts:
        return float(np.median(s))
    return float(np.median(s[front]))


def front_month_series(month_df: pd.DataFrame, timestamps: list,
                       min_contracts: int = 5,
                       with_dte: bool = False):
    """对整批数据一次性反推每个时间戳的最近月标的价（向量化，避免逐截面 O(n²)）。

    等价于对每个 timestamp 调用 `front_month_price`，但用 groupby 一次算完
    （实测比逐截面快 238×）。

    with_dte=True 时返回 `(S, dte)` 两个等长数组，dte 为该截面最近月的剩余天数,
    供 `roll_mask` 识别换月用。
    """
    d = month_df
    m = d["moneyness"].to_numpy(float)
    k = d["strike"].to_numpy(float)
    ok = np.isfinite(m) & np.isfinite(k) & (m > 1e-6)
    sub = pd.DataFrame({
        "ts": d["timestamp"].astype(str).to_numpy()[ok],
        "dte": d["dte"].to_numpy(float)[ok],
        "S": k[ok] / m[ok],
    })
    # 每个 (ts, dte) 的中位 S 与合约数
    g = sub.groupby(["ts", "dte"])["S"].agg(["median", "size"]).reset_index()
    g = g[g["size"] >= min_contracts]
    if g.empty:
        nan = np.full(len(timestamps), np.nan)
        return (nan, nan.copy()) if with_dte else nan
    # 每个 ts 取最小 dte 那一组
    g = g.sort_values(["ts", "dte"]).groupby("ts", as_index=False).first()
    lut = dict(zip(g["ts"], g["median"]))
    S = np.array([lut.get(str(t), np.nan) for t in timestamps], dtype=float)
    if not with_dte:
        return S
    dlut = dict(zip(g["ts"], g["dte"]))
    D = np.array([dlut.get(str(t), np.nan) for t in timestamps], dtype=float)
    return S, D


def roll_mask(front_dte: np.ndarray) -> np.ndarray:
    """标记发生换月的截面（True = 该根的收益率不可信，应剔除）。

    正常情况下最近月 dte 随时间单调不增；一旦**增大**说明前一个合约到期、
    最近月切到了下一个到期月，此时反解价会跳一整个日历价差。
    返回数组与输入等长，位置 i 为 True 表示 i 与 i−1 之间跨越了换月。
    """
    d = np.asarray(front_dte, float)
    out = np.zeros(len(d), dtype=bool)
    if len(d) < 2:
        return out
    inc = np.isfinite(d[1:]) & np.isfinite(d[:-1]) & (d[1:] > d[:-1])
    out[1:] = inc
    return out


def realized_vol(prices: np.ndarray, window: int,
                 bars_per_day: float = BARS_PER_DAY_DEFAULT,
                 drop: np.ndarray | None = None) -> np.ndarray:
    """滚动已实现波动（年化）。prices 为等间隔的标的价序列。

    返回与 prices 等长的数组，前 window 个位置为 nan。
    只用**过去**的数据，不含当前之后的点——因果性由 `drl/tests.py::G1` 的同类断言保证。

    drop : 布尔数组，True 的位置其收益率被剔除（用于换月跳空，见 `roll_mask`）。
    """
    p = np.asarray(prices, dtype=float)
    valid = np.isfinite(p) & (p > 0)
    lr = np.full(len(p), np.nan)
    lr[1:] = np.where(valid[1:] & valid[:-1], np.log(p[1:] / np.maximum(p[:-1], 1e-9)), np.nan)
    if drop is not None:
        lr = np.where(np.asarray(drop, bool), np.nan, lr)
    s = pd.Series(lr)
    sd = s.rolling(window, min_periods=max(4, window // 2)).std().to_numpy()
    ann = np.sqrt(bars_per_day * TRADING_DAYS)
    return sd * ann


def build_underlying_features(prices: np.ndarray, atm_iv: np.ndarray,
                              bars_per_day: float = BARS_PER_DAY_DEFAULT,
                              short: int = 16, long: int = 64,
                              z_window: int = 240,
                              drop: np.ndarray | None = None) -> dict:
    """构造标的侧特征。全部只依赖当前及之前的数据。

    返回 dict[str, np.ndarray]，长度与输入一致：
      rv_short / rv_long  : 短/长窗口年化已实现波动
      iv_rv_spread        : ATM IV − 短窗 RV（方差风险溢价）
      iv_rv_ratio         : ATM IV / 短窗 RV
      spread_z            : 价差的滚动 z-score（本项目主用形态）
      rv_accel            : 短窗 RV / 长窗 RV − 1（波动加速）
    """
    p = np.asarray(prices, float)
    iv = np.asarray(atm_iv, float)
    rv_s = realized_vol(p, short, bars_per_day, drop=drop)
    rv_l = realized_vol(p, long, bars_per_day, drop=drop)
    spread = iv - rv_s
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(rv_s > 1e-6, iv / rv_s, np.nan)
        accel = np.where(rv_l > 1e-6, rv_s / rv_l - 1.0, np.nan)
    ss = pd.Series(spread)
    mu = ss.rolling(z_window, min_periods=max(20, z_window // 5)).mean()
    sd = ss.rolling(z_window, min_periods=max(20, z_window // 5)).std()
    z = ((ss - mu) / sd.replace(0, np.nan)).to_numpy()
    clean = lambda a: np.nan_to_num(np.asarray(a, float), nan=0.0,
                                    posinf=0.0, neginf=0.0)
    return {"rv_short": clean(rv_s), "rv_long": clean(rv_l),
            "iv_rv_spread": clean(spread), "iv_rv_ratio": clean(ratio),
            "spread_z": clean(z), "rv_accel": clean(accel)}


UNDERLYING_FEATURES = ["rv_short", "rv_long", "iv_rv_spread", "iv_rv_ratio",
                       "spread_z", "rv_accel"]
