"""drl.dataset — 从 anchor 数据集构造 DRL 训练用的 episode。

数据来源：`data_out/anchor/{symbol}/{YYYY-MM}/features.parquet`
（由 scripts/build_anchor_dataset.py 生成，26 维特征 + `_timestamp` + 标签 `y`）

本模块做三件事：
1. **复现风险起点**：重算 `identify_risk_windows` 的判定（20 日回看 q95、horizon=5、
   120min 去重），从而拿到每个截面到下一个风险起点的**分钟数**——奖励里的提前时间
   加权需要它，而 anchor parquet 只存了二值标签 `y`。
2. **自校验**：用重算出的风险起点按 horizon=8 反推标签，与 parquet 里存的 `y`
   逐行比对。不一致即抛异常——这保证本模块的风险判定与赛题评分口径**逐位对齐**，
   而不是"我以为对齐了"。
3. **按时间切分**：train / val / test 严格按月份先后切，杜绝未来信息回流。

注：这里刻意不 import vol_surface.quant_metrics。后者在模块顶层 import 了
scipy 依赖链（interpolation），而本模块只需要其中纯 pandas 的那一小段判定逻辑。
两者的一致性由上面第 2 步的逐行比对来保证。
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 与 scripts/build_anchor_dataset.NUM_FEATURES 保持一致，顺序即状态向量顺序
FEATURES = [
    "atm_iv", "atm_iv_z", "atm_iv_vel_z", "convexity_violation", "convexity_violation_z",
    "convexity_vel_z", "term_slope", "term_slope_roc", "skew_percentile", "skew_val",
    "gamma_concentration", "vega_concentration", "iv_change", "iv_change_rate",
    "iv_spike", "fit_degradation", "arb_score", "arb_score_z", "arb_calendar_n",
    "arb_butterfly_n", "arb_parity_n", "liquidity_ratio", "liquidity_z", "iv_near",
    "iv_far", "n_contracts",
]

# 「曲面之外」的标的侧特征（scripts/build_underlying.py 产出，见 README §7.9/§7.10）。
# **必须追加在 FEATURES 之后、不得插在中间**：drl.baseline.RulePolicy 依赖
# `enumerate(FEATURES)` 与 X 前 26 列的位置对应关系来复现规则引擎判定。
UNDERLYING = ["rv_short", "rv_long", "iv_rv_spread", "iv_rv_ratio",
              "spread_z", "rv_accel"]

ALL_FEATURES = FEATURES + UNDERLYING


def feature_names(underlying: bool) -> list:
    """当前特征集的名字表。26 维（仅曲面）或 32 维（曲面 + 标的）。"""
    return list(ALL_FEATURES) if underlying else list(FEATURES)

LABEL_HORIZON = 8       # 标签口径：未来 8 个截面（=120min）内是否有风险起点
RISK_HORIZON = 5        # 赛题定义：未来 5 个时间点内曲面显著恶化
RISK_LOOKBACK_DAYS = 20
RISK_Q = 0.95
DEDUP_MIN = 120.0       # 风险起点去重窗口（分钟）

# ---- 结构性死区 ------------------------------------------------------------
# 精确率的命中窗口是**墙钟分钟制**的 `[a, a+120min]`（drl/metrics.py），
# 而风险起点是**根数制**的（未来 RISK_HORIZON 根）。两者口径不一致，
# 后果是：**收盘前发出的预警，它承诺的那 120 分钟里几乎没有可交易时间，
# 结构上不可能命中**。
#
# 实测（anchor 测试集 29,696 个截面，全样本命中率 24.39%）：
#   15:00 → 0.65% | 14:45 → 1.58% | 14:30 → 2.88% | 02:30 → 1.89%
#   按「(t, t+120min] 内可交易根数」分组总体上升：0 根 8.8% → 8 根 36.6%
#   （**并非逐档单调**，5 根 32.6% → 6 根 30.8% 有一处回落）
#
# **注意「可交易根数」不是判别死区的充分特征**：同为 0 根，15:00 是 0.65%
#   而 11:30 是 22.05%——差别不来自可交易根数，而来自「风险起点本身落在该根上」
#   的概率（11:30 是午休前最后一根，收盘则不是）。
#   故死区**按「其后是否遭遇长休市」定义，而不是按可交易根数定义**；
#   上面那条梯度是**佐证**，不是判据。
#
# **休市阈值 150 分钟这个数不能凭印象设，必须由真实盘口结构决定**：
#   实测该数据集的日内休市是 **135 分钟**（早市末根 11:30 → 下午首根 13:45），
#   **不是想当然的 120 分钟**。测试集 29,637 个相邻间隔里，**日内间隔有两档**：
#   1075 个 135 分钟（午休 11:30→13:45）与 1075 个 30 分钟（10:15→10:45 小节休息）；
#   隔夜/跨节则 ≥375 分钟。**日内最大值 135 与隔夜最小值 375 之间有 240 分钟空隙**，
#   150 落在其中，但**下方余量只有 15 分钟**（再低 15 分钟就会误伤午休）。
#   若阈值取 ≤135（例如凭「午休 120 分钟」写成 130），午休前的预警会被误杀，
#   而**精确率照样上升**（它们命中率 22~23%，略低于全样本 24.39%）——
#   指标变好、系统变差，且没有任何数字门会报警。故由 **G12 从数据现算**校准，
#   不再把 120 或 135 写死在断言里。
#
#   注：午休前的预警之所以仍接近基础率，**不是**因为窗口边界上正好有一根
#   （[11:30, 13:30] 里其实一根都没有，下一根是 13:45），
#   而是因为**风险起点本身常常就落在 11:30 那一根上**（闭区间含 a 自身）。
#   早前写的「13:30 那根正好落在闭区间边界上」是错的，已更正。
#
# **只处理「休市」，不处理「幕尾」**：按月切幕会让每幕最后几根也无法命中
# （因为指标逐幕评估、不跨幕匹配），但那是**评测脚手架的产物**而非市场结构，
# 且实测只占 0.60% 截面、命中率 12.4%（远没有休市前那么极端）。
# 拿系统侧的过滤去补评测侧的切分 artifact 是错位的，故不做。
DEAD_ZONE_BARS = 3           # 休市前多少根算死区
# 阈值取空隙 [135, 375] 的**中点**，两侧各留 120min 余量。
# 为什么不是原来的 150：150 是我按「午休 120min」这个**错误前提**推出来的
# （120 + 30 余量），而实测午休是 135min，于是真实下余量只剩 15min——
# **它能用是巧合，不是设计**。改到中点后两侧余量对称且充裕。
# **这次改动是可证明的空操作**：全量 263 幕实测，观测间隔落在 (135, 375) 内的
# 个数为 **0**，阈值 150 与 255 的死区掩码逐位差异为 **0 个截面**。
# 故改它不动任何交付数字，只提高对「将来加入新交易时段品种」的稳健性。
DEAD_ZONE_BREAK_MIN = 255.0  # 空隙 [135,375] 的中点；由 G12 从数据现算校准余量


def session_dead_zone(ts: list, k_bars: int = DEAD_ZONE_BARS,
                      min_break_min: float = DEAD_ZONE_BREAK_MIN) -> np.ndarray:
    """标记「其后 120min 内几乎无可交易时间」的截面。

    判据：某根之后到下一根的间隔 > `min_break_min`（即遇到休市），
    则该根及其前 `k_bars-1` 根标记为死区。

    **不标记幕尾**——见上方注释：那是按月切幕的产物，不是市场结构。

    返回长度与 `ts` 相同的 bool 数组，True 表示落在结构性死区内。
    """
    n = len(ts)
    if n == 0:
        return np.zeros(0, dtype=bool)
    t = (pd.to_datetime(pd.Series(ts), format="%Y%m%d%H%M%S")
         .astype("int64").to_numpy() // 10 ** 9)
    # 非升序输入会让 `t[1:]-t[:-1]` 出现负值，从而**静默返回全 False**——
    # 即「过滤看起来生效了、其实一根都没标」。这类失败没有任何下游断言能发现，
    # 故在此显式拦下。（当前 `load_episode` 一律排序，本断言是第二道保险。）
    if n > 1:
        assert np.all(t[1:] >= t[:-1]), (
            "session_dead_zone 收到非升序时间戳——死区判定依赖相邻间隔，"
            "乱序会静默退化为「不标记任何根」。请先按时间排序")
    dead = np.zeros(n, dtype=bool)
    brk = np.flatnonzero((t[1:] - t[:-1]) / 60.0 > min_break_min)
    for i in brk:
        dead[max(0, i - k_bars + 1):i + 1] = True
    return dead


def suppress_dead_zone(ts: list, levels: np.ndarray, **kw) -> np.ndarray:
    """把落在结构性死区里的预警降级为 0（不预警）。

    **必须对规则引擎与 DRL 同时施加**——只给一方加就不是公平对照。
    实测两者都会获得同等量级的精确率提升，判别力几乎不变，报数时须一并说明。
    """
    levels = np.asarray(levels).copy()
    levels[session_dead_zone(ts, **kw)] = 0
    return levels



# ------------------------------------------------------------------ 风险起点

def _rolling_q95(s: pd.Series, window: int) -> pd.Series:
    """与 quant_metrics._rolling_q95 同参：min_periods = max(window//4, 20)。"""
    return s.rolling(window=window, min_periods=max(window // 4, 20)).quantile(RISK_Q)


def identify_risk_starts(ts: list, atm_iv: np.ndarray, convexity: np.ndarray) -> list:
    """复现 quant_metrics.identify_risk_windows，返回风险起点在序列中的**索引**。

    判定：若 t 之后 RISK_HORIZON 个截面内，atm_iv 或 convexity_violation 超过
    过去 20 交易日（≈320 个 15min 截面）的 95 分位，则 t 记为风险区间起始。
    相邻 120min 内的起点合并，只保留最早的那个。
    """
    n = len(ts)
    if n == 0:
        return []
    win = RISK_LOOKBACK_DAYS * 16
    iv_thr = _rolling_q95(pd.Series(atm_iv), win).to_numpy()
    cx_thr = _rolling_q95(pd.Series(convexity), win).to_numpy()

    raw = []
    for i in range(n - RISK_HORIZON):
        j0, j1 = i + 1, i + 1 + RISK_HORIZON
        # NaN 阈值（历史不足）时比较恒为 False，与 pandas 行为一致
        if np.any(atm_iv[j0:j1] > iv_thr[i]) or np.any(convexity[j0:j1] > cx_thr[i]):
            raw.append(i)

    dt = pd.to_datetime(pd.Series(ts), format="%Y%m%d%H%M%S")
    keep = []
    for i in raw:
        if not keep or abs((dt.iloc[i] - dt.iloc[keep[-1]]).total_seconds()) / 60.0 > DEDUP_MIN:
            keep.append(i)
    return keep


def labels_from_risk_starts(n: int, risk_idx: list, horizon: int = LABEL_HORIZON) -> np.ndarray:
    """复现 build_anchor_dataset._label_slices：i ∈ [i0-horizon, i0-1] 置 1。"""
    y = np.zeros(n, dtype=np.int64)
    for i0 in risk_idx:
        for i in range(max(0, i0 - horizon), i0):
            y[i] = 1
    return y


# ------------------------------------------------------------------ Episode

@dataclass
class Episode:
    """一个 (品种, 月份) 的连续截面序列 = 一幕。"""
    symbol: str
    ym: str                      # "YYYY-MM"
    ts: list                     # 时间戳字符串
    X: np.ndarray                # (n, 26) 或 (n, 32) 原始特征
    y: np.ndarray                # (n,) 标签：未来 120min 内有风险起点
    risk_idx: np.ndarray         # 风险起点索引
    lead_min: np.ndarray         # (n,) 到下一个风险起点的分钟数，无则 inf
    dt: np.ndarray = field(repr=False, default=None)   # (n,) datetime64
    feat_names: list = field(repr=False, default=None)  # X 各列的名字

    def __len__(self) -> int:
        return len(self.ts)


def _lead_minutes(dt: pd.Series, risk_idx: list) -> np.ndarray:
    """每个截面到**其后（含自身）**最近一个风险起点的分钟数；之后没有则 inf。

    「含自身」很关键：赛题精确率判定是「预警后 2h 内存在风险区间」，窗口为
    **闭区间** [a, a+120min]；召回判定窗口同样是闭区间 [rs−120min, rs]。因此恰好
    在风险起点当根 K 线发出的预警，赛题算命中、也算召回。若这里用严格大于，
    环境会把这类预警同时记成「误报」和「漏报」，与评分口径相悖。
    """
    n = len(dt)
    out = np.full(n, np.inf)
    if not risk_idx:
        return out
    rs = sorted(risk_idx)
    secs = dt.astype("int64").to_numpy() / 1e9
    ptr = 0
    for i in range(n):
        while ptr < len(rs) and rs[ptr] < i:
            ptr += 1
        if ptr < len(rs):
            out[i] = (secs[rs[ptr]] - secs[i]) / 60.0
    return out


def load_episode(path: str, symbol: str, ym: str, verify: bool = True,
                 underlying: bool = False) -> Episode | None:
    """读一个月的 features.parquet，重算风险起点并自校验标签一致性。

    underlying=True 时把同目录 `underlying.parquet` 的 6 维标的特征拼在后面
    （由 scripts/build_underlying.py 生成）。文件缺失或行数对不上直接报错——
    静默退回 26 维会让「重训后指标没变化」这类问题极难定位。
    """
    df = pd.read_parquet(path)
    if df.empty or len(df) < RISK_HORIZON + 2:
        return None
    df = df.sort_values("_timestamp").reset_index(drop=True)
    ts = df["_timestamp"].astype(str).tolist()
    X = df[FEATURES].to_numpy(dtype=np.float64)
    names = list(FEATURES)

    if underlying:
        up = os.path.join(os.path.dirname(path), "underlying.parquet")
        if not os.path.exists(up):
            raise FileNotFoundError(
                f"{symbol} {ym}: 缺 underlying.parquet——请先跑 "
                f"`python scripts/build_underlying.py --stage all`")
        ud = pd.read_parquet(up).sort_values("_timestamp").reset_index(drop=True)
        if len(ud) != len(df) or ud["_timestamp"].astype(str).tolist() != ts:
            raise ValueError(f"{symbol} {ym}: underlying.parquet 与 features.parquet "
                             f"时间戳不对齐（{len(ud)} vs {len(df)}）")
        X = np.hstack([X, ud[UNDERLYING].to_numpy(dtype=np.float64)])
        names = names + list(UNDERLYING)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_stored = df["y"].to_numpy(dtype=np.int64)

    risk_idx = identify_risk_starts(ts, df["atm_iv"].to_numpy(),
                                    df["convexity_violation"].to_numpy())
    if verify:
        y_rebuilt = labels_from_risk_starts(len(ts), risk_idx)
        if not np.array_equal(y_rebuilt, y_stored):
            bad = int((y_rebuilt != y_stored).sum())
            raise ValueError(
                f"{symbol} {ym}: 重算标签与存档 y 不一致（{bad}/{len(ts)} 行）——"
                f"风险起点判定与赛题口径不对齐，拒绝继续训练")

    dt = pd.to_datetime(df["_timestamp"].astype(str), format="%Y%m%d%H%M%S")
    return Episode(symbol=symbol, ym=ym, ts=ts, X=X, y=y_stored,
                   risk_idx=np.array(risk_idx, dtype=np.int64),
                   lead_min=_lead_minutes(dt, risk_idx),
                   dt=dt.to_numpy(), feat_names=names)


def load_episodes(anchor_dir: str, symbols: list | None = None,
                  verify: bool = True, underlying: bool = False) -> list:
    """加载全部 (品种, 月) episode，按 (ym, symbol) 排序。

    symbols=None 时**自动发现** anchor 目录下的所有品种，而不是写死一份名单——
    早前这里硬编码了 ["ag","au","sc","si"]，扩样到 7 个品种后如果忘了同步，
    新数据会被静默忽略、指标看着"没变化"，属于最难察觉的一类 bug。
    """
    if symbols is None:
        symbols = sorted(
            os.path.basename(d.rstrip("/"))
            for d in glob.glob(os.path.join(anchor_dir, "*/"))
            if os.path.isdir(d) and not os.path.basename(d.rstrip("/")).startswith("_")
        )
    eps = []
    for sym in symbols:
        for d in sorted(glob.glob(os.path.join(anchor_dir, sym, "*/"))):
            ym = os.path.basename(d.rstrip("/"))
            p = os.path.join(d, "features.parquet")
            if not os.path.exists(p):
                continue
            e = load_episode(p, sym, ym, verify=verify, underlying=underlying)
            if e is not None:
                eps.append(e)
    eps.sort(key=lambda e: (e.ym, e.symbol))
    return eps


# ------------------------------------------------------------------ 切分/归一

def split_episodes(eps: list, train_end: str = "2024-12",
                   val_end: str = "2025-06") -> tuple:
    """按月份先后严格切分，不打乱、不交叉——测试集全部晚于训练集。"""
    tr = [e for e in eps if e.ym <= train_end]
    va = [e for e in eps if train_end < e.ym <= val_end]
    te = [e for e in eps if e.ym > val_end]
    return tr, va, te


# ------------------------------------------------------ 跨月连续的风险起点

def apply_continuous_risk(eps: list) -> dict:
    """把风险起点判定改为**按品种跨月连续**计算，就地改写各 Episode。

    为什么必须这么做
    ----------------
    赛题的风险区间定义是「超过**过去 20 个交易日**的 95 分位」。原管线
    (`build_anchor_dataset` 第 178 行) 对每个 (品种, 月) 单独调用
    `identify_risk_windows`，滚动窗口每月**重置**——而同一个脚本对特征却做了
    跨月预热 (`_load_history_priors`)。标签与特征口径不一致，是实现缺陷，
    不是赛题要求。

    后果是量化过的：`_rolling_q95` 的 `min_periods=80`，故每幕**前 72 根截面
    在结构上不可能有正标签**（实测 59 幕测试集里这片区域的风险起点数 = 0/1534）。
    它占 14.1% 的截面，却吃掉 DRL 全部误报的 20.2%、规则基线的 21.9%——
    在那里发的预警**必然**是误报，与模型好坏无关。

    改为连续后，冷启动只在每个品种的**首幕**出现一次（7 个品种共 ~560 根，
    占全样本 0.4%），而非每月一次。`lead_min` 同样在连续时间轴上算再切回各幕，
    这样月末的预警也能被下月初的风险起点正确计入——与线上实况一致。
    跨幕的时间间隔由 `_lead_minutes` 的真实分钟差自然处理（跨周末超 120min 即无效）。

    返回改动统计，供报告引用。
    """
    by_sym = {}
    for e in eps:
        by_sym.setdefault(e.symbol, []).append(e)

    stat = {"symbols": 0, "risk_before": 0, "risk_after": 0, "episodes": 0}
    i_iv, i_cx = FEATURES.index("atm_iv"), FEATURES.index("convexity_violation")

    for sym, group in by_sym.items():
        group.sort(key=lambda e: e.ym)
        stat["risk_before"] += sum(len(e.risk_idx) for e in group)

        ts_all, iv_all, cx_all = [], [], []
        for e in group:
            ts_all.extend(e.ts)
            iv_all.append(e.X[:, i_iv])
            cx_all.append(e.X[:, i_cx])
        iv_all = np.concatenate(iv_all)
        cx_all = np.concatenate(cx_all)

        risk_all = identify_risk_starts(ts_all, iv_all, cx_all)
        y_all = labels_from_risk_starts(len(ts_all), risk_all)
        dt_all = pd.to_datetime(pd.Series(ts_all), format="%Y%m%d%H%M%S")
        lead_all = _lead_minutes(dt_all, risk_all)
        rs = np.array(risk_all, dtype=np.int64)

        off = 0
        for e in group:
            n = len(e)
            sl = slice(off, off + n)
            local = rs[(rs >= off) & (rs < off + n)] - off
            e.risk_idx = local.astype(np.int64)
            e.y = y_all[sl].copy()
            e.lead_min = lead_all[sl].copy()
            off += n
            stat["episodes"] += 1
        stat["risk_after"] += len(risk_all)
        stat["symbols"] += 1

    return stat


class Normalizer:
    """z-score 归一化，统计量**只在训练集上**估计，再套用到 val/test。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray, clip: float = 5.0):
        self.mean, self.std, self.clip = mean, std, clip

    @classmethod
    def fit(cls, eps: list, clip: float = 5.0) -> "Normalizer":
        X = np.concatenate([e.X for e in eps], axis=0)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0
        return cls(mean, std, clip)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean) / self.std, -self.clip, self.clip)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "clip": self.clip}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.array(d["mean"]), np.array(d["std"]), d.get("clip", 5.0))
