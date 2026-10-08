"""cleaning — 曲面数据清洗。

基于对数据集的实测发现，每个时间截面里存在以下系统性质量问题，需在插值/特征
之前清除，否则会污染曲面形状与风险指标：

1. 休市段：suspend_flag == 1（实测约 7% 行）
2. 僵尸合约：volume == 0 且 open_interest == 0，且 close == pre_close（实测大量
   深度 OTM 合约长时间无成交、价格卡死）
3. IV 边界失效：iv <= lo 或 iv >= hi（实测深度 ITM 的 iv 频繁 >1.0 甚至到 5.0，
   系 BS 数值不稳定；iv == 0.001 是底值占位）
4. Greeks 卡死：iv 极小且 |delta|==1 且 gamma==0 且 vega==0，说明 BS 反推已退化
5. 邻居跳变：同一截面按 moneyness 排序后，IV 一阶差分超 k×MAD 视为离群跳变
   （实测 ag2306C4850 iv=0.489 与邻居 0.80 严重偏离）
6. 陈价过久（可选，默认关闭）：价格连续多根 K 线未变，IV 实为过期报价。
   见 `annotate_staleness` 的说明——原「僵尸合约」规则因 pre_close 是逐根 K 线口径
   而几乎从不触发，这一维度此前是缺失的。

清洗过程记录每一步剔除的条数，便于审计与可解释性。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CleaningReport:
    """记录每个清洗步骤剔除的条数与原因。"""

    n_in: int = 0
    dropped_suspended: int = 0
    dropped_inactive: int = 0
    dropped_iv_bound: int = 0
    dropped_dead_greeks: int = 0
    dropped_stale: int = 0
    dropped_neighbor_jump: int = 0
    by_reason: dict = field(default_factory=dict)
    # 逐行剔除原因 {输入df的索引: [reason_code, ...]}，仅当 clean_slice(with_reasons=True)
    # 时填充；被保留的行为空列表。默认留空以免给 21 处既有调用增加开销。
    reason_codes: dict = field(default_factory=dict)

    @property
    def n_out(self) -> int:
        return self.n_in - (
            self.dropped_suspended
            + self.dropped_inactive
            + self.dropped_stale
            + self.dropped_iv_bound
            + self.dropped_dead_greeks
            + self.dropped_neighbor_jump
        )

    def as_dict(self) -> dict:
        return {
            "n_in": self.n_in,
            "n_out": self.n_out,
            "dropped_suspended": self.dropped_suspended,
            "dropped_inactive": self.dropped_inactive,
            "dropped_stale": self.dropped_stale,
            "dropped_iv_bound": self.dropped_iv_bound,
            "dropped_dead_greeks": self.dropped_dead_greeks,
            "dropped_neighbor_jump": self.dropped_neighbor_jump,
        }

    def summary(self) -> str:
        return (
            f"cleaning: in={self.n_in} out={self.n_out} "
            f"(suspend={self.dropped_suspended} inactive={self.dropped_inactive} "
            f"stale={self.dropped_stale} "
            f"iv_bound={self.dropped_iv_bound} dead_greeks={self.dropped_dead_greeks} "
            f"neighbor_jump={self.dropped_neighbor_jump})"
        )


def _is_dead_greeks(df: pd.DataFrame) -> pd.Series:
    """Greeks 卡死：iv 极小、delta 触 ±1、gamma/vega 归零。"""
    return (
        (df["iv"] <= 0.005)
        & (df["delta"].abs() >= 0.999)
        & (df["gamma"].abs() <= 1e-9)
        & (df["vega"].abs() <= 1e-6)
    )


def _drop_neighbor_jumps(group: pd.DataFrame, k: float) -> pd.DataFrame:
    """对单个 (option_type, dte) 组按 moneyness 排序后剔除 IV 离群跳变。

    IV 微笑/期限结构在 ATM 附近呈 V 形，一阶差分本就很大，是真实形态而非噪声。
    因此只剔除**单调段内的逆向离群跳变**：先估计局部趋势方向（窗口内中位差分
    的符号），再标记与趋势方向相反且幅度超 k×MAD 的点。V 形拐点（方向连续反转
    的连续段）予以保留，不误杀微笑形态。

    返回保留行的布尔索引（与 group 索引对齐）。
    """
    g = group.sort_values("moneyness")
    iv = g["iv"].to_numpy()
    n = iv.size
    if n < 4:
        return pd.Series(True, index=group.index)
    d = np.diff(iv, prepend=iv[0])
    abs_d = np.abs(d)
    med = np.median(abs_d)
    mad = np.median(np.abs(abs_d - med)) if med > 0 else 0.0
    scale = 1.4826 * mad if mad > 0 else (np.std(d) if np.std(d) > 0 else 0.0)
    if scale <= 0:
        return pd.Series(True, index=group.index)
    # 局部趋势方向：窗口内中位差分的符号（+1 上升 / -1 下降）
    half = max(2, n // 4)
    signs = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        med_d = np.median(d[lo:hi])
        signs[i] = 1.0 if med_d >= 0 else -1.0
    # 仅当与局部趋势方向相反且超阈时判为离群；V 形拐点两侧符号本就相反，不在此列
    against = np.sign(d) == -signs
    outlier = against & (abs_d > k * scale)
    keep = ~outlier
    return pd.Series(keep, index=g.index).reindex(group.index)


def annotate_staleness(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """给整段数据标注每个 (合约, K线) 的**陈价年龄**：价格已连续多少根 K 线未变。

    为什么需要这个维度
    ------------------
    原清洗把「僵尸合约」定义为 `volume==0 且 open_interest==0 且 close==pre_close`
    三条同时成立。但实测发现 `pre_close` 是**上一根 K 线**的收盘（98% 吻合），
    因此 `close==pre_close` 只说明这 15 分钟内没成交——商品期权绝大多数合约本就
    不是每根 K 线都有成交，该条件几乎恒真（实测保留行里 79.6% 满足），
    而 `open_interest==0` 又极少成立，导致这条规则实际上几乎从不触发。

    真正有害的不是「这一根没成交」，而是「已经很久没成交」：实测 ag 2025-08 全月
    价格连续未变根数的中位数是 14 根（约 3.5 小时）、90 分位 34 根（约一个交易日）。
    这类合约的 IV 是隔日陈价，拿去拟合曲面会把过期形态带进当前截面。

    本函数按合约分组、沿时间累计「价格未变的连续根数」，输出列 `stale_age`
    （0 表示本根有价格变化）。需在 `clean_slice` 之前对整段数据调用一次。
    """
    if price_col not in df.columns or "contract" not in df.columns:
        return df
    d = df.sort_values(["contract", "timestamp"]).copy()
    p = d[price_col].to_numpy()
    cid = d["contract"].to_numpy()
    age = np.zeros(len(d), dtype=np.int32)
    for i in range(1, len(d)):
        if cid[i] == cid[i - 1] and p[i] == p[i - 1]:
            age[i] = age[i - 1] + 1
    d["stale_age"] = age
    return d.reindex(df.index) if df.index.equals(d.index) else d


def clean_slice(
    df: pd.DataFrame,
    *,
    iv_lo: float = 0.01,
    iv_hi: float = 3.0,
    jump_k: float = 6.0,
    max_stale_bars: int | None = None,
    drop_suspended: bool = True,
    drop_inactive: bool = True,
    drop_dead_greeks: bool = True,
    drop_neighbor_jumps: bool = True,
    with_reasons: bool = False,
) -> tuple[pd.DataFrame, CleaningReport]:
    """清洗一个时间截面（或整段）的期权数据。

    顺序：suspend -> inactive -> stale -> iv bound -> dead greeks -> neighbor jumps。
    每步后状态立即用于下一步，保证后续判断基于已清洗子集。

    Parameters
    ----------
    iv_lo, iv_hi : IV 有效区间，超出视为 BS 反推失效
    jump_k : 邻居跳变剔除的 MAD 倍数阈值（默认 6，较宽松以免误杀微笑形态）
    max_stale_bars : 陈价年龄上限。为 None 时不启用（保持与原有行为完全一致）；
        给定整数 N 时，剔除价格已连续 N 根以上 K 线未变的合约。
        需先用 `annotate_staleness` 给数据加 `stale_age` 列，否则该项静默跳过。
    with_reasons : 额外在 `CleaningReport.reason_codes` 里给出**逐行**剔除原因
        （被保留的行为空列表），索引与**输入** df 对齐。**默认关闭**。

        为什么加这个（借鉴自另一实现的 `quality_flags.annotate_quality`）
        ------------------------------------------------------------------
        本函数返回的是**已删减**的 DataFrame，`CleaningReport` 只有各步的汇总条数。
        于是「这一条为什么被删」在返回值里查不到——排查某个截面为何合约骤减时
        只能靠改代码打印。对方的做法是**只打标不删行**，把判定写成布尔列 +
        `quality_reason_codes`，下游自行选档，逐行可追溯。

        本项目有 21 处调用依赖现有的两元组返回，改签名代价过大且易引入回归，
        故改为**可选开关**：默认路径逐位不变（已由 `tests/test_cleaning_reasons.py`
        断言），需要追溯时按需打开。
    """
    rep = CleaningReport(n_in=len(df))
    out = df
    # 逐行原因：以**输入** df 的索引为键，仅在 with_reasons 时维护
    reasons: dict = {i: [] for i in df.index} if with_reasons else None

    def _mark(mask, code):
        """把一步的剔除原因记到逐行表上。mask 的索引是当前 out 的子集。"""
        if reasons is None:
            return
        for i in mask[mask].index:
            reasons[i].append(code)

    # 1. 休市
    if drop_suspended:
        mask = out["suspend_flag"].fillna(0).astype(int) == 1
        rep.dropped_suspended = int(mask.sum())
        _mark(mask, "suspended")
        out = out[~mask]

    # 2. 僵尸合约：零成交且零持仓且价格未动（close==pre_close）
    if drop_inactive:
        mask = (
            (out["volume"].fillna(0) == 0)
            & (out["open_interest"].fillna(0) == 0)
            & (out["close"] == out["pre_close"])
        )
        rep.dropped_inactive = int(mask.sum())
        _mark(mask, "inactive_zombie")
        out = out[~mask]

    # 2.5 陈价过久：价格连续多根 K 线未变，IV 实为过期报价
    if max_stale_bars is not None and "stale_age" in out.columns:
        mask = out["stale_age"] > int(max_stale_bars)
        rep.dropped_stale = int(mask.sum())
        _mark(mask, "stale_price")
        out = out[~mask]

    # 3. IV 边界
    mask = (out["iv"] <= iv_lo) | (out["iv"] >= iv_hi)
    rep.dropped_iv_bound = int(mask.sum())
    _mark(mask, "iv_out_of_bounds")
    out = out[~mask]

    # 4. Greeks 卡死（在 IV 已过滤的基础上再确认）
    if drop_dead_greeks:
        mask = _is_dead_greeks(out)
        rep.dropped_dead_greeks = int(mask.sum())
        _mark(mask, "dead_greeks")
        out = out[~mask]

    # 5. 邻居跳变：在 (option_type, dte) 分组内按 moneyness 剔除离群
    if drop_neighbor_jumps and len(out) > 0:
        keep_parts = []
        for (_ot, _dte), grp in out.groupby(["option_type", "dte"], sort=False):
            keep_parts.append(_drop_neighbor_jumps(grp, jump_k))
        keep = pd.concat(keep_parts).reindex(out.index).fillna(True).astype(bool)
        rep.dropped_neighbor_jump = int((~keep).sum())
        _mark(~keep, "neighbor_jump")
        out = out[keep]

    if reasons is not None:
        rep.reason_codes = reasons
    return out.reset_index(drop=True), rep
