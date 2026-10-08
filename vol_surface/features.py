"""features — 曲面风险特征工程。

每个时间截面 -> 一个特征向量。设计目标是把赛题明确点名的风险指标全部落到
可量化、可滚动统计、可喂给规则/ML 的标量：

1. 曲面凸性违反程度 convexity_violation   —— 蝶式二阶差分<0 的数量×幅度
2. 偏度分位数 skew_percentile             —— 25-delta skew 的历史分位
3. 期限结构斜率变化率 term_slope_roc       —— (IV_far-IV_near)/IV_near 的一阶变化
4. Gamma 截面集中度 gamma_concentration    —— Gamma×OI 的 HHI
5. Vega 截面集中度 vega_concentration      —— Vega×OI 的 HHI
6. ATM IV 水平 / z-score
7. IV 短期变化率 / 急升幅度
8. 曲面拟合退化（SVI rmse 综合）
9. 无套利违反综合分
10. 流动性退化指标

滚动统计由 FeatureHistory 维护，支持 z-score 化与分位计算。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .arbitrage import check_arbitrage
from .interpolation import fit_slice_iv


def _hhi(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index 集中度：sum(p_i^2)，p_i=权重归一化。1=完全集中。"""
    w = np.asarray(weights, float)
    w = w[w > 0]
    if w.size == 0:
        return 0.0
    p = w / w.sum()
    return float(np.sum(p**2))


def _pct(x: float, history: np.ndarray) -> float:
    """x 在 history 中的分位（0~1）。"""
    h = history[np.isfinite(history)]
    if h.size == 0:
        return 0.5
    return float(np.searchsorted(np.sort(h), x) / h.size)


@dataclass
class FeatureHistory:
    """滚动窗口维护各特征的历史，用于 z-score 与分位。

    职责边界（重要）：
      - 本类只管"存"与"查"——维护 maxlen 滚动缓冲，提供 zscore / percentile /
        velocity / velocity_z 的查询接口。
      - 决定"何时 push"的职责由调用方（compute_features 或 replay.FHReplay）
        承担：push 前算的特征（atm_iv_z / liquidity_z / skew_percentile）不含
        当前点；push 后算的特征（convexity_violation_z / arb_score_z /
        velocity_z）含当前点。
      - 不要在评估路径里"先 push 再覆盖"或反复 push 同一截面；一个截面
        只 push 一次，且要在所有 push 前特征已经入库之后。
      - 冷启动（history 历史少于 5 个） zscore 返回 0.0；不足 5 个也得 0.0
        （由 zscore 的实现条款保证）。
    """

    window: int = 500
    _buf: dict = field(default_factory=dict)

    def push(self, feats: dict) -> None:
        for k, v in feats.items():
            if k in ("timestamp", "symbol", "date", "level", "triggers", "top_features"):
                continue
            if not isinstance(v, (int, float, np.floating, np.integer)) or not np.isfinite(v):
                continue
            if k not in self._buf:
                self._buf[k] = deque(maxlen=self.window)
            self._buf[k].append(float(v))

    def history(self, key: str) -> np.ndarray:
        return np.array(self._buf.get(key, []), float)

    def zscore(self, key: str, value: float) -> float:
        h = self.history(key)
        if h.size < 5:
            return 0.0
        mu, sd = float(np.mean(h)), float(np.std(h))
        return 0.0 if sd < 1e-9 else float((value - mu) / sd)

    def percentile(self, key: str, value: float) -> float:
        return _pct(value, self.history(key))

    def velocity(self, key: str, n: int = 4) -> float:
        """最近 n 个截面的变化速率（末值-首值）。用于领先于水平的导数预警：
        IV 加速上升时 velocity 先变大，早于 IV 本身突破阈值。"""
        h = self.history(key)
        if h.size < n:
            return 0.0
        return float(h[-1] - h[-n])

    def velocity_z(self, key: str, n: int = 4) -> float:
        """velocity 的 z-score（相对历史的滑动差分）。

        语义契约（与全项目基线一致，重放器必须逐 bit 保留）：
          - 当前点自身的 velocity 参与 z-score 分母（在 meh 里当最后一个样本）。
          - 看 push 后、再算 z 的特征，当前点计入历史；push 前算的（atm_iv_z
            /liquidity_z/skew_percentile）不计入。
          这不是 bug 而是校准基线的口径；改动必须带来指标逐点 diff。
        """
        h = self.history(key)
        if h.size < n + 5:
            return 0.0
        win_diffs = np.array([h[i + n] - h[i]
                              for i in range(max(0, h.size - self.window),
                                             h.size - n)])
        if win_diffs.size < 5:
            return 0.0
        mu, sd = float(np.mean(win_diffs)), float(np.std(win_diffs))
        cur = float(h[-1] - h[-n])
        return 0.0 if sd < 1e-9 else float((cur - mu) / sd)


def compute_features(
    clean: pd.DataFrame,
    iv_result: dict | None = None,
    prev_features: dict | None = None,
    history: FeatureHistory | None = None,
    lightweight: bool = False,
) -> dict:
    """计算单截面的风险特征向量。

    Parameters
    ----------
    clean : 清洗后的截面 DataFrame
    iv_result : fit_slice_iv 的输出；为 None 时内部调用（lightweight 时跳过 SVI）
    prev_features : 上一截面的特征，用于计算变化率（term_slope_roc 等）
    history : FeatureHistory，用于 z-score/分位；会原地 push 本次特征
    lightweight : 跳过 SVI 拟合，只算 PCHIP 重建 + 必要特征，用于赛题口径量化
        评测这类需连续跑大量截面的场景（速度约 3 倍）。SVI 参数特征置 NaN。
    """
    if iv_result is None:
        iv_result = fit_slice_iv(clean, with_svi=not lightweight)
    arb = check_arbitrage(iv_result)
    dte_grid = iv_result["dte_grid"]
    mgrid = iv_result["moneyness_grid"]
    call_surf = iv_result["iv_surface"].get("call", np.full((dte_grid.size, mgrid.size), np.nan))
    put_surf = iv_result["iv_surface"].get("put", np.full((dte_grid.size, mgrid.size), np.nan))

    # ATM 索引（moneyness≈1.0）
    atm = int(np.argmin(np.abs(mgrid - 1.0)))
    otm25c = int(np.argmin(np.abs(mgrid - 1.05)))  # 5% OTM call
    otm25p = int(np.argmin(np.abs(mgrid - 0.95)))  # 5% OTM put

    # 临到期调整：当存在 dte<7 的近月时，近月 IV 极陡是 BS 机制性常态，
    # 凸性违反/期限斜率的"异常"部分按近月占比降权，避免临到期常态化误报。
    real_dtes = sorted({d for d in clean["dte"].astype(float).unique()})
    near_dtes = [d for d in real_dtes if d < 7]
    expiry_factor = 0.35 if near_dtes else 1.0  # 临到期时降权 65%

    # --- 1. 凸性违反程度（蝶式二阶差分<0 的数量×平均幅度）---
    def _convexity(surf):
        if surf.shape[1] < 3:
            return 0.0
        w = np.clip(surf, 0, 3) ** 2
        d2 = w[:, :-2] - 2 * w[:, 1:-1] + w[:, 2:]
        bad = d2 < 0
        n = int(np.nansum(bad))
        mag = float(np.nanmean(-d2[bad])) if n else 0.0
        return float(n * mag)

    convexity_violation = max(_convexity(call_surf), _convexity(put_surf)) * expiry_factor

    # --- 2. 偏度分位数：25-delta skew = IV_otm_put - IV_otm_call（ATM 附近）---
    near_idx = 0  # 最近月
    put_v = float(np.nanmean(put_surf[near_idx, otm25p])) if np.isfinite(put_surf[near_idx, otm25p]).any() else 0.0
    call_v = float(np.nanmean(call_surf[near_idx, otm25c])) if np.isfinite(call_surf[near_idx, otm25c]).any() else 0.0
    skew_val = put_v - call_v
    if history is not None:
        skew_percentile = history.percentile("skew_val", skew_val)
    else:
        skew_percentile = 0.5

    # --- 3. 期限结构斜率：(IV_far - IV_near) / IV_near ---
    iv_near = float(np.nanmean(call_surf[0, atm]))
    iv_far = float(np.nanmean(call_surf[-1, atm]))
    term_slope = float((iv_far - iv_near) / (iv_near + 1e-6))
    if prev_features and "term_slope" in prev_features:
        term_slope_roc = float((term_slope - prev_features["term_slope"]) * expiry_factor)
    else:
        term_slope_roc = 0.0

    # --- 4/5. Gamma / Vega 截面集中度（HHI of Greek×OI）---
    g = clean.copy()
    g["gamma_oi"] = g["gamma"].abs() * g["open_interest"].fillna(0).astype(float)
    g["vega_oi"] = g["vega"].abs() * g["open_interest"].fillna(0).astype(float)
    gamma_concentration = _hhi(g["gamma_oi"].to_numpy())
    vega_concentration = _hhi(g["vega_oi"].to_numpy())

    # --- 6. ATM IV 水平（取中段 dte 的 ATM IV 中位，剔除近月临到期极端值）---
    # 近月 dte 切片临到期时 ATM IV 会被 BS 反推抬到 0.8+，不能代表曲面主体水平；
    # 取 dte_grid 中段（去掉最近 1 个与最远 1 个切片）的 ATM IV 中位数作为"整体水平"。
    if dte_grid.size >= 4:
        mid_atm = call_surf[1:-1, atm]
        atm_iv = float(np.nanmedian(mid_atm)) if np.isfinite(mid_atm).any() else float(np.nanmean(call_surf[:, atm]))
    else:
        atm_iv = float(np.nanmean(call_surf[:, atm]))
    atm_iv_z = history.zscore("atm_iv", atm_iv) if history else 0.0

    # --- 7. IV 短期变化率 ---
    if prev_features and "atm_iv" in prev_features:
        iv_change = float(atm_iv - prev_features["atm_iv"])
        iv_change_rate = float(iv_change / (prev_features["atm_iv"] + 1e-6))
    else:
        iv_change = 0.0
        iv_change_rate = 0.0
    iv_spike = max(abs(iv_change), abs(term_slope_roc) * iv_near) if prev_features else 0.0

    # --- 8. 曲面拟合退化（SVI rmse 综合）---
    svi_rmses = [f.rmse for f in iv_result["svi"].get("call", []) + iv_result["svi"].get("put", []) if np.isfinite(f.rmse)]
    fit_degradation = float(np.nanmean(svi_rmses)) if svi_rmses else 0.0

    # --- 9. 无套利违反综合分 ---
    arb_score = (
        arb.calendar_violations * max(arb.calendar_avg_mag, 0.01)
        + arb.butterfly_violations * max(arb.butterfly_avg_mag, 0.01)
        + arb.parity_violations * max(arb.parity_avg_mag, 0.05)
    )

    # --- 10. 流动性退化：活跃合约占比 ---
    n_active = int((clean["volume"].fillna(0) > 0).sum()) + int((clean["open_interest"].fillna(0) > 0).sum())
    liquidity_ratio = float(n_active / (2 * max(len(clean), 1)))
    liquidity_z = history.zscore("liquidity_ratio", liquidity_ratio) if history else 0.0

    feats = {
        # 赛题点名指标
        "convexity_violation": convexity_violation,
        "skew_percentile": float(skew_percentile),
        "skew_val": skew_val,
        "term_slope": term_slope,
        "term_slope_roc": term_slope_roc,
        "gamma_concentration": gamma_concentration,
        "vega_concentration": vega_concentration,
        # 补充指标
        "atm_iv": atm_iv,
        "atm_iv_z": atm_iv_z,
        "iv_change": iv_change,
        "iv_change_rate": iv_change_rate,
        "iv_spike": iv_spike,
        "fit_degradation": fit_degradation,
        "arb_score": arb_score,
        "arb_calendar_n": arb.calendar_violations,
        "arb_butterfly_n": arb.butterfly_violations,
        "arb_parity_n": arb.parity_violations,
        "liquidity_ratio": liquidity_ratio,
        "liquidity_z": liquidity_z,
        "iv_near": iv_near,
        "iv_far": iv_far,
        "n_contracts": int(len(clean)),
    }
    # SVI 参数特征（取近月 call）
    svi_call = iv_result["svi"].get("call", [])
    if svi_call:
        f0 = svi_call[0]
        feats.update({
            "svi_rho": f0.rho, "svi_b": f0.b, "svi_sigma": f0.sigma,
            "svi_m": f0.m, "svi_rmse": f0.rmse,
        })
    if history is not None:
        # 先 push 基础值，再补算相对历史的 z-score 与导数特征（供规则做自适应阈值与提前预警）
        history.push(feats)
        feats["convexity_violation_z"] = history.zscore("convexity_violation", convexity_violation)
        feats["arb_score_z"] = history.zscore("arb_score", arb_score)
        # 导数（velocity）特征：领先于水平，用于提前预警、降低事件期全程响的密集预警
        feats["atm_iv_vel_z"] = history.velocity_z("atm_iv", n=4)
        feats["convexity_vel_z"] = history.velocity_z("convexity_violation", n=4)
    else:
        feats["convexity_violation_z"] = 0.0
        feats["arb_score_z"] = 0.0
        feats["atm_iv_vel_z"] = 0.0
        feats["convexity_vel_z"] = 0.0
    return feats


FEATURE_COLUMNS = [
    "convexity_violation", "skew_percentile", "skew_val", "term_slope", "term_slope_roc",
    "gamma_concentration", "vega_concentration", "atm_iv", "atm_iv_z", "iv_change",
    "iv_change_rate", "iv_spike", "fit_degradation", "arb_score", "arb_calendar_n",
    "arb_butterfly_n", "arb_parity_n", "liquidity_ratio", "liquidity_z",
    "iv_near", "iv_far", "n_contracts", "svi_rho", "svi_b", "svi_sigma", "svi_m", "svi_rmse",
    "convexity_violation_z", "arb_score_z", "atm_iv_vel_z", "convexity_vel_z",
]
