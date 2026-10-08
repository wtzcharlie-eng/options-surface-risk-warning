"""interpolation — IV 曲面插值与参数化拟合。

商品期权每个时间截面上，到期月数量有限（实测 6~10 个）、每个到期月下行权价
间距固定。为得到连续的 IV(dte, moneyness) 曲面供特征计算与无套利校验，本
模块采用两套机制：

1. **曲面重建（主路径）**：稳健插值，不依赖易陷入局部极小的非线性拟合。
   - moneyness 维度：PCHIP 单调三次 Hermite 插值（保形、不振荡、不过冲），
     正确还原微笑形态。
   - dte 维度：基于 sqrt(T) 的方差线性插值（符合 w∝T 的无套利期限结构），仅
     内插不外推，避免近月 dte=1 的极端曲线污染远月网格。
2. **SVI 参数化（特征提取）**：对每个 (option_type, dte) 切片拟合 SVI 五参数
   [a, b, rho, m, sigma]，作为偏度/倾斜/曲率的可解释特征。SVI 不要求完美重建
   曲面，只取其参数；重建由 PCHIP 负责。

输出规则网格 IV(dte_grid, moneyness_grid) + 每切片 SVI 参数与拟合质量。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, RBFInterpolator
from scipy.optimize import least_squares

DEFAULT_DTE_GRID = np.array([7.0, 14.0, 30.0, 60.0, 90.0, 180.0])
DEFAULT_MONEYNESS_GRID = np.round(np.arange(0.85, 1.151, 0.01), 3)


@dataclass
class SVIFit:
    """单个 (option_type, dte) 切片的 SVI 拟合结果（用于特征提取）。"""

    option_type: str
    dte: float
    params: np.ndarray  # [a, b, rho, m, sigma]
    rmse: float
    n_points: int
    ok: bool

    @property
    def a(self) -> float:
        return float(self.params[0])

    @property
    def b(self) -> float:
        return float(self.params[1])

    @property
    def rho(self) -> float:
        return float(self.params[2])

    @property
    def m(self) -> float:
        return float(self.params[3])

    @property
    def sigma(self) -> float:
        return float(self.params[4])

    def predict(self, k: np.ndarray) -> np.ndarray:
        return np.sqrt(np.clip(_svi(self.params, k), 0.0, 25.0))

    def as_dict(self) -> dict:
        return {
            "option_type": self.option_type,
            "dte": self.dte,
            "svi_a": self.a,
            "svi_b": self.b,
            "svi_rho": self.rho,
            "svi_m": self.m,
            "svi_sigma": self.sigma,
            "svi_rmse": self.rmse,
            "svi_n": self.n_points,
            "svi_ok": self.ok,
        }


def _svi(params, k):
    a, b, rho, m, sigma = params
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma**2))


def _residual(params, k, iv):
    """IV 空间残差：sqrt(svi) - iv。"""
    return np.sqrt(np.clip(_svi(params, k), 0.0, 25.0)) - iv


def _fit_svi(k: np.ndarray, iv: np.ndarray) -> SVIFit | None:
    """对单切片做 SVI 拟合，提取五参数特征（不强求完美重建）。"""
    w0 = iv**2
    a0 = float(np.clip(np.median(w0), 1e-4, 5.0))
    dk = k[-1] - k[0]
    slope = (w0[-1] - w0[0]) / (dk + 1e-6)
    b0 = max(1e-3, min(2.0, abs(slope) + 0.01))
    rho0 = float(np.clip(slope / (b0 + 1e-6), -0.9, 0.9)) if b0 > 1e-3 else 0.0
    m0 = float(np.median(k))
    sigma0 = max(1e-3, 0.3 * (dk + 1e-6))
    x0 = np.array([a0, b0, rho0, m0, sigma0])
    lb = [1e-5, 1e-6, -0.999, k.min() - 1.0, 1e-3]
    ub = [5.0, 2.0, 0.999, k.max() + 1.0, 2.0]
    try:
        res = least_squares(_residual, x0, bounds=(lb, ub), args=(k, iv), max_nfev=2000, xtol=1e-9)
    except Exception:
        return None
    rmse = float(np.sqrt(np.mean(res.fun**2))) if res.fun.size else float("nan")
    ok = bool(res.success) and np.isfinite(rmse)
    return SVIFit(params=res.x, rmse=rmse, n_points=len(k), ok=ok, option_type="", dte=0.0)


def _interp_moneyness(k_sorted, iv_sorted, grid):
    """PCHIP 单调插值，外推用边界值。x 去重保证严格递增。"""
    if k_sorted.size < 2:
        return np.full(grid.size, np.nan)
    # 去重：重复 moneyness 取均值，保证 x 严格递增
    k_sorted = np.asarray(k_sorted, float)
    iv_sorted = np.asarray(iv_sorted, float)
    order = np.argsort(k_sorted)
    k_sorted, iv_sorted = k_sorted[order], iv_sorted[order]
    k_uniq, inv = np.unique(k_sorted, return_inverse=True)
    if k_uniq.size != k_sorted.size:
        iv_uniq = np.array([iv_sorted[inv == j].mean() for j in range(k_uniq.size)])
        k_sorted, iv_sorted = k_uniq, iv_uniq
    if k_sorted.size < 2:
        return np.full(grid.size, np.nan)
    pchip = PchipInterpolator(k_sorted, iv_sorted, extrapolate=False)
    pred = pchip(grid)
    pred = np.where(np.isnan(pred), np.interp(grid, k_sorted, iv_sorted), pred)
    return np.clip(pred, 0.0, 3.0)


def fit_slice_iv(
    df: pd.DataFrame,
    dte_grid: np.ndarray | None = None,
    moneyness_grid: np.ndarray | None = None,
    with_svi: bool = True,
) -> dict:
    """对一个时间截面重建 IV 曲面 + 提取 SVI 特征。

    with_svi=False 时跳过 SVI 拟合（仅 PCHIP 重建），用于只需曲面形态、不需要 SVI
    参数的场景（如赛题口径的量化指标评测），可显著加速。
    """
    dte_grid = DEFAULT_DTE_GRID if dte_grid is None else np.asarray(dte_grid, float)
    mgrid = DEFAULT_MONEYNESS_GRID if moneyness_grid is None else np.asarray(moneyness_grid, float)
    out = {"iv_surface": {}, "svi": {}, "raw_curves": {}, "dte_grid": dte_grid,
           "moneyness_grid": mgrid, "fit_ok": True}

    for ot in ["call", "put"]:
        sub = df[df["option_type"] == ot]
        surface = np.full((dte_grid.size, mgrid.size), np.nan)
        svi_list = []
        raw = []
        if len(sub) == 0:
            out["iv_surface"][ot] = surface
            out["svi"][ot] = []
            out["raw_curves"][ot] = []
            out["fit_ok"] = False
            continue
        # 每个真实 dte 切片：PCHIP 求曲线 + SVI 提参数
        real_dtes = np.sort(sub["dte"].astype(float).unique())
        slice_curves = []  # [(dte, curve_on_mgrid)]
        for d in real_dtes:
            g = sub[sub["dte"].astype(float) == d].sort_values("moneyness")
            k = g["moneyness"].to_numpy(float)
            iv = g["iv"].to_numpy(float)
            if k.size < 2:
                continue
            curve = _interp_moneyness(k, iv, mgrid)
            slice_curves.append((float(d), curve))
            raw.append((float(d), k, iv))
            if with_svi:
                f = _fit_svi(k, iv)
                if f is not None:
                    f.option_type = ot
                    f.dte = float(d)
                    svi_list.append(f)
        if not slice_curves:
            out["iv_surface"][ot] = surface
            out["svi"][ot] = []
            out["raw_curves"][ot] = []
            out["fit_ok"] = False
            continue
        # dte 维度：sqrt(T) 空间线性插值（w=iv²），仅内插
        real_ds = np.array([c[0] for c in slice_curves])
        real_st = np.sqrt(real_ds / 365.0)
        curves = np.array([c[1] for c in slice_curves])  # (n_slice, n_money)
        for j, tg in enumerate(dte_grid):
            gt = np.sqrt(tg / 365.0)
            if gt <= real_st[0]:
                surface[j] = curves[0]
            elif gt >= real_st[-1]:
                surface[j] = curves[-1]
            else:
                idx = int(np.searchsorted(real_st, gt) - 1)
                t1, t2 = real_st[idx], real_st[idx + 1]
                alpha = (gt - t1) / (t2 - t1 + 1e-9)
                w1 = np.clip(curves[idx], 0, 3) ** 2
                w2 = np.clip(curves[idx + 1], 0, 3) ** 2
                w = (1 - alpha) * w1 + alpha * w2
                surface[j] = np.sqrt(np.clip(w, 0, 9))
        out["iv_surface"][ot] = surface
        out["svi"][ot] = svi_list
        out["raw_curves"][ot] = raw
    return out
