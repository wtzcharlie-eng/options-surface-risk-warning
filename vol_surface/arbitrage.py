"""arbitrage — 无套利条件校验。

无套利的两条基本要求（针对 IV 曲面）：

1. **日历套利（calendar arbitrage）**：同一行权价，IV 应随到期时间非降（更长期
   限的方差不能小于更短期限）。等价地：对每个 k，w(k, T) = iv(k,T)^2 * T 关于 T
   单调非降。本数据集 BS 反推常导致近月 IV 异常高于远月（实测 ag2403 dte=1 的
   ATM IV=0.20 与 ag2404 dte=31 的 0.14 相比已偏高，深度 ITM/OTM 更甚），日历
   套利违反是临到期与极端行情的重要信号。

2. **蝶式套利（butterfly arbitrage）**：对每个 T，w(k) 必须关于 k 凸（蝶式组合
   价格非负）。等价于 w 的二阶差分 >= 0，即 ∂²w/∂k² >= 0。SVI 在 |rho|<1 且
   b*sigma*(1-rho^2) 充分大时自动满足，本模块对插值后的网格直接做二阶差分校验。

3. **put-call parity 偏离**：同 strike、同 dte 的 call/put 由平价关系约束，IV
   间应接近一致；偏离过大提示流动性/定价异常。

本模块直接在拟合后的规则网格上工作，输出违反条数、平均幅度、最大幅度与位置，
供特征工程与预警规则使用。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ArbReport:
    calendar_violations: int = 0
    calendar_avg_mag: float = 0.0
    calendar_max_mag: float = 0.0
    butterfly_violations: int = 0
    butterfly_avg_mag: float = 0.0
    butterfly_max_mag: float = 0.0
    parity_violations: int = 0
    parity_avg_mag: float = 0.0
    parity_max_mag: float = 0.0

    def as_dict(self) -> dict:
        return {
            "arb_calendar_n": self.calendar_violations,
            "arb_calendar_avg": self.calendar_avg_mag,
            "arb_calendar_max": self.calendar_max_mag,
            "arb_butterfly_n": self.butterfly_violations,
            "arb_butterfly_avg": self.butterfly_avg_mag,
            "arb_butterfly_max": self.butterfly_max_mag,
            "arb_parity_n": self.parity_violations,
            "arb_parity_avg": self.parity_avg_mag,
            "arb_parity_max": self.parity_max_mag,
        }


def check_calendar(surface: np.ndarray, dte_grid: np.ndarray, moneyness_grid: np.ndarray) -> dict:
    """日历套利：w(k,T)=iv^2 * T 随 T 单调非降。

    surface 形状 (n_dte, n_money)。返回违反条数与幅度。
    """
    n_dte, n_money = surface.shape
    if n_dte < 2:
        return {"n": 0, "avg": 0.0, "max": 0.0}
    T = dte_grid / 365.0
    w = surface**2 * T[None].T  # (n_dte, n_money)
    violations = 0
    mags = []
    for i in range(n_dte - 1):
        diff = w[i] - w[i + 1]  # 早期减晚期，>0 即违反（早期方差更大）
        bad = diff > 0
        violations += int(bad.sum())
        mags.extend(diff[bad].tolist())
    return {
        "n": violations,
        "avg": float(np.mean(mags)) if mags else 0.0,
        "max": float(np.max(mags)) if mags else 0.0,
    }


def check_butterfly(surface: np.ndarray) -> dict:
    """蝶式套利：w(k) 关于 k 凸，二阶差分 >= 0。

    surface 形状 (n_dte, n_money)，按 moneyness 升序。返回违反条数与幅度。
    """
    n_dte, n_money = surface.shape
    if n_money < 3:
        return {"n": 0, "avg": 0.0, "max": 0.0}
    w = surface**2  # 用 iv^2 近似 w（省 T，凸性结论一致）
    # 二阶差分: w[i,j-1] - 2 w[i,j] + w[i,j+1]
    d2 = w[:, :-2] - 2 * w[:, 1:-1] + w[:, 2:]
    bad = d2 < 0
    mags = (-d2[bad]).tolist()
    return {
        "n": int(bad.sum()),
        "avg": float(np.mean(mags)) if mags else 0.0,
        "max": float(np.max(mags)) if mags else 0.0,
    }


def check_parity(call_surface: np.ndarray, put_surface: np.ndarray) -> dict:
    """put-call parity 偏离：同点 IV 差的绝对值。"""
    if call_surface.shape != put_surface.shape:
        return {"n": 0, "avg": 0.0, "max": 0.0}
    diff = call_surface - put_surface
    mag = np.abs(diff)
    # 超过 5 个 vol point 视为显著违反
    bad = mag > 0.05
    mags = mag[bad].tolist()
    return {
        "n": int(bad.sum()),
        "avg": float(np.mean(mags)) if mags else 0.0,
        "max": float(np.max(mags)) if mags else 0.0,
    }


def check_arbitrage(iv_result: dict, parity_threshold: float = 0.05) -> ArbReport:
    """对一个 fit_slice_iv 的结果做全套无套利校验。"""
    dte_grid = iv_result["dte_grid"]
    mgrid = iv_result["moneyness_grid"]
    call = iv_result["iv_surface"].get("call")
    put = iv_result["iv_surface"].get("put")
    rep = ArbReport()
    if call is not None and np.isfinite(call).any():
        cal = check_calendar(call, dte_grid, mgrid)
        rep.calendar_violations = cal["n"]
        rep.calendar_avg_mag = cal["avg"]
        rep.calendar_max_mag = cal["max"]
        bf = check_butterfly(call)
        rep.butterfly_violations += bf["n"]
        rep.butterfly_avg_mag += bf["avg"] * (bf["n"] > 0)
        rep.butterfly_max_mag = max(rep.butterfly_max_mag, bf["max"])
    if put is not None and np.isfinite(put).any():
        cal = check_calendar(put, dte_grid, mgrid)
        rep.calendar_violations = max(rep.calendar_violations, cal["n"])
        rep.calendar_max_mag = max(rep.calendar_max_mag, cal["max"])
        bf = check_butterfly(put)
        rep.butterfly_violations += bf["n"]
        rep.butterfly_avg_mag += bf["avg"] * (bf["n"] > 0)
        rep.butterfly_max_mag = max(rep.butterfly_max_mag, bf["max"])
    if call is not None and put is not None:
        par = check_parity(call, put)
        rep.parity_violations = par["n"]
        rep.parity_avg_mag = par["avg"]
        rep.parity_max_mag = par["max"]
    if rep.butterfly_violations > 0:
        rep.butterfly_avg_mag /= 2.0
    return rep
