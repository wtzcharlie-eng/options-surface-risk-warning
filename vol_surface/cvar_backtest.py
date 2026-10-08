"""cvar_backtest — 赛题核心业务指标：CVaR 改善率回测。

赛题定义（量化技术指标2）：
  测试组合：期货多头、期货空头各 1 手。
  风控规则：系统发出预警（等级≥2）时，在预警时点的下一个交易日开盘将持仓减半
    （多头平一半，空头回补一半），减仓后持有至测试期结束，不再恢复；多次预警仅
    首次减仓。
  无预警策略：始终持有原头寸。
  评价：算测试期（≥3 个月连续数据）日收益率序列的 CVaR(95%)，多头改善率与空头
    改善率取算术平均，>10% 视为显著有效。

标的期货价格：优先用 AKShare 取主力连续日线；不可用时从期权数据 strike/moneyness
反推每个截面最活跃到期月的标的价（数据自洽，不依赖外部网络）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_out")

# 品种族 -> AKShare 主力连续代码
AK_SYMBOL = {"ag": "AG0", "au": "AU0", "sc": "SC0", "si": "SI0", "cu": "CU0",
             "lc": "LC0", "rb": "RB0"}


def fetch_futures_price(symbol: str, start: str = "20230101", end: str = "20260401") -> pd.DataFrame:
    """用 AKShare 取主力连续日线，返回 DataFrame[date, open, close]（date 为 Timestamp）。"""
    import akshare as ak
    code = AK_SYMBOL.get(symbol, symbol.upper() + "0")
    df = ak.futures_main_sina(symbol=code, start_date=start, end_date=end)
    df = df.rename(columns={"日期": "date", "开盘价": "open", "收盘价": "close"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "open", "close"]].sort_values("date").reset_index(drop=True)


def infer_futures_price(symbol: str) -> pd.DataFrame:
    """兜底：从期权数据反推标的价。取每个时间戳最活跃到期月（持仓量最大）的中位反推价，
    再按日取收盘。返回 DataFrame[date, open, close]。
    """
    from .io_loader import load_symbol_family, ts_to_date
    df = load_symbol_family(symbol)
    df = df[df["open_interest"].fillna(0) > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=["date", "open", "close"])
    df["px"] = df["strike"] / df["moneyness"]
    # 每个时间戳取所有合约反推价的中位（最活跃到期月权重已隐含在 OI>0 过滤里）
    ts_px = df.groupby("timestamp")["px"].median()
    ts_px.index = pd.to_datetime(ts_px.index, format="%Y%m%d%H%M%S")
    daily = ts_px.resample("D").last().dropna()
    daily.index = daily.index.normalize()
    out = pd.DataFrame({"date": daily.index, "close": daily.values})
    out["open"] = out["close"]  # 无开盘价，用收盘近似
    return out[["date", "open", "close"]].reset_index(drop=True)


def get_futures_price(symbol: str, use_akshare: bool = True) -> pd.DataFrame:
    if use_akshare:
        try:
            px = fetch_futures_price(symbol)
            if not px.empty:
                return px
        except Exception as e:
            print(f"  AKShare 取价失败({e})，改用反推")
    return infer_futures_price(symbol)


def _daily_returns(price: pd.DataFrame) -> pd.Series:
    """日收益率：基于收盘。"""
    p = price.sort_values("date").reset_index(drop=True)
    return p["close"].pct_change().dropna()


def _cvar95(returns: pd.Series) -> float:
    """CVaR(95%)：损失分布 5% 尾部条件期望。返回正值表示损失大小。"""
    r = returns.dropna().to_numpy()
    if r.size < 20:
        return 0.0
    var = np.percentile(r, 5)  # 5% 分位（最差的 5% 收益）
    tail = r[r <= var]
    return float(-tail.mean())  # 正值 = 平均尾部损失


@dataclass
class CvarResult:
    long_cvar_no_alert: float
    long_cvar_alert: float
    long_improve: float
    short_cvar_no_alert: float
    short_cvar_alert: float
    short_improve: float
    avg_improve: float
    first_alert_date: pd.Timestamp | None
    n_alert_days: int


def backtest_cvar(
    alerts: pd.DataFrame,
    price: pd.DataFrame,
    direction: str = "both",
) -> CvarResult:
    """CVaR 改善率回测。

    ⚠️ **本函数的口径已被证实是退化的，保留仅为复现旧结果，请改用
    `backtest_cvar_path`。** 规则是「首次预警次日减半、之后持有至期末」，
    而实测预警密度下首次预警总是落在窗口第一个交易日，于是它等价于「全程半仓」，
    结果恒为 +50.0%，与预警质量无关。详见 `backtest_cvar_path` 的说明与 README §7.11。

    Parameters
    ----------
    alerts : 含 timestamp(str)、level 的预警序列
    price : date(Timestamp)、open、close 的日线
    direction : "long" / "short" / "both"
    """
    alerts = alerts.copy()
    alerts["dt"] = pd.to_datetime(alerts["timestamp"], format="%Y%m%d%H%M%S", errors="coerce")
    price = price.sort_values("date").reset_index(drop=True).copy()
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()

    # 首次 level>=2 预警日期
    hi = alerts[alerts["level"] >= 2].sort_values("dt")
    first_alert_dt = hi["dt"].iloc[0].normalize() if not hi.empty else None

    def _run(long: bool) -> tuple[float, float]:
        """返回 (无预警CVaR, 预警策略CVaR)。多头 long=True，空头 long=False。"""
        rets = _daily_returns(price)
        rets = rets.copy()
        # 对齐日期
        rets.index = price["date"].iloc[1:].values
        no_alert = rets.copy()
        if long:
            no_alert = no_alert  # 多头收益 = 标的收益
        else:
            no_alert = -no_alert  # 空头收益 = -标的收益
        alert = no_alert.copy()
        if first_alert_dt is not None:
            # 首次预警次日开盘减半 → 减半后该日起收益减半
            cut_idx = alert.index >= first_alert_dt + pd.Timedelta(days=1)
            alert[cut_idx] = alert[cut_idx] * 0.5
        return _cvar95(no_alert), _cvar95(alert)

    long_no, long_al = _run(True) if direction in ("long", "both") else (0.0, 0.0)
    short_no, short_al = _run(False) if direction in ("short", "both") else (0.0, 0.0)
    long_imp = (long_no - long_al) / long_no if long_no > 1e-9 else 0.0
    short_imp = (short_no - short_al) / short_no if short_no > 1e-9 else 0.0
    return CvarResult(
        long_cvar_no_alert=long_no, long_cvar_alert=long_al, long_improve=long_imp,
        short_cvar_no_alert=short_no, short_cvar_alert=short_al, short_improve=short_imp,
        avg_improve=(long_imp + short_imp) / 2,
        first_alert_date=first_alert_dt,
        n_alert_days=int((alerts["level"] >= 2).sum()),
    )


def backtest_cvar_path(
    alerts: pd.DataFrame,
    price: pd.DataFrame,
    hold_days: int = 5,
    cut_to: float = 0.5,
    direction: str = "both",
    n_random: int = 200,
    seed: int = 0,
) -> dict:
    """**非退化版** CVaR 改善率：仓位在每次预警后只压 `hold_days` 天，之后恢复。

    为什么必须换这个口径
    --------------------
    原 `backtest_cvar` 的规则是「首次预警次日减半，之后持有至期末」。实测 7 个
    品种的**首次预警全部落在窗口第一个交易日**（预警密度 150~1500 条 / 45~95 日），
    于是该策略退化为「全程半仓」，CVaR 被机械地按比例缩小——7 个品种算出来
    **一模一样都是 +50.0%**。那个数字与预警质量完全无关：恒预警、随机预警、
    甚至瞎报都会得到同样的 +50.0%。它衡量的是降杠杆，不是风控。

    本函数改为**事件驱动的仓位路径**：预警日起 `hold_days` 个交易日内仓位压到
    `cut_to`，随后恢复满仓。这样「什么时候报」才会影响结果。

    并且必须配对照
    --------------
    仅有「减仓 vs 不减仓」的对比仍不足以说明预警有效——**任何**降低平均仓位的
    做法都会降低 CVaR。故本函数同时给出：
      - `always`：恒定 `cut_to` 仓位（仓位下限对照，改善率的"白拿"部分）
      - `random`：随机挑同样多的预警日、重复 `n_random` 次取均值
        （**同等仓位占用下的无信息对照**，这才是真正的基准线）
    判据是 `excess = improve − random_improve`：只有它显著为正，
    才说明改善来自「预警报得准」而非「仓位压得低」。

    返回 dict，键含 improve / random_improve / excess / avg_position 等。
    """
    rng = np.random.default_rng(seed)
    price = price.sort_values("date").reset_index(drop=True).copy()
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    dates = price["date"].to_numpy()
    n = len(price)
    if n < 25:
        raise ValueError(f"日线样本过少（{n} 天），CVaR 尾部估计不可靠")

    a = alerts.copy()
    a["dt"] = pd.to_datetime(a["timestamp"], format="%Y%m%d%H%M%S", errors="coerce")
    hi = a[a["level"] >= 2]
    alert_days = pd.to_datetime(pd.Series(hi["dt"].dt.normalize().unique())).to_numpy()
    idx = np.unique(np.searchsorted(dates, alert_days))
    idx = idx[idx < n]

    def _pos_from(hit_idx) -> np.ndarray:
        pos = np.ones(n)
        for i in hit_idx:
            pos[i + 1: i + 1 + hold_days] = cut_to      # 次日开盘执行
        return pos

    ret = price["close"].pct_change().to_numpy()

    def _cvar_for(pos, long: bool) -> float:
        r = ret * pos if long else -ret * pos
        return _cvar95(pd.Series(r[1:]))

    def _improve(pos) -> float:
        outs = []
        for long in ((True, False) if direction == "both" else (direction == "long",)):
            base = _cvar_for(np.ones(n), long)
            cut = _cvar_for(pos, long)
            outs.append((base - cut) / base if base > 1e-12 else 0.0)
        return float(np.mean(outs))

    pos_real = _pos_from(idx)
    imp_real = _improve(pos_real)
    imp_always = _improve(np.full(n, cut_to))

    rnd = []
    for _ in range(n_random):
        pick = rng.choice(n, size=min(len(idx), n), replace=False)
        rnd.append(_improve(_pos_from(np.sort(pick))))
    rnd = np.array(rnd)

    return {
        "n_days": n, "n_alert_days": int(len(idx)),
        "avg_position": float(pos_real.mean()),
        "improve": imp_real,
        "always_improve": imp_always,
        "random_improve": float(rnd.mean()),
        "random_sd": float(rnd.std(ddof=1)),
        # 超额改善：扣掉「同等仓位占用下随机报」能拿到的部分
        "excess": float(imp_real - rnd.mean()),
        # 分位：真实改善在随机分布中的位置，>0.95 才算显著
        "pctile": float((rnd < imp_real).mean()),
        "hold_days": hold_days, "cut_to": cut_to,
    }


def format_cvar(r: CvarResult) -> str:
    return (
        f"CVaR 改善率回测: 首次预警={r.first_alert_date} 预警截面(≥2)={r.n_alert_days}\n"
        f"  多头: 无预警CVaR={r.long_cvar_no_alert:.4f} → 预警CVaR={r.long_cvar_alert:.4f} 改善={r.long_improve:.1%}\n"
        f"  空头: 无预警CVaR={r.short_cvar_no_alert:.4f} → 预警CVaR={r.short_cvar_alert:.4f} 改善={r.short_improve:.1%}\n"
        f"  平均改善率={r.avg_improve:.1%} (目标>10% 视为显著有效)"
    )
