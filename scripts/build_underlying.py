"""build_underlying — 为 anchor 数据集补算「曲面之外」的标的侧特征。

背景：README §7.9。原 26 维特征全部来自隐含波动率曲面自身，信息同源；
标的已实现波动（RV）来自价格路径，是唯一被实测证明能把 P-R 前沿外推的信息源。

产物：`data_out/anchor/{symbol}/{YYYY-MM}/underlying.parquet`
      列 = `_timestamp` + `UNDERLYING_FEATURES`(6) + `_S`(反解标的价) + `_front_dte`

两阶段设计（都可断点续跑）
--------------------------
`price` 阶段：逐 (品种, 月) 读原始期权 parquet，反解最近月标的价，只落盘
             时间戳/价格/最近月 dte 三列。重的 IO 全在这里，跑一次即可。
`feat`  阶段：**按品种把所有月份拼成一条连续序列**再算特征，最后切回各月落盘。

为什么特征必须按品种连续算，而不是逐月各算各的
----------------------------------------------
`spread_z` 的滚动窗口是 240 根，逐月计算时每月开头约 48 根（≈13%）拿不到有效
z-score 只能填 0；`rv_long` 也要 32 根热身。按品种连续算等于用上月末尾做热身，
这些空窗除品种首月外全部消失。这既更接近线上实况（线上永远有历史），
也没有引入未来信息——所有滚动窗口都只向后看，因果性由 G1 门断言。

代价：§7.9 的实验数是逐月口径算的，改连续口径后需重新测量，不能直接沿用。

用法::

    python scripts/build_underlying.py --stage price      # 慢，支持中断续跑
    python scripts/build_underlying.py --stage feat       # 快
    python scripts/build_underlying.py --stage all
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.io_loader import _files_for_root
from vol_surface.underlying import (TRADING_DAYS, UNDERLYING_FEATURES,
                                    build_underlying_features,
                                    front_month_series, roll_mask)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "data_out", "anchor")

PRICE_FILE = "_underlying_price.parquet"
OUT_FILE = "underlying.parquet"


def _months(symbol: str) -> list:
    ds = sorted(glob.glob(os.path.join(ANCHOR, symbol, "*/")))
    return [os.path.basename(d.rstrip("/")) for d in ds]


def _symbols() -> list:
    return sorted(os.path.basename(d.rstrip("/"))
                  for d in glob.glob(os.path.join(ANCHOR, "*/"))
                  if os.path.isdir(d) and not os.path.basename(d.rstrip("/")).startswith("_"))


# ------------------------------------------------------------------ price 阶段

def build_price(symbols: list, budget: float = 1e9, force: bool = False) -> dict:
    """逐 (品种, 月) 反解最近月标的价并落盘。返回 {done, skipped, missing}。"""
    t0 = time.time()
    stat = {"done": 0, "skipped": 0, "missing": 0, "timeout": False}
    for sym in symbols:
        for ym in _months(sym):
            d = os.path.join(ANCHOR, sym, ym)
            out = os.path.join(d, PRICE_FILE)
            if os.path.exists(out) and not force:
                stat["skipped"] += 1
                continue
            if time.time() - t0 > budget:
                stat["timeout"] = True
                return stat
            feat = os.path.join(d, "features.parquet")
            if not os.path.exists(feat):
                stat["missing"] += 1
                continue
            ts = pd.read_parquet(feat).sort_values("_timestamp")["_timestamp"] \
                   .astype(str).tolist()
            files = sorted(_files_for_root(sym, year=int(ym[:4]), month=int(ym[5:7])))
            if not files:
                stat["missing"] += 1
                continue
            raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            S, D = front_month_series(raw, ts, with_dte=True)
            del raw
            pd.DataFrame({"_timestamp": ts, "_S": S, "_front_dte": D}).to_parquet(
                out, index=False)
            stat["done"] += 1
            print(f"  [price] {sym} {ym}: {len(ts)} 截面, "
                  f"有效价 {np.isfinite(S).mean()*100:.1f}%", flush=True)
    return stat


# ------------------------------------------------------------------ feat 阶段

def build_feat(symbols: list, short: int = 16, long: int = 64,
               z_window: int = 240, no_roll_fix: bool = False) -> dict:
    """按品种拼成连续序列算特征，再切回各月落盘。"""
    stat = {"symbols": 0, "months": 0, "rolls": 0}
    for sym in symbols:
        yms, parts, ivs = [], [], []
        for ym in _months(sym):
            d = os.path.join(ANCHOR, sym, ym)
            p, f = os.path.join(d, PRICE_FILE), os.path.join(d, "features.parquet")
            if not (os.path.exists(p) and os.path.exists(f)):
                continue
            pr = pd.read_parquet(p)
            fe = pd.read_parquet(f).sort_values("_timestamp")
            if len(pr) != len(fe):
                raise ValueError(f"{sym} {ym}: price 行数 {len(pr)} ≠ features {len(fe)}，"
                                 f"请用 --force 重跑 price 阶段")
            yms.append((ym, len(pr)))
            parts.append(pr)
            ivs.append(fe["atm_iv"].to_numpy(float))
        if not parts:
            continue

        cat = pd.concat(parts, ignore_index=True)
        S = cat["_S"].to_numpy(float)
        D = cat["_front_dte"].to_numpy(float)
        iv = np.concatenate(ivs)
        drop = None if no_roll_fix else roll_mask(D)
        if drop is not None:
            stat["rolls"] += int(drop.sum())

        # 每日 K 线数用实测值：总截面数 / 交易日数
        ndays = len({t[:8] for t in cat["_timestamp"].astype(str)})
        bpd = max(8.0, len(cat) / max(ndays, 1))

        fx = build_underlying_features(S, iv, bars_per_day=bpd, short=short,
                                       long=long, z_window=z_window, drop=drop)

        off = 0
        for ym, n in yms:
            sl = slice(off, off + n)
            df = pd.DataFrame({"_timestamp": cat["_timestamp"].to_numpy()[sl]})
            for k in UNDERLYING_FEATURES:
                df[k] = fx[k][sl]
            df["_S"] = S[sl]
            df["_front_dte"] = D[sl]
            df.to_parquet(os.path.join(ANCHOR, sym, ym, OUT_FILE), index=False)
            off += n
            stat["months"] += 1
        stat["symbols"] += 1
        print(f"  [feat] {sym}: {len(cat)} 截面 / {len(yms)} 月, bars_per_day={bpd:.1f}, "
              f"换月 {0 if drop is None else int(drop.sum())} 处, "
              f"rv_long 中位 {np.nanmedian(fx['rv_long'][fx['rv_long'] > 0])*100:.1f}%",
              flush=True)
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["price", "feat", "all"])
    ap.add_argument("--symbols", default=None, help="逗号分隔，默认全部")
    ap.add_argument("--budget", type=float, default=1e9, help="price 阶段秒数预算")
    ap.add_argument("--force", action="store_true", help="price 阶段重算已存在的月份")
    ap.add_argument("--no-roll-fix", action="store_true",
                    help="不剔除换月跳空（用于消融对照）")
    a = ap.parse_args()
    syms = a.symbols.split(",") if a.symbols else _symbols()

    if a.stage in ("price", "all"):
        s = build_price(syms, budget=a.budget, force=a.force)
        print(f"price 阶段: 新算 {s['done']}, 已有 {s['skipped']}, 缺数据 {s['missing']}"
              + ("  [预算用尽，重跑本命令继续]" if s["timeout"] else ""))
        if s["timeout"]:
            return
    if a.stage in ("feat", "all"):
        s = build_feat(syms, no_roll_fix=a.no_roll_fix)
        print(f"feat 阶段: {s['symbols']} 个品种 / {s['months']} 个月, "
              f"共剔除换月跳空 {s['rolls']} 处")


if __name__ == "__main__":
    main()
