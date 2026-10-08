"""scan — 定时增量扫描入口。

读取某品种最新若干截面，跑清洗→特征→预警，与历史特征合并后输出预警。
支持单次扫描与定时循环两种模式，用于近实时风控。

用法
----
  # 单次扫描最新截面
  python scripts/scan.py --symbol ag --latest 1
  # 定时每 15 分钟扫描
  python scripts/scan.py --symbol ag --interval 15
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.alert_engine import evaluate, LEVEL_NAMES
from vol_surface.alert_model import AlertModel
from vol_surface.cleaning import clean_slice
from vol_surface.features import compute_features, FeatureHistory
from vol_surface.io_loader import load_symbol_family, slice_at_timestamp, timestamps_in, ts_to_date

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_out")


def scan_once(symbol: str, latest: int = 1, verbose: bool = True, df: pd.DataFrame | None = None) -> list:
    """扫描某品种最新 latest 个截面，输出预警。

    Parameters
    ----------
    df : 可选，已加载的品种全量数据；为 None 时只读最新一个数据月文件（增量场景）。
    """
    if df is None:
        # 只读最新月份文件，避免全量加载
        from vol_surface.io_loader import _files_for_root
        files = _files_for_root(symbol)
        if not files:
            raise FileNotFoundError(f"no data for {symbol}")
        # 选文件名时间戳最大的几个文件（最新月）
        files = sorted(files, key=lambda f: f.split("options_")[-1])[-2:]
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    tss = timestamps_in(df)
    targets = tss[-latest:]
    # 加载历史特征重建滚动窗口
    feat_path = os.path.join(DATA_OUT, f"features_{symbol}.parquet")
    hist = FeatureHistory(window=500)
    if os.path.exists(feat_path):
        fdf = pd.read_parquet(feat_path)
        for _, r in fdf.iterrows():
            hist.push(r.to_dict())
    model_path = os.path.join(DATA_OUT, f"model_{symbol}.joblib")
    model = AlertModel.load(model_path) if os.path.exists(model_path) else None

    # 用历史最后一截面作为 prev
    prev = fdf.iloc[-1].to_dict() if os.path.exists(feat_path) and len(fdf) else None
    results = []
    for ts in targets:
        sl = slice_at_timestamp(df, ts)
        if len(sl) < 15:
            continue
        clean, _ = clean_slice(sl)
        if len(clean) < 10:
            continue
        feats = compute_features(clean, prev_features=prev, history=hist)
        prev = feats
        res = evaluate(feats, model=model)
        results.append({
            "timestamp": ts, "symbol": symbol, "level": res["level"],
            "name": LEVEL_NAMES[res["level"]], "ml_score": res["ml_score"],
            "triggers": "; ".join(t["reason"] for t in res["triggers"]),
        })
        if verbose:
            r = results[-1]
            print(f"[{ts}] L{r['level']} {r['name']} ml={r['ml_score']:.2f}")
            if r["triggers"]:
                for t in r["triggers"].split("; "):
                    print(f"    - {t}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ag")
    ap.add_argument("--latest", type=int, default=1, help="扫描最新 N 个截面")
    ap.add_argument("--interval", type=int, default=0, help=">0 时定时每 N 分钟扫描")
    args = ap.parse_args()
    if args.interval > 0:
        print(f"定时扫描启动：每 {args.interval} 分钟扫描 {args.symbol} 最新截面")
        while True:
            print(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} 扫描 ===")
            try:
                scan_once(args.symbol, args.latest)
            except Exception as e:
                print(f"扫描失败: {e}")
            time.sleep(args.interval * 60)
    else:
        scan_once(args.symbol, args.latest)


if __name__ == "__main__":
    main()
