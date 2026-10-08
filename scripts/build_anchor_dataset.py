"""build_anchor_dataset — 构造锚点预测器的监督数据集。

对 (symbol, month) 逐月跑连续 15min 截面, 标注
    y = 1[t+1 .. t+H] 个截面内出现风险起点
风险起点严格复用 vol_surface/quant_metrics.identify_risk_windows 的判定
(horizon=5, q95, 20 日回看), 使标签与赛题评分公式完全对齐。

增量续跑: 若 {symbol}/{year}-{month}/features.parquet 已存在且 meta 完整则跳过。
跨月 history: 首月从空开始; 后续月重放前一月的特征行回填 history(不重跑清洗)。
落盘: data_out/anchor/{symbol}/{year}-{month:02d}/features.parquet + meta.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.cleaning import clean_slice
from vol_surface.features import FeatureHistory, compute_features
from vol_surface.io_loader import _files_for_root, slice_at_timestamp, timestamps_in
from vol_surface.quant_metrics import identify_risk_windows

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data_out", "anchor")

SYMBOL_YEARS = {
    "ag": [2023, 2024, 2025, 2026],
    "au": [2023, 2024, 2025, 2026],
    "sc": [2023, 2024, 2025, 2026],
    "si": [2023, 2024, 2025],  # si 2026 无月度文件
    # 扩样品种：原 4 品种只覆盖 extreme_event.csv 里 17 个事件中的 7 个，
    # 且 ag/au 同属贵金属、sc 与 si 各自单独，跨板块传导无从验证。
    # 补这三个后事件覆盖升到 15/17，板块从 3 个扩到 5 个（+黑色系 +有色）。
    "lc": [2023, 2024, 2025, 2026],  # 碳酸锂，对应 3 个未覆盖事件
    "cu": [2023, 2024, 2025, 2026],  # 铜，对应 2 个未覆盖事件
    "rb": [2023, 2024, 2025, 2026],  # 螺纹钢，黑色系，对应 3 个未覆盖事件
                                     # （rb 另有 2022 数据，为与其他品种对齐不纳入）
}

NUM_FEATURES = [
    "atm_iv", "atm_iv_z", "atm_iv_vel_z", "convexity_violation", "convexity_violation_z",
    "convexity_vel_z", "term_slope", "term_slope_roc", "skew_percentile", "skew_val",
    "gamma_concentration", "vega_concentration", "iv_change", "iv_change_rate",
    "iv_spike", "fit_degradation", "arb_score", "arb_score_z", "arb_calendar_n",
    "arb_butterfly_n", "arb_parity_n", "liquidity_ratio", "liquidity_z", "iv_near",
    "iv_far", "n_contracts",
]


@dataclass
class MonthTask:
    symbol: str
    year: int
    month: int


@dataclass
class MonthMeta:
    symbol: str
    year: int
    month: int
    n_slices: int
    n_pos: int
    pos_rate: float
    nan_rate: float
    n_days: int
    path: str
    sec: float


def _months_for_year(year: int) -> list:
    now = pd.Timestamp.now()
    return list(range(1, (now.month if year == now.year else 12) + 1))


def _load_history_priors(symbol: str, year: int, month: int) -> FeatureHistory:
    """用前一月的特征行回填 FeatureHistory(避免月切分后 z-score 失效)。

    若前一月 parquet 缺失(首月或断档)则返回空 history。
    """
    hist = FeatureHistory(window=500)
    prev_m = month - 1 if month > 1 else 12
    prev_y = year if month > 1 else year - 1
    prev_path = os.path.join(DATA_OUT, symbol, f"{prev_y}-{prev_m:02d}", "features.parquet")
    if not os.path.exists(prev_path):
        return hist
    try:
        df = pd.read_parquet(prev_path, columns=[c for c in NUM_FEATURES])
        for _, row in df.iterrows():
            # 与 compute_features 的 push 一致: 非有限值跳过
            feats = {k: (float(row[k]) if k in row and np.isfinite(row[k]) else np.nan)
                     for k in NUM_FEATURES}
            hist.push({k: v for k, v in feats.items() if np.isfinite(v)})
    except Exception:
        return FeatureHistory(window=500)
    return hist


def _label_slices(ts_list: list, risk_starts: list, horizon: int) -> np.ndarray:
    """对每个截面 i, y=1 当且仅当某条风险起点落在 ts_list[i+1 .. i+horizon]。

    风险起点本身是 identify_risk_windows 返回的 timestamp(即评分中的 rs)。
    预警命中判定是 risk_start ∈ [alert_t, alert_t+120min], 因此标签对应
    "当前截面之后 120min(8×15min) 内是否有 rs 落在"——与评分一致。
    """
    n = len(ts_list)
    y = np.zeros(n, dtype=int)
    if not risk_starts:
        return y
    rs = sorted(risk_starts)
    j_map = {t: i for i, t in enumerate(ts_list)}
    for r in rs:
        if r not in j_map:
            continue
        i0 = j_map[r]  # 风险起点本身的截面索引
        # 对 i 满足 i+1 <= i0 <= i+horizon ⇒ i ∈ [i0-horizon, i0-1]
        for i in range(max(0, i0 - horizon), i0):
            y[i] = 1
    return y


def _process_month(task: MonthTask, force: bool = False) -> MonthMeta | None:
    t0 = time.time()
    sym, y, m = task.symbol, task.year, task.month
    out_dir = os.path.join(DATA_OUT, sym, f"{y}-{m:02d}")
    out_path = os.path.join(out_dir, "features.parquet")
    meta_path = os.path.join(out_dir, "meta.parquet")
    if os.path.exists(out_path) and os.path.exists(meta_path) and not force:
        try:
            r = pd.read_parquet(meta_path).iloc[0]
            return MonthMeta(sym, y, m, int(r["n_slices"]), int(r["n_pos"]),
                             float(r["pos_rate"]), float(r["nan_rate"]),
                             int(r["n_days"]), out_path, 0.0)
        except Exception:
            pass

    files = sorted(_files_for_root(sym, year=y, month=m))
    if not files:
        return None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    month_str = f"{y}{m:02d}"
    tss = sorted(t for t in timestamps_in(df) if t.startswith(month_str))
    if not tss:
        return None

    hist = _load_history_priors(sym, y, m)
    rows, prev = [], None
    for ts in tss:
        sl = slice_at_timestamp(df, ts)
        if len(sl) < 15:
            continue
        clean, _ = clean_slice(sl)
        if len(clean) < 10:
            continue
        feats = compute_features(clean, prev_features=prev, history=hist,
                                  lightweight=True)
        rows.append({**{k: (feats.get(k, np.nan) if np.isfinite(feats.get(k, np.nan))
                                else np.nan) for k in NUM_FEATURES},
                     "_timestamp": ts})
        prev = feats

    if not rows:
        return None
    out = pd.DataFrame(rows).sort_values("_timestamp").reset_index(drop=True)

    cont = pd.DataFrame({"timestamp": out["_timestamp"],
                         "atm_iv": out["atm_iv"],
                         "convexity_violation": out["convexity_violation"]})
    risk_starts = identify_risk_windows(cont)
    y_arr = _label_slices(out["_timestamp"].tolist(), risk_starts, horizon=8)
    out["y"] = y_arr

    os.makedirs(out_dir, exist_ok=True)
    out.to_parquet(out_path, index=False)
    days = pd.to_datetime(out["_timestamp"], format="%Y%m%d%H%M%S").dt.date.nunique()
    meta = MonthMeta(sym, y, m, len(out), int(y_arr.sum()), float(y_arr.mean()),
                     float(out[NUM_FEATURES].isna().mean().mean()),
                     int(days), out_path, time.time() - t0)
    pd.DataFrame([asdict(meta)]).to_parquet(meta_path, index=False)
    return meta


def run(tasks: list, workers: int, force: bool) -> None:
    t0 = time.time()
    metas = []
    if workers <= 1:
        for t in tasks:
            r = _process_month(t, force=force)
            if r:
                metas.append(r)
                print(f"[{r.symbol} {r.year}-{r.month:02d}] slices={r.n_slices} "
                      f"pos={r.n_pos}({r.pos_rate:.1%}) nan={r.nan_rate:.1%} "
                      f"t={r.sec:.0f}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_process_month, t, force=force): t for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                if r:
                    metas.append(r)
                    print(f"[{r.symbol} {r.year}-{r.month:02d}] slices={r.n_slices} "
                          f"pos={r.n_pos}({r.pos_rate:.1%}) nan={r.nan_rate:.1%} "
                          f"t={r.sec:.0f}s", flush=True)
    if metas:
        summary = pd.DataFrame([asdict(m) for m in metas]).sort_values(
            ["symbol", "year", "month"]).reset_index(drop=True)
        os.makedirs(DATA_OUT, exist_ok=True)
        summary.to_parquet(os.path.join(DATA_OUT, "_build_summary.parquet"), index=False)
        print(f"\n汇总 {len(metas)} 月, {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", default=",".join(SYMBOL_YEARS))
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", nargs=2, metavar=("SYMBOL", "YYYY-MM"),
                    help="冒烟测试: 只处理一个品种的一个月, 忽略增量续跑, 打印前后样本")
    args = ap.parse_args()

    if args.smoke:
        sym, ym = args.smoke
        y, m = int(ym.split("-")[0]), int(ym.split("-")[1])
        t = MonthTask(sym, y, m)
        r = _process_month(t, force=True)
        if r:
            print(f"\n冒烟结果: slices={r.n_slices} pos={r.n_pos}({r.pos_rate:.1%}) "
                  f"nan={r.nan_rate:.1%} days={r.n_days} t={r.sec:.0f}s")
            df = pd.read_parquet(r.path)
            print("\n前 5 行 (特征缩略):")
            cols = ["_timestamp", "atm_iv", "convexity_violation", "atm_iv_z",
                    "term_slope_roc", "n_contracts", "y"]
            print(df[cols].head(5).to_string())
            print(f"\ny=1 的行数: {int(df['y'].sum())} / {len(df)}")
            print(f"y=1 的样例 ts: {df[df['y']==1]['_timestamp'].head(5).tolist()}")
        return

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tasks = [MonthTask(sym, y, m)
             for sym in syms
             for y in SYMBOL_YEARS.get(sym, [2023, 2024, 2025])
             for m in _months_for_year(y)]
    print(f"待处理月份: {len(tasks)}  ({', '.join(syms)}, {args.workers} workers, force={args.force})",
          flush=True)
    run(tasks, args.workers, args.force)


if __name__ == "__main__":
    main()
