"""build_features — 离线特征与预警生成。

读取 archive 中目标品种族，逐时间截面跑清洗→插值→特征→预警，产出：
  data_out/features_<symbol>.parquet   每截面一行特征向量
  data_out/alerts_<symbol>.parquet      每截面一行预警结果（level/triggers/top_features）
  data_out/model_<symbol>.joblib        训练好的 IsolationForest
  data_out/surfaces_<symbol>.parquet    抽样截面的 IV 曲面（供仪表板热力图）

策略：全量跑太慢（单品种 265 文件、数千万行），故按 `--max-slices` 在**全局**时间序列上
均匀采样截面，单遍扫描算特征 → 训练 IsolationForest → 出预警 → 对抽中的少数截面落曲面。

> 早前是**两遍**扫描（先算特征训 IF，再重放一遍出预警）。实测两遍调的是完全相同的
> `compute_features(clean, prev_features=prev, history=<各自新建的空 hist>)`、
> 遍历同一批时间戳，故结果逐位相同——第二遍纯属重复劳动。合并后结果不变、耗时减半。
> （顺带更正：本文档原先写「两遍共用同一 FeatureHistory」，与代码不符，两遍各建各的。）

用法
----
  python scripts/build_features.py [--symbols ag,si,au,sc] [--event-window 2]

  # 单次进程时长受限时：算够 N 秒就存盘退出，反复运行直到跑完
  python scripts/build_features.py --symbols lc --budget 150

  缓存（`--cache-dir`，默认在系统临时目录）只加速、不改变结果；**不要指向 data_out**。
"""

from __future__ import annotations

import argparse
import os
import pickle as _pickle
import sys
import tempfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.alert_engine import evaluate
from vol_surface.alert_model import AlertModel, MODEL_FEATURES
from vol_surface.cleaning import clean_slice
from vol_surface.features import compute_features, FeatureHistory, FEATURE_COLUMNS
from vol_surface.io_loader import load_symbol_family, timestamps_in, ts_to_date

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_out")
EVENT_SYMBOLS = ["ag", "si", "au", "sc", "cu", "lc", "rb"]


def _iter_slices(df: pd.DataFrame, sampled_ts: list):
    """单次 groupby 遍历，按 timestamp 依次 yield (ts, slice_df)。比逐次布尔索引快得多。"""
    ts_set = set(sampled_ts)
    for ts, grp in df.groupby("timestamp", sort=True):
        if ts in ts_set:
            yield ts, grp


def _load_sampled(symbol: str, max_slices: int | None,
                  cache_dir: str | None = None) -> pd.DataFrame:
    """按年流式载入，**只保留被采样到的截面**，避免全量 concat 撑爆内存。

    为什么需要
    ----------
    原实现 `load_symbol_family(symbol)` 一次性读入该品种族全部合约全部年份。
    ag/au/sc/si 尚可，但 lc 有 341 个合约文件，实测直接 **OOM 被 SIGKILL**
    （与 `vol_surface/quant_metrics._iter_slices` 当初踩的是同一类坑）。

    而下游只用到 `max_slices` 个采样截面，全量载入纯属浪费。

    **采样口径与原实现逐位一致**：先只读 `timestamp` 一列（内存极小）跨年汇总，
    在**全局**时间序列上按 `tss[::step]` 取样——而不是每年各取一批，
    后者会改变采样点、使新旧品种口径不可比。

    `cache_dir`
    -----------
    这一步要扫该品种族全部合约全部月份，是整个脚本最贵的一环（实测单品种 1.5~3 分钟），
    而它的结果**只由 (symbol, max_slices) 决定**。给定缓存目录时把结果落盘，
    重跑（换模型、改预警逻辑、分批跑）就不必重扫归档。
    缓存只是加速，**删掉它不影响任何结果**。
    """
    cp = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        # 用 pickle 而非 parquet：原始帧的 `date` 列是混合类型的 object，
        # parquet 会报 ArrowTypeError；而把它强制转成 str 会让
        # **命中缓存与不命中缓存得到不同的数据**——缓存绝不能改变结果。
        cp = os.path.join(cache_dir, f"{symbol}_{max_slices}.pkl")
        if os.path.exists(cp):
            d = pd.read_pickle(cp)
            print(f"  命中载入缓存 {os.path.basename(cp)}（{len(d):,} 行）", flush=True)
            return d

    # 只读排序所需的 4 列（而非全部 23 列）。**不能只读 timestamp**——
    # `load_symbol_family` 内部按 4 列排序，少给会 KeyError。
    # 初版把这个异常 `except: continue` 吞掉了，表现为"没有可用数据"，
    # 排查了一轮才发现——静默吞异常正是本项目反复踩的坑，故此处显式打印。
    _SORT_COLS = ["timestamp", "underlying", "option_type", "strike"]
    tss = []
    for y in range(2022, 2027):
        try:
            d = load_symbol_family(symbol, year=y, columns=_SORT_COLS)
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  [{symbol}] {y} 年读取失败：{type(e).__name__}: {e}", flush=True)
            continue
        if len(d):
            tss.extend(d["timestamp"].unique().tolist())
        del d
    tss = sorted(set(tss))
    if not tss:
        raise RuntimeError(f"{symbol} 没有可用数据")
    if max_slices:
        tss = tss[::max(1, len(tss) // max_slices)]
    keep = set(tss)
    print(f"  全局采样 {len(keep)} 个截面，按年载入...", flush=True)

    parts = []
    # **按月而非按年**：ag 单年全量载入仍会 OOM（实测 SIGKILL），
    # 与 `quant_metrics._iter_slices` 当初的结论一致——粒度必须细到月。
    # 过滤后每月只剩几个采样截面，累计内存很小。
    for y in range(2022, 2027):
        for mo in range(1, 13):
            try:
                d = load_symbol_family(symbol, year=y, month=mo)
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"  [{symbol}] {y}-{mo:02d} 读取失败：{type(e).__name__}: {e}",
                      flush=True)
                continue
            if len(d):
                d = d[d["timestamp"].isin(keep)]
                if len(d):
                    parts.append(d.copy())
            del d
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if cp is not None and len(out):
        # 先写临时文件再改名：若本次被中途杀掉，不会留下半截缓存被下次当成完整的用
        tmp = cp + ".part"
        out.to_pickle(tmp)
        # **写完立刻读回来逐值比对**：缓存只许加速，不许改变数据。
        # 不验的话，任何一个 dtype 在往返中被悄悄改掉，都会让「命中缓存」
        # 与「不命中缓存」跑出两套结果，而且没有任何门看得见。
        back = pd.read_pickle(tmp)
        if not back.equals(out):
            os.remove(tmp)
            raise RuntimeError(
                f"[{symbol}] 载入缓存往返后与原始帧不一致，已放弃缓存以免污染结果")
        os.replace(tmp, cp)
        print(f"  已缓存载入结果 → {os.path.basename(cp)}（往返校验通过）", flush=True)
    return out


def build_symbol(symbol: str, event_window_months: int = 2, max_slices: int | None = 400,
                 save_surfaces_every: int = 20, cache_dir: str | None = None,
                 budget: float = 0.0) -> dict:
    """对一个品种完整跑特征+预警+训练，落盘。"""
    os.makedirs(DATA_OUT, exist_ok=True)
    t0 = time.time()
    print(f"\n=== [{symbol}] 加载数据（按年流式，只留采样截面）===", flush=True)
    df = _load_sampled(symbol, max_slices, cache_dir=cache_dir)
    print(f"  {len(df):,} 行, {df['underlying'].nunique()} 到期月, {df['timestamp'].nunique():,} 时间点", flush=True)

    # ---- 单遍扫描：算特征 + 留下曲面候选切片
    # 原实现分两遍，且两遍**算的是同一批特征**——那个已删除的 `_collect_features` 与下面的循环
    # 都调 `compute_features(clean, prev_features=prev, history=<各自新建的 hist>)`，
    # 遍历同一批时间戳、同一顺序、同样 window=500 的空历史，故逐位相同。
    # 实证：拿落盘特征重训 IF，与当初落盘的模型对同一输入给出的 raw 分数
    #      **逐位相同（最大差 0.00e+00，ag/lc/rb 三品种）**，
    #      即落盘的第二遍特征就是当初训练用的第一遍特征。
    # 于是第二遍纯属重复劳动，占掉一半运行时。合并后结果不变、耗时减半。
    print(f"  单遍扫描：算特征（max_slices={max_slices}）", flush=True)
    tss = timestamps_in(df)
    step = max(1, len(tss) // max_slices) if max_slices else 1
    sampled = set(tss[::step])

    # ---- 特征级断点：算特征是本脚本最贵的一步（lc 实测 ~200s），
    # 而有些运行环境对单次进程时长有上限。给定 budget 时算到点就存盘退出，
    # 再次运行从断点续算。**续算与一次跑完等价**：`FeatureHistory` 与 `prev`
    # 一并存盘，恢复后 z-score 的滚动窗状态与不中断时逐位相同。
    fc = (os.path.join(cache_dir, f"{symbol}_{max_slices}_feats.pkl")
          if cache_dir else None)
    st = {"feats": [], "ts": [], "hist": FeatureHistory(window=500),
          "prev": None, "done": 0}
    if fc and os.path.exists(fc):
        with open(fc, "rb") as f:
            st = _pickle.load(f)
        print(f"  命中特征断点：已算 {st['done']} 个截面，续算", flush=True)

    t_feat = time.time()
    n_seen, hit_budget = 0, False
    for ts, sl in _iter_slices(df, sampled):
        n_seen += 1
        if n_seen <= st["done"]:            # 续算：跳过已完成的（遍历本身很廉价）
            continue
        if len(sl) >= 15:
            clean, rep = clean_slice(sl)
            if len(clean) >= 10:
                feats = compute_features(clean, prev_features=st["prev"],
                                         history=st["hist"])
                st["prev"] = feats
                st["feats"].append(feats)
                st["ts"].append(ts)
        st["done"] = n_seen
        if budget and time.time() - t_feat > budget:
            hit_budget = True
            break
    if fc:
        tmp = fc + ".part"
        with open(tmp, "wb") as f:
            _pickle.dump(st, f)
        os.replace(tmp, fc)
    if hit_budget:
        print(f"  预算用尽：已算 {st['done']}/{len(sampled)} 个截面并存盘，"
              f"再次运行本命令可继续", flush=True)
        return {"symbol": symbol, "incomplete": True,
                "done": st["done"], "total": len(sampled)}

    feats_list, ts_list = st["feats"], st["ts"]
    fdf = pd.DataFrame(feats_list)
    print(f"  特征行: {len(fdf)}", flush=True)

    # 训练 IsolationForest
    if len(fdf) >= 30:
        X = fdf[MODEL_FEATURES].to_numpy()
        model = AlertModel(contamination=0.06).fit(X)
        model.save(os.path.join(DATA_OUT, f"model_{symbol}.joblib"))
        print(f"  IsolationForest 已训练并保存", flush=True)
    else:
        model = None
        print(f"  特征不足，跳过 ML 训练", flush=True)

    # ---- 出预警（只需 feats，无需再遍历原始切片）
    print(f"  生成预警", flush=True)
    alert_rows, surface_rows, need_surf = [], [], []
    for i, (ts, feats) in enumerate(zip(ts_list, feats_list)):
        res = evaluate(feats, model=model)
        alert_rows.append({
            "timestamp": ts, "date": ts_to_date(ts), "symbol": symbol,
            "level": res["level"], "ml_score": res["ml_score"],
            "rule_level": res["rule_level"], "ml_level": res["ml_level"],
            "n_triggers": len(res["triggers"]),
            "triggers": "; ".join(t["reason"] for t in res["triggers"]),
            "trigger_rules": ",".join(t["rule"] for t in res["triggers"]),
            **{k: feats.get(k, np.nan) for k in FEATURE_COLUMNS},
        })
        # 抽样落曲面（高等级优先 + 定期采样）——判据与合并前逐字一致
        if (i % save_surfaces_every == 0 and res["level"] >= 1) or \
           (i % (save_surfaces_every * 3) == 0):
            need_surf.append(ts)

    # ---- 曲面：只对上面选中的那 ~30 个截面重新清洗一次（相对全量可忽略）
    print(f"  抽样曲面（{len(need_surf)} 个截面）", flush=True)
    if need_surf:
        from vol_surface.interpolation import fit_slice_iv
        want = set(need_surf)
        for ts, sl in _iter_slices(df, want):
            if len(sl) < 15:
                continue
            clean, _ = clean_slice(sl)
            if len(clean) < 10:
                continue
            ivres = fit_slice_iv(clean)
            for ot in ["call", "put"]:
                surf = ivres["iv_surface"].get(ot)
                if surf is None:
                    continue
                surface_rows.append({
                    "timestamp": ts, "symbol": symbol, "option_type": ot,
                    "dte_grid": ivres["dte_grid"].tolist(),
                    "moneyness_grid": ivres["moneyness_grid"].tolist(),
                    "surface": surf.tolist(),
                })

    adf = pd.DataFrame(alert_rows)
    fdf_out = adf[["timestamp", "date", "symbol"] + FEATURE_COLUMNS].copy()
    fdf_out.to_parquet(os.path.join(DATA_OUT, f"features_{symbol}.parquet"), index=False)
    adf.to_parquet(os.path.join(DATA_OUT, f"alerts_{symbol}.parquet"), index=False)
    if surface_rows:
        pd.DataFrame(surface_rows).to_parquet(os.path.join(DATA_OUT, f"surfaces_{symbol}.parquet"), index=False)
    print(f"  落盘: features/alerts/surfaces_{symbol}.parquet, 预警分布:", flush=True)
    print(adf["level"].value_counts().sort_index().to_string(), flush=True)
    print(f"  耗时 {time.time()-t0:.0f}s", flush=True)
    return {"symbol": symbol, "n_alerts": len(adf), "n_features": len(fdf_out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ag,si,au,sc", help="逗号分隔品种族")
    ap.add_argument("--event-window", type=int, default=2)
    ap.add_argument("--max-slices", type=int, default=400, help="每品种采样截面数上限")
    # 缓存默认写**系统临时目录**，不写 data_out。
    # 教训：初版默认 `data_out/.bf_cache`，7 个品种跑完积了 122MB——
    # 交付目录从 81MB 变成 203MB，而**所有验证门都看不见目录体积**
    # （本项目此前已因训练断点栽过一次，1.9G 里 1.78G 是残渣）。
    # 且部分挂载环境不允许删除，一旦写进去就清不掉。
    ap.add_argument("--cache-dir",
                    default=os.path.join(tempfile.gettempdir(), "vol_surface_bf_cache"),
                    help="载入/特征的缓存目录（只加速，删掉不影响结果）；传空串禁用。"
                         "默认在系统临时目录，**不要指向 data_out**")
    ap.add_argument("--budget", type=float, default=0.0,
                    help="算特征跑够这么多秒就存盘退出，可反复调用直到跑完（需 --cache-dir）")
    args = ap.parse_args()
    syms = args.symbols.split(",")
    summary = []
    for s in syms:
        s = s.strip()
        if not s:
            continue
        try:
            summary.append(build_symbol(s, args.event_window, args.max_slices,
                                        cache_dir=args.cache_dir or None,
                                        budget=args.budget))
        except Exception as e:
            print(f"  [{s}] 失败: {e}", flush=True)
            import traceback; traceback.print_exc()
    print("\n=== 汇总 ===")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
