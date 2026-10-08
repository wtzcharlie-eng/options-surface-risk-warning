"""calibrate_symbol_params — 把规则里的**原始量纲**阈值按品种改标定为等价分位数。

问题
----
`best_params.json` 的阈值是在 ag/sc 上网格搜索出来的，其中一部分作用在**原始量纲**
特征上（`r1_min`/`r1_min2` 用 convexity_violation、`r5_min`/`r5_min2` 用 arb_score、
`r2_*` 用 term_slope_roc、`r7_*` 用 iv_spike）。这些量的尺度跨品种差异极大——
`convexity_violation` 的 90 分位 ag 3.96、rb 0.73、cu 0.49，差近 8 倍。
于是同一个绝对阈值在校准品种上「刚好」，在别的品种上要么几乎不触发、要么频繁误触发。

做法
----
1. 在**校准品种**（ag+sc）的**训练期**分布里，求每个原始量纲阈值所处的分位 p；
2. 对每个品种，用它自己**训练期**分布的 p 分位作为新阈值。
即「保持在各自分布中的相对严格程度不变」。

**严禁用验证/测试期数据标定**——那等于把答案抄进阈值里。本脚本硬编码只读
`split_episodes` 的训练段。

实测效果（README §7.13）
------------------------
单独用是**负效果**（总体判别力 +10.5 → +8.2pp）——因为它同时放松了 R4 的阈值，
让本就有害的 R4 在 rb/cu 上触发得更多。**必须先关掉 R4**（现已是默认）。
关掉 R4 之后是小幅正效果，且验证集与测试集方向一致：
  同召回精确率 +0.41~+1.42pp（测试集）、+0.36~+1.18pp（验证集）。
量级远小于关闭 R4 本身（+1.9~+7.3pp），属于锦上添花。

用法::

    python scripts/calibrate_symbol_params.py      # 写出 data_out/symbol_params.json
    # 使用：evaluate_rules(feats, params=json.load(...)[symbol])
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.baseline import load_best_params
from drl.dataset import FEATURES, load_episodes, split_episodes
from vol_surface.alert_rules import DEFAULT_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 阈值 -> 它作用的原始量纲特征；None 表示 max(gamma_conc, vega_conc)
RAW_THRESHOLDS = {
    "r1_min": "convexity_violation", "r1_min2": "convexity_violation",
    "r2_watch": "term_slope_roc", "r2_warn": "term_slope_roc", "r2_ser": "term_slope_roc",
    "r4_watch": None, "r4_warn": None, "r4_ser": None,
    "r5_min": "arb_score", "r5_min2": "arb_score",
    "r7_watch": "iv_spike", "r7_warn": "iv_spike", "r7_ser": "iv_spike",
}
CALIB_SYMBOLS = ("ag", "sc")      # 原阈值就是在这两个品种上搜出来的


def _column(episodes: list, feat: str | None) -> np.ndarray:
    if feat is None:
        g = np.concatenate([e.X[:, FEATURES.index("gamma_concentration")] for e in episodes])
        v = np.concatenate([e.X[:, FEATURES.index("vega_concentration")] for e in episodes])
        return np.maximum(g, v)
    return np.concatenate([e.X[:, FEATURES.index(feat)] for e in episodes])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out", "symbol_params.json"))
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=False)
    train, _, _ = split_episodes(eps)          # 只用训练段
    if not train:
        raise SystemExit("训练集为空")

    base = {**DEFAULT_PARAMS, **load_best_params()}
    calib = [e for e in train if e.symbol in CALIB_SYMBOLS]
    if not calib:
        raise SystemExit(f"训练集中没有校准品种 {CALIB_SYMBOLS}")

    pct = {}
    for key, feat in RAW_THRESHOLDS.items():
        col = _column(calib, feat)
        pct[key] = float((col < base[key]).mean())

    symbols = sorted({e.symbol for e in train})
    out = {"_meta": {"calib_symbols": list(CALIB_SYMBOLS),
                     "train_episodes": len(train),
                     "equivalent_percentile": {k: round(v, 5) for k, v in pct.items()},
                     "note": "阈值只用训练期标定；R4 默认关闭，本表里的 r4_* 仅在 "
                             "r4_enabled=True 时才生效"}}
    for s in symbols:
        g = [e for e in train if e.symbol == s]
        p = dict(base)
        for key, feat in RAW_THRESHOLDS.items():
            p[key] = float(np.quantile(_column(g, feat), pct[key]))
        out[s] = p

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("等价分位：" + " ".join(f"{k}={v:.3f}" for k, v in pct.items()))
    for s in symbols:
        print(f"  {s}: r1_min2 {base['r1_min2']:.3f}→{out[s]['r1_min2']:.3f}  "
              f"r5_min2 {base['r5_min2']:.1f}→{out[s]['r5_min2']:.1f}  "
              f"r7_warn {base['r7_warn']:.4f}→{out[s]['r7_warn']:.4f}")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
