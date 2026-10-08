"""build_kg — 标定并装配期权风险知识图谱（问题2）。

用法::

    python scripts/build_kg.py                 # 全量标定 + 装配 + 落盘
    python scripts/build_kg.py --n-perm 500    # 提高置换检验精度（p 值下限 = 1/(N+1)）
    python scripts/build_kg.py --quick         # 少量幕 + 少量置换，冒烟用

产物（data_out/kg/）::

    kg_edges.json    全部候选边的标定统计（含未通过检验的，供审计）
    kg_cross.json    跨品种共现的标定统计
    kg_graph.json    装配好的图谱（纯 JSON，可 diff、可审计）

纪律：边在**训练期** 2023-01..2024-12 标定，在**测试期** 2025-07..2026-04 独立复算；
只有通过支撑度 + 置换检验 + lift 置信下界三条的边才进入推理图。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.dataset import FEATURES, load_episodes, split_episodes
from kg.build import build, summary
from kg.calibrate import (calibrate, calibrate_cross_symbol, save,
                          session_profile)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "data_out", "anchor")
OUT = os.path.join(ROOT, "data_out", "kg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-support", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--include-weak", action="store_true",
                    help="把未通过检验的边也放进图（默认不放，只留在 json 里备查）")
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    print("载入 anchor 数据集 ...")
    eps = load_episodes(args.anchor, verify=True)
    tr, va, te = split_episodes(eps)
    if args.quick:
        tr, te = tr[:20], te[:8]
        args.n_perm = min(args.n_perm, 30)
    print(f"  训练 {len(tr)} 幕 / {sum(len(e) for e in tr)} 截面；"
          f"测试 {len(te)} 幕 / {sum(len(e) for e in te)} 截面\n")

    print(f"标定 异常→后果 边（置换 {args.n_perm} 次）...")
    calib = calibrate(tr, te, FEATURES, n_perm=args.n_perm, seed=args.seed,
                      min_support=args.min_support)
    m = calib["meta"]
    print(f"\n  候选 {m['n_candidate']} 条 → 通过 {m['n_strong']} 条"
          f"（其中测试期按同一标准复现 {m['n_replicated']} 条、"
          f"仅满足宽松判据 {m.get('n_replicated_loose', 0)} 条）")

    print(f"\n标定 跨品种共现 边 ...")
    cross = calibrate_cross_symbol(tr, FEATURES, n_perm=args.n_perm, seed=args.seed)
    calib["cross_symbol"] = cross
    n_cs = sum(1 for c in cross if c["strong"])
    print(f"  候选 {len(cross)} 条 → 通过 {n_cs} 条")

    save({"edges": calib["edges"], "meta": calib["meta"]},
         os.path.join(args.out, "kg_edges.json"))
    save({"cross_symbol": cross}, os.path.join(args.out, "kg_cross.json"))

    # 交易时段画像：为「跨品种共现多半是日内季节性」这一判断留下可复算证据
    prof = session_profile(tr, FEATURES)
    save(prof, os.path.join(args.out, "kg_session.json"))
    if prof.get("by_symbol"):
        print("\n交易时段画像（R6 流动性异常）:")
        for s_, d in prof["by_symbol"].items():
            top = ", ".join(f"{h}点:{v:.0%}" for h, v in d["top_hours"][:3])
            print(f"    {s_}: n={d['n_fired']:5d} 夜盘占比={d['night_share']:.1%} "
                  f"最集中→{top}")
        print("    交易小时 Jaccard: " + "  ".join(
            f"{k}={v:.2f}" for k, v in prof["jaccard"].items()))

    g = build(calib, include_weak=args.include_weak)
    g.save(os.path.join(args.out, "kg_graph.json"))
    print("\n" + summary(g))
    print(f"\n完成，用时 {time.time() - t0:.0f}s → {args.out}/")


if __name__ == "__main__":
    main()
