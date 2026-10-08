"""coverage_2022 — 量化「2022 年数据未纳入」这条局限到底有多大。

为什么要有这个脚本
------------------
README §12 长期挂着「2022 年数据未纳入——`archive/` 下有 747 个月度文件，
而 anchor 从 2023-01 起，**补进去可增加历史 regime 覆盖**」。
那句话暗示存在可观的上行空间，但**从未量化过**。

实测后结论相反：对交付的 7 个品种而言，2022 年数据近乎不存在。本脚本把这个判断
变成可复算的数字，避免它停留在「我看了一眼觉得不值得」。

结论要点（由本脚本产出，勿手抄）
--------------------------------
- lc / cu **完全没有** 2022 数据；ag / si / rb 各只有 1 个日历月（均为 2022-12）；
  只有 au 有 6 个月。合计约 12 个品种月，相对现有 263 幕仅 +4.6%。
- 增量**极不均衡**：一半来自 au，会把本已不平衡的品种分布推得更偏。
- 「给首月预热滚动窗」这个潜在收益也不成立：实测各品种首月的 |atm_iv_z| 均值
  与随后三个月同量级，零值占比也接近，看不出系统性的冷启动损失。
- 代价却是全量的：训练集一改 → 15 个 seed 全部重训 → 所有指标、所有文档数字、
  十轮盲评的材料基线全部失效。

用法::

    python scripts/coverage_2022.py       # → data_out/coverage_2022.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYMS = ("ag", "au", "sc", "si", "lc", "cu", "rb")


def archive_coverage() -> dict:
    """按品种统计 archive 里各年的**不同日历月**数（不是文件数——一个日历月
    下有多个合约月文件，按文件数会把覆盖度夸大数倍）。"""
    mo = defaultdict(lambda: defaultdict(set))
    size = defaultdict(int)
    pat = re.compile(r"archive/[^/]+/([a-z]+)\d+/options/(\d{4})/(\d{2})/")
    for f in glob.glob(os.path.join(ROOT, "archive", "*", "*", "options",
                                    "*", "*", "*.parquet")):
        m = pat.search(f.replace(os.sep, "/"))
        if not m:
            continue
        sym, year, month = m.groups()
        mo[sym][year].add(month)
        if year == "2022":
            size[sym] += os.path.getsize(f)
    return {s: {"months_2022": sorted(mo[s].get("2022", [])),
                "n_months_2022": len(mo[s].get("2022", [])),
                "mb_2022": round(size[s] / 2 ** 20, 1),
                "n_months_by_year": {y: len(v) for y, v in sorted(mo[s].items())}}
            for s in SYMS}


def warmup_check() -> dict:
    """首月是否存在滚动窗冷启动损失。

    若存在，2022 数据的价值就不只是「多 4.6% 样本」，而是给 2023-01 预热——
    那会是纳入它的**真正理由**。故必须单独查，不能想当然。
    判据：首月的 |atm_iv_z| 均值与零值占比，对比随后三个月。
    """
    from drl.dataset import FEATURES, load_episodes
    eps = load_episodes(os.path.join(ROOT, "data_out", "anchor"), verify=False)
    j = FEATURES.index("atm_iv_z")
    by = defaultdict(list)
    for e in eps:
        by[e.symbol].append(e)
    out = {}
    for s, v in by.items():
        v.sort(key=lambda e: e.ym)
        z0 = v[0].X[:, j]
        rest = v[1:4]
        zr = np.concatenate([x.X[:, j] for x in rest]) if rest else z0
        out[s] = {"first_ym": v[0].ym,
                  "first_abs_z_mean": round(float(np.abs(z0).mean()), 3),
                  "first_zero_frac": round(float((z0 == 0).mean()), 4),
                  "next3_abs_z_mean": round(float(np.abs(zr).mean()), 3),
                  "next3_zero_frac": round(float((zr == 0).mean()), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out",
                                                  "coverage_2022.json"))
    args = ap.parse_args()

    cov = archive_coverage()
    warm = warmup_check()

    n_add = sum(c["n_months_2022"] for c in cov.values())
    from drl.dataset import load_episodes
    n_now = len(load_episodes(os.path.join(ROOT, "data_out", "anchor"),
                              verify=False))
    zero_syms = [s for s, c in cov.items() if c["n_months_2022"] == 0]
    top = max(cov.items(), key=lambda kv: kv[1]["n_months_2022"])

    res = {
        "per_symbol": cov,
        "warmup": warm,
        "summary": {
            "n_symbol_months_2022": n_add,
            "n_episodes_now": n_now,
            "relative_gain": round(n_add / n_now, 4),
            "symbols_with_zero_2022": zero_syms,
            "largest_contributor": {"symbol": top[0],
                                    "n_months": top[1]["n_months_2022"],
                                    "share": round(top[1]["n_months_2022"] / n_add, 3)
                                    if n_add else 0.0},
            # 冷启动是否成立：首月 |z| 均值与随后三月相差是否超过 30%
            "warmup_deficit_symbols": [
                s for s, w in warm.items()
                if w["next3_abs_z_mean"] > 0
                and abs(w["first_abs_z_mean"] - w["next3_abs_z_mean"])
                / w["next3_abs_z_mean"] > 0.30],
        },
    }
    # 冷启动与可补性的交集：有冷启动损失、**且** 2022 确实有数据可补的品种，
    # 才是「纳入 2022」能真正改善的对象。实测这个交集为空——唯一有冷启动迹象的
    # cu 恰好是 2022 零数据的两个品种之一，补也补不了。
    deficit = res["summary"]["warmup_deficit_symbols"]
    fixable = [s for s in deficit if cov.get(s, {}).get("n_months_2022", 0) > 0]
    res["summary"]["warmup_deficit_fixable_by_2022"] = fixable

    res["verdict"] = (
        "不纳入。2022 年对交付品种近乎无数据（"
        + "、".join(f"{s} {cov[s]['n_months_2022']} 月" for s in SYMS)
        + f"），合计 {n_add} 个品种月、相对现有 {n_now} 幕仅 "
        f"+{res['summary']['relative_gain']:.1%}；"
        f"且 {res['summary']['largest_contributor']['share']:.0%} 来自单一品种 "
        f"{res['summary']['largest_contributor']['symbol']}，会加剧品种不平衡。"
        f"「给首月预热滚动窗」这个理由也不成立：有冷启动迹象的品种是 "
        f"{deficit or '无'}，其中 2022 有数据可补的是 {fixable or '无'}"
        f"——{'交集为空，补也补不到需要它的地方' if not fixable else '仅 ' + str(fixable)}。"
        "而代价是全量重训与全部文档数字失效。")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"{'品种':>6s} {'2022 月数':>9s} {'月份':>22s} {'大小':>8s}")
    for s in SYMS:
        c = cov[s]
        print(f"{s:>6s} {c['n_months_2022']:>9d} "
              f"{','.join(c['months_2022']) or '—':>22s} {c['mb_2022']:>7.1f}MB")
    print(f"\n合计 {n_add} 个品种月 / 现有 {n_now} 幕 = "
          f"+{res['summary']['relative_gain']:.1%}")
    print(f"零数据品种：{zero_syms or '无'}")
    print(f"冷启动确有损失的品种：{res['summary']['warmup_deficit_symbols'] or '无'}")
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
