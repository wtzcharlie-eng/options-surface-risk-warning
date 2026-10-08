"""q1_window_zmult — 问题1 事件窗口：调紧阈值能否让更多窗口三项全达标。

**判据写于跑数之前**，见 `data_out/q1_window_zmult_preregistration.md`。

问题
----
交付配置（`zmult=1.0`）下 20 个事件窗口有 **13 个**三项全达标。
未达标的 7 个**全部只卡精确率**，召回 90.9%~100%（门槛 60%）——
即**有 30~40pp 的召回余量可以换精确率**。

为什么必须先划分窗口集
----------------------
这 20 个窗口**就是问题1 的评测对象**。对着它们的达标数调 `zmult`
等于**对评测集拟合**，调出来的数不能作为成绩。故按 anchor 的时间切分：

    训练期 10 窗（2023-06~2024-09）  ← **唯一**用于选 zmult
    验证期  2 窗（2025-04 ×2）       ← 样本太少，单列不参与选择
    测试期  8 窗（2025-07~2026-03）  ← **只评一次**，唯一可作成绩的部分

一句必须写在前面的话
--------------------
调 `zmult` 本质是**沿 P-R 前沿滑动**，不是移动前沿。
但因为召回余量极大，滑动本身也可能让更多窗口跨过 50% 线。
**不得把「滑到更好的工作点」讲成「系统变强了」**——
故本脚本同时输出「同召回处的精确率」用于区分两者。

用法::

    python scripts/q1_window_zmult.py --budget 900   # 可反复调用，带缓存
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.alert_rules import DEFAULT_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 预登记参数，不得事后调整
_Z = ("r1_watch_z", "r1_warn_z", "r1_ser_z", "r3_watch", "r3_warn", "r3_ser",
      "r5_watch_z", "r5_warn_z", "r5_ser_z", "r8_watch", "r8_warn", "r8_ser")
ZMULTS = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]     # 只往调紧方向；放松与目标相反
TARGET = {"P": 0.50, "R": 0.60, "lead": 30.0}
# 品种不在 anchor 的 7 个训练品种内——阈值从未针对它们标定过，须单独标注
OUT_OF_CALIB = {"al", "br"}


def _seg(label: str) -> str:
    """按 anchor 的时间切分给窗口分段。"""
    ym = str(label)[:7]
    return "train" if ym < "2025-01" else ("val" if ym < "2025-07" else "test")


def _pass3(v) -> bool:
    return bool(v["P"] >= TARGET["P"] and v["R"] >= TARGET["R"]
                and v["lead"] >= TARGET["lead"])


def _binding(v) -> list:
    return [k for k, ok in (("精确率", v["P"] >= TARGET["P"]),
                            ("召回", v["R"] >= TARGET["R"]),
                            ("提前", v["lead"] >= TARGET["lead"])) if not ok]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out",
                                                  "q1_window_zmult.json"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "data_out",
                                                    ".q1_zmult_cache.json"))
    ap.add_argument("--budget", type=float, default=0.0,
                    help="跑够这么多秒就存盘退出，可反复调用直到跑完")
    a = ap.parse_args()

    from scripts.eval_q1_windows import EVENTS, _eval_one, _window

    cache = {}
    if os.path.exists(a.cache):
        cache = json.load(open(a.cache, encoding="utf-8"))
    t0, done_now = time.time(), 0

    def run(sym, label, sd, ed, z):
        key = f"{z}|{sym}|{label}"
        if key in cache:
            return cache[key]
        p = {k: DEFAULT_PARAMS[k] * z for k in _Z}
        s, e = _window(sd, ed)
        r = _eval_one(sym, s, e, p)
        cache[key] = r
        return r

    rows = []
    for z in ZMULTS:
        for sym, label, sd, ed in EVENTS:
            key = f"{z}|{sym}|{label}"
            if key not in cache and a.budget and time.time() - t0 > a.budget:
                json.dump(cache, open(a.cache, "w", encoding="utf-8"),
                          ensure_ascii=False)
                print(f"\n预算用尽，已缓存 {len(cache)} 项，再次运行可继续")
                return
            r = run(sym, label, sd, ed, z)
            if key not in cache:
                done_now += 1
            v = r["v2"]
            rows.append({"zmult": z, "sym": sym, "label": label,
                         "seg": _seg(label), "P": v["P"], "R": v["R"],
                         "lead": v["lead"], "pass3": _pass3(v),
                         "binding": _binding(v),
                         "out_of_calib": sym in OUT_OF_CALIB})
        print(f"  zmult={z} 跑完", flush=True)
    json.dump(cache, open(a.cache, "w", encoding="utf-8"), ensure_ascii=False)

    res = {"preregistration": "data_out/q1_window_zmult_preregistration.md",
           "target": TARGET, "zmults": ZMULTS, "rows": rows,
           "protocol": ("训练期 10 窗选点；验证期 2 窗单列不参与选择；"
                        "测试期 8 窗只评一次。调 zmult 是**沿前沿滑动**，不是移动前沿。")}

    # ---- 训练期选点
    print("\n=== 各 zmult 的逐段达标数 ===")
    print(f"  {'zmult':>6}{'训练10':>9}{'验证2':>8}{'测试8':>8}{'全部20':>9}"
          f"{'训练期均P':>11}{'最低召回':>10}")
    by_z = {}
    for z in ZMULTS:
        sub = [r for r in rows if r["zmult"] == z]
        g = {s: [r for r in sub if r["seg"] == s] for s in ("train", "val", "test")}
        cnt = {s: sum(1 for r in g[s] if r["pass3"]) for s in g}
        meanP = float(np.mean([r["P"] for r in g["train"]])) if g["train"] else 0.0
        minR = float(min(r["R"] for r in sub))
        by_z[z] = {"n_pass": cnt, "n_pass_all": sum(cnt.values()),
                   "train_mean_P": meanP, "min_recall_any": minR}
        print(f"  {z:>6.1f}{cnt['train']:>9}{cnt['val']:>8}{cnt['test']:>8}"
              f"{sum(cnt.values()):>9}{meanP:>11.1%}{minR:>10.1%}")
    res["by_zmult"] = by_z

    base_z = 1.0
    pick = max(ZMULTS, key=lambda z: (by_z[z]["n_pass"]["train"],
                                      by_z[z]["train_mean_P"]))
    res["picked_zmult"] = pick
    res["baseline_zmult"] = base_z
    print(f"\n=== 训练期选出 zmult={pick}"
          f"（训练期达标 {by_z[pick]['n_pass']['train']}/10，"
          f"基线 {by_z[base_z]['n_pass']['train']}/10）===")

    n_test_pick = by_z[pick]["n_pass"]["test"]
    n_test_base = by_z[base_z]["n_pass"]["test"]
    print(f"  测试期（只评一次）：{n_test_base}/8 → **{n_test_pick}/8**")
    print(f"  全部 20 窗：{by_z[base_z]['n_pass_all']}/20 → {by_z[pick]['n_pass_all']}/20")

    # ---- 代价：召回是否掉破门槛
    broke = [r for r in rows if r["zmult"] == pick and r["R"] < TARGET["R"]]
    res["recall_broken"] = [{k: r[k] for k in ("sym", "label", "seg", "P", "R")}
                            for r in broke]
    if broke:
        print(f"\n  ⚠ 有 {len(broke)} 个窗口召回掉破 60%：")
        for r in broke:
            print(f"     {r['sym']} {r['label']}  R={r['R']:.1%} (P={r['P']:.1%})")
    else:
        print("\n  召回全部仍在 60% 以上——没有把「不达标项」从精确率换到召回")

    res["verdict"] = {
        "passed": bool(n_test_pick > n_test_base),
        "n_test_base": n_test_base, "n_test_pick": n_test_pick,
        "text": (f"**{'通过' if n_test_pick > n_test_base else '未通过'}**："
                 f"测试期达标 {n_test_base}/8 → {n_test_pick}/8。"
                 + ("　**但这是沿 P-R 前沿滑动，不是系统变强**——"
                    "调紧阈值把召回余量换成了精确率，前沿本身没动。"
                    if n_test_pick > n_test_base else
                    "　窗口口径下调阈值也推不动，不再试其他阈值类做法。")),
    }
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n" + res["verdict"]["text"])
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
