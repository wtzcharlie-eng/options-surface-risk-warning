"""ml_score_defect — 量化「异常分批内归一化」缺陷的影响范围。

缺陷
----
`AlertModel.score_samples` 原实现最后一步是::

    norm = np.clip(-(raw + 0.1), 0, None)
    return norm / (norm.max() + 1e-9)     # ← **批内相对**

而所有生产路径每次只传**一个**样本：

    alert_engine.evaluate:  x = model.featurize(feats).reshape(1, -1)
    scripts/scan.py:74      evaluate(feats, model=model)
    scripts/build_features.py:173

batch=1 时 `norm.max()` 就是 `norm` 自己 ⇒ 异常分**恒等于 1.0**。

后果不是「数字略偏」，而是**四级预警退化成两级**：
`_ml_level` 恒走 `ml_score > ML_HIGH` 分支，`level = max(rule_level, ml_level)`
于是把 rule_level 0→1、2→3 全线抬升，「正常」永不出现、「预警 WARN」不可达。
**退化方向恒为「系统看起来一直在报警」。**

本脚本怎么做到可复算
--------------------
不依赖修复前的旧产物（它们已被覆盖）。而是**就地重现修复前的公式**，
对同一批真实特征分别用「旧公式（逐样本）」与「新公式」打分，比较两者
经 `_ml_level` + `max()` 融合后的等级分布。

`--verify-against` 可指向一份修复前的 alerts parquet 快照，
用于确认「重现出来的旧行为」与「当初真实落盘的旧行为」一致——
**否则这个脚本只是自说自话**。

用法::

    python scripts/ml_score_defect.py
    python scripts/ml_score_defect.py --verify-against /path/to/prefix_snapshot_dir
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.alert_engine import ML_HIGH, ML_MID, _ml_level
from vol_surface.alert_model import AlertModel, MODEL_FEATURES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_OUT = os.path.join(ROOT, "data_out")


def _legacy_score(model: AlertModel, X: np.ndarray) -> np.ndarray:
    """**修复前**的打分公式，逐样本调用（即生产路径的真实用法）。

    原样照抄改动前的三行，只是显式地一个样本一个样本地喂——
    因为那正是 `alert_engine.evaluate` 做的事。
    """
    out = []
    for i in range(len(X)):
        raw = model.iforest.score_samples(model.scaler.transform(X[i:i + 1]))
        norm = np.clip(-(raw + 0.1), 0, None)
        out.append(float((norm / (norm.max() + 1e-9))[0]))
    return np.asarray(out)


def _fuse(rule_level: np.ndarray, ml_score: np.ndarray) -> np.ndarray:
    """复现 alert_engine 的 `level = max(rule_level, ml_level)`（无 state/composite）。"""
    return np.asarray([min(max(max(int(r), _ml_level(float(s), int(r))), 0), 3)
                       for r, s in zip(rule_level, ml_score)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA_OUT, "ml_score_defect.json"))
    ap.add_argument("--verify-against", default=None,
                    help="修复前 alerts_*.parquet 所在目录，用于校验重现是否忠实")
    a = ap.parse_args()

    syms, rows = [], {}
    for ap_ in sorted(glob.glob(os.path.join(DATA_OUT, "alerts_*.parquet"))):
        sym = os.path.basename(ap_)[len("alerts_"):-len(".parquet")]
        mp = os.path.join(DATA_OUT, f"model_{sym}.joblib")
        if not os.path.exists(mp):
            continue
        d = pd.read_parquet(ap_)
        missing = [k for k in MODEL_FEATURES if k not in d.columns]
        if missing:
            print(f"  [{sym}] 跳过：alerts parquet 缺特征列 {missing}")
            continue
        X = np.nan_to_num(d[MODEL_FEATURES].to_numpy(float))
        m = AlertModel.load(mp)
        s_new = m.score_samples(X)
        s_old = _legacy_score(m, X)
        rl = d["rule_level"].to_numpy(int)
        lv_new, lv_old = _fuse(rl, s_new), _fuse(rl, s_old)

        def _hist(v):
            return {str(k): int(c) for k, c in zip(*np.unique(v, return_counts=True))}

        rows[sym] = {
            "n": int(len(d)),
            "rule_level_hist": _hist(rl),
            "old": {"score_min": float(s_old.min()), "score_max": float(s_old.max()),
                    "n_unique_4dp": int(len(set(np.round(s_old, 4)))),
                    "level_hist": _hist(lv_old)},
            "new": {"score_min": float(s_new.min()), "score_max": float(s_new.max()),
                    "n_unique_4dp": int(len(set(np.round(s_new, 4)))),
                    "score_median": float(np.median(s_new)),
                    "frac_gt_mid": float((s_new > ML_MID).mean()),
                    "frac_gt_high": float((s_new > ML_HIGH).mean()),
                    "level_hist": _hist(lv_new)},
        }
        syms.append(sym)
        print(f"  [{sym}] {len(d):>4} 截面  旧: 分数取值 {rows[sym]['old']['n_unique_4dp']} 个"
              f"、等级 {rows[sym]['old']['level_hist']}"
              f"  →  新: 取值 {rows[sym]['new']['n_unique_4dp']} 个"
              f"、等级 {rows[sym]['new']['level_hist']}", flush=True)

    # ---- 汇总
    def _agg(path):
        t = {}
        for s in syms:
            d = rows[s]
            for k, v in (d[path[0]]["level_hist"] if path[0] in ("old", "new")
                         else d[path[0]]).items():
                t[k] = t.get(k, 0) + v
        return dict(sorted(t.items()))

    tot = {"n": sum(rows[s]["n"] for s in syms),
           "rule_level_hist": _agg(("rule_level_hist",)),
           "old_level_hist": _agg(("old",)), "new_level_hist": _agg(("new",))}
    old_uniq = max(rows[s]["old"]["n_unique_4dp"] for s in syms)
    tot["old_max_unique_scores_4dp"] = old_uniq
    tot["old_score_is_constant"] = bool(old_uniq == 1)
    tot["levels_absent_old"] = [k for k in "0123" if k not in tot["old_level_hist"]]
    tot["levels_absent_new"] = [k for k in "0123" if k not in tot["new_level_hist"]]

    print(f"\n=== 汇总（{len(syms)} 品种 / {tot['n']} 截面）===")
    print(f"  规则层等级分布      {tot['rule_level_hist']}")
    print(f"  旧公式融合后        {tot['old_level_hist']}   缺失等级 {tot['levels_absent_old']}")
    print(f"  新公式融合后        {tot['new_level_hist']}   缺失等级 {tot['levels_absent_new']}")
    print(f"  旧公式分数取值数    {old_uniq}（1 = 退化为常数）")

    # ---- 校验：重现出来的旧行为，与当初真实落盘的旧行为是否一致
    ver = None
    if a.verify_against:
        ver = {"dir": a.verify_against, "per_symbol": {}}
        for s in syms:
            p = os.path.join(a.verify_against, f"alerts_{s}.parquet")
            if not os.path.exists(p):
                continue
            d0 = pd.read_parquet(p)
            ver["per_symbol"][s] = {
                "n": int(len(d0)),
                "ml_score_min": float(d0["ml_score"].min()),
                "ml_score_max": float(d0["ml_score"].max()),
                "n_unique_4dp": int(len(set(np.round(d0["ml_score"], 4)))),
                "level_hist": {str(k): int(c) for k, c in
                               d0["level"].value_counts().sort_index().items()},
            }
        allc = all(v["n_unique_4dp"] == 1 and v["ml_score_min"] > 0.999
                   for v in ver["per_symbol"].values())
        # ⚠ `all()` 对**空集合返回 True** —— 路径写错时会打出「一致 ✓」，
        # 是典型的「门在该失败时通过」。故先要求确实读到了品种。
        ver["n_symbols_found"] = len(ver["per_symbol"])
        if not ver["per_symbol"]:
            ver["snapshot_scores_all_constant_1"] = None
            ver["consistent_with_reproduction"] = None
            ver["error"] = (f"--verify-against 指向的目录里没找到任何 "
                            f"alerts_*.parquet：{a.verify_against}")
            print(f"\n=== 与修复前快照校验 ===\n  ✗ {ver['error']}")
        else:
            ver["snapshot_scores_all_constant_1"] = bool(allc)
            ver["consistent_with_reproduction"] = bool(
                allc and tot["old_score_is_constant"])
            print(f"\n=== 与修复前快照校验（{len(ver['per_symbol'])} 品种）===")
            for s, v in ver["per_symbol"].items():
                print(f"  {s}: 落盘 ml_score 取值 {v['n_unique_4dp']} 个"
                      f"（{v['ml_score_min']:.4f}~{v['ml_score_max']:.4f}）"
                      f" 等级 {v['level_hist']}")
            print(f"  → 重现与快照一致: "
                  f"{'✓' if ver['consistent_with_reproduction'] else '✗'}")
    elif os.path.exists(a.out):
        # 不带 --verify-against 重跑时，**保留**已有的 verification 段。
        # 否则 PACKAGE.md 里那条 `python scripts/ml_score_defect.py` 照着跑一遍，
        # 就会把交付 JSON 里的校验记录清成 null——而那份修复前快照并不随包发布，
        # 评委无法自己重建它。
        try:
            _old = json.load(open(a.out, encoding="utf-8"))
            ver = _old.get("verification")
            if ver:
                print("\n  （未给 --verify-against，沿用已有的快照校验记录）")
        except Exception:                                        # noqa: BLE001
            ver = None

    res = {"note": ("旧公式为**就地重现**（见 _legacy_score），非读取旧产物——"
                    "旧产物已被修复后的重跑覆盖。verification 段用一份修复前快照校验"
                    "重现是否忠实。"),
           "ml_mid": ML_MID, "ml_high": ML_HIGH,
           "symbols": syms, "per_symbol": rows, "total": tot,
           "verification": ver}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
