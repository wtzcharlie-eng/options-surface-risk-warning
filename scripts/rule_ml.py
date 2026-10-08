"""rule_ml — 问题1「规则 + ML 融合」在 anchor 连续口径上的达标研究。

**判据写于读测试集之前**，见 `data_out/rule_ml_preregistration.md`。

为什么走这条路
--------------
§7.21 已证伪**过滤类杠杆**（冷却 −2.8~−3.7pp、流动性门 +0.32pp，36 配置无一可行），
纯规则的 P-R 前沿不含目标角点。剩下只有**加信息**这一类。

赛题问题1 本就要求「规则 + ML 融合」，但当前所有 anchor 连续口径评测都跑
`model=None`（纯规则）。而**既有 ML 模型不能用**：
  ① 泄漏 —— `train_anchor_predictor.py` 训练区间「2023–2025」覆盖了测试期 2025-07~2026-04；
  ② 方向错 —— `alert_engine` 的融合是 `max(rule, ml)`，**只加不减**，与「缺精确率」相反。

故本脚本：**只用 anchor 训练集重训** + **改成否决式融合**。

否决式融合
----------
    if ml_prob < thr: level = 0   # ML 认为不像，就压掉规则的预警
    else:             level = rule_level

**不用 max()**——那只会增加预警、拉低精确率。

用法::

    python scripts/rule_ml.py --underlying
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.dataset import (Normalizer, apply_continuous_risk, feature_names,
                         load_episodes, split_episodes, suppress_dead_zone)
from drl.metrics import aggregate
from drl.metrics_v2 import aggregate_v2, evaluate_v2
from vol_surface.alert_engine import evaluate as engine_eval
from vol_surface.alert_rules import DEFAULT_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 预登记参数，不得事后调整
_Z = ("r1_watch_z", "r1_warn_z", "r1_ser_z", "r3_watch", "r3_warn", "r3_ser",
      "r5_watch_z", "r5_warn_z", "r5_ser_z", "r8_watch", "r8_warn", "r8_ser")
ZMULTS = [0.40, 0.55, 0.70, 0.85]
THRS = [0.0, 0.15, 0.25, 0.35, 0.45]
TARGET = {"precision": 0.50, "recall": 0.60, "avg_lead_min": 30.0}
MODEL_KW = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=0)


def _pass3(m) -> bool:
    return bool(m["precision"] >= TARGET["precision"]
                and m["recall"] >= TARGET["recall"]
                and m["avg_lead_min"] >= TARGET["avg_lead_min"])


def rule_levels(group, FN, sp, z):
    """纯规则等级序列（死区过滤已施加）——走 alert_engine 交付本体。"""
    out = []
    for e in group:
        p = dict(sp.get(e.symbol) or {})
        for k in _Z:
            p[k] = DEFAULT_PARAMS[k] * z
        lv = np.array([int(engine_eval({k: float(e.X[i, j]) for j, k in enumerate(FN)},
                                       params=p, state=None, model=None)["level"])
                       for i in range(len(e))])
        out.append(suppress_dead_zone(e.ts, lv))
    return out


def fuse(levels, probs, thr):
    """否决式融合：ML 概率低于阈值则压掉该条预警。"""
    L = levels.copy()
    if thr > 0:
        L[probs < thr] = 0
    return L


def evaluate_cfg(group, lv_list, pv_list, thr) -> dict:
    ros = []
    for e, lv, pv in zip(group, lv_list, pv_list):
        L = fuse(lv, pv, thr)
        ros.append({"symbol": e.symbol, "ym": e.ym, "ts": e.ts, "actions": L,
                    "risk_idx": e.risk_idx, "total_reward": 0.0,
                    "n_alert": int((L >= 2).sum())})
    m = aggregate(ros)["micro"]
    return {k: m[k] for k in ("precision", "recall", "avg_lead_min", "n_alert")}


def _slice(probs, group):
    out, off = [], 0
    for e in group:
        out.append(probs[off:off + len(e)]); off += len(e)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out", "rule_ml.json"))
    ap.add_argument("--underlying", action="store_true")
    a = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    FN = feature_names(a.underlying)
    sp = json.load(open(os.path.join(ROOT, "data_out", "symbol_params.json"),
                        encoding="utf-8"))
    print(f"训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)} 幕；特征 {len(FN)} 维")

    Xtr = np.vstack([e.X for e in tr]); ytr = np.concatenate([e.y for e in tr])
    Xva = np.vstack([e.X for e in va]); yva = np.concatenate([e.y for e in va])
    Xte = np.vstack([e.X for e in te])

    res = {"preregistration": "data_out/rule_ml_preregistration.md",
           "target": TARGET, "model": "HistGradientBoostingClassifier",
           "model_params": MODEL_KW,
           "train_span": "anchor 训练集 2023-01~2024-12（验证/测试一律不参与训练）",
           "fusion": "否决式：ml_prob < thr → level=0（**不用 max()**）"}

    # ---------------- 训练 + 泄漏门 ----------------
    clf = HistGradientBoostingClassifier(**MODEL_KW).fit(Xtr, ytr)
    pva, pte = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yva, pva))

    rng = np.random.default_rng(0)
    clf_s = HistGradientBoostingClassifier(**MODEL_KW).fit(Xtr, rng.permutation(ytr))
    auc_s = float(roc_auc_score(yva, clf_s.predict_proba(Xva)[:, 1]))
    leak_ok = abs(auc_s - 0.5) <= 0.05
    res["leakage_gate"] = {"val_auc": auc, "val_auc_shuffled": auc_s,
                           "passed": leak_ok}
    print(f"\n泄漏门：验证集 AUC={auc:.4f}，打乱标签后 AUC={auc_s:.4f} "
          f"→ {'通过' if leak_ok else '**未通过，本研究作废**'}")
    if not leak_ok:
        res["verdict"] = {"passed": False, "text": "泄漏门未通过，本研究作废，不报任何数字。"}
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return

    # ---------------- 落盘模型与训练指纹 ----------------
    # 交付配置若只存在于评测脚本里，就是「有结果、没系统」：
    # 报告能复算，但 scan.py / 仪表板调不到，也无法部署。
    # 指纹用于门核对——**特征名与顺序**尤其关键：模型按列序吃数，
    # 顺序变了不会报错，只会**静默给出错误概率**。
    import joblib
    _mp = os.path.join(ROOT, "data_out", "rule_ml_model.joblib")
    joblib.dump(clf, _mp)
    fp = {"feature_names": list(FN), "n_features": len(FN),
          "train_span": ["2023-01", "2024-12"],
          "n_train_rows": int(Xtr.shape[0]),
          "pos_rate_train": float(ytr.mean()),
          "model_params": MODEL_KW,
          "sklearn": __import__("sklearn").__version__}
    res["model_artifact"] = {"path": "data_out/rule_ml_model.joblib",
                             "fingerprint": fp}
    json.dump(fp, open(os.path.join(ROOT, "data_out", "rule_ml_model.fingerprint.json"),
                       "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  模型已落盘 → data_out/rule_ml_model.joblib "
          f"（{fp['n_features']} 维特征，{fp['n_train_rows']:,} 行，"
          f"正样本率 {fp['pos_rate_train']:.1%}）")

    pv_by_ep, pt_by_ep = _slice(pva, va), _slice(pte, te)

    # ---------------- 验证集选点（测试集不读） ----------------
    print("\n=== 验证集网格 ===")
    grid = []
    for z in ZMULTS:
        lv = rule_levels(va, FN, sp, z)
        for thr in THRS:
            m = evaluate_cfg(va, lv, pv_by_ep, thr)
            grid.append({"zmult": z, "thr": thr, **m, "pass3": _pass3(m)})
            print(f"  z={z:.2f} thr={thr:.2f} | P={m['precision']:.2%} "
                  f"R={m['recall']:.2%} lead={m['avg_lead_min']:.1f}m "
                  f"n={m['n_alert']} {'✓' if _pass3(m) else ''}")
    res["val_grid"] = grid
    feas = [r for r in grid if r["pass3"]]
    if not feas:
        res["picked"] = None
        res["verdict"] = {"passed": False,
                          "text": "验证集上无可行点 → ML 否决也推不动前沿，不评测试集。"}
        print("\n" + res["verdict"]["text"])
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return
    pick = max(feas, key=lambda r: r["precision"])
    res["picked"] = pick
    print(f"\n=== 验证集选中 zmult={pick['zmult']} thr={pick['thr']} "
          f"(P={pick['precision']:.2%} R={pick['recall']:.2%}) ===")

    # ---------------- 测试集：只评一次 ----------------
    lv_te = rule_levels(te, FN, sp, pick["zmult"])
    m_ml = evaluate_cfg(te, lv_te, pt_by_ep, pick["thr"])
    m_base = evaluate_cfg(te, lv_te, pt_by_ep, 0.0)          # 同 zmult 的纯规则
    # **两个口径都要报**：概览卡引用的是新口径（事件级），而选点用的是旧口径。
    # 只报一个会让人以为另一个也这样——本项目最常见的一类误导。
    def _v2_of(thr):
        per = []
        for e, lv, pv in zip(te, lv_te, pt_by_ep):
            per.append(evaluate_v2(e.ts, fuse(lv, pv, thr), e.risk_idx))
        m = aggregate_v2(per)
        return {k: m[k] for k in ("precision", "recall", "avg_lead_min")}
    v2_ml, v2_base = _v2_of(pick["thr"]), _v2_of(0.0)
    res["test"] = {"fused": {**m_ml, "pass3": _pass3(m_ml)},
                   "rule_only_same_zmult": {**m_base, "pass3": _pass3(m_base)},
                   "fused_v2": {**v2_ml, "pass3": _pass3(v2_ml)},
                   "rule_only_same_zmult_v2": {**v2_base, "pass3": _pass3(v2_base)}}
    print(f"  [新口径] 融合   P={v2_ml['precision']:.2%} R={v2_ml['recall']:.2%} "
          f"lead={v2_ml['avg_lead_min']:.0f}m {'✓' if _pass3(v2_ml) else '✗'}")
    print(f"  [新口径] 纯规则 P={v2_base['precision']:.2%} R={v2_base['recall']:.2%} "
          f"lead={v2_base['avg_lead_min']:.0f}m {'✓' if _pass3(v2_base) else '✗'}")
    print(f"  融合   P={m_ml['precision']:.2%} R={m_ml['recall']:.2%} "
          f"lead={m_ml['avg_lead_min']:.1f}m n={m_ml['n_alert']} "
          f"{'✓ 三项全达标' if _pass3(m_ml) else '✗'}")
    print(f"  纯规则 P={m_base['precision']:.2%} R={m_base['recall']:.2%} "
          f"lead={m_base['avg_lead_min']:.1f}m n={m_base['n_alert']}")

    # ---------------- 同召回处对比：证明是移动前沿 ----------------
    pure = []
    for z in ZMULTS:
        lv = rule_levels(te, FN, sp, z)
        pure.append({"zmult": z, **evaluate_cfg(te, lv, pt_by_ep, 0.0)})
    ok60 = [r for r in pure if r["recall"] >= 0.60]
    p_at60 = min(ok60, key=lambda r: r["recall"]) if ok60 else None
    res["test_pure_frontier"] = pure
    res["test_p_at_recall60_pure"] = p_at60
    if p_at60:
        res["frontier_gain_pp"] = (m_ml["precision"] - p_at60["precision"]) * 100
        print(f"\n  纯规则在 R≈60% 处 P={p_at60['precision']:.2%}"
              f"（z={p_at60['zmult']}, R={p_at60['recall']:.2%}）")
        print(f"  → 融合后 {m_ml['precision']:.2%}，"
              f"**同召回量级上 {res['frontier_gain_pp']:+.2f}pp**")

    # ---------------- 并列 DRL ----------------
    wp = json.load(open(os.path.join(ROOT, "data_out", "drl", "workpoint.json"),
                        encoding="utf-8"))
    res["drl_same_caliber"] = {"lean": wp["lean"], "precision": wp["precision"],
                               "recall": wp["recall"], "avg_lead_min": wp["avg_lead_min"]}
    print(f"  同口径 DRL：P={wp['precision']:.2%} R={wp['recall']:.2%} "
          f"lead={wp['avg_lead_min']:.1f}m")

    res["verdict"] = {
        "passed": bool(_pass3(m_ml)),
        "text": (("**通过**：规则+ML 融合在 anchor 连续口径上三项全达标。"
                  "但必须同时写明：① §7.21「纯规则够不到」的结论**仍然成立**，"
                  "本结论只适用于**融合后**的系统；② 这已**不是纯固定阈值系统**；"
                  "③ 这不改变「固定阈值僵化、需要学习成分」的结论——恰恰再次印证了它。")
                 if _pass3(m_ml) else
                 "**未通过**：验证集上可行、测试集未达标，计入「验证集→测试集选择误差」，"
                 "不重选工作点、不换模型、不调网格。"),
    }
    print("\n" + res["verdict"]["text"])
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
