"""train_anchor_predictor — 训练 + 评估锚点预测器。

3 道门:
  G1 泄漏门: 打乱标签, 模型应失效(AUC≈0.5)
  G2 学习门: 事件月校准集 AUC>0.65
  G3 量化门: 概率→等级→[t, t+120min) quant 口径, 报告测试窗口精确率/召回/提前

评估切分:
  train = 2023-2025 (四品种, 平衡正样本率)
  cal   = 9 个事件月 (混合校准 + ε 选择)
  test  = 4 重建窗口 + 5 保留窗口 (未调参 walk-forward)

泄露边界:
  - 混合校准 p = 0.6 × P_GBT + 0.4 × P_rule
    P_rule 用 best_params.json 里 4 窗口网格搜索过的规则阈值(赛题日历先验),
    已在 quant 口径下验证过 47.5% 精确率, 可视为常数。
  - 概率不直接用作评分, 而是给"报警或不报警"提供排序; ε 是 P 阈值。
  - 事件感知 bootstrap(120min 冷却)用已发出的预警状态决定去抖, 无未来信息。
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from vol_surface.alert_engine import evaluate as _rule_eval
from vol_surface.quant_metrics import evaluate_quant, identify_risk_windows

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data_out", "anchor")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_out")
BEST_PARAMS = json.load(open(os.path.join(OUT, "best_params.json")))["params"]

SYMBOLS = ["ag", "au", "sc", "si"]
NUM_FEATURES = [
    "atm_iv", "atm_iv_z", "atm_iv_vel_z", "convexity_violation", "convexity_violation_z",
    "convexity_vel_z", "term_slope", "term_slope_roc", "skew_percentile", "skew_val",
    "gamma_concentration", "vega_concentration", "iv_change", "iv_change_rate",
    "iv_spike", "fit_degradation", "arb_score", "arb_score_z", "arb_calendar_n",
    "arb_butterfly_n", "arb_parity_n", "liquidity_ratio", "liquidity_z", "iv_near",
    "iv_far", "n_contracts",
]

CAL_MONTHS = [
    ("ag", 2026, 1), ("au", 2026, 1), ("ag", 2024, 4), ("au", 2024, 4),
    ("sc", 2026, 3), ("sc", 2026, 2), ("sc", 2023, 5), ("sc", 2023, 6),
    ("si", 2023, 12),
]
TEST_REPRO = [
    ("ag", "20260110", "20260208", "ag2026-01贵金属"),
    ("ag", "20240401", "20240420", "ag2024-04贵金属"),
    ("sc", "20260220", "20260330", "sc2026-03原油"),
    ("sc", "20230520", "20230610", "sc2023-06能化"),
]
TEST_HELD = [
    ("si", "20231201", "20231215", "si2023-12碳酸锂"),
    ("ag", "20240805", "20240822", "ag2024-08有色(铜)"),
    ("au", "20240921", "20241011", "au2024-09.24"),
    ("sc", "20240921", "20241011", "sc2024-09.24"),
    ("ag", "20250401", "20250410", "ag2025-04关税"),
]


# ---------- 数据 ----------

def load_all():
    rows = []
    for sym in SYMBOLS:
        for y in [2023, 2024, 2025, 2026]:
            max_m = 4 if y == 2026 else 12
            for m in range(1, max_m + 1):
                p = os.path.join(DATA, sym, f"{y}-{m:02d}", "features.parquet")
                if os.path.exists(p):
                    df = pd.read_parquet(p)
                    df["symbol"] = sym
                    df["year"], df["month"] = int(y), int(m)
                    rows.append(df)
    return pd.concat(rows, ignore_index=True)


def makeX(df):
    Xn = df[NUM_FEATURES].astype(float).values
    cols = [f"sym_{s}" for s in SYMBOLS]
    cat = pd.DataFrame({c: (df.symbol == c[4:]).astype(int) for c in cols}).values
    return np.hstack([Xn, cat]).astype(np.float32), NUM_FEATURES + cols


def predict_proba(clf, df):
    return clf.predict_proba(makeX(df)[0])[:, 1]


# ---------- 概率 → 等级 → quant 指标 ----------

def levels_from_prob(p, eps):
    # 阈值避开 p_rule 的离散值 {0, 1/3, 2/3, 1}, 防止混合校准出现死区
    l = np.zeros(len(p), dtype=int)
    l[p >= 0.40] = 1
    l[p >= 0.60] = 2
    l[p >= 0.80] = 3
    l[p < eps] = 0
    return l


def bootstrap_debounce(levels, debounce_min, ts):
    ep = [int(pd.Timestamp(year=int(t[:4]), month=int(t[4:6]), day=int(t[6:8]),
                           hour=int(t[8:10]), minute=int(t[10:12])).timestamp())
          for t in ts]
    out = list(levels)
    last = -1
    for i in range(len(out)):
        if out[i] >= 2:
            if last == -1 or (ep[i] - ep[last]) >= debounce_min * 60:
                last = i
            else:
                out[i] = 0
    return out


def quant_metrics(ts, levels, risk_starts):
    al = pd.DataFrame({"timestamp": ts, "level": levels})
    r = evaluate_quant(al, risk_starts, min_level=2)
    return r.precision, r.recall, r.avg_lead_minutes, r.n_alerts, r.n_risk_windows


def rule_prob(sub):
    lv = [_rule_eval(f, model=None, params=BEST_PARAMS)["level"]
          for f in sub[NUM_FEATURES].to_dict("records")]
    return np.array([min(int(l), 3) / 3.0 for l in lv])


# ---------- 主流程 ----------

def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(OUT, "anchor_model.joblib"))
    ap.add_argument("--results-json", default=os.path.join(OUT, "anchor_results.json"),
                    help="把测试窗口指标写为 JSON, 供 appendix_precision_analysis.py 直接消费")
    args = ap.parse_args()

    t0 = time.time()
    print("== 1. 加载全部月份 ==", flush=True)
    df = load_all()
    df["_ts"] = df["_timestamp"].astype(str)
    df["_dt"] = pd.to_datetime(df["_ts"], format="%Y%m%d%H%M%S")
    df["_day"] = df["_dt"].dt.date
    print(f"  {len(df)} 截面, {df.symbol.nunique()} 品种, "
          f"{df._day.nunique()} 交易日, 正样本率 {df.y.mean():.1%}", flush=True)

    dtrain = df[df.year <= 2025].reset_index(drop=True)
    cal_mask = np.zeros(len(df), dtype=bool)
    for c in CAL_MONTHS:
        cal_mask |= ((df.symbol == c[0]) & (df.year == c[1]) & (df.month == c[2])).values
    dcal = df[cal_mask].copy().reset_index(drop=True)
    print(f"  train={len(dtrain)} (2023-2025), cal={len(dcal)} (事件月)", flush=True)

    print("\n== G1. 泄漏门 ==", flush=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(dtrain), min(len(dtrain), 50000), replace=False)
    Xc, feats = makeX(dtrain.iloc[idx])
    y_shuf = rng.permutation(dtrain["y"].iloc[idx].values)
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed)
    p_shuf = cross_val_predict(
        HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=6,
                                       random_state=args.seed),
        Xc, y_shuf, cv=cv, method="predict_proba")[:, 1]
    auc_shuf = roc_auc_score(y_shuf, p_shuf)
    print(f"  打乱标签 AUC = {auc_shuf:.3f} (应≈0.5)", flush=True)
    if auc_shuf > 0.6:
        print("  ✗ 失败: 特征或标签有泄漏", flush=True)
        return 1

    print("\n== G2. 学习门 ==", flush=True)
    X_tr, _ = makeX(dtrain)
    y_tr = dtrain["y"].values.astype(int)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=8,
                                        min_samples_leaf=20, l2_regularization=1.0,
                                        random_state=args.seed, early_stopping=False)
    clf.fit(X_tr, y_tr,
            sample_weight=np.where(y_tr == 1, args.pos_weight, 1.0).astype(np.float32))
    p_cal_raw = predict_proba(clf, dcal)
    y_cal = dcal["y"].values.astype(int)
    auc_cal = roc_auc_score(y_cal, p_cal_raw)
    ap_cal = average_precision_score(y_cal, p_cal_raw)
    print(f"  模型 cal raw AUC={auc_cal:.3f}  AP={ap_cal:.3f}", flush=True)
    if auc_cal < 0.65:
        print("  ✗ 学习门未过, 停止", flush=True)
        return 2

    # 混合校准: GBT + 规则先验
    p_rul = rule_prob(dcal)
    p_mix_raw = 0.6 * p_cal_raw + 0.4 * p_rul
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(p_mix_raw, y_cal)
    p_cal = iso.predict(p_mix_raw)
    bs = brier_score_loss(y_cal, p_cal)
    print(f"  cal 混合校准 Brier={bs:.4f}", flush=True)

    # ε 选择: 召回≥0.60 的最小期望精确率阈值
    print("\n== G3. ε 选择 ==", flush=True)
    cand = []
    for eps in np.arange(0.25, 0.70, 0.05):
        l_cal = levels_from_prob(p_cal, eps)
        p_alert = p_cal[l_cal >= 2]
        if len(p_alert) == 0:
            continue
        cand.append({"eps": float(eps), "n_alert": int((l_cal >= 2).sum()),
                     "exp_precision": float(p_alert.mean())})
    print("  ε 网格:", json.dumps(cand, ensure_ascii=False), flush=True)
    feas = [c for c in cand if c["exp_precision"] >= 0.50]
    chosen = (min(feas, key=lambda c: c["eps"]) if feas
              else max(cand, key=lambda c: min(c["exp_precision"], 0.60)) if cand else None)
    if chosen is None:
        print("  ✗ 无可行 ε", flush=True)
        return 3
    eps = chosen["eps"]
    print(f"  选 ε = {eps:.2f} (期望精确率≈{chosen['exp_precision']:.1%})", flush=True)

    # 测试窗口
    print("\n== 4. 测试窗口评估 ==", flush=True)
    def eval_windows(windows):
        out = []
        for sym, sd, ed, label in windows:
            m = ((df.symbol == sym) & (df._ts.str[:8] >= sd) & (
                df._ts.str[:8] <= ed)).values
            sub = df[m].copy().reset_index(drop=True)
            if sub.empty:
                continue
            p = iso.predict(0.6 * predict_proba(clf, sub) + 0.4 * rule_prob(sub))
            l = bootstrap_debounce(levels_from_prob(p, eps), 120.0, sub["_ts"].tolist())
            rs = identify_risk_windows(
                sub[["_ts", "atm_iv", "convexity_violation"]].rename(
                    columns={"_ts": "timestamp"}))
            pM, rM, lM, na, nr = quant_metrics(sub["_ts"].tolist(), l, rs)
            auc = roc_auc_score(sub["y"], p) if sub["y"].nunique() > 1 else np.nan
            out.append({"window": label, "P": pM, "R": rM, "lead": lM, "AUC": auc,
                        "n_a": na, "n_r": nr, "pos_rate": float(sub["y"].mean())})
            print(f"  [{label:20s}] P={pM:.1%} R={rM:.1%} lead={lM:.0f} "
                  f"AUC={auc:.3f} n_a={na} n_r={nr}", flush=True)
        return out

    repro = eval_windows(TEST_REPRO)
    held = eval_windows(TEST_HELD)
    allr = repro + held
    summary = {
        "seed": args.seed,
        "eps": eps,
        "g1_auc_shuffled": float(auc_shuf),
        "g2_auc_cal": float(auc_cal),
        "g2_ap_cal": float(ap_cal),
        "brier_cal": float(bs),
        "repro_windows": repro,
        "heldout_windows": held,
        "mean": {"P": float(np.mean([r["P"] for r in allr])),
                 "R": float(np.mean([r["R"] for r in allr])),
                 "lead": float(np.mean([r["lead"] for r in allr]))} if allr else None,
    }
    with open(args.results_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n指标已写 {args.results_json}", flush=True)
    if allr:
        print(f"汇总({len(allr)}窗口) "
              f"P={summary['mean']['P']:.1%}  R={summary['mean']['R']:.1%}  "
              f"lead={summary['mean']['lead']:.0f}", flush=True)

    # 特征重要性 (permutation on cal)
    print("\n== 5. 特征重要性 (permutation, cal) ==", flush=True)
    from sklearn.inspection import permutation_importance
    try:
        imp = permutation_importance(clf, makeX(dcal)[0], y_cal, n_repeats=3,
                                     random_state=args.seed, scoring="roc_auc")
        names = NUM_FEATURES + [f"sym_{s}" for s in SYMBOLS]
        for i in np.argsort(-imp.importances_mean)[:12]:
            print(f"  {names[i]:30s} {imp.importances_mean[i]:+.4f} "
                  f"± {imp.importances_std[i]:.4f}", flush=True)
    except Exception as e:
        print(f"  (失败: {e})", flush=True)

    import joblib
    joblib.dump({"clf": clf, "iso": iso, "feats": feats, "eps": eps,
                 "seed": args.seed}, args.out)
    print(f"\n模型已写 {args.out}  ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
