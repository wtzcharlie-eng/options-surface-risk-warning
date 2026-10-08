"""cvar_workpoint — 为 CVaR 目标单独选工作点的补充研究。

**判据写于跑数之前**，见 `data_out/cvar_workpoint_preregistration.md`。
**前一份预登记的「未通过」结论（hold_days=5 @ lean=0.3）不被本脚本修改或覆盖。**

问题
----
前一次 CVaR 评测在交付工作点 `lean=0.3` 上未通过，根因经实测是**指标退化**：
该工作点下 189/191 个交易日都有预警（99.0%），`hold_days=5` 让每一天都被
某个预警的持有期覆盖，平均仓位恒等于下限 0.5，于是 `improve` 对所有策略恒为
+50%、`excess ≈ 0`。佐证：`恒报警` 与 `每8步机械报警` 的 excess 恰为 0.00%，
而唯一 excess 显著为正的是最稀疏的「只在开头报一次」（+2.30%）。

即：**累计奖励口径奖励密集预警，CVaR 口径奖励稀疏预警，两者要求相反的密度，
却共用了同一个工作点。** 本脚本为 CVaR 目标单独选点，回答「那次失败是
模型没有 CVaR 价值，还是工作点选错了目标」。

协议（预登记，不得事后调）
--------------------------
- 工作点**只在验证集上选**，测试集在选择过程中一次都不读；
- 候选点必须在验证集上同时满足**新口径召回 ≥60% 且精确率 ≥50%**——
  这条约束是防塌缩的关键：不加它，选择会直接跑到「只在开头报一次」，
  那是平凡策略不是预警系统；
- 满足约束者中取**验证集 CVaR excess 最大**；
- **规则引擎用完全相同的协议选它自己的稀疏工作点**，否则不是公平对照；
- CVaR 参数与前一份预登记完全一致（hold_days=5 / cut_to=0.5 / n_random=200）。

达标判据（与前一份相同，不放宽）
--------------------------------
`excess` 七品种等权均值 > 0，且 ≥4/7 品种 excess > 0。

用法::

    python scripts/cvar_workpoint.py --underlying
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

from drl.api import AlertAgentAPI
from drl.baseline import RulePolicy
from drl.dataset import (Normalizer, apply_continuous_risk, load_episodes,
                         split_episodes, suppress_dead_zone)
from drl.dqn import DQNAgent
from drl.env import AlertEnv, RewardSpec
from drl.metrics_v2 import aggregate_v2, evaluate_v2
from vol_surface.alert_rules import DEFAULT_PARAMS
from vol_surface.cvar_backtest import backtest_cvar_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "data_out", "anchor")

# ---- 预登记参数，不得事后调整
HOLD_DAYS, CUT_TO, N_RANDOM, SEED = 5, 0.5, 200, 0
LEAN_GRID = [0.3, 1.0, 2.5, 5.0, 7.0, 10.0, 15.0, 20.0]
ZMULT_GRID = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]   # 规则引擎的稀疏化旋钮
MIN_RECALL_V2, MIN_PRECISION_V2 = 0.60, 0.50
SYMS = ["ag", "au", "sc", "si", "lc", "cu", "rb"]


def daily_price(symbol: str) -> pd.DataFrame:
    fs = sorted(glob.glob(os.path.join(ANCHOR, symbol, "*", "underlying.parquet")))
    if not fs:
        return pd.DataFrame(columns=["date", "open", "close"])
    d = pd.concat([pd.read_parquet(f, columns=["_timestamp", "_S"]) for f in fs])
    ts = pd.to_datetime(d["_timestamp"].str.decode("utf-8"), format="%Y%m%d%H%M%S")
    px = (pd.DataFrame({"date": ts.dt.normalize(), "close": d["_S"].to_numpy()})
          .groupby("date", as_index=False).last())
    px["open"] = px["close"]
    return px[["date", "open", "close"]]


def alerts_from(rollouts: list) -> pd.DataFrame:
    ts, lv = [], []
    for r in rollouts:
        ts.append(np.asarray(r["ts"]))
        lv.append(np.asarray(r["actions"]))
    if not ts:
        return pd.DataFrame(columns=["timestamp", "level"])
    return pd.DataFrame({"timestamp": np.concatenate(ts),
                         "level": np.concatenate(lv)})


def _dz(rollouts):
    """死区过滤——与交付一致，规则与 DRL 同时施加。"""
    out = []
    for ro in rollouts:
        a = suppress_dead_zone(ro["ts"], ro["actions"])
        out.append({**ro, "actions": a, "n_alert": int((a >= 2).sum())})
    return out


def _v2(rollouts) -> dict:
    """新口径（事件级精确率 / 召回不设上限）——用于施加防塌缩约束。"""
    per = [evaluate_v2(r["ts"], r["actions"], r["risk_idx"]) for r in rollouts]
    return aggregate_v2(per)


def _cvar(rollouts, px_by_sym) -> dict:
    """把 rollout 按品种切开跑 CVaR，返回逐品种与汇总。"""
    by = {}
    for r in rollouts:
        by.setdefault(r["symbol"], []).append(r)
    per, ex, imp, rnd, pos = {}, [], [], [], []
    for sym, ros in by.items():
        px = px_by_sym.get(sym)
        if px is None or px.empty:
            continue
        res = backtest_cvar_path(alerts_from(ros), px, hold_days=HOLD_DAYS,
                                 cut_to=CUT_TO, direction="both",
                                 n_random=N_RANDOM, seed=SEED)
        e = float(res["improve"] - res["random_improve"])
        per[sym] = {"improve": float(res["improve"]),
                    "random_improve": float(res["random_improve"]),
                    "excess": e,
                    "avg_position": float(res["avg_position"])}
        ex.append(e); imp.append(float(res["improve"]))
        rnd.append(float(res["random_improve"])); pos.append(float(res["avg_position"]))
    return {"per_symbol": per,
            "excess_mean": float(np.mean(ex)) if ex else float("nan"),
            "improve_mean": float(np.mean(imp)) if imp else float("nan"),
            "random_improve_mean": float(np.mean(rnd)) if rnd else float("nan"),
            "avg_position_mean": float(np.mean(pos)) if pos else float("nan"),
            "n_positive": int(sum(1 for x in ex if x > 0)), "n_symbol": len(ex)}


def _alert_day_frac(rollouts) -> float:
    fr = []
    by = {}
    for r in rollouts:
        by.setdefault(r["symbol"], []).append(r)
    for sym, ros in by.items():
        ts = np.concatenate([np.asarray(r["ts"]) for r in ros])
        lv = np.concatenate([np.asarray(r["actions"]) for r in ros])
        d = pd.to_datetime(pd.Series(ts), format="%Y%m%d%H%M%S").dt.date
        tot = d.nunique()
        ad = d[lv >= 2].nunique()
        if tot:
            fr.append(ad / tot)
    return float(np.mean(fr)) if fr else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out",
                                                  "cvar_workpoint.json"))
    ap.add_argument("--underlying", action="store_true")
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()
    px = {s: daily_price(s) for s in SYMS}
    print(f"验证 {len(va)} 幕 / 测试 {len(te)} 幕")

    files = sorted(f for f in glob.glob(os.path.join(a.drl_dir, "agent_seed*.npz"))
                   if "shuf_" not in os.path.basename(f))
    agents, extra = [], None
    for f in files:
        ag, ex = DQNAgent.load(f)
        agents.append(ag); extra = extra or ex
    api = AlertAgentAPI(agents, Normalizer.from_dict(extra["normalizer"]), extra)

    sp = {}
    spp = os.path.join(ROOT, "data_out", "symbol_params.json")
    if os.path.exists(spp):
        sp = json.load(open(spp, encoding="utf-8"))

    res = {"preregistration": "data_out/cvar_workpoint_preregistration.md",
           "note": ("本研究只改工作点，不改 CVaR 口径参数；"
                    "前一份预登记的『未通过』结论（hold_days=5 @ lean=0.3）不被修改"),
           "params": {"hold_days": HOLD_DAYS, "cut_to": CUT_TO,
                      "n_random": N_RANDOM, "seed": SEED,
                      "min_recall_v2": MIN_RECALL_V2,
                      "min_precision_v2": MIN_PRECISION_V2},
           "selection": {}}

    # ================= 在验证集上选工作点（测试集一次都不读） =================
    def select(kind: str, grid, make_ro):
        rows = []
        for g in grid:
            ro = _dz(make_ro(va, g))
            v2 = _v2(ro)
            cv = _cvar(ro, px)
            ok = (v2["recall"] >= MIN_RECALL_V2
                  and v2["precision"] >= MIN_PRECISION_V2)
            rows.append({"knob": g, "val_precision_v2": v2["precision"],
                         "val_recall_v2": v2["recall"],
                         "val_excess": cv["excess_mean"],
                         "val_avg_position": cv["avg_position_mean"],
                         "val_alert_day_frac": _alert_day_frac(ro),
                         "meets_constraint": bool(ok)})
            print(f"  [{kind}] {g:>5} | P={v2['precision']:.1%} R={v2['recall']:.1%}"
                  f" | excess={cv['excess_mean']:+.2%}"
                  f" 仓位={cv['avg_position_mean']:.3f}"
                  f" 预警日={rows[-1]['val_alert_day_frac']:.0%}"
                  f" {'✓可选' if ok else '✗不满足约束'}")
        feas = [r for r in rows if r["meets_constraint"]]
        pick = max(feas, key=lambda r: r["val_excess"]) if feas else None
        return {"grid": rows, "picked": pick}

    def drl_ro(group, lean):
        api.set_confidence_threshold(lean)
        return [AlertEnv(e, norm, spec).rollout(api.policy()) for e in group]

    # 规则引擎的稀疏化旋钮：把全部 z 类阈值同乘 zmult（与 rule_v2_search.py 同一做法）
    _Z = ("r1_watch_z", "r1_warn_z", "r1_ser_z", "r3_watch", "r3_warn", "r3_ser",
          "r5_watch_z", "r5_warn_z", "r5_ser_z", "r8_watch", "r8_warn", "r8_ser")

    def rule_ro(group, zmult):
        out = []
        for e in group:
            p = dict(sp.get(e.symbol) or {})
            for k in _Z:
                p[k] = DEFAULT_PARAMS[k] * zmult
            out.append(AlertEnv(e, norm, spec).rollout(RulePolicy(e, params=p)))
        return out

    print("\n=== 验证集：DRL 选点 ===")
    res["selection"]["drl"] = select("DRL", LEAN_GRID, drl_ro)
    print("\n=== 验证集：规则引擎选点（同一协议）===")
    res["selection"]["rule"] = select("RULE", ZMULT_GRID, rule_ro)

    # ================= 测试集：只评一次 =================
    print("\n=== 测试集（只评一次）===")
    res["test"] = {}
    for kind, mk, key in (("drl", drl_ro, "drl"), ("rule", rule_ro, "rule")):
        pick = res["selection"][key]["picked"]
        if pick is None:
            res["test"][key] = {"error": "验证集上没有满足约束的工作点"}
            print(f"  [{kind}] 验证集无可行点")
            continue
        ro = _dz(mk(te, pick["knob"]))
        cv = _cvar(ro, px)
        v2 = _v2(ro)
        res["test"][key] = {"knob": pick["knob"], **cv,
                            "test_precision_v2": v2["precision"],
                            "test_recall_v2": v2["recall"],
                            "test_alert_day_frac": _alert_day_frac(ro)}
        print(f"  [{kind}] knob={pick['knob']} excess={cv['excess_mean']:+.2%} "
              f"({cv['n_positive']}/{cv['n_symbol']} 品种为正) "
              f"improve={cv['improve_mean']:+.1%} 随机={cv['random_improve_mean']:+.1%} "
              f"仓位={cv['avg_position_mean']:.3f} 预警日={res['test'][key]['test_alert_day_frac']:.0%}")

    # ---- 平凡策略对照（同一 CVaR 口径）
    print("\n=== 平凡策略对照（测试集）===")
    triv = {}
    n_te = {s: sum(len(e) for e in te if e.symbol == s) for s in SYMS}
    for name, fn in (("恒报警", lambda n: np.full(n, 3)),
                     ("每8步机械报警", lambda n: np.where(np.arange(n) % 8 == 0, 3, 0)),
                     ("只在开头报一次", lambda n: np.concatenate([[3], np.zeros(n - 1, int)]))):
        ro = [{"symbol": e.symbol, "ym": e.ym, "ts": e.ts,
               "actions": fn(len(e)), "risk_idx": e.risk_idx,
               "total_reward": 0.0, "n_alert": 0} for e in te]
        cv = _cvar(_dz(ro), px)
        triv[name] = cv
        print(f"  {name:<14} excess={cv['excess_mean']:+.2%} "
              f"improve={cv['improve_mean']:+.1%} 仓位={cv['avg_position_mean']:.3f}")
    res["trivial"] = triv

    # ================= 预登记判据裁决 =================
    d = res["test"].get("drl", {})
    r = res["test"].get("rule", {})
    passed = bool(d and d.get("excess_mean", -1) > 0 and d.get("n_positive", 0) >= 4)
    rule_ge = bool(r and d and r.get("excess_mean", -9) >= d.get("excess_mean", 9))
    res["preregistered_verdict"] = {
        "passed": passed,
        "excess_mean": d.get("excess_mean"),
        "n_positive": d.get("n_positive"), "n_symbol": d.get("n_symbol"),
        "rule_excess_mean": r.get("excess_mean"),
        "rule_not_worse_than_drl": rule_ge,
        "escaped_saturation": bool(d and d.get("avg_position_mean", 0.5) > 0.52),
        "text": (("**通过**：为 CVaR 目标单独选点后，DRL 的 excess 均值 > 0 且 ≥4/7 品种为正。"
                  if passed else
                  "**未通过**：即便为 CVaR 目标单独选点，DRL 仍未显示出超越随机预警的风控价值。")
                 + ("　⚠ 且**规则引擎的 excess 不低于 DRL**，须与 DRL 数字并列报出。"
                    if rule_ge else "")),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("\n" + res["preregistered_verdict"]["text"])
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
