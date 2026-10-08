"""cvar_drl — 问题3 的 **CVaR 改善率** 口径（补上 README §12 列为局限的那一项）。

为什么要有这个脚本
------------------
赛题允许问题3 用「累计奖励**或** CVaR 改善率」二选一，本项目此前只走累计奖励口径。
命题方 2026-08 的答复给出了改善率公式，故补出该口径，使两种口径可以并列对照。

**判据在跑数之前已写定**，见 `data_out/cvar_preregistration.md`：
主判据是 `excess = improve − random_improve` 的 7 品种等权均值 > 0 且 ≥4/7 品种为正。

为什么不能只看 improve
----------------------
这个指标在本项目翻过一次车：问题1 早前用 `backtest_cvar` 算出 **7 个品种一模一样
都是 +50.0%**——因为那个函数的仓位规则退化成了「全程半仓」，CVaR 被机械地按比例缩小，
**与预警质量完全无关**，恒预警/随机预警/瞎报都会得到同样的数。
故这里一律用 `backtest_cvar_path`（事件驱动仓位路径），并强制并列两条对照：
  - `always` 恒定半仓 —— 改善率里「白拿」的部分；
  - `random` 随机挑同样多的预警日 —— **同等仓位占用下的无信息对照**。

标的价从哪来
------------
不走 AKShare（离线不可用），也不走 `infer_futures_price`（它 `load_symbol_family(symbol)`
不带年月会加载该品种全部数据，实测 170s 跑不完）。
改用 anchor 数据集里已经落盘、且**已与评测用的 episode 对齐**的 `underlying.parquet::_S`。
这样价格与预警来自同一份数据，不会引入对齐误差。

用法::

    python scripts/cvar_drl.py                 # 全部 7 个品种
    python scripts/cvar_drl.py --symbols ag,cu # 指定品种
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

from drl.baseline import RulePolicy
from drl.dataset import Normalizer, load_episodes, split_episodes
from drl.dqn import DQNAgent
from drl.env import AlertEnv, RewardSpec
from drl.metrics_v2 import trivial_policies
from vol_surface.cvar_backtest import backtest_cvar_path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "data_out", "anchor")

# 预登记参数——**不得事后调整**（见 data_out/cvar_preregistration.md）
HOLD_DAYS, CUT_TO, N_RANDOM, SEED = 5, 0.5, 200, 0


def daily_price(symbol: str) -> pd.DataFrame:
    """从 anchor 的 underlying.parquet 取标的价日线（date, open, close）。"""
    fs = sorted(glob.glob(os.path.join(ANCHOR, symbol, "*", "underlying.parquet")))
    if not fs:
        return pd.DataFrame(columns=["date", "open", "close"])
    d = pd.concat([pd.read_parquet(f, columns=["_timestamp", "_S"]) for f in fs])
    ts = pd.to_datetime(d["_timestamp"].str.decode("utf-8"), format="%Y%m%d%H%M%S")
    px = (pd.DataFrame({"date": ts.dt.normalize(), "close": d["_S"].to_numpy()})
          .groupby("date", as_index=False).last())
    px["open"] = px["close"]          # 无开盘价，用收盘近似（与原实现一致）
    return px[["date", "open", "close"]]


def alerts_from(rollouts: list) -> pd.DataFrame:
    """把若干幕的 rollout 结果拼成 backtest_cvar_path 需要的 alerts 表。"""
    ts, lv = [], []
    for r in rollouts:
        ts.append(np.asarray(r["ts"]))
        lv.append(np.asarray(r["actions"]))
    if not ts:
        return pd.DataFrame(columns=["timestamp", "level"])
    return pd.DataFrame({"timestamp": np.concatenate(ts),
                         "level": np.concatenate(lv)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out", "cvar_drl.json"))
    ap.add_argument("--sweep-hold", default="",
                    help="事后探索：扫 hold_days（逗号分隔）。**不改预登记结论**，"
                         "只用于定位该口径在多大预警密度下失去判别力。")
    args = ap.parse_args()

    print("载入 anchor（32 维，与交付模型一致）...")
    eps = load_episodes(ANCHOR, verify=True, underlying=True)
    tr, va, te = split_episodes(eps)
    norm = Normalizer.fit(tr)
    spec = RewardSpec()

    files = sorted(f for f in glob.glob(os.path.join(args.drl_dir, "agent_seed*.npz"))
                   if "shuf_" not in os.path.basename(f))
    if not files:
        raise SystemExit("没有 checkpoint，请先跑 scripts/train_drl.py")
    agents = [DQNAgent.load(f)[0] for f in files]
    print(f"  测试期 {len(te)} 幕 · 集成 {len(agents)} 个 seed")

    def ensemble_policy(env):
        """多数投票集成：与 results.json 的 ensemble 口径一致（取各 seed 贪心动作的中位）。"""
        pols = [a.greedy_policy() for a in agents]

        def _p(s, i):
            return int(np.median([p(s, i) for p in pols]))
        return _p

    syms = ([s for s in args.symbols.split(",") if s]
            or sorted({e.symbol for e in te}))
    out: dict = {"params": {"hold_days": HOLD_DAYS, "cut_to": CUT_TO,
                            "n_random": N_RANDOM, "seed": SEED,
                            "n_seed": len(agents)},
                 "per_symbol": {}}

    for sym in syms:
        seps = [e for e in te if e.symbol == sym]
        px = daily_price(sym)
        if not seps or len(px) < 25:
            print(f"  {sym}: 跳过（{len(seps)} 幕 / {len(px)} 天）")
            continue
        # 价格只截到该品种测试期覆盖的日期范围
        lo = min(pd.Timestamp(e.ts[0][:8]) for e in seps)
        hi = max(pd.Timestamp(e.ts[-1][:8]) for e in seps)
        pxt = px[(px["date"] >= lo) & (px["date"] <= hi)].reset_index(drop=True)
        if len(pxt) < 25:
            print(f"  {sym}: 跳过（测试期日线 {len(pxt)} < 25）")
            continue

        envs = [AlertEnv(e, norm, spec) for e in seps]
        pol = ensemble_policy(None)
        cand = {
            "drl": alerts_from([env.rollout(pol) for env in envs]),
            "rule": alerts_from([AlertEnv(e, norm, spec).rollout(RulePolicy(e))
                                 for e in seps]),
        }
        # 平凡策略：直接按动作数组回放，不经过 env（它们不看状态）
        for name in ("恒报警", "每8步机械报警", "只在开头报一次"):
            rs = []
            for e in seps:
                acts = trivial_policies(len(e))[name]
                rs.append({"ts": e.ts, "actions": acts})
            cand[name] = alerts_from(rs)

        if args.sweep_hold:
            # ---- 事后探索：同一批预警，扫不同 hold_days ----
            # 预登记结论**不因本段改变**（hold_days=5 已写死）。本段回答的是
            # 另一个问题：「这个口径在多大的预警密度下才还有判别力？」
            sw = out.setdefault("sweep", {}).setdefault(sym, {})
            for hd in [int(x) for x in args.sweep_hold.split(",")]:
                sw[str(hd)] = {}
                for name, al in cand.items():
                    try:
                        r = backtest_cvar_path(al, pxt, hold_days=hd, cut_to=CUT_TO,
                                               n_random=N_RANDOM, seed=SEED)
                        sw[str(hd)][name] = {
                            "improve": float(r["improve"]),
                            "random_improve": float(r["random_improve"]),
                            "excess": float(r["excess"]),
                            "avg_position": float(r["avg_position"]),
                        }
                    except Exception as e:                        # noqa: BLE001
                        sw[str(hd)][name] = {"error": str(e)}

        res = {}
        for name, al in cand.items():
            try:
                r = backtest_cvar_path(al, pxt, hold_days=HOLD_DAYS, cut_to=CUT_TO,
                                       n_random=N_RANDOM, seed=SEED)
                res[name] = {k: (float(v) if isinstance(v, (int, float, np.floating))
                                 else v)
                             for k, v in r.items()
                             if not isinstance(v, (pd.Timestamp, type(None)))}
            except Exception as e:                                # noqa: BLE001
                res[name] = {"error": f"{type(e).__name__}: {e}"}
        out["per_symbol"][sym] = {"n_days": len(pxt), "n_ep": len(seps), **res}
        d = res.get("drl", {})
        print(f"  {sym}: DRL improve {d.get('improve', float('nan')):+.1%} · "
              f"random {d.get('random_improve', float('nan')):+.1%} · "
              f"excess {d.get('excess', float('nan')):+.1%}")

    # ---- 汇总：主判据只看 excess
    def _agg(key: str, field: str) -> list:
        return [v[key][field] for v in out["per_symbol"].values()
                if key in v and field in v[key]]

    summary = {}
    for key in ("drl", "rule", "恒报警", "每8步机械报警", "只在开头报一次"):
        ex = _agg(key, "excess")
        if not ex:
            continue
        summary[key] = {
            "improve_mean": float(np.mean(_agg(key, "improve"))),
            "random_improve_mean": float(np.mean(_agg(key, "random_improve"))),
            "always_improve_mean": float(np.mean(_agg(key, "always_improve")))
            if _agg(key, "always_improve") else None,
            "excess_mean": float(np.mean(ex)),
            "n_positive": int(sum(1 for x in ex if x > 0)),
            "n_symbol": len(ex),
        }
    out["summary"] = summary
    # 预登记判据：excess 均值 >0 且 ≥4/7 品种为正
    d = summary.get("drl", {})
    out["preregistered_verdict"] = {
        "criterion": "excess_mean > 0 且 n_positive >= 4/7",
        "excess_mean": d.get("excess_mean"),
        "n_positive": d.get("n_positive"),
        "n_symbol": d.get("n_symbol"),
        "passed": bool(d and d["excess_mean"] > 0 and d["n_positive"] >= 4),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n→ {args.out}")
    for k, v in summary.items():
        print(f"  {k:12s} improve {v['improve_mean']:+.1%} · "
              f"random {v['random_improve_mean']:+.1%} · "
              f"**excess {v['excess_mean']:+.1%}** · "
              f"{v['n_positive']}/{v['n_symbol']} 品种为正")
    print(f"\n预登记判据：{'通过' if out['preregistered_verdict']['passed'] else '**未通过**'}")


if __name__ == "__main__":
    main()
