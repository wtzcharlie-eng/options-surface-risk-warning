"""ensemble_scaling — 集成规模能否把 P-R 前沿整体抬高？

**判据在跑数之前已写定**，见 `data_out/ensemble_preregistration.md`：
验证集判别力 N=15 比 N=5 高 ≥1.0pp 才认定有效，届时才跑**唯一一次**测试集评估；
不足 1.0pp 则判无效且**不看测试集**（避免多次窥视）。

为什么值得试
------------
前面五条路径（架构 / 选点协议 / 加特征 / 滚动重训 / 召回偏移先验）都在试图
**抵消** 2025-07 之后的 regime 变化，全部失败。集成规模不同——它不消除 regime
变化，只把整条前沿抬高。而实测单 seed 判别力 16.63% → 5 seed 集成 21.06%，
**+4.43pp 纯来自方差削减**，说明这个方向的斜率是真实存在的，只是从没测过 5 以上。

混杂控制
--------
「N 越大越好」必须与「恰好抽到几个好 seed」分开：对每个 N<15 随机抽多组子集，
报均值 ± 标准差。若 N 增大的提升小于同 N 下子集间的标准差，不算证据。

用法::

    python scripts/ensemble_scaling.py --budget 150   # 反复调用直到训完
    python scripts/ensemble_scaling.py --eval
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(ROOT, "data_out", "drl")          # 交付版 seed 42~46
MORE_DIR = os.path.join(ROOT, "data_out", "drl_more")     # 新增 seed 47~56
NEW_SEEDS = list(range(47, 57))
SIZES = [1, 3, 5, 10, 15]
N_SUBSET = 5          # 每个 N 最多抽多少组子集（控制"抽到好 seed"的混杂）
GAIN_THRESHOLD = 1.0  # 预先登记的判定阈值（pp）
GRID = [0.0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .8, 1.0, 1.3]
CACHE = os.path.join(ROOT, "data_out", ".ensemble_cache.json")


def train(budget: float) -> None:
    os.makedirs(MORE_DIR, exist_ok=True)
    src = os.path.join(BASE_DIR, "sweep.json")
    dst = os.path.join(MORE_DIR, "sweep.json")
    if not os.path.exists(src):
        raise SystemExit(f"缺 {src}，无法与交付版对齐超参")
    if not os.path.exists(dst):
        with open(src, encoding="utf-8") as f:
            blob = f.read()
        with open(dst, "w", encoding="utf-8") as f:
            f.write(blob)

    t0 = time.time()
    for sd in NEW_SEEDS:
        if os.path.exists(os.path.join(MORE_DIR, f"metrics_seed{sd}.json")):
            continue
        left = budget - (time.time() - t0)
        if left < 45:
            print(f"预算剩 {left:.0f}s，退出（再跑本命令可续）")
            return
        n_ep = max(1, int(left // 40))
        print(f"seed{sd} 训练（本次最多 {n_ep} epoch）...")
        r = subprocess.run(
            [sys.executable, os.path.join(ROOT, "scripts", "train_drl.py"),
             "--stage", "train", "--seed", str(sd), "--underlying",
             "--out", MORE_DIR, "--epochs-per-call", str(n_ep)],
            capture_output=True, text=True)
        tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:]
        print("   " + (tail[0] if tail else r.stderr.strip()[-200:]))
    print("全部 seed 训练完成，可跑 --eval")


def _all_agents():
    from drl.dqn import DQNAgent
    out = {}
    for d in (BASE_DIR, MORE_DIR):
        for f in sorted(glob.glob(os.path.join(d, "agent_seed*.npz"))):
            if "shuf_" in os.path.basename(f):
                continue
            sd = int(os.path.basename(f)[len("agent_seed"):-len(".npz")])
            out[sd] = DQNAgent.load(f)
    return out


def evaluate(budget: float = 0.0) -> None:
    from drl.api import AlertAgentAPI
    from drl.dataset import (Normalizer, apply_continuous_risk, load_episodes,
                             split_episodes)
    from drl.env import AlertEnv, RewardSpec
    from drl.metrics import aggregate
    from drl.train import hit_base_rate

    eps = load_episodes(os.path.join(ROOT, "data_out", "anchor"),
                        verify=True, underlying=True)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()
    base_v, base_t = hit_base_rate(va), hit_base_rate(te)

    agents = _all_agents()
    seeds = sorted(agents)
    print(f"可用 seed {len(seeds)} 个：{seeds}")
    if len(seeds) < max(SIZES):
        print(f"不足 {max(SIZES)} 个，先跑训练")
        return

    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            cache = json.load(f)

    def api_of(sub):
        ags = [agents[s][0] for s in sub]
        ex = agents[sub[0]][1]
        return AlertAgentAPI(ags, Normalizer.from_dict(ex["normalizer"]), ex)

    def val_dp(sub):
        """按交付协议在验证集上选工作点，返回 (lean, 判别力)。带缓存，可分批跑。

        **状态含「距上次预警时长」与「上次等级」，依赖历史动作**，故 Q 值不能
        脱离策略预先算好再按子集平均——只能逐个子集实跑。这也是本评估必须
        分批 + 缓存的原因（5 个规模 × 5 组子集 × 14 个 lean 一次跑不完）。
        """
        key = ",".join(map(str, sub))
        if key in cache:
            return cache[key]["lean"], cache[key]["dp"]
        api = api_of(list(sub))
        best = None
        for t in GRID:
            api.set_confidence_threshold(t)
            m = aggregate([AlertEnv(e, norm, spec).rollout(api.policy())
                           for e in va])["micro"]
            feasible = m["recall"] >= .60 and m["avg_lead_min"] >= 30
            rank = (feasible, m["precision"])
            if best is None or rank > best[0]:
                best = (rank, t, m)
        cache[key] = {"lean": best[1], "dp": best[2]["precision"] - base_v,
                      "P": best[2]["precision"], "R": best[2]["recall"],
                      "feasible": bool(best[0][0])}
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, default=float)
        return cache[key]["lean"], cache[key]["dp"]

    # ---- 组装待评子集（种子固定，保证可复现）
    rng = np.random.default_rng(0)
    plan = []
    for n in SIZES:
        if n >= len(seeds):
            plan.append((n, tuple(seeds)))
            continue
        seen = set()
        for _ in range(N_SUBSET * 4):
            s = tuple(sorted(rng.choice(seeds, n, replace=False).tolist()))
            if s not in seen:
                seen.add(s)
                plan.append((n, s))
            if len(seen) >= N_SUBSET:
                break

    todo = [(n, s) for n, s in plan if ",".join(map(str, s)) not in cache]
    print(f"待评 {len(todo)} 组（缓存已有 {len(cache)} 组）")
    t0 = time.time()
    for n, s in todo:
        if budget and time.time() - t0 > budget:
            print(f"预算用尽，已完成 {len(cache)} 组，可再次运行续跑")
            return
        val_dp(s)
        print(f"  N={n:2d} {str(s)[:46]:46s} DP={cache[','.join(map(str,s))]['dp']*100:6.2f}%")

    rows = []
    print(f"\n{'N':>3s}{'验证判别力(均值±标准差)':>26s}{'子集数':>8s}")
    for n in SIZES:
        dps = [cache[",".join(map(str, s))]["dp"] * 100
               for nn, s in plan if nn == n]
        rows.append({"n": n, "mean": float(np.mean(dps)),
                     "std": float(np.std(dps, ddof=1)) if len(dps) > 1 else 0.0,
                     "n_subsets": len(dps), "dps": dps})
        r = rows[-1]
        print(f"{n:3d}{r['mean']:17.2f}% ± {r['std']:5.2f}{len(dps):8d}")

    d5 = next(r for r in rows if r["n"] == 5)["mean"]
    d15 = next(r for r in rows if r["n"] == max(SIZES))["mean"]
    gain = d15 - d5
    std5 = next(r for r in rows if r["n"] == 5)["std"]
    print(f"\nN=5 → N={max(SIZES)} 验证集判别力提升 {gain:+.2f}pp "
          f"（预先登记阈值 ≥{GAIN_THRESHOLD:.1f}pp；N=5 子集间标准差 {std5:.2f}pp）")

    out = {"rows": rows, "gain_val_dp_pp": gain,
           "threshold_pp": GAIN_THRESHOLD, "base_val": base_v}
    if gain >= GAIN_THRESHOLD:
        print("→ 达到预先登记阈值，执行**唯一一次**测试集评估")
        sub = tuple(seeds)
        lean, _ = val_dp(sub)
        api = api_of(list(sub))
        api.set_confidence_threshold(lean)
        mt = aggregate([AlertEnv(e, norm, spec).rollout(api.policy())
                        for e in te])["micro"]
        ok = (mt["precision"] >= .5 and mt["recall"] >= .6
              and mt["avg_lead_min"] >= 30)
        print(f"  验证集选出 lean={lean:.2f}")
        print(f"  测试集 P={mt['precision']:.2%} {'✓' if mt['precision']>=.5 else '✗'}"
              f"  R={mt['recall']:.2%} {'✓' if mt['recall']>=.6 else '✗'}"
              f"  lead={mt['avg_lead_min']:.1f}m "
              f"{'✓' if mt['avg_lead_min']>=30 else '✗'}"
              f"  三项全达标：{'是' if ok else '否'}")
        out["test"] = {"lean": lean, "precision": mt["precision"],
                       "recall": mt["recall"], "avg_lead_min": mt["avg_lead_min"],
                       "DP": mt["precision"] - base_t, "pass3": bool(ok)}
    else:
        print("→ 未达阈值，按预先登记规则**不看测试集**，记为第六条被证伪的路径")
        out["test"] = None

    p = os.path.join(ROOT, "data_out", "ensemble_scaling.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"→ {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    evaluate(a.budget) if a.eval else train(a.budget)


if __name__ == "__main__":
    main()
