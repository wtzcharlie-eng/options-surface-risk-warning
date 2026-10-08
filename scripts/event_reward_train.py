"""event_reward_train — 事件级奖励能否在新口径下做得更好？

**判据已预先登记**于 `data_out/event_reward_preregistration.md`：
验证集（新口径事件级精确率）提升 ≥2.0pp 才进入测试集评估；不足则判无效且**不看测试集**。
**我事先写明预计这一条会失败**——见登记文件，防止事后把"本来也没指望"当成不算一次尝试。

动机
----
当前奖励按**旧口径**逐截面罚误报，而新口径按**合并后的预警事件**计精确率：
同一段风险内连报十条与只报一条代价相同。奖励与评测口径不一致。
`RewardSpec(event_level_fp=True)` 把误报计价改为「只罚每段连续误报的第一条」，与新口径对齐。

对照设计
--------
- 新旧奖励各训 seed 42/43/44，**同超参、同切分、同 epoch 数**
- 对照组**不复用现有 15-seed 集成**（规模不同不可比），而是用相同 3 个 seed 重训旧奖励
- 两组在同一选点协议下**各自**选工作点，避免"给新模型调参、旧模型不调"

用法::

    python scripts/event_reward_train.py --budget 150   # 分批训练，可反复调用
    python scripts/event_reward_train.py --eval
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 43, 44]
DIRS = {"old": os.path.join(ROOT, "data_out", "drl_ctrl_old"),
        "new": os.path.join(ROOT, "data_out", "drl_evrew")}
LEAN_GRID = [0.0, .35, .6, 1.0, 1.3, 1.8, 2.5, 3.5, 5.0, 7.0, 10.0]
GAIN_THRESHOLD = 2.0   # 预先登记的判定阈值（pp）


def train(budget: float) -> None:
    src = os.path.join(ROOT, "data_out", "drl", "sweep.json")
    if not os.path.exists(src):
        raise SystemExit(f"缺 {src}")
    t0 = time.time()
    for arm, d in DIRS.items():
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "sweep.json")
        if not os.path.exists(dst):
            with open(src, encoding="utf-8") as f:
                blob = f.read()
            with open(dst, "w", encoding="utf-8") as f:
                f.write(blob)
        for sd in SEEDS:
            if os.path.exists(os.path.join(d, f"metrics_seed{sd}.json")):
                continue
            left = budget - (time.time() - t0)
            if left < 45:
                print(f"预算剩 {left:.0f}s，退出（再跑本命令可续）")
                return
            n_ep = max(1, int(left // 40))
            print(f"[{arm}] seed{sd}（本次最多 {n_ep} epoch）...")
            env = dict(os.environ)
            if arm == "new":
                env["DRL_EVENT_LEVEL_FP"] = "1"
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts", "train_drl.py"),
                 "--stage", "train", "--seed", str(sd), "--underlying",
                 "--out", d, "--epochs-per-call", str(n_ep)],
                capture_output=True, text=True, env=env)
            tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:]
            print("   " + (tail[0] if tail else r.stderr.strip()[-200:]))
    print("两组训练完成，可跑 --eval")


def evaluate() -> None:
    from drl.api import AlertAgentAPI
    from drl.dataset import (Normalizer, apply_continuous_risk, load_episodes,
                             split_episodes)
    from drl.dqn import DQNAgent
    from drl.env import AlertEnv, RewardSpec
    from drl.metrics_v2 import aggregate_v2, evaluate_v2

    eps = load_episodes(os.path.join(ROOT, "data_out", "anchor"),
                        verify=True, underlying=True)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()   # 评测恒用默认 spec，与口径无关

    def api_of(d):
        fs = sorted(f for f in glob.glob(os.path.join(d, "agent_seed*.npz"))
                    if "shuf_" not in os.path.basename(f))
        if len(fs) < len(SEEDS):
            return None
        ags, ex = [], None
        for f in fs:
            a, e = DQNAgent.load(f)
            ags.append(a)
            ex = ex or e
        return AlertAgentAPI(ags, Normalizer.from_dict(ex["normalizer"]), ex)

    def curve(api, grp):
        out = []
        for t in LEAN_GRID:
            api.set_confidence_threshold(t)
            ms = [evaluate_v2(ro["ts"], ro["actions"], ro["risk_idx"])
                  for ro in (AlertEnv(e, norm, spec).rollout(api.policy())
                             for e in grp)]
            out.append((t, aggregate_v2(ms)))
        return out

    def pick(c):
        feas = [(t, m) for t, m in c
                if m["recall"] >= .6 and m["avg_lead_min"] >= 30]
        return max(feas, key=lambda x: x[1]["precision"]) if feas else \
            max(c, key=lambda x: x[1]["precision"])

    res = {}
    for arm, d in DIRS.items():
        api = api_of(d)
        if api is None:
            print(f"{arm} 尚未训完")
            return
        lean, mv = pick(curve(api, va))
        res[arm] = {"lean": lean, "val": mv, "api": api}
        print(f"  [{arm}] 验证集选出 lean={lean:.2f}  "
              f"P={mv['precision']:.2%}  R={mv['recall']:.2%}")

    gain = (res["new"]["val"]["precision"] - res["old"]["val"]["precision"]) * 100
    print(f"\n新奖励 − 旧奖励（验证集事件级精确率）= {gain:+.2f}pp "
          f"（预先登记阈值 ≥{GAIN_THRESHOLD:.1f}pp）")

    out = {"seeds": SEEDS, "threshold_pp": GAIN_THRESHOLD,
           "old": {k: res["old"][k] for k in ("lean", "val")},
           "new": {k: res["new"][k] for k in ("lean", "val")},
           "gain_val_pp": gain}
    if gain >= GAIN_THRESHOLD:
        print("→ 达阈值，执行**唯一一次**测试集评估")
        api = res["new"]["api"]
        api.set_confidence_threshold(res["new"]["lean"])
        ms = [evaluate_v2(ro["ts"], ro["actions"], ro["risk_idx"])
              for ro in (AlertEnv(e, norm, spec).rollout(api.policy()) for e in te)]
        mt = aggregate_v2(ms)
        ok = (mt["precision"] >= .5 and mt["recall"] >= .6
              and mt["avg_lead_min"] >= 30)
        print(f"  测试集 P={mt['precision']:.2%} R={mt['recall']:.2%} "
              f"lead={mt['avg_lead_min']:.0f}m → 三项{'全达标' if ok else '未全达标'}")
        out["test"] = mt
        out["pass3"] = bool(ok)
    else:
        print("→ 未达阈值，按预先登记规则**不看测试集**，记为负结果")
        out["test"] = None

    p = os.path.join(ROOT, "data_out", "event_reward.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"→ {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    evaluate() if a.eval else train(a.budget)


if __name__ == "__main__":
    main()
