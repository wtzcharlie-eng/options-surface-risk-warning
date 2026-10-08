"""roll_retrain — 扩展窗滚动重训：能否补回测试期 regime 变化造成的落差？

背景
----
`scripts/drift_study.py` 已证明：val→test 的 −6.5pp 判别力落差**与模型无关**
（规则引擎 −7.10pp、DRL −7.33pp，差额 −0.24pp；见 scripts/architecture_drift.py，早前此处引的 −6.57/−6.56/0.00 无落盘出处），是 2025-07 之后的 regime 变化；
且该落差无法从 2023-07~2025-06 的历史窗外推（历史漂移 +1.34/+0.06/+2.59pp，符号相反）。
既然估不出 margin，就只剩「缩短外推距离」这条路——交付模型训练截止 2024-12、
测试期延到 2026-04，等于外推 16 个月。

实验设计（关键：同段对比）
--------------------------
每折 训练 → 选点窗(3个月) → 评估段(3个月)，滚动前移：

    折  训练截止   选点窗        评估段
    F2  2025-03   2025-04~06   2025-07~09
    F3  2025-06   2025-07~09   2025-10~12
    F4  2025-09   2025-10~12   2026-01~03

**冻结基线**＝交付模型（训练截止 2024-12、工作点在 2025-01~06 上选），
在**同样三段**上评估。两者评估段完全相同、只有训练截止与选点窗不同，
故差异可归因于「重训 + 就近选点」。

公平性
------
滚动模型用 3 个 seed 集成，冻结基线**也只取同样 3 个 seed**——
集成规模影响判别力（实测单 seed 16.63% → 5 seed 集成 21.06%），
拿 3 seed 打 5 seed 是不公平的。

每折的工作点只在该折的选点窗上选，**评估段一次都不参与选择**。

用法（可反复调用直到跑完）::

    python scripts/roll_retrain.py --budget 150     # 训练，按预算分批
    python scripts/roll_retrain.py --eval           # 训练完成后出对比表
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [42, 43, 44]
FOLDS = [
    ("f2", "2025-03", "2025-06", "2025-07", "2025-09"),
    ("f3", "2025-06", "2025-09", "2025-10", "2025-12"),
    ("f4", "2025-09", "2025-12", "2026-01", "2026-03"),
]
GRID = [0.0, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .8, 1.0, 1.3]


def _done(out: str, seed: int) -> bool:
    return os.path.exists(os.path.join(out, f"metrics_seed{seed}.json"))


def train(budget: float) -> None:
    t0 = time.time()
    for tag, tr_end, va_end, _, _ in FOLDS:
        out = os.path.join(ROOT, "data_out", "roll", tag)
        os.makedirs(out, exist_ok=True)
        # 超参沿用交付模型的搜索结果（`_load_cfg` 读的是 out 目录下的 sweep.json），
        # 不在每折重搜——重搜会把折间差异混进「超参运气」，那样就归因不了了；
        # 更要紧的是：若滚动折退回 DEFAULT_CFG 而交付模型用的是搜索后的配置，
        # 这个对比从一开始就不公平。
        src = os.path.join(ROOT, "data_out", "drl", "sweep.json")
        dst = os.path.join(out, "sweep.json")
        if not os.path.exists(src):
            raise SystemExit(f"缺 {src}，无法与交付模型对齐超参")
        if not os.path.exists(dst):
            with open(src, encoding="utf-8") as f:
                blob = f.read()
            with open(dst, "w", encoding="utf-8") as f:
                f.write(blob)
        for sd in SEEDS:
            if _done(out, sd):
                continue
            left = budget - (time.time() - t0)
            if left < 45:
                print(f"预算剩 {left:.0f}s，不足一轮，退出（再跑本命令可续）")
                return
            n_ep = max(1, int(left // 40))
            print(f"[{tag}] seed{sd} 训练（本次最多 {n_ep} epoch）...")
            r = subprocess.run(
                [sys.executable, os.path.join(ROOT, "scripts", "train_drl.py"),
                 "--stage", "train", "--seed", str(sd), "--underlying",
                 "--train-end", tr_end, "--val-end", va_end,
                 "--out", out, "--epochs-per-call", str(n_ep)],
                capture_output=True, text=True)
            tail = [l for l in r.stdout.strip().splitlines() if l.strip()][-1:]
            print("   " + (tail[0] if tail else r.stderr.strip()[-200:]))
    print("全部折训练完成，可跑 --eval")


def _load_api(d: str):
    from drl.api import AlertAgentAPI
    from drl.dataset import Normalizer
    from drl.dqn import DQNAgent
    fs = sorted(f for f in glob.glob(os.path.join(d, "agent_seed*.npz"))
                if "shuf_" not in os.path.basename(f)
                and int(os.path.basename(f)[10:-4]) in SEEDS)
    if len(fs) < len(SEEDS):
        return None
    ags, ex = [], None
    for f in fs:
        a, e = DQNAgent.load(f)
        ags.append(a)
        ex = ex or e
    return AlertAgentAPI(ags, Normalizer.from_dict(ex["normalizer"]), ex)


def evaluate() -> None:
    from drl.dataset import (Normalizer, apply_continuous_risk, load_episodes,
                             split_episodes)
    from drl.env import AlertEnv, RewardSpec
    from drl.metrics import aggregate
    from drl.train import hit_base_rate

    eps = load_episodes(os.path.join(ROOT, "data_out", "anchor"),
                        verify=True, underlying=True)
    apply_continuous_risk(eps)
    tr0, _, _ = split_episodes(eps)            # 归一化恒用交付模型的训练集口径
    norm, spec = Normalizer.fit(tr0), RewardSpec()
    seg = lambda lo, hi: [e for e in eps if lo <= e.ym <= hi]

    def curve(api, g, base):
        out = []
        for t in GRID:
            api.set_confidence_threshold(t)
            m = aggregate([AlertEnv(e, norm, spec).rollout(api.policy())
                           for e in g])["micro"]
            out.append({"lean": t, "P": m["precision"], "R": m["recall"],
                        "lead": m["avg_lead_min"], "DP": m["precision"] - base})
        return out

    def pick(c):
        ok = [r for r in c if r["R"] >= .60 and r["lead"] >= 30]
        return max(ok, key=lambda r: r["P"]) if ok else max(c, key=lambda r: r["P"])

    frozen = _load_api(os.path.join(ROOT, "data_out", "drl"))
    if frozen is None:
        raise SystemExit("交付模型缺 seed 42/43/44")
    # 冻结基线的工作点：在它自己的验证窗 2025-01~06 上选，与交付协议一致
    vg = seg("2025-01", "2025-06")
    f_pick = pick(curve(frozen, vg, hit_base_rate(vg)))
    print(f"冻结基线（训练截止 2024-12，3 seed）工作点 lean={f_pick['lean']:.2f}"
          f"（选自 2025-01~06）\n")

    rows = []
    print(f"{'评估段':16s}{'':4s}{'冻结 P/R/DP':>26s}{'滚动 P/R/DP':>26s}{'ΔDP':>8s}")
    for tag, tr_end, va_end, lo, hi in FOLDS:
        api = _load_api(os.path.join(ROOT, "data_out", "roll", tag))
        if api is None:
            print(f"  {tag}: 尚未训练完，跳过")
            continue
        g, vg2 = seg(lo, hi), seg(tr_end[:7], va_end)
        vg2 = [e for e in eps if tr_end < e.ym <= va_end]
        if not g or not vg2:
            print(f"  {tag}: 段为空，跳过")
            continue
        base = hit_base_rate(g)
        r_pick = pick(curve(api, vg2, hit_base_rate(vg2)))

        api.set_confidence_threshold(r_pick["lean"])
        mr = aggregate([AlertEnv(e, norm, spec).rollout(api.policy())
                        for e in g])["micro"]
        frozen.set_confidence_threshold(f_pick["lean"])
        mf = aggregate([AlertEnv(e, norm, spec).rollout(frozen.policy())
                        for e in g])["micro"]
        dpf, dpr = mf["precision"] - base, mr["precision"] - base
        ok = lambda m: "✓" if (m["precision"] >= .5 and m["recall"] >= .6
                               and m["avg_lead_min"] >= 30) else "✗"
        print(f"  {lo}~{hi}  {mf['precision']:6.1%}/{mf['recall']:5.1%}/"
              f"{dpf:+6.2%} {ok(mf)}"
              f"  {mr['precision']:6.1%}/{mr['recall']:5.1%}/{dpr:+6.2%} {ok(mr)}"
              f"  {(dpr-dpf)*100:+7.2f}")
        rows.append({"fold": tag, "seg": [lo, hi], "base": base,
                     "frozen": {"lean": f_pick["lean"], **{k: mf[k] for k in
                                ("precision", "recall", "avg_lead_min")},
                                "DP": dpf},
                     "rolling": {"lean": r_pick["lean"], **{k: mr[k] for k in
                                 ("precision", "recall", "avg_lead_min")},
                                 "DP": dpr}})
    if rows:
        df = sum(r["frozen"]["DP"] for r in rows) / len(rows)
        dr = sum(r["rolling"]["DP"] for r in rows) / len(rows)
        print(f"\n三段平均判别力：冻结 {df:+.2%}  滚动 {dr:+.2%}  "
              f"差 {(dr-df)*100:+.2f}pp")
        nf = sum(1 for r in rows if r["frozen"]["precision"] >= .5
                 and r["frozen"]["recall"] >= .6
                 and r["frozen"]["avg_lead_min"] >= 30)
        nr = sum(1 for r in rows if r["rolling"]["precision"] >= .5
                 and r["rolling"]["recall"] >= .6
                 and r["rolling"]["avg_lead_min"] >= 30)
        print(f"三项全达标段数：冻结 {nf}/{len(rows)}  滚动 {nr}/{len(rows)}")
        p = os.path.join(ROOT, "data_out", "roll_retrain.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"seeds": SEEDS, "folds": rows,
                       "mean_DP_frozen": df, "mean_DP_rolling": dr},
                      f, ensure_ascii=False, indent=2, default=float)
        print(f"→ {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=150.0)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    evaluate() if a.eval else train(a.budget)


if __name__ == "__main__":
    main()
