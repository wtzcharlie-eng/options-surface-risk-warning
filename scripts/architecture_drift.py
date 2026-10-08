"""architecture_drift — 「换架构改不动精确率缺口」这个论断的**真实出处**。

为什么单独写这个脚本
--------------------
README §12 长期用一句话作为该论断的唯一定量证据：

    「漂移与模型无关。val→test 判别力漂移：规则引擎 −6.57pp、DRL 集成 −6.56pp，
      差额 0.00pp（`scripts/drift_study.py`，结果落 `data_out/drift_study.json`）」

**但那个出处是假的**（交付前独立审计查出）：
  - `data_out/drift_study.json` 的顶层键只有 windows/grid/curves/bases/pairs/constraint，
    **没有任何 rule-vs-DRL 的 val→test 对比**；
  - `scripts/drift_study.py` 自己声明「真测试集（2025-07 起）**一次都不读**」，
    它在设计上就**不可能**算出 val→test 漂移；
  - 全仓 grep 显示 −6.57/−6.56 只存在于 README、platform、以及另一个脚本的转述里，
    三处都是手写。

论断本身很可能成立，缺的是**能复算的出处**。本脚本把它真算一遍并落盘。
这正是本项目自己立下的规矩（README §7）：**用小样本临时算出的对照数不要写进文档——
要么落盘跑全，要么不写。**

口径
----
判别力（discrimination power, DP）= 该策略的精确率 − 同一数据集上的**命中基础率**。
减基础率是必须的：验证集与测试集的基础率本身就不同，直接比精确率会把
「数据变难了」误算成「模型退化了」。

漂移 = DP(test) − DP(val)。若规则引擎（**零个学习参数**）与 DRL 集成
（15 个 seed 的学习模型）漂移幅度相近，说明这段落差是**数据性质**而非模型缺陷——
换架构改不动它。反之则说明模型侧仍有空间。

**本脚本会读测试集**（这正是它与 `drift_study.py` 的区别，后者刻意不读）。
读测试集在这里是正当的：本脚本不做任何选择或调参，只是**报告已冻结模型的表现**。

用法::

    python scripts/architecture_drift.py     # → data_out/architecture_drift.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.baseline import RulePolicy, load_best_params
from drl.dataset import (Normalizer, apply_continuous_risk, load_episodes,
                         split_episodes)
from drl.dqn import DQNAgent
from drl.env import AlertEnv, RewardSpec
from drl.metrics import aggregate
from drl.train import hit_base_rate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "data_out", "anchor")


def _eval(eps, norm, spec, policy_factory) -> dict:
    ros = [AlertEnv(e, norm, spec).rollout(policy_factory(e)) for e in eps]
    # `aggregate` 返回 {"micro": {...}, "macro": {...}, "per_episode": df}。
    # **取 micro**：它等价于把所有幕拼成一条长序列，不会被小样本幕放大——
    # 与 results.json / metrics_v2 的主口径一致，换成 macro 会得到另一组数。
    m = aggregate(ros)["micro"]
    return {"precision": float(m["precision"]), "recall": float(m["recall"]),
            "avg_lead_min": float(m["avg_lead_min"]), "n_alert": int(m["n_alert"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out",
                                                  "architecture_drift.json"))
    args = ap.parse_args()

    print("载入 anchor（32 维，与交付模型一致）...")
    eps = load_episodes(ANCHOR, verify=True, underlying=True)
    # **必须先应用连续风险口径**，否则用的是已被废弃的「逐月」标签：
    # 那个口径下每幕前 72 根截面结构上不可能有正标签，测试集命中基础率
    # 会算成 21.06% 而非交付口径的 24.41%（README §7.10）。
    # 漏这一步会让整个漂移计算建立在错误的标签上——初版就漏了，
    # 是靠「基础率与 results.json 对不上」才发现的。
    st = apply_continuous_risk(eps)
    print(f"  连续风险口径：风险起点 {st['risk_before']} → {st['risk_after']}")
    tr, va, te = split_episodes(eps)
    norm = Normalizer.fit(tr)
    spec = RewardSpec()
    bp = load_best_params()

    files = sorted(f for f in glob.glob(os.path.join(args.drl_dir, "agent_seed*.npz"))
                   if "shuf_" not in os.path.basename(f))
    if not files:
        raise SystemExit("没有 checkpoint，请先跑 scripts/train_drl.py")
    agents = [DQNAgent.load(f)[0] for f in files]
    pols = [a.greedy_policy() for a in agents]

    def _ens(_e):
        def _p(s, i):
            return int(np.median([p(s, i) for p in pols]))
        return _p

    def _rule(e):
        return RulePolicy(e, params=bp)

    # 基础率必须**逐数据集**各算各的——验证集与测试集的难度本就不同，
    # 不减基础率就会把「数据变难」误算成「模型退化」。
    base = {"val": float(hit_base_rate(va)), "test": float(hit_base_rate(te))}
    print(f"  命中基础率：val {base['val']:.2%} · test {base['test']:.2%} "
          f"（差 {(base['test'] - base['val']) * 100:+.2f}pp）")

    out = {"note": __doc__.strip().splitlines()[0],
           "n_seed": len(agents), "hit_base_rate": base,
           "caliber": "DP = precision − hit_base_rate（逐数据集各算各的）；"
                      "drift = DP(test) − DP(val)",
           "models": {}}

    for name, fac in (("rule", _rule), ("drl_ensemble", _ens)):
        r = {}
        for split, E in (("val", va), ("test", te)):
            m = _eval(E, norm, spec, fac)
            m["dp"] = m["precision"] - base[split]
            r[split] = m
        r["drift_pp"] = (r["test"]["dp"] - r["val"]["dp"]) * 100
        out["models"][name] = r
        print(f"  {name:12s} val DP {r['val']['dp'] * 100:+.2f}pp → "
              f"test DP {r['test']['dp'] * 100:+.2f}pp   漂移 {r['drift_pp']:+.2f}pp")

    d_rule = out["models"]["rule"]["drift_pp"]
    d_drl = out["models"]["drl_ensemble"]["drift_pp"]
    out["drift_gap_pp"] = d_drl - d_rule
    out["verdict"] = (
        f"规则引擎（**零个学习参数**）漂移 {d_rule:+.2f}pp、"
        f"DRL 集成（15 seed）漂移 {d_drl:+.2f}pp，差额 {d_drl - d_rule:+.2f}pp。"
        + ("两者量级相近，说明这段落差是**数据性质**而非模型缺陷——换架构改不动它。"
           if abs(d_drl - d_rule) < 2.0 else
           "两者差异明显，**不能据此断言「换架构无用」**——模型侧仍可能有空间。"))
    print(f"\n差额 {out['drift_gap_pp']:+.2f}pp")
    print(out["verdict"])

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
