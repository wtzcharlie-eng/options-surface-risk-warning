"""rule_v2_search — 新口径下重检那些「旧口径因杀召回而被否决」的提精手段。

为什么值得重检
--------------
旧口径要求召回 ≥60%，于是下列四项都被否掉了（README §6、§7.7、§7.8）：
  - **composite**（level≥2 需多条规则协同）：精确率能到 60%+，但召回跌破 60%
  - **cooldown**（触发后冷却，抑制重复升级）：同上
  - **流动性质量门**（`liquidity_ratio<0.55` 时抑制预警）：给规则基线 +3.10pp，但损召回
  - **ML 层**（IsolationForest）：交付评测一直走纯规则路径

**新口径下召回几乎免费**——实测所有平凡策略召回均达 98.9%（`drl/metrics_v2.py` 的退化检验）。
既然约束松了，这些手段应当重新检验。这不是翻旧账，是约束变了之后的必要复查。

为什么必须做交叉组合而非逐项开关
--------------------------------
这四项互相耦合：cooldown 会减少预警条数，从而改变 composite 的"多规则协同"判定基数；
质量门抑制的低流动性截面又与 cooldown 的冷却窗重叠。**逐项单独开关会给出与组合相反的结论**
——本项目此前在「评估改动须先排除耦合」上已栽过一次。故此处跑全组合。

选点协议
--------
与全项目一致：**所有配置只在验证集上比较、选出一个**，测试集只评一次。
任何"看了测试集再挑配置"的做法都被禁止。

用法::

    python scripts/rule_v2_search.py --underlying
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.dataset import (Normalizer, apply_continuous_risk, feature_names,
                         load_episodes, split_episodes)
from drl.env import AlertEnv, RewardSpec
from drl.metrics_v2 import aggregate_v2, evaluate_v2
from vol_surface.alert_engine import evaluate as engine_eval
from vol_surface.alert_rules import DEFAULT_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_Z = ("r1_watch_z", "r1_warn_z", "r1_ser_z", "r3_watch", "r3_warn", "r3_ser",
      "r5_watch_z", "r5_warn_z", "r5_ser_z", "r8_watch", "r8_warn", "r8_ser")
ZMULT = [1.0, 1.6, 2.0, 2.5, 3.0]
LIQ_GATE = 0.55
_ok = lambda m: (m["precision"] >= .5 and m["recall"] >= .6
                 and m["avg_lead_min"] >= 30)


def levels_for(e, FN, sp, zmult, composite, cooldown, qgate):
    """按给定配置算一幕的等级序列。

    直接调用 `alert_engine.evaluate`（问题1 的交付本体），不另写一套判定逻辑——
    评测与交付走同一条代码路径，否则"评测通过"说明不了交付系统的行为。
    """
    from vol_surface.alert_engine import AlertState
    p = dict(sp.get(e.symbol, {}))
    for k in _Z:
        p[k] = DEFAULT_PARAMS[k] * zmult
    st = AlertState(cooldown=cooldown) if cooldown else None
    j_liq = FN.index("liquidity_ratio") if "liquidity_ratio" in FN else None
    out = np.zeros(len(e), dtype=np.int64)
    for i in range(len(e)):
        feats = {k: float(e.X[i, j]) for j, k in enumerate(FN)}
        r = engine_eval(feats, params=p, state=st, model=None,
                        composite=composite, composite_min_rules=2)
        lv = int(r["level"] if isinstance(r, dict) else r)
        if qgate and j_liq is not None and float(e.X[i, j_liq]) < LIQ_GATE:
            lv = 0
        out[i] = lv
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=os.path.join(ROOT, "data_out", "anchor"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out", "rule_v2_search.json"))
    ap.add_argument("--underlying", action="store_true")
    a = ap.parse_args()

    eps = load_episodes(a.anchor, verify=True, underlying=a.underlying)
    apply_continuous_risk(eps)
    tr, va, te = split_episodes(eps)
    norm, spec = Normalizer.fit(tr), RewardSpec()
    FN = feature_names(underlying=a.underlying)
    with open(os.path.join(ROOT, "data_out", "symbol_params.json"),
              encoding="utf-8") as f:
        sp = json.load(f)
    has_liq = "liquidity_ratio" in FN
    if not has_liq:
        print("注意：特征集中没有 liquidity_ratio，质量门维度被跳过")

    def run(grp, cfg):
        ms = []
        for e in grp:
            ro = AlertEnv(e, norm, spec).rollout(lambda s, i: 0)
            lv = levels_for(e, FN, sp, **cfg)
            ms.append(evaluate_v2(ro["ts"], lv, ro["risk_idx"]))
        return aggregate_v2(ms)

    grid = []
    for z, comp, cd, qg in itertools.product(
            ZMULT, (False, True), (0, 8), (False, True) if has_liq else (False,)):
        grid.append({"zmult": z, "composite": comp, "cooldown": cd, "qgate": qg})

    print(f"验证集扫描 {len(grid)} 组配置（2×2×2 交叉 × {len(ZMULT)} 阈值系数）...")
    rows = []
    for cfg in grid:
        mv = run(va, cfg)
        rows.append({"cfg": cfg, "val": mv})
        flag = "可行" if (mv["recall"] >= .6 and mv["avg_lead_min"] >= 30) else "  — "
        print(f"  z={cfg['zmult']:.1f} comp={int(cfg['composite'])} "
              f"cd={cfg['cooldown']:d} qg={int(cfg['qgate'])} │ "
              f"事件 {mv['n_alert_event']:5d}  P={mv['precision']:6.1%}  "
              f"R={mv['recall']:6.1%}  {flag}")

    feas = [r for r in rows if r["val"]["recall"] >= .6
            and r["val"]["avg_lead_min"] >= 30]
    if not feas:
        raise SystemExit("验证集可行域为空")
    pick = max(feas, key=lambda r: r["val"]["precision"])
    mt = run(te, pick["cfg"])
    print(f"\n验证集选出：{pick['cfg']}")
    print(f"  验证 P={pick['val']['precision']:.2%} R={pick['val']['recall']:.2%}")
    print(f"  测试 P={mt['precision']:.2%} {'✓' if mt['precision'] >= .5 else '✗'}  "
          f"R={mt['recall']:.2%} {'✓' if mt['recall'] >= .6 else '✗'}  "
          f"提前={mt['avg_lead_min']:.0f}min  → 三项全达标：{'是' if _ok(mt) else '否'}")

    base = [r for r in rows if r["cfg"] == {"zmult": 2.0, "composite": False,
                                            "cooldown": 0, "qgate": False}]
    out = {"grid": rows, "picked": {"cfg": pick["cfg"], "val": pick["val"],
                                    "test": mt, "pass3": bool(_ok(mt))},
           "baseline_val": base[0]["val"] if base else None,
           "caliber": "新口径（事件级精确率 / 召回不设上限），见 drl/metrics_v2.py",
           "note": "所有配置只在验证集比较并选出一个；测试集只评一次。"}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
