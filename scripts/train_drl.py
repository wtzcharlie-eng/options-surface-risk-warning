"""train_drl — 问题3 DRL 预警智能体的训练入口（分阶段，可断点续跑）。

用法::

    python scripts/train_drl.py --stage all                    # 一条龙（推荐）
    python scripts/train_drl.py --stage sweep                  # 只跑超参搜索
    python scripts/train_drl.py --stage train --seed 42        # 训练单个 seed
    python scripts/train_drl.py --stage report                 # 汇总 + 测试集评估
    python scripts/train_drl.py --stage shuffle                # G1 泄漏门对照实验

分阶段是为了让每一步都能独立重跑/并行，也便于在算力受限的机器上分批完成。

产物（默认 data_out/drl/）::

    sweep.json            超参搜索结果（只看验证集）
    agent_seed{N}.npz     该 seed 的最优 checkpoint（按验证集累计奖励选）
    history_seed{N}.csv   逐 epoch 训练日志（收敛曲线）
    metrics_seed{N}.json  该 seed 在 val/test 上的指标
    results.json          基线 + DRL 汇总（报告直接引用这个文件）
    results_shuffled.json 打乱风险起点后的对照结果

纪律（硬编码在流程里，不靠自觉）：
  - 归一化统计量与网络参数只见训练集 2023-01..2024-12
  - 超参与 checkpoint 只用验证集 2025-01..2025-06 选
  - 测试集 2025-07..2026-04 只在 report 阶段评估一次，不参与任何选择
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.dataset import (Normalizer, _lead_minutes, apply_continuous_risk,
                         load_episodes, split_episodes)
from drl.dqn import DQNAgent
from drl.env import RewardSpec
from drl.train import (baseline_table, evaluate_agent, evaluate_policy,
                       hit_base_rate, train)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "data_out", "anchor")
OUT = os.path.join(ROOT, "data_out", "drl")

DEFAULT_CFG = {"gamma": 0.95, "lr": 1e-3, "hidden": 128, "weight_decay": 1e-4}
# 实测本任务约 1 个 epoch 后就开始过拟合，故搜索重点放在折扣因子与权重衰减，
# 而非更长的训练；epoch 数少但每 epoch 内多次评估、保留验证集最优 checkpoint。
SWEEP_GRID = [
    {"gamma": 0.95, "lr": 1e-3, "hidden": 128, "weight_decay": 0.0},
    {"gamma": 0.95, "lr": 1e-3, "hidden": 128, "weight_decay": 1e-4},
    {"gamma": 0.90, "lr": 1e-3, "hidden": 128, "weight_decay": 1e-4},
]
_CFG_KEYS = ("gamma", "lr", "hidden", "weight_decay")


def _fmt(name: str, m: dict) -> str:
    return (f"  {name:26s} R={m['total_reward']:9.1f}  P={m['precision']:6.1%}  "
            f"Rec={m['recall']:6.1%}  lead={m['avg_lead_min']:5.1f}m  "
            f"n_alert={m['n_alert']:6d}  rate={m['alert_rate']:6.1%}")


def prepare(args, shuffle: bool = False):
    """载入数据 → 修正风险口径 → 切分 → 归一化。

    shuffle=True 时随机重排风险起点（泄漏门用）。

    风险起点默认按**品种跨月连续**判定（见 dataset.apply_continuous_risk）。
    注意顺序：先 `verify=True` 校验存档标签与逐月口径逐位一致（证明 parquet 没坏），
    再改写为连续口径。两道保障都要，不能因为要修口径就把校验关掉。
    """
    eps = load_episodes(args.anchor, verify=True,
                        underlying=getattr(args, "underlying", False))
    if not getattr(args, "monthly_risk", False):
        apply_continuous_risk(eps)
    if shuffle:
        rng = np.random.default_rng(12345)
        for e in eps:
            k = len(e.risk_idx)
            if k:
                e.risk_idx = np.sort(rng.choice(np.arange(len(e)), size=k, replace=False))
            e.lead_min = _lead_minutes(pd.Series(pd.to_datetime(e.dt)),
                                       e.risk_idx.tolist())
    tr, va, te = split_episodes(eps, train_end=args.train_end, val_end=args.val_end)
    # 事件级误报计价的开关走环境变量，避免给所有 stage 都加一个参数。
    # **默认关闭**：不设该变量时行为与既往逐位一致（已实测交付指标不变）。
    # 仅 scripts/event_reward_train.py 的对照实验会设置它。
    _ev = os.environ.get("DRL_EVENT_LEVEL_FP") == "1"
    spec = RewardSpec(event_level_fp=True) if _ev else RewardSpec()
    if _ev:
        print("  [奖励] event_level_fp=True（误报按预警事件计价，仅罚每段第一条）")
    return tr, va, te, Normalizer.fit(tr), spec


def _fingerprint(tr, va, te) -> dict:
    """数据集指纹：幕数/截面数/品种表/特征数/风险起点数。用于拦截「新旧数据混算」。

    `n_features` 与 `n_risk` 必须在指纹里：26 维与 32 维的 checkpoint 无法互换；
    逐月与连续两种风险口径下的指标也不可比（命中基础率不同）。
    历史上这里漏过一次——扩样品种后只重训了部分 seed，旧结果被混进集成平均值。
    """
    allg = tr + va + te
    return {"n_episodes": len(allg),
            "n_slices": int(sum(len(e) for e in allg)),
            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
            "symbols": sorted({e.symbol for e in allg}),
            "n_features": int(allg[0].X.shape[1]) if allg else 0,
            "n_risk": int(sum(len(e.risk_idx) for e in allg))}


def _load_cfg(out: str) -> dict:
    p = os.path.join(out, "sweep.json")
    if not os.path.exists(p):
        return dict(DEFAULT_CFG)
    with open(p, encoding="utf-8") as f:
        sweep = json.load(f)
    best = max(sweep, key=lambda r: r["val_reward"])
    return {k: best[k] for k in _CFG_KEYS}


# ---------------------------------------------------------------- stages

def stage_sweep(args) -> None:
    tr, va, te, norm, spec = prepare(args)
    print(f"超参搜索：{len(SWEEP_GRID)} 组 × {args.sweep_epochs} epoch（只看验证集）")
    sweep = []
    for g in SWEEP_GRID:
        print(f"\n  -> {g}")
        ag, _ = train(tr, va, norm, spec, epochs=args.sweep_epochs,
                      seed=args.seed, verbose=True, **g)
        m = evaluate_agent(va, norm, spec, ag)
        sweep.append({**g, "val_reward": m["total_reward"],
                      "val_precision": m["precision"], "val_recall": m["recall"],
                      "val_lead": m["avg_lead_min"]})
        print(_fmt("   ↳ val", m))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sweep.json"), "w", encoding="utf-8") as f:
        json.dump(sweep, f, ensure_ascii=False, indent=2)
    best = max(sweep, key=lambda r: r["val_reward"])
    print(f"\n选中: " + " ".join(f"{k}={best[k]}" for k in _CFG_KEYS) +
          f"  (val R={best['val_reward']:.0f})")


class _Paused(Exception):
    """内部信号：本次调用的 epoch 预算用尽，状态已存盘，等待下次续跑。"""


def stage_train(args, shuffle: bool = False) -> None:
    tr, va, te, norm, spec = prepare(args, shuffle=shuffle)
    cfg = _load_cfg(args.out) if not shuffle else dict(DEFAULT_CFG)
    tag = f"{'shuf_' if shuffle else ''}seed{args.seed}"
    os.makedirs(args.out, exist_ok=True)
    wip = os.path.join(args.out, f"_wip_{tag}.pkl")
    fp = _fingerprint(tr, va, te)

    # 续跑：状态文件必须与当前数据/超参完全对应，否则宁可从头
    resume, done_epochs = None, 0
    if os.path.exists(wip):
        with open(wip, "rb") as f:
            st = pickle.load(f)
        if st.get("_cfg") == cfg and st.get("_fp") == fp:
            resume, done_epochs = st, st["next_epoch"]
            print(f"续跑 {tag}：已完成 {done_epochs}/{args.epochs} epoch")
        else:
            print(f"⚠ {wip} 与当前配置/数据不匹配，丢弃并从头训练")

    budget = args.epochs_per_call or args.epochs

    def _on_epoch_end(ep_i, state):
        state["_cfg"], state["_fp"] = cfg, fp
        with open(wip, "wb") as f:
            pickle.dump(state, f, protocol=4)
        if ep_i + 1 < args.epochs and (ep_i + 1 - done_epochs) >= budget:
            raise _Paused

    print(f"训练 {tag}：{args.epochs} epoch，配置 {cfg}")
    t0 = time.time()
    try:
        ag, hist = train(tr, va, norm, spec, epochs=args.epochs, seed=args.seed,
                         verbose=True, resume=resume, on_epoch_end=_on_epoch_end,
                         **cfg)
    except _Paused:
        with open(wip, "rb") as f:
            st = pickle.load(f)
        print(f"已完成 {st['next_epoch']}/{args.epochs} epoch，状态存于 {os.path.basename(wip)}"
              f"（用时 {time.time() - t0:.0f}s）。重跑同一命令继续。")
        return

    pd.DataFrame(hist).to_csv(os.path.join(args.out, f"history_{tag}.csv"), index=False)
    ag.save(os.path.join(args.out, f"agent_{tag}.npz"),
            extra={"normalizer": norm.to_dict(), "cfg": cfg, "seed": args.seed,
                   "reward": spec.describe(),
                   "features": list(tr[0].feat_names or [])})
    # 成品已落盘 → 断点是死重，就地清掉。
    #
    # 为什么必须自动清：`_wip` 里含**整个回放池**，单个 56MB。早前训练完不清，
    # 实测 `data_out/` 累积到 **1.9G，其中 1.78G 是这些断点**——真正的交付产物
    # （15 个 agent + 图谱 + 报告）只有约 77MB。一份要提交的作品里 95% 是训练残渣。
    # 放在 `ag.save` **之后**：只有成品确实写成功了才删，中途失败仍可续跑。
    if os.path.exists(wip):
        try:
            os.remove(wip)
        except OSError as e:                                  # noqa: BLE001
            print(f"⚠ 断点 {os.path.basename(wip)} 未能删除（{e}），可手工清理")
    mv, mt = evaluate_agent(va, norm, spec, ag), evaluate_agent(te, norm, spec, ag)
    with open(os.path.join(args.out, f"metrics_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "cfg": cfg, "shuffled": shuffle,
                   "fingerprint": fp,
                   "val": mv, "test": mt}, f, ensure_ascii=False, indent=2, default=float)
    if os.path.exists(wip):
        # 清理失败（只读挂载等）不该让整次训练白跑——续跑逻辑靠 _cfg/_fp 校验，
        # 残留的旧状态文件会在下次被识别为不匹配并丢弃，不会污染结果。
        try:
            os.remove(wip)
        except OSError as e:
            print(f"⚠ 未能删除中间状态 {os.path.basename(wip)}：{e}")
    print(_fmt(f"{tag} VAL", mv))
    print(_fmt(f"{tag} TEST", mt))
    print(f"用时 {time.time() - t0:.0f}s")


def stage_report(args, shuffle: bool = False) -> None:
    tr, va, te, norm, spec = prepare(args, shuffle=shuffle)
    pat = os.path.join(args.out, f"metrics_{'shuf_' if shuffle else ''}seed*.json")
    files = sorted(f for f in glob.glob(pat)
                   if ("shuf_" in os.path.basename(f)) == shuffle)
    if not files:
        raise SystemExit(f"找不到任何 {pat}，请先跑 --stage train")
    per = [json.load(open(f, encoding="utf-8")) for f in files]
    seeds = [p["seed"] for p in per]

    # 数据指纹一致性：扩样品种后若只重训了部分 seed，旧结果会与新结果混在一起被平均，
    # 报告数字看着正常但实际横跨两个数据集——这类错误极难事后察觉，故在此硬拦。
    fp_now = _fingerprint(tr, va, te)
    stale = [(p["seed"], p.get("fingerprint")) for p in per
             if p.get("fingerprint") != fp_now]
    if stale:
        raise SystemExit(
            f"以下 seed 的结果与当前数据集不匹配（可能是扩样前训练的旧结果）：{stale}\n"
            f"当前数据指纹: {fp_now}\n"
            f"请对这些 seed 重跑 --stage train，或删除其 metrics_seed*.json 后再汇总。")
    print(f"汇总 {len(per)} 个 seed: {seeds}（数据指纹 {fp_now['n_episodes']} 幕 / "
          f"{len(fp_now['symbols'])} 品种，全部一致）\n")

    print("=" * 100)
    print("对照基线（VAL 2025-01..2025-06）")
    base_va = baseline_table(va, norm, spec)
    for k, m in base_va.items():
        print(_fmt(k, m))

    print("\n" + "=" * 100)
    print("对照基线（TEST 2025-07..2026-04）")
    base_te = baseline_table(te, norm, spec)
    for k, m in base_te.items():
        print(_fmt(k, m))

    print("\nDRL 各 seed（TEST）")
    for p in per:
        print(_fmt(f"seed{p['seed']}", p["test"]))

    def agg(key, split="test"):
        v = np.array([p[split][key] for p in per], dtype=float)
        return float(v.mean()), float(v.std())

    print("\nDRL 汇总（TEST，均值 ± 标准差）")
    for key, lab in [("total_reward", "累计奖励"), ("precision", "精确率"),
                     ("recall", "召回率"), ("avg_lead_min", "平均提前(min)"),
                     ("alert_rate", "预警率")]:
        m, s = agg(key)
        print(f"    {lab:14s} {m:10.4f} ± {s:.4f}")

    # ---- 集成：多 seed Q 值平均。单 seed 方差大，集成才是合理的生产配置
    ens_va = ens_te = None
    try:
        from drl.api import AlertAgentAPI
        api = AlertAgentAPI.from_dir(args.out,
                                     pattern=f"agent_{'shuf_' if shuffle else ''}seed*.npz")
        ens_va = evaluate_policy(va, norm, spec, lambda e: api.policy())
        ens_te = evaluate_policy(te, norm, spec, lambda e: api.policy())
        print("\n集成（%d 个 seed 的 Q 值平均）" % len(api.agents))
        print(_fmt("ensemble VAL", ens_va))
        print(_fmt("ensemble TEST", ens_te))
    except Exception as exc:                                   # noqa: BLE001
        print(f"\n（集成评估跳过: {type(exc).__name__}: {exc}）")

    # ---- 判别力：精确率相对 hit 基础率的提升。盲策略恒为 0，是"是否学到信号"的硬指标
    base = hit_base_rate(te)
    print(f"\n判别力（精确率 − hit 基础率 {base:.2%}）——盲策略恒≈0")
    disc = {}
    for k, m in base_te.items():
        if np.isfinite(m.get("precision", np.nan)) and m["n_alert"] > 0:
            disc[k] = m["precision"] - base
            print(f"    {k:26s} {disc[k]:+.2%}")
    disc["DRL(单seed均值)"] = agg("precision")[0] - base
    print(f"    {'DRL(单seed均值)':26s} {disc['DRL(单seed均值)']:+.2%}")
    if ens_te:
        disc["DRL(集成)"] = ens_te["precision"] - base
        print(f"    {'DRL(集成)':26s} {disc['DRL(集成)']:+.2%}")

    rule = base_te["规则基线(校准阈值)"]
    orc = base_te["预言机(上界)"]["total_reward"]
    blind = next(v for k, v in base_te.items() if k.startswith("最佳盲节奏"))
    drl_r, drl_s = agg("total_reward")
    gap = orc - rule["total_reward"]
    recovered = (drl_r - rule["total_reward"]) / abs(gap) if gap else float("nan")
    rel_abs = (drl_r - rule["total_reward"]) / abs(rule["total_reward"]) \
        if rule["total_reward"] else float("nan")
    ens_rel = ((ens_te["total_reward"] - rule["total_reward"]) / abs(rule["total_reward"])
               if ens_te and rule["total_reward"] else float("nan"))
    print(f"\n  规则基线 {rule['total_reward']:.1f} → DRL 单seed均值 {drl_r:.1f} (±{drl_s:.1f})"
          + (f" → 集成 {ens_te['total_reward']:.1f}" if ens_te else ""))
    print(f"  最佳盲节奏（不看特征）{blind['total_reward']:.1f}；预言机上界 {orc:.1f}")
    print(f"  相对规则基线提升：单seed均值 {rel_abs:+.1%}"
          + (f"，集成 {ens_rel:+.1%}" if ens_te else ""))
    print(f"  收复「基线→上界」差距：{recovered:+.1%}")

    out = {
        "seeds": seeds, "cfg": per[0]["cfg"], "shuffled": shuffle,
        "reward_spec": spec.describe(),
        "split": {"train_end": args.train_end, "val_end": args.val_end,
                  "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                  "risk_train": int(sum(len(e.risk_idx) for e in tr)),
                  "risk_val": int(sum(len(e.risk_idx) for e in va)),
                  "risk_test": int(sum(len(e.risk_idx) for e in te))},
        "baseline_val": base_va, "baseline_test": base_te,
        "drl_per_seed": per,
        "drl_test_mean": {k: agg(k)[0] for k in
                          ("total_reward", "precision", "recall", "avg_lead_min", "alert_rate")},
        "drl_test_std": {k: agg(k)[1] for k in
                         ("total_reward", "precision", "recall", "avg_lead_min", "alert_rate")},
        "drl_val_mean": {k: agg(k, "val")[0] for k in
                         ("total_reward", "precision", "recall", "avg_lead_min", "alert_rate")},
        "ensemble_val": ens_va, "ensemble_test": ens_te,
        "hit_base_rate_test": base, "discrimination_test": disc,
        "lift_vs_rule_abs": rel_abs, "lift_gap_recovered": recovered,
        "lift_ensemble_vs_rule": ens_rel,
    }
    name = "results_shuffled.json" if shuffle else "results.json"
    with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n→ {os.path.join(args.out, name)}")


def stage_ablation(args) -> None:
    """奖励设计消融：证明第一版「每步下注」奖励可被「恒不预警」套利。

    产物 ablation_reward.json 供报告引用——这段结论因此不是手写的历史数字，
    而是每次都能重跑出来的实测值。
    """
    tr, va, te, norm, _ = prepare(args)
    from drl.train import hit_base_rate
    out = {"hit_base_rate": {"train": hit_base_rate(tr), "val": hit_base_rate(va),
                             "test": hit_base_rate(te),
                             "all": hit_base_rate(tr + va + te)}}
    print(f"hit 基础率: " + "  ".join(f"{k}={v:.4f}" for k, v in out["hit_base_rate"].items()))
    for mode in ("perstep_v1", "event"):
        spec = RewardSpec(mode=mode)
        tbl = baseline_table(te, norm, spec)
        out[mode] = {k: v["total_reward"] for k, v in tbl.items()}
        rule = tbl["规则基线(校准阈值)"]["total_reward"]
        never = tbl["恒不预警(level0)"]["total_reward"]
        out[mode + "_never_beats_rule"] = bool(never > rule)
        print(f"\n[{mode}] (TEST)")
        for k, v in tbl.items():
            print(f"    {k:26s} {v['total_reward']:10.1f}")
        print(f"    → 恒不预警是否优于规则基线: "
              f"{'是（奖励可被套利）' if never > rule else '否'}")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ablation_reward.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n→ {os.path.join(args.out, 'ablation_reward.json')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all",
                    choices=["all", "sweep", "train", "report", "shuffle",
                             "shuffle-report", "ablation"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epochs-per-call", type=int, default=0,
                    help="每次调用最多跑几个 epoch，跑满即存盘退出（0=不限）。"
                         "用于单次执行时长受限的环境，重跑同一命令自动续训")
    ap.add_argument("--sweep-epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="42,43,44", help="--stage all 时用")
    ap.add_argument("--train-end", default="2024-12")
    ap.add_argument("--val-end", default="2025-06")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--underlying", action="store_true",
                    help="状态里并入 6 维标的特征（26→32 维），需先跑 "
                         "scripts/build_underlying.py")
    ap.add_argument("--monthly-risk", action="store_true",
                    help="风险起点退回逐月判定（滚动窗口每月重置）。**仅用于复现旧结果**——"
                         "该口径下每幕前 72 根截面不可能有正标签，见 README §7.10")
    args = ap.parse_args()

    if args.stage == "sweep":
        stage_sweep(args)
    elif args.stage == "train":
        stage_train(args)
    elif args.stage == "report":
        stage_report(args)
    elif args.stage == "shuffle":
        stage_train(args, shuffle=True)
    elif args.stage == "shuffle-report":
        stage_report(args, shuffle=True)
    elif args.stage == "ablation":
        stage_ablation(args)
    else:
        stage_ablation(args)
        stage_sweep(args)
        for sd in [int(s) for s in args.seeds.split(",") if s.strip()]:
            args.seed = sd
            stage_train(args)
        stage_report(args)


if __name__ == "__main__":
    main()
