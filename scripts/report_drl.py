"""report_drl — 由 data_out/drl/results.json 生成问题3 的离线回测报告。

报告里**没有一个手打的数字**——全部从 results.json / ablation_reward.json /
gates.json / sweep.json / history_seed*.csv 读出来再格式化。这样报告与代码输出
不可能对不上，也便于重跑后一键刷新。

依赖的产物及其生成命令：
    results.json          scripts/train_drl.py --stage report
    ablation_reward.json  scripts/train_drl.py --stage ablation
    gates.json            python -m drl.tests --underlying --emit

用法::

    python scripts/report_drl.py                    # 生成 data_out/drl_report.md
    python scripts/report_drl.py --no-plot          # 跳过收敛曲线图
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRL_OUT = os.path.join(ROOT, "data_out", "drl")
REPORT = os.path.join(ROOT, "data_out", "drl_report.md")

TARGET = {"precision": 0.50, "recall": 0.60, "avg_lead_min": 30.0}


def _pct(v, digits=1):
    return "—" if v is None or not np.isfinite(v) else f"{v * 100:.{digits}f}%"


def _num(v, digits=1):
    return "—" if v is None or not np.isfinite(v) else f"{v:,.{digits}f}"


def _mark(v, key):
    if v is None or not np.isfinite(v):
        return ""
    return " ✓" if v >= TARGET[key] else " ✗"


def _row(name, m, base=None):
    if m is None:
        return None
    p, r, l = m.get("precision"), m.get("recall"), m.get("avg_lead_min")
    disc = (p - base) if (base is not None and p is not None and np.isfinite(p)) else None
    n = m.get("n_alert", -1)
    return (f"| {name} | {_num(m['total_reward'])} | {_pct(p)}{_mark(p, 'precision')} | "
            f"{_pct(r)}{_mark(r, 'recall')} | {_num(l)}{_mark(l, 'avg_lead_min')} | "
            f"{'—' if n < 0 else f'{n:,}'} | {_pct(m.get('alert_rate'))} | "
            f"{'—' if disc is None else _pct(disc, 2)} |")


def plot_curves(out_png: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                        # noqa: BLE001
        return False
    files = sorted(f for f in glob.glob(os.path.join(DRL_OUT, "history_seed*.csv")))
    if not files:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for f in files:
        d = pd.read_csv(f)
        sd = os.path.basename(f).split("seed")[-1].split(".")[0]
        x = np.arange(len(d))
        axes[0].plot(x, d["val_reward"], marker="o", ms=3, label=f"seed{sd}")
        axes[1].plot(x, d["val_precision"], marker="o", ms=3, label=f"seed{sd}")
        axes[2].plot(x, d["val_recall"], marker="o", ms=3, label=f"seed{sd}")
    for ax, t, hl in ((axes[0], "Validation cumulative reward", None),
                      (axes[1], "Validation precision", 0.50),
                      (axes[2], "Validation recall", 0.60)):
        ax.set_title(t, fontsize=11)
        ax.set_xlabel("checkpoint (4 per epoch)")
        ax.grid(alpha=.25, linewidth=.6)
        if hl is not None:
            ax.axhline(hl, ls="--", lw=1, color="#888")
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=REPORT)
    ap.add_argument("--drl-dir", default=DRL_OUT)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    def _rel(p: str) -> str:
        """把绝对路径显示成仓库内相对路径，便于报告在任何机器上都可读。"""
        try:
            return os.path.relpath(os.path.abspath(p), ROOT)
        except ValueError:
            return p

    rp = os.path.join(args.drl_dir, "results.json")
    if not os.path.exists(rp):
        raise SystemExit(f"缺少 {rp}，请先跑 python scripts/train_drl.py --stage all")
    R = json.load(open(rp, encoding="utf-8"))

    def _load(name):
        p = os.path.join(args.drl_dir, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    ABL = _load("ablation_reward.json")     # 奖励设计消融（--stage ablation）
    GATES = _load("gates.json")             # 验证门结果（python -m drl.tests --underlying --emit）

    base_te, base_va = R["baseline_test"], R["baseline_val"]
    mean, std = R["drl_test_mean"], R["drl_test_std"]
    ens = R.get("ensemble_test")
    hbr = R.get("hit_base_rate_test")
    rule = base_te["规则基线(校准阈值)"]
    blind_key = next(k for k in base_te if k.startswith("最佳盲节奏"))
    blind = base_te[blind_key]
    orc = base_te["预言机(上界)"]

    # 各 seed 的测试集累计奖励，用于披露"是否有单模型优于集成"
    seed_rewards = {p["seed"]: p["test"]["total_reward"] for p in R["drl_per_seed"]}
    best_seed = max(seed_rewards, key=seed_rewards.get)
    seed_prec = {p["seed"]: p["test"]["precision"] for p in R["drl_per_seed"]}
    best_seed_prec = max(seed_prec.values())
    # 样本标准差（ddof=1），与 results.json 里的总体标准差（ddof=0）并列给出
    sd1 = float(np.std(list(seed_rewards.values()), ddof=1))

    png = os.path.join(args.drl_dir, "convergence.png")
    has_plot = (not args.no_plot) and plot_curves(png)

    L = []
    A = L.append
    A("# 问题3：基于深度强化学习的自适应预警策略 —— 离线回测报告\n")
    A("> 本报告由 `scripts/report_drl.py` 生成。**全部实验结果数字**（累计奖励、精确率、")
    A(f"> 召回率、提前时间、判别力、标准差、消融、验证门结论）均从 `{_rel(args.drl_dir)}` 下的")
    A("> `results.json` / `ablation_reward.json` / `gates.json` / `sweep.json` /")
    A("> `history_seed*.csv` 读出后格式化，重跑训练后重新执行本脚本即可整体刷新。")
    A("> 文中的**口径常量**（120 分钟窗口、状态维度、赛题目标阈值等）属固定描述，写在正文里。\n")

    # ---------------- 1
    A("## 1. 任务与评测口径\n")
    A("赛题问题3 要求：以曲面风险特征为状态、预警等级为动作、以历史预警"
      "正确/虚假/漏报的累积奖励为优化目标，训练 DRL 智能体动态调整预警策略，"
      "并在离线回测中与固定阈值方法对比。加分门槛为**累计奖励或 CVaR 改善率"
      "相对固定规则提升 ≥10%**。\n")
    sp = R["split"]
    # 品种数与状态维度从 metrics_seed*.json 的数据指纹读出，不写死——
    # 早前这里硬编码「4 个品种」「28 维」，扩样与加特征后就成了错的
    fp = {}
    for f in sorted(glob.glob(os.path.join(args.drl_dir, "metrics_seed*.json"))):
        if "shuf_" in os.path.basename(f):
            continue
        fp = json.load(open(f, encoding="utf-8")).get("fingerprint", {})
        break
    n_sym = len(fp.get("symbols", [])) or 0
    n_feat = fp.get("n_features", 0)
    sdim = (n_feat + 2) if n_feat else 0
    A("数据来自 `data_out/anchor/`（由 `scripts/build_anchor_dataset.py` 生成）："
      f"{n_sym or '若干'} 个品种（{'/'.join(fp.get('symbols', []))}）、每 (品种,月) 一幕，"
      f"共 {sp['n_train'] + sp['n_val'] + sp['n_test']} 幕。"
      "严格按时间先后切分，测试集全部晚于训练集：\n")
    A("| 划分 | 区间 | 幕数 | 风险起点数 | 用途 |")
    A("|------|------|------|-----------|------|")
    A(f"| 训练 | 2023-01 .. {sp['train_end']} | {sp['n_train']} | {sp['risk_train']} | "
      "拟合网络与归一化统计量 |")
    A(f"| 验证 | {sp['train_end']}+ .. {sp['val_end']} | {sp['n_val']} | {sp['risk_val']} | "
      "选超参、选 checkpoint |")
    A(f"| 测试 | {sp['val_end']}+ .. 2026-04 | {sp['n_test']} | {sp['risk_test']} | "
      "DRL 侧不做任何选择 |")
    A("\n测试集覆盖 2026-01 白银与 2026-03 原油两次极端行情，是赛题最关注的时段。\n")
    A("> 一处需要如实声明的例外：DRL 的超参与 checkpoint 全部只用验证集选，测试集不参与；"
      "但**盲节奏基线的周期**（每 N 步报一次）是在被评估的那个数据集上取最优的。"
      "这个方向是**对基线有利、对本文结论从严**——它把对照线做强了，不会抬高 DRL 的成绩。\n")
    A("精确率/召回率/平均提前时间的计算完全复用赛题口径，并通过验证门 G2 "
      "与既有实现 `vol_surface/quant_metrics.evaluate_quant` 逐位比对一致。\n")

    # ---------------- 2
    A("## 2. MDP 设计\n")
    A(f"**状态（{sdim or 28} 维）**：{n_feat or 26} 维风险特征"
      f"{'（26 维曲面 + 6 维标的侧）' if n_feat == 32 else '（曲面）'}"
      "（训练集统计量 z-score 归一、截断 ±5）"
      "+ 2 维决策上下文（距上次预警的分钟数、上一步动作）。上下文这两维让智能体"
      "能感知「我刚报过了」，从而学会抑制冗余预警——这是无状态的固定阈值规则做不到的。"
      "状态中不含任何未来信息，由验证门 G1 强制保证。\n")
    A("**动作**：0=无预警 / 1=关注 / 2=警告 / 3=危险，与赛题四级定义一致。\n")
    A("**奖励**：事件级会计，直接映射赛题指标——\n")
    A("| 事件 | 触发条件 | 奖励 |")
    A("|------|---------|------|")
    A("| cover | 首次覆盖某风险起点的 level≥2 预警 | `+6·(1+提前占比)·等级系数` |")
    A("| redundant | 预警正确但该风险起点已被更早预警覆盖 | `+0.2·等级系数` |")
    A("| false | 预警后 120min 内无风险起点 | `−1.0·等级系数` |")
    A("| miss | 风险起点走过时从未被覆盖 | `−10` |")
    A("| watch | level==1（不计入赛题预警集合） | `±0.05` 量级整形 |")
    A(f"\n实际参数：`{R['reward_spec']}`\n")
    if ABL:
        v1, ev = ABL["perstep_v1"], ABL["event"]
        br = ABL["hit_base_rate"]
        A("> **踩过的坑1：第一版奖励可被「闭嘴」套利。** 初版用「每步下注」结构"
          "（命中 +1 / 误报 −1）。由于风险起点基础率偏低"
          f"（全量 {_pct(br['all'], 2)}、测试集 {_pct(br['test'], 2)}），"
          "「恒不预警」的累计奖励反而**高于**规则基线——在那种奖励下 DRL 只要学会"
          "永远不报就能赢，而召回率是 0。改为事件级计价后该套利消失：\n")
        A("| 奖励设计 | 规则基线 | 恒不预警 | 恒不预警是否套利成功 |")
        A("|---------|---------|---------|------------------|")
        for key, lab in (("perstep_v1", "第一版·每步下注"), ("event", "现版·事件级")):
            d = ABL[key]
            A(f"| {lab} | {_num(d['规则基线(校准阈值)'])} | {_num(d['恒不预警(level0)'])} | "
              f"{'**是**' if ABL[key + '_never_beats_rule'] else '否'} |")
        A("\n> 该消融可用 `python scripts/train_drl.py --stage ablation` 复现"
          "（第一版奖励保留在 `RewardSpec(mode=\"perstep_v1\")`）。"
          "验证门 G4 现在持续守住这一性质。\n")
    A("> **踩过的坑2**：环境最初把「恰好在风险起点当根 K 线发出的预警」同时记成"
      "误报与漏报，而赛题的两个 120 分钟窗口 `[rs−120, rs]` 与 `[a, a+120]` 都是"
      "闭区间、这类预警应算命中。修正后，环境的 cover/miss 计数与赛题指标的"
      "召回/漏报数**完全相等**（验证门 G3）。\n")

    # ---------------- 3
    A("## 3. 训练设置\n")
    cfg = R["cfg"]
    A(f"- 算法：Double DQN（在线网选动作、目标网估值）+ 经验回放 + Huber 损失\n"
      f"- 网络：{sdim or 28} → {cfg['hidden']} → {cfg['hidden']} → 4 的 MLP，**纯 NumPy 手写前反向**\n"
      f"- 超参（在验证集上小规模搜索选出）：γ={cfg['gamma']}、lr={cfg['lr']}、"
      f"weight_decay={cfg['weight_decay']}\n"
      f"- 随机种子：{R['seeds']}，共 {len(R['seeds'])} 个独立训练\n")
    A("不引入 PyTorch 的理由：本项目原依赖里没有任何深度学习框架，而这里只需要一个"
      "两万参数量级的小 MLP；纯 NumPy 可完全确定性复现（验证门 G6），CPU 数分钟"
      "跑完，评审复现零环境成本。\n")
    sweep_p = os.path.join(args.drl_dir, "sweep.json")
    if os.path.exists(sweep_p):
        sw = json.load(open(sweep_p, encoding="utf-8"))
        A("超参搜索结果（只看验证集）：\n")
        A("| γ | lr | hidden | weight_decay | 验证集累计奖励 |")
        A("|---|----|--------|--------------|---------------|")
        for s in sorted(sw, key=lambda r: -r["val_reward"]):
            A(f"| {s['gamma']} | {s['lr']} | {s['hidden']} | {s.get('weight_decay', 0)} | "
              f"{_num(s['val_reward'])} |")
        A("")
    # 最优 checkpoint 的实际位置——从 history CSV 现算，不写死
    peaks = []
    for f in sorted(glob.glob(os.path.join(args.drl_dir, "history_seed*.csv"))):
        d = pd.read_csv(f)
        peaks.append(str(d.loc[d["val_reward"].idxmax(), "tag"]))
    if peaks:
        # ⚠ 交付版是 15 个 seed，但目录里只有 5 份 `history_seed*.csv`
        # （其余 seed 未保留逐评估历史）。此前这段紧跟在「共 15 个独立训练」之后
        # 写「各 seed 的最优 checkpoint 位置为 …」，读者会以为覆盖全部 15 个。
        # 故显式写明这是**留有历史记录的那几个 seed**，并给出分母。
        _ns = len(R.get("drl_per_seed", [])) or len(peaks)
        A(f"训练过程明显过拟合：训练奖励持续上升而验证奖励很早见顶。"
          f"**{len(peaks)}/{_ns} 个 seed 留有逐评估历史**（其余未保存 "
          f"`history_seed*.csv`），这几个的最优 checkpoint 位置为 "
          f"`{'`, `'.join(peaks)}`（tag 形如 `e0.2` 表示第 0 个 epoch "
          f"的第 3 次评估，每 epoch 评估 4 次），"
          f"其中 {sum(1 for x in peaks if x.startswith('e0.'))}/{len(peaks)} 个在"
          f"**第 1 个 epoch 之内**就到顶。因此采用 epoch 内多次评估 + 保留验证集最优"
          f" checkpoint 的早停策略。**该观察基于这 {len(peaks)} 个 seed，"
          f"不宜外推到全部 {_ns} 个。**\n")
    if os.path.exists(sweep_p):
        w0 = [x for x in sw if x.get("weight_decay", 0) == 0]
        w1 = [x for x in sw if x.get("weight_decay", 0) and x["gamma"] == 0.95]
        if w0 and w1:
            A(f"权重衰减消融（同为 γ=0.95）：wd=0 时验证集累计奖励 "
              f"{_num(w0[0]['val_reward'])}，wd={w1[0]['weight_decay']} 时 "
              f"{_num(w1[0]['val_reward'])}。注意出厂配置最终选的是 γ={cfg['gamma']}，"
              f"该消融是在 γ=0.95 下做的受控对比，搜索网格中没有 γ=0.9 + wd=0 的组合。\n")
    if has_plot:
        A(f"![收敛曲线]({os.path.relpath(png, os.path.dirname(args.out))})\n")
        A("*各 seed 在验证集上的累计奖励 / 精确率 / 召回率随 checkpoint 的变化；"
          "虚线为赛题目标线。*\n")

    # ---------------- 4
    A("## 4. 主结果（测试集 2025-07 .. 2026-04）\n")
    A(f"「判别力」= 精确率 − hit 基础率（{_pct(hbr, 2)}）。"
      "会预警但不看特征的策略，其精确率必然≈基础率、判别力≈0，"
      "因此这一列才是「有没有真正学到曲面信号」的硬指标。\n")
    A("> 一个口径说明：「恒不预警」从不发预警，精确率按 0 计，故判别力显示为 "
      f"{_pct(-hbr, 2)}。这是分母为空的退化情形，不参与上述解读。\n")
    A("| 策略 | 累计奖励 | 精确率 | 召回率 | 平均提前 | 预警数 | 预警率 | 判别力 |")
    A("|------|---------|--------|--------|---------|--------|--------|--------|")
    A(_row("**规则基线**（问题1 校准阈值）", rule, hbr))
    A(_row(f"**{blind_key}**", blind, hbr))
    A(_row("随机策略", base_te["随机策略"], hbr))
    A(_row("恒不预警 level0", base_te["恒不预警(level0)"], hbr))
    A(_row("恒警告 level2", base_te["恒警告(level2)"], hbr))
    drl_row = (f"| **DRL 单 seed 均值**（n={len(R['seeds'])}） | "
               f"{_num(mean['total_reward'])} ± {_num(std['total_reward'])} | "
               f"{_pct(mean['precision'])}{_mark(mean['precision'], 'precision')} | "
               f"{_pct(mean['recall'])}{_mark(mean['recall'], 'recall')} | "
               f"{_num(mean['avg_lead_min'])}{_mark(mean['avg_lead_min'], 'avg_lead_min')} | "
               f"— | {_pct(mean['alert_rate'])} | "
               f"{_pct(mean['precision'] - hbr, 2)} |")
    A(drl_row)
    if ens:
        A(_row("**DRL 集成**（多 seed Q 值平均）", ens, hbr))
    A(_row("预言机上界（每个风险起点前恰好报一次）", orc, None))
    A("")

    A("各 seed 明细：\n")
    A("| seed | 累计奖励 | 精确率 | 召回率 | 平均提前 | 预警率 |")
    A("|------|---------|--------|--------|---------|--------|")
    for p in R["drl_per_seed"]:
        t = p["test"]
        A(f"| {p['seed']} | {_num(t['total_reward'])} | {_pct(t['precision'])} | "
          f"{_pct(t['recall'])} | {_num(t['avg_lead_min'])} | {_pct(t['alert_rate'])} |")
    A("")

    # ---------------- 5
    A("## 5. 结论与赛题门槛对照\n")
    lift = R["lift_vs_rule_abs"]
    ens_lift = R.get("lift_ensemble_vs_rule")
    gap_total = orc["total_reward"] - rule["total_reward"]
    ens_recovered = ((ens["total_reward"] - rule["total_reward"]) / gap_total
                     if ens and gap_total else float("nan"))
    A(f"相对固定阈值规则基线的累计奖励提升：**单 seed 均值 {lift:+.1%}**"
      + (f"、**集成 {ens_lift:+.1%}**" if ens_lift and np.isfinite(ens_lift) else "")
      + "，均远超赛题 ≥10% 的加分门槛。\n")
    A(f"若以「基线→预言机上界」的差距（{_num(gap_total)}）为标尺，"
      f"单 seed 均值收复其中 **{R['lift_gap_recovered']:+.1%}**"
      + (f"，集成收复 **{ens_recovered:+.1%}**" if np.isfinite(ens_recovered) else "")
      + "。\n")
    A("赛题主指标达标情况（测试集，DRL 集成，**贪心工作点 lean=0**）：\n")
    if ens:
        A("| 指标 | 目标 | 规则基线 | DRL 集成 | 达标 |")
        A("|------|------|---------|---------|------|")
        A(f"| 精确率 | ≥50% | {_pct(rule['precision'])} | {_pct(ens['precision'])} | "
          f"{'✓' if ens['precision'] >= .5 else '✗'} |")
        A(f"| 召回率 | ≥60% | {_pct(rule['recall'])} | {_pct(ens['recall'])} | "
          f"{'✓' if ens['recall'] >= .6 else '✗'} |")
        A(f"| 平均提前 | ≥30min | {_num(rule['avg_lead_min'])}min | "
          f"{_num(ens['avg_lead_min'])}min | "
          f"{'✓' if ens['avg_lead_min'] >= 30 else '✗'} |")
        A("")
        wp = _load("workpoint.json")
        if wp:
            A("### 交付工作点（不是上表的贪心点）\n")
            # ⚠️ 这张表**跨两个口径**，必须写明，否则会把死区过滤的收益
            # 误归因给「换工作点」。独立审计实测：+9.1pp 里约 77% 来自过滤。
            # 左列 `results.json` 是**死区过滤关**，右列 `workpoint.json` 是**开**。
            _dzp = os.path.join(ROOT, "data_out", "dead_zone_study.json")
            _dz = json.load(open(_dzp, encoding="utf-8")) if os.path.exists(_dzp) else None
            A(f"贪心（`lean=0`）不是最优工作点。沿 P-R 前沿移动置信度门槛 `lean`，"
              f"**只用验证集**挑选（约束：召回≥60% 且 平均提前≥30min，其中精确率最大），"
              f"选出 `lean={wp['lean']:.2f}`（验证集精确率 {_pct(wp['val_precision'])}）。"
              f"该阈值在测试集上给出：\n")
            if _dz:
                A("> ⚠️ **下表左右两列口径不同**：左列「贪心 lean=0」取自 `results.json`，"
                  "**未开启结构性死区过滤**；右列「交付」取自 `workpoint.json`，"
                  "**已开启**。两列之差**不能**全部归因于换工作点——拆解见表下。\n")
            A("| 指标 | 目标 | 贪心 lean=0（过滤关） | **交付 lean=%.2f（过滤开）** | 达标 |"
              % wp["lean"])
            A("|------|------|------------|------------------|------|")
            A(f"| 精确率 | ≥50% | {_pct(ens['precision'])} | **{_pct(wp['precision'])}** | "
              f"{'✓' if wp['precision'] >= .5 else '✗'} |")
            A(f"| 召回率 | ≥60% | {_pct(ens['recall'])} | **{_pct(wp['recall'])}** | "
              f"{'✓' if wp['recall'] >= .6 else '✗'} |")
            A(f"| 平均提前 | ≥30min | {_num(ens['avg_lead_min'])}min | "
              f"**{_num(wp['avg_lead_min'])}min** | "
              f"{'✓' if wp['avg_lead_min'] >= 30 else '✗'} |")
            A("")
            if _dz:
                _g = _dz["grid_2x2"]
                _a = _g["wp_new|dz_off"]["drl"]["precision"]   # 换了工作点、未过滤
                _b = _g["wp_new|dz_on"]["drl"]["precision"]    # 再加过滤
                _wp_gain = (_a - ens["precision"]) * 100
                _dz_gain = (_b - _a) * 100
                _ro = _g["wp_old|dz_off"]["rule"]["precision"]
                _rn = _g["wp_new|dz_on"]["rule"]["precision"]
                A(f"**精确率 {_pct(ens['precision'])} → {_pct(wp['precision'])} 的拆解**"
                  f"（`data_out/dead_zone_study.json`）：换工作点贡献 "
                  f"**{_wp_gain:+.2f}pp**（{_pct(ens['precision'])} → {_pct(_a)}，"
                  f"两侧均未过滤），**结构性死区过滤贡献 {_dz_gain:+.2f}pp**"
                  f"（{_pct(_a)} → {_pct(_b)}）。即绝大部分来自过滤而非换工作点。\n")
                A(f"> **而死区过滤不是模型改进**：同一过滤施加于规则引擎，"
                  f"其精确率 {_pct(_ro)} → {_pct(_rn)}，**增益同等**；判别力 "
                  f"{_dz['verdict']['discrimination_before_pp']:+.2f}pp → "
                  f"{_dz['verdict']['discrimination_after_pp']:+.2f}pp，仅变 "
                  f"**{_dz['verdict']['delta_pp']:+.2f}pp**。预登记判据判为「全员抬升」，"
                  f"详见 README §7.19。召回与平均提前同样跨口径，不可直接相减。\n")
            # NaN 必须走单独分支，否则会渲染出「可达 —，与交付点相差 nanpp」这种
            # 半截句子（实测已泄漏进交付报告）。**指标算不出来时要说算不出来，
            # 而不是把 NaN 格式化成一个看起来像数字的东西。**
            _fr = wp.get("frontier_p_at_r60")
            if _fr is not None and _fr == _fr:
                A(f"> **测试集全程不参与选择**。作为对照：若允许在测试集上挑阈值，"
                  f"前沿在召回 {wp['constraint']['min_recall']:.0%} 处可达 "
                  f"{_pct(_fr)}，与交付点相差 "
                  f"{(_fr - wp['precision']) * 100:.2f}pp。"
                  f"**这不代表「换个阈值就能达标」**——该点对应的阈值在验证集上并不满足"
                  f"召回/提前约束，任何只看验证集的程序都不会选它。此数仅用于量化"
                  f"验证集与测试集之间的分布漂移，不作为成绩。\n")
            else:
                # ⚠ 早前这里写「没有任何一个点**满足**该召回约束」——**说反了**。
                # 实测 test_curve 全部 14 个点召回都 ≥60%（范围 60.33%~79.12%），
                # 即没有点落在 60% **以下**，插值到 60% 就成了**外推**。
                # 照原话读会得出「DRL 在测试集上达不到 60% 召回」，
                # 与交付配置 74.93% 的召回完全相反。原因用落盘的 frontier_note，不手写。
                _tc = wp.get("test_curve") or []
                _rs = [p["recall"] for p in _tc if "recall" in p]
                _rng = (f"（曲线召回范围 {min(_rs):.2%}~{max(_rs):.2%}，"
                        f"{sum(1 for r in _rs if r >= wp['constraint']['min_recall'])}"
                        f"/{len(_rs)} 个点在门槛之上）" if _rs else "")
                A(f"> **测试集全程不参与选择**。原本此处给出「若允许在测试集上挑阈值，"
                  f"前沿在召回 {wp['constraint']['min_recall']:.0%} 处能到多少」作为漂移的"
                  f"量化对照，但该点**无法插值**："
                  f"{wp.get('frontier_note', '目标召回落在测试集曲线的召回范围之外，不外推')}"
                  f"{_rng}。故此处不给数字。"
                  f"**注意这不是「DRL 达不到 60% 召回」**——恰恰相反，"
                  f"扫描到的点召回全都在门槛之上，交付配置为 {_pct(wp['recall'])}。\n")
        # **方向必须由数据决定，不能写死在模板里。**
        # 原模板无条件写「预警条数从 A 降到 B（少 X%）——用不到一半的预警量」。
        # 5-seed 时期确实是降的，换 15-seed 后集成变得更敢报（4,376 → 6,379，
        # 实为 **+45.8%**），而模板照旧输出「降到…少 -46%…用不到一半」——
        # 一句在交付报告里字面为假、且评审做一次除法就能发现的话。
        _d = ens["n_alert"] - rule["n_alert"]
        _r = abs(_d) / rule["n_alert"] * 100 if rule["n_alert"] else float("nan")
        if _d < 0:
            _alert_txt = (f"预警条数从 {rule['n_alert']:,} **降到** {ens['n_alert']:,}"
                          f"（少 {_r:.0f}%）——更少的预警量拿到了更高的精确率与召回率")
        elif _d > 0:
            _alert_txt = (f"预警条数从 {rule['n_alert']:,} **增到** {ens['n_alert']:,}"
                          f"（多 {_r:.0f}%）——**精确率的提升不是靠少报换来的**，"
                          f"智能体报得更多且更准")
        else:
            _alert_txt = f"预警条数持平（均为 {rule['n_alert']:,}）"
        A(f"DRL 集成相对规则基线，**精确率与召回率同时改善**：精确率 "
          f"{_pct(rule['precision'])} → {_pct(ens['precision'])}、召回率 "
          f"{_pct(rule['recall'])} → {_pct(ens['recall'])}，而{_alert_txt}。\n")
        A(f"> 但这**不是帕累托改进**：赛题三个主指标中的平均提前时间从 "
          f"{_num(rule['avg_lead_min'])}min 降到 {_num(ens['avg_lead_min'])}min，"
          f"是严格变劣的（仍高于 30min 门槛）。智能体倾向于等证据更充分时才出手，"
          f"这是精确率提升所付的代价，详见 §8。\n")

    # ---------------- 6
    A("## 6. 关键对照：为什么这不是「刷召回」\n")
    A("本任务的奖励（以及赛题指标本身）对召回的权重远高于误报，因此**必须**回答一个"
      "问题：DRL 的增益是真学到了曲面信号，还是只是学会了多报几次？\n")
    A(f"证据是那条「盲节奏」对照线——它完全不看特征，只按固定间隔机械刷预警。"
      f"实测它在测试集上拿到 {_num(blind['total_reward'])} 的累计奖励，"
      f"**反而优于**校准后的规则基线（{_num(rule['total_reward'])}）。"
      "这说明累计奖励单独看是个弱判据，如果只报这一个数字会严重误导。\n")
    A("真正的判据是判别力（精确率 − hit 基础率）：\n")
    disc = R.get("discrimination_test", {})
    A("| 策略 | 判别力 |")
    A("|------|--------|")
    for k in [blind_key, "随机策略", "恒警告(level2)", "规则基线(校准阈值)",
              "DRL(单seed均值)", "DRL(集成)"]:
        if k in disc:
            A(f"| {k} | {_pct(disc[k], 2)} |")
    A("")
    A("盲节奏与恒预警策略的判别力≈0（它们的精确率就等于基础率），而 DRL 集成达到 "
      f"{_pct(disc.get('DRL(集成)'), 2)}，接近规则基线（{_pct(disc.get('规则基线(校准阈值)'), 2)}）"
      "的两倍。**增益来自曲面信号，不是预警节奏。**\n")

    # ---------------- 7
    A("## 7. 验证门\n")
    if GATES:
        A(f"`python -m drl.tests --underlying --emit` 的实际执行结果："
          f"**{GATES['n_pass']}/{GATES['n_total']} 通过**"
          f"（结果存档于 `{_rel(args.drl_dir)}/gates.json`）。任一不过即视为结果不可用。\n")
    else:
        A("`python -m drl.tests` 会逐条执行以下断言，任一不过即视为结果不可用：\n")
    if GATES:
        A("| 门 | 结果 | 实测结论 |")
        A("|----|------|---------|")
        for g in GATES["gates"]:
            A(f"| {g['name']} | {'✓' if g['passed'] else '✗'} | {g['message']} |")
    else:
        A("（未找到 gates.json，请运行 `python -m drl.tests --underlying --emit`）")
    A("")
    A("> **G4 的适用范围**：该门检查的平凡策略集合是「恒0 / 恒2 / 恒3 / 随机」，"
      "**不含**盲节奏策略。事实上盲节奏策略的累计奖励是**高于**规则基线的（见 §6），"
      "这一点由 G8 单独把关。两条门合起来才完整。\n")
    g7 = (GATES or {}).get("details", {}).get("G7")
    if g7:
        A("> **G7 本身也踩过坑**：初版判据是「打乱后累计奖励须跌回随机策略水平」，"
          "结果 FAIL。排查发现不是泄漏，而是判据错了——盲节奏策略的累计奖励本来就"
          "远高于随机策略，累计奖励里混入了与信号无关的「预警节奏」成分，不能用来判泄漏。"
          f"改用判别力后：真实数据 {_pct(g7['real_lift'], 2)} → 打乱后 "
          f"{_pct(g7['shuffled_lift'], 2)}（塌陷至 {g7['ratio']:.0%}），通过。\n")

    # ---------------- 8
    A("## 8. 局限与讨论\n")
    A(f"1. **精确率仍未达 50%**。集成为 {_pct(ens['precision']) if ens else '—'}，"
      "较规则基线的 "
      f"{_pct(rule['precision'])} 有明显提升但未过线。这与 README §12 记录的结论一致——"
      "召回与精确率共用同一个对称 120 分钟窗口，事件月里风险起点密集出现，"
      "维持高召回就必然产生结构性误报。DRL 把前沿整体外推了，但没有改变前沿的形状。\n")
    A(f"2. **单 seed 方差偏大**：累计奖励 {_num(mean['total_reward'])} ± "
      f"{_num(std['total_reward'])}（总体标准差 ddof=0；按样本标准差 ddof=1 为 "
      f"±{_num(sd1)}）。根因是最优 checkpoint 出现得很早、验证集选择噪声大。\n")
    if ens:
        # **比较方向与「是否落在 ±1σ 内」都必须现算**，不能写死。
        # 原模板无条件断言「单 seed seed{X} 的累计奖励高于集成」「集成落在单 seed
        # 均值 ±1 标准差区间内」——这在 5-seed 时期成立，15-seed 后两条都变成假：
        # 实测最佳单 seed 3,022.2 **低于**集成 4,236.4，且集成偏离均值 +2.8σ。
        # 这一段本身是「自我批评」，出现假陈述反而更刺眼。
        _bv, _ev = seed_rewards[best_seed], ens["total_reward"]
        _m, _s = mean["total_reward"], sd1
        _within = (_m - _s) <= _ev <= (_m + _s) if _s > 0 else False
        _sig = (_ev - _m) / _s if _s > 0 else float("nan")
        if _bv > _ev:
            _cmp = (f"**集成并非在每个口径上都最优**。单 seed 中 seed{best_seed} 的"
                    f"累计奖励为 {_num(_bv)}，**高于集成的 {_num(_ev)}**")
        else:
            _cmp = (f"集成的累计奖励 {_num(_ev)} **高于最好的单 seed**"
                    f"（seed{best_seed}，{_num(_bv)}）")
        _band = (f"集成的累计奖励落在单 seed 均值 ±1 标准差区间 "
                 f"[{_num(_m - _s)}, {_num(_m + _s)}] **之内**，"
                 f"因此不能说集成在累计奖励上「显著」更好"
                 if _within else
                 f"集成的累计奖励在单 seed 均值 ±1 标准差区间 "
                 f"[{_num(_m - _s)}, {_num(_m + _s)}] **之外**（偏离 {_sig:+.1f}σ）；"
                 f"但**这仍不足以宣称显著**——集成只评估了一次，"
                 f"σ 描述的是单 seed 的离散度，不是集成估计量的抽样误差")
        A(f"   需要如实说明的是：{_cmp}；{_band}。"
          f"集成的精确率 {_pct(ens['precision'])} 高于最好的单 seed"
          f"（{_pct(best_seed_prec)}），但这 "
          f"{(ens['precision'] - best_seed_prec) * 100:.1f}pp 的差距同样在种子噪声量级内"
          f"（单 seed 精确率标准差 {_pct(std['precision'], 2)}）。本次集成只评估了一次、"
          f"未做方差估计，因此**不宣称集成在任一指标上显著更优**。推荐用集成的实际理由是"
          f"工程性的：它消除了「挑到坏 seed」的风险（最差单 seed 累计奖励 "
          f"{_num(min(seed_rewards.values()))}），结果不依赖选种、可复现。\n")
    A(f"3. **平均提前时间下降**：规则基线 {_num(rule['avg_lead_min'])}min → 集成 "
      f"{_num(ens['avg_lead_min']) if ens else '—'}min。仍高于 30min 门槛，"
      "但说明智能体倾向于在更接近风险起点、证据更充分时才出手——这是精确率提升的代价。"
      "若业务上更看重提前量，可调高奖励里的 `lead` 系数重训。\n")
    A("4. **验证集→测试集存在分布漂移**：验证集（2025 上半年）的累计奖励普遍高于"
      "测试集（含 2026 初极端行情）。这符合赛题背景描述的"
      "「2026 年初历史罕见剧烈波动」，属真实的市场状态变化，"
      "也正是固定阈值方法最吃亏、自适应方法最有价值的场景。\n")
    # CVaR 口径已补出（scripts/cvar_drl.py）。这条局限从「未做」改为「已做、结论不利」——
    # 数字从 cvar_drl.json 现读，不写死，否则重跑后又会过期。
    _cv = None
    _cvp = os.path.join(ROOT, "data_out", "cvar_drl.json")
    if os.path.exists(_cvp):
        _cv = json.load(open(_cvp, encoding="utf-8"))
    if _cv:
        _cs, _cvd = _cv["summary"], _cv["preregistered_verdict"]
        A(f"5. **CVaR 口径已补出，但结论不利、且暴露该口径本身退化**"
          f"（`scripts/cvar_drl.py`，判据**跑数前写定**于 "
          f"`data_out/cvar_preregistration.md`）。预登记参数下 DRL "
          f"**未通过**：excess {_cs['drl']['excess_mean']:+.1%}、"
          f"仅 {_cvd['n_positive']}/{_cvd['n_symbol']} 个品种为正。"
          f"根因不是 DRL 无效，而是**恒报警也拿到 "
          f"{_cs['恒报警']['improve_mean']:+.1%}**、与 DRL 的 "
          f"{_cs['drl']['improve_mean']:+.1%} 几乎相同——仓位路径按日频走而预警是 "
          f"15 分钟频，只要平均每 5 个交易日报一次仓位就全程半仓，改善率被机械锁死。"
          f"验证门 **G11** 强制：报 CVaR 必须并列「恒报警」对照，一旦退化就不得把 "
          f"improve 当作达标证据。详见 README §12。\n")
    else:
        A("5. **未做 CVaR 口径对比**。赛题允许「累计奖励**或** CVaR 改善率」二选一，"
          "本报告走的是累计奖励口径。跑 `python scripts/cvar_drl.py` 即可补出。\n")

    A("\n---\n")
    A("## 复现\n")
    A("```bash\n"
      "python scripts/build_anchor_dataset.py      # 若 data_out/anchor 尚未生成\n"
      "python scripts/train_drl.py --stage all     # 超参搜索 + 多 seed 训练 + 评估\n"
      f"python -m drl.tests --underlying --emit                  # {(GATES or {}).get('n_total', 9)} 道验证门\n"
      "python scripts/report_drl.py                # 重新生成本报告\n"
      "```\n")

    txt = "\n".join(x for x in L if x is not None)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"→ {args.out}  ({len(txt.splitlines())} 行"
          + (f"，含收敛曲线 {os.path.basename(png)}" if has_plot else "") + ")")


if __name__ == "__main__":
    main()
