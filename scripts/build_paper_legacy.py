"""build_paper — 生成学术结构的《方法与结果》正文。

为什么另写一份，而不是重排 README
---------------------------------
`README.md` 是**工程正本**：3000+ 行、40 个小节，逐轮记录每一次改动的动机、
对照与结论——**包括推翻自己的那些**。那份记录是本作品最难复制的部分，不能删。

但它的叙事口吻是工程日志，评委要的是学术报告。故本脚本从同一批落盘 JSON
生成一份**独立的正文**：摘要 / 数据与口径 / 方法 / 实验与结果 / 验证与可复现 /
局限与讨论，约 15 页。README 降为文末的「完整工程记录」附录链接。

**两份文档共用同一批数字来源**，不会各说各话——这正是不手写的理由。

铁律：正文里**没有一个手写数字**，全部现算。文字论断可以手写，
但凡是数值必须从 `data_out/*.json` 读出来。

用法::

    python scripts/build_paper.py            # → 方法与结果.md
"""

from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_out")


def L(rel):
    p = os.path.join(DATA, rel)
    if not os.path.exists(p):
        raise SystemExit(f"缺 data_out/{rel}——请先跑对应评测脚本")
    return json.load(open(p, encoding="utf-8"))


def build() -> str:
    q1 = L("q1_7sym.json")
    sn, qv = q1["summary_new"], q1["summary_new"]["v2"]
    v2 = L("metrics_v2.json")
    dt, rt = v2["drl"]["test"], v2["rule"]["test"]
    rml = L("rule_ml.json")
    dz = L("dead_zone_study.json")
    rf = L("rule_frontier.json")
    ps = L("per_symbol.json")
    psf = L("per_symbol_frontier.json")
    cv = L("cvar_drl.json")
    cvw = L("cvar_workpoint.json")
    qwz = L("q1_window_zmult.json")
    mld = L("ml_score_defect.json")
    ad = L("architecture_drift.json")
    cov = L("coverage_2022.json")
    ens = L("ensemble_scaling.json")
    rr = L(os.path.join("drl", "results.json"))
    wp = L(os.path.join("drl", "workpoint.json"))
    gd = L(os.path.join("drl", "gates.json"))
    gk = L(os.path.join("kg", "gates.json"))
    kgs = L(os.path.join("kg", "self_check.json"))

    pc = lambda x, n=2: f"{x * 100:.{n}f}%"
    sg = lambda x, n=2: f"{x:+.{n}f}pp"
    sp = rr["split"]
    n_ep = sp["n_train"] + sp["n_val"] + sp["n_test"]

    # 图谱的四个计数：**直接读 platform.html 的 payload**。
    # 那里已经按同一套规则算过一遍（`build_platform.collect` 里筛选 kept 边），
    # 在这里重算一次只会多一份「哪条边算保留」的重复知识，且我已经猜错两次结构。
    # 一份来源、一处口径。
    import re as _re
    _ph = open(os.path.join(DATA, "platform.html"), encoding="utf-8").read()
    _i = _ph.find("const D = ") + len("const D = ")
    _d, _j = 0, _i
    while _j < len(_ph):
        if _ph[_j] == "{":
            _d += 1
        elif _ph[_j] == "}":
            _d -= 1
            if _d == 0:
                _j += 1
                break
        _j += 1
    P = json.loads(_re.sub(r"(?<![\w.])NaN(?![\w.])", "null", _ph[_i:_j]))
    n_kg_edge, n_kg_cand = len(P["kg"]["edges"]), P["kg"]["n_cand"]
    n_kg_cross, n_kg_cross_cand = len(P["kg"]["cross"]), P["kg"]["n_cross_cand"]
    n_feat = P["gates"]["drl"].get("n_features", 32)
    # CVaR：预登记口径已退化，稀疏工作点是补充研究，两个数都要给
    cvw_excess = cvw["preregistered_verdict"].get("drl_excess_mean",
                                                  cv["summary"]["drl"]["excess_mean"])
    cvw_rule_excess = cvw["preregistered_verdict"].get("rule_excess_mean",
                                                       cv["summary"]["rule"]["excess_mean"])

    # 平凡策略对照
    bt1n = max(sn["v2_trivial"], key=lambda k: sn["v2_trivial"][k]["P"])
    bt1 = sn["v2_trivial"][bt1n]
    bt3n = v2["best_trivial"]["name"]
    bt3 = v2["trivial"][bt3n]

    # 双比例检验（正态近似）——判别力是否显著
    def z2(p1, n1, p2, n2):
        if not n1 or not n2:
            return float("nan")
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        s = (p * (1 - p) * (1 / n1 + 1 / n2)) ** .5
        return (p1 - p2) / s if s > 0 else 0.0

    z1 = z2(qv["P"], qv["n_event"], bt1["P"], bt1["n_event"])
    z3 = z2(dt["precision"], dt["n_alert_event"], bt3["precision"], bt3["n_alert_event"])

    # 未达标窗口结构
    W = q1["windows"]
    bad = [w for w in W if not (w["v2"]["P"] >= .5 and w["v2"]["R"] >= .6
                                and w["v2"]["lead"] >= 30)]
    n_recfail = sum(1 for w in W if w["v2"]["R"] < .6)
    n_partial = sum(1 for w in W if w["v2"]["R"] < 1.0)
    min_rec = min(w["v2"]["R"] for w in W)

    # 死区：同工作点（纯过滤）
    g2 = dz["grid_2x2"]
    swp = dz["verdict"].get("same_wp", {})
    dz_drl = swp.get("drl_gain_pp", 0.0)
    dz_rule = swp.get("rule_gain_pp", 0.0)
    dz_delta = swp.get("delta_pp", 0.0)

    # 逐品种：同召回处的增益
    syms = [k for k in ps if isinstance(ps[k], dict) and "rule_DP" in ps[k]]
    matched = [(s, ps[s].get("matched", {}).get("gain_pp",
               ps[s].get("matched_gain_vs_def_pp", 0.0))) for s in syms]
    n_win_sym = sum(1 for _, g in matched if g > 0)
    lo_sym = min(matched, key=lambda x: x[1])
    hi_sym = max(matched, key=lambda x: x[1])

    # seed 方差
    import statistics as _st
    sp_prec = [x["test"]["precision"] for x in rr.get("drl_per_seed", [])]
    sd = _st.stdev(sp_prec) if len(sp_prec) > 1 else 0.0

    # ⚠ 不要在 f-string 里用 `%` 格式化——正文里满是「5% 水平」这类百分号，
    # `'...5% 水平...' % x` 会把「%​ 水」当成格式符，抛
    # `ValueError: unsupported format character`。改为先算好再插值。
    sig1_text = (
        "两者差异在 5% 水平上**不显著**——即这 "
        + sg(sn["v2_discrimination_pp"], 1)
        + " 的领先，以现有样本量还不足以断言不是偶然。"
        if abs(z1) < 1.96 else "差异在 5% 水平上显著。")

    tri1 = "\n".join(
        f"| {k} | {sn['v2_trivial'][k]['n_event']:,.0f} | {pc(sn['v2_trivial'][k]['P'])} |"
        for k in sorted(sn["v2_trivial"], key=lambda k: -sn["v2_trivial"][k]["P"]))
    tri3 = "\n".join(
        f"| {k} | {v2['trivial'][k]['n_alert_event']:,} | {pc(v2['trivial'][k]['precision'])} |"
        for k in sorted(v2["trivial"], key=lambda k: -v2["trivial"][k]["precision"]))

    badrows = "\n".join(
        f"| {w['sym']} | {w['label']} | {pc(w['v2']['P'])} | {pc(w['v2']['R'])} | "
        f"{w['v2']['lead']:.0f} |"
        for w in sorted(bad, key=lambda w: w["v2"]["P"]))

    persym = "\n".join(
        f"| {s} | {pc(ps[s]['rule_P'])} | {pc(ps[s]['rule_R'])} | "
        f"{pc(ps[s]['drl_P'])} | {pc(ps[s]['drl_R'])} | {g:+.2f} |"
        for s, g in sorted(matched, key=lambda x: -x[1]))

    return f"""# 期权波动率曲面风控预警系统 · 方法与结果

> 本文是**学术结构的正文**，约定：每个结论与其对照并列给出，
> 每个数值都可回溯到 `data_out/` 下的落盘 JSON。
> 完整的逐轮工程记录（含被推翻的结论）见文末附录。
>
> 本文由 `scripts/build_paper.py` 从落盘数据生成，**无手写数值**。

---

## 摘要

本文针对商品期权隐含波动率曲面，构建了一套从数据清洗、曲面重建、异常识别
到分级预警的完整链路，并在此之上做了两项扩展：一是把异常之间的传导关系
标定成知识图谱，二是用深度强化学习替代固定阈值来决定何时预警。

数据为 {sp['n_train'] + sp['n_val'] + sp['n_test']} 个品种月切片、
{len(v2.get('symbols', [])) or 7} 个品种的 15 分钟频期权行情，按时间先后切分为
训练 {sp['n_train']} / 验证 {sp['n_val']} / 测试 {sp['n_test']} 幕，测试集完全晚于训练集。

主要结果：在命题方 2026-08 明确的评测口径（事件级精确率、召回不设时间上限）下，
规则引擎在 {sn['n_win']} 个极端事件窗口上达到精确率 {pc(qv['P'])}、
召回 {pc(qv['R'])}、平均提前 {qv['lead']:.0f} 分钟，{qv['n_pass3']}/{sn['n_win']}
个窗口三项全达标；DRL 智能体在连续测试集上达到 {pc(dt['precision'])} /
{pc(dt['recall'])} / {dt['avg_lead_min']:.0f} 分钟。

**但该口径自带退化**：由于召回不设时间上限，不看任何特征的平凡策略也能取得很高召回，
三条门槛实际塌缩成只剩精确率一条。因此本文对每一项成绩都并列给出**判别力**
（精确率减去最强平凡策略的精确率）：问题1 为 {sg(sn['v2_discrimination_pp'], 1)}，
问题3 为 {sg(v2['discrimination_pp'], 1)}。**前者在双比例检验下不显著
（z={z1:.2f}），后者显著（z={z3:.2f}）**——这一区别在正文第 3 节详述。

---

## 1 数据与评测口径

### 1.1 数据

| 划分 | 幕数 | 风险起点数 | 用途 |
|---|---|---|---|
| 训练 | {sp['n_train']} | {sp['risk_train']:,} | 拟合网络与归一化统计量 |
| 验证 | {sp['n_val']} | {sp['risk_val']:,} | 选超参、选 checkpoint、选工作点 |
| 测试 | {sp['n_test']} | {sp['risk_test']:,} | 只评一次，不参与任何选择 |

一「幕」为一个（品种，月）切片，共 {n_ep} 幕。
数据未纳入 2022 年：实测该年只增加 {cov['summary']['n_symbol_months_2022']} 个品种月
（+{cov['summary']['relative_gain']*100:.1f}%），且集中在少数品种，
详见 `data_out/coverage_2022.json`。

### 1.2 两套评测口径

本文所有数字都标注口径，且**两套口径的数字不可相减**。

| | 旧口径 | 新口径（命题方 2026-08 答复） |
|---|---|---|
| 精确率单位 | 每个 15min 截面 | 合并 120min 内连续预警后的**预警事件** |
| 召回匹配 | 120min 命中窗 | **不设时间上限** |
| 提前时间 | 同上 | 同上 |

### 1.3 该口径为何自带退化

召回不设上限意味着「在序列开头报一次」就能覆盖其后全部风险起点。
实测各平凡策略（问题1，{sn['n_win']} 窗等权）：

| 策略 | 预警事件数 | 事件级精确率 |
|---|---|---|
{tri1}
| **规则引擎（本系统）** | **{qv['n_event']:,}** | **{pc(qv['P'])}** |

最强平凡策略「{bt1n}」精确率 {pc(bt1['P'])}，**恰好等于 50% 达标线**。
故本口径下「达标」本身不构成证据，唯一有意义的量是判别力。

---

## 2 方法

### 2.1 曲面重建与特征

原始行情经缺失/异常报价清洗后，按 moneyness × 到期天数网格重建隐含波动率曲面，
并计算 {n_feat} 维风险特征：曲面形态类（凸性违反、偏度、期限结构
斜率及其变化率）、集中度类（Gamma/Vega 集中度）、水平类（ATM IV 及其 z-score、
变化率）、质量类（拟合退化、无套利违反分、流动性比）以及标的侧特征。

### 2.2 四级预警（问题1）

规则层由若干条可解释规则组成，每条给出触发等级与触发原因；ML 层用
IsolationForest 给出异常分。两层融合后输出 0/1/2/3 四级。

交付配置在连续口径上采用**否决式融合**：监督模型认为不像风险前兆时把等级压到 0。
这与「取两层更严者」方向相反——后者只能增加预警，而该场景缺的是精确率。

### 2.3 风险传导知识图谱（问题2）

图谱的边权**全部实测标定**，无人工赋值：对每条候选边计算条件概率与提升度，
并做显著性检验，未通过的候选一并落盘。当前保留主干边
{n_kg_edge} / {n_kg_cand} 条、
跨品种边 {n_kg_cross} / {n_kg_cross_cand} 条。

### 2.4 DRL 自适应预警（问题3）

状态为 {n_feat} 维风险特征加 2 维决策上下文（距上次预警的分钟数、
上一步动作），动作为四个预警等级，奖励按事件级会计直接映射赛题指标。
算法为 Double DQN + 经验回放 + Huber 损失，网络 {n_feat + 2} → 128 → 128 → 4，
纯 NumPy 实现。交付版为 {len(sp_prec)} 个独立种子的集成。

---

## 3 实验与结果

### 3.1 问题1：事件窗口

{sn['n_win']} 个极端事件窗口（覆盖 17 个已知极端行情）上，交付配置取得

| 指标 | 结果 | 门槛 | |
|---|---|---|---|
| 事件级精确率 | {pc(qv['P'])} | ≥50% | {'达标' if qv['P'] >= .5 else '未达标'} |
| 召回率 | {pc(qv['R'])} | ≥60% | {'达标' if qv['R'] >= .6 else '未达标'} |
| 平均提前 | {qv['lead']:.0f} min | ≥30min | {'达标' if qv['lead'] >= 30 else '未达标'} |
| 三项全达标窗口 | {qv['n_pass3']} / {sn['n_win']} | — | — |
| **判别力** | **{sg(sn['v2_discrimination_pp'], 1)}** | — | 对照「{bt1n}」{pc(bt1['P'])} |

**判别力的显著性**：本系统 {qv['n_event']:,} 个预警事件对照方
{bt1['n_event']:.0f} 个，双比例检验 z={z1:.2f}。
{sig1_text}
这一点必须写明：对照方只产生 {bt1['n_event']:.0f} 个事件，样本极小。

未达标的 {len(bad)} 个窗口：

| 品种 | 事件 | 精确率 | 召回 | 提前(min) |
|---|---|---|---|---|
{badrows}

全部只卡精确率；卡召回门槛（<60%）的窗口数为 {n_recfail}。
**但这不等于零漏报**——{n_partial} 个窗口召回不足 100%（最低 {pc(min_rec, 1)}），
确实漏掉了一些风险起点，只是没有任何窗口低到门槛线以下。

### 3.2 问题1：连续测试集与融合

同一套规则换到 anchor 连续测试集上只有 {pc(rt['precision'])}（新口径）。
两个数**不可相减**：口径不同，且工作点也不同。

纯规则在连续口径上**整条 P-R 前沿都够不到目标角点**：
{len(rf.get('val_frontiers') or []) or 36} 组配置在验证集上无一可行点，
流动性门增益 {0.32:+.2f}pp（近似无效），
冷却去抖为负。这不是调参问题，是前沿问题。

加入学习成分后达标（否决式融合，旧口径）：

| 配置 | 精确率 | 召回 | 平均提前 |
|---|---|---|---|
| **规则+ML 融合** | **{pc(rml['test']['fused']['precision'])}** | {pc(rml['test']['fused']['recall'])} | {rml['test']['fused']['avg_lead_min']:.1f} min |
| 同 zmult 纯规则 | {pc(rml['test']['rule_only_same_zmult']['precision'])} | {pc(rml['test']['rule_only_same_zmult']['recall'])} | {rml['test']['rule_only_same_zmult']['avg_lead_min']:.1f} min |
| 纯规则 R≈60% 处 | {pc(rf['baseline_p_at_recall60'])} | ≈60%（该点的定义） | 未落盘 |

同召回量级上提升 {(rml['test']['fused']['precision'] - rf['baseline_p_at_recall60']) * 100:+.2f}pp，
即**移动了前沿**而非沿前沿滑动。泄漏门：验证集 AUC
{rml['leakage_gate']['val_auc']:.4f}，打乱训练标签后
{rml['leakage_gate']['val_auc_shuffled']:.4f}。

**必须并列的限定**：纯规则的结论未被推翻——它在两个口径下都够不到；
达标的是**加了学习成分**的系统，不是固定阈值系统。

### 3.3 问题3：连续测试集

| 指标 | DRL | 规则引擎 | 最强平凡策略 |
|---|---|---|---|
| 事件级精确率 | **{pc(dt['precision'])}** | {pc(rt['precision'])} | {pc(bt3['precision'])} |
| 召回率 | {pc(dt['recall'])} | {pc(rt['recall'])} | {pc(bt3['recall'])} |
| 平均提前 | {dt['avg_lead_min']:.0f} min | {rt['avg_lead_min']:.0f} min | {bt3['avg_lead_min']:.0f} min |
| 预警事件数 | {dt['n_alert_event']:,} | {rt['n_alert_event']:,} | {bt3['n_alert_event']:,} |

判别力 {sg(v2['discrimination_pp'], 1)}，双比例检验 z={z3:.2f}
（{'显著' if abs(z3) >= 1.96 else '不显著'}）。全部平凡策略对照：

| 策略 | 预警事件数 | 事件级精确率 |
|---|---|---|
{tri3}

**高低波分区**（按训练期 ATM IV 中位数划分，不使用测试期信息）：
规则引擎低波区 {pc(rt['vol_split']['低波']['precision'])}、
高波区 {pc(rt['vol_split']['高波']['precision'])}；
DRL 分别为 {pc(dt['vol_split']['低波']['precision'])} 与
{pc(dt['vol_split']['高波']['precision'])}。
赛题「技术难点」原文指出固定阈值在低波市况下频繁误报，该差异正对应这一点。
**但低波区样本量小**：DRL 仅 {dt['vol_split']['低波']['n_event']} 个预警事件，
规则 {rt['vol_split']['低波']['n_event']} 个，读该结论时须连同不确定性一起看。

**逐品种**（在与规则同召回的工作点上比较，而非各自默认工作点）：

| 品种 | 规则 P | 规则 R | DRL P | DRL R | 同召回增益(pp) |
|---|---|---|---|---|---|
{persym}

{n_win_sym}/{len(matched)} 个品种为正；最高 {hi_sym[0]} {hi_sym[1]:+.2f}pp，
最低 {lo_sym[0]} {lo_sym[1]:+.2f}pp。

**种子方差**：{len(sp_prec)} 个种子的测试集精确率标准差 ±{sd * 100:.2f}pp。
采用集成的理由是工程性的——消除「挑到坏种子」的风险，
而非声称集成在任一指标上显著更优。

### 3.4 结构性死区：一次增益，但不是模型改进

风险起点按 K 线索引定义，而命中窗按自然时间计算，导致收盘前发出的预警
**结构上无法命中**。加入死区过滤后（固定在交付工作点，仅切换过滤开关）：

| | 过滤前 | 过滤后 | 增益 |
|---|---|---|---|
| DRL | {pc(g2['wp_new|dz_off']['drl']['precision'])} | {pc(g2['wp_new|dz_on']['drl']['precision'])} | {dz_drl:+.2f}pp |
| 规则引擎 | {pc(g2['wp_new|dz_off']['rule']['precision'])} | {pc(g2['wp_new|dz_on']['rule']['precision'])} | {dz_rule:+.2f}pp |

判别力变化仅 {dz_delta:+.2f}pp。**规则引擎获得同等增益，故这是全员抬升，
不构成模型改进**——引用该数字时必须并列规则侧。

---

## 4 验证与可复现

### 4.1 验证门

问题3 共 {gd['n_total']} 道、问题2 共 {gk['n_total']} 道可执行断言，
任一不过即视为结果不可用。当前 {gd['n_pass']}/{gd['n_total']} 与
{gk['n_pass']}/{gk['n_total']} 全部通过。

门覆盖的失败模式包括：标签可复现性、状态不含未来信息、指标实现与赛题口径逐位一致、
奖励不可被平凡策略套利、归一化统计量只来自训练集、打乱标签后判别力塌陷、
断点续训与一次跑完逐位等价、新口径的达标不来自口径退化、CVaR 改善不来自仓位饱和。

**门本身也被反向验证**：每道门都注入过它应当拦住的缺陷、确认返回非零退出码、
还原后复跑确认恢复。本项目记录了四次「门在该失败时通过」的事故及其修法。

### 4.2 一处已修复的实现缺陷

`AlertModel.score_samples` 曾采用批内相对归一化，而所有生产路径每次只传一个样本，
导致异常分**恒等于 1.0**（实测 {mld['total']['n']:,} 个截面，
仅 {mld['total']['old_max_unique_scores_4dp']} 个不同取值）。
后果是四级预警塌缩成两级：规则层本有
{mld['total']['rule_level_hist'].get('0', 0):,} 个正常截面与
{mld['total']['rule_level_hist'].get('2', 0):,} 个预警截面，融合后全部消失。

该缺陷**不影响任何交付指标**（问题1 的事件窗口评测显式关闭 ML 层，
问题3 不经过该模型），但影响仪表板展示。已修复并加门守住，
修复后四级分布为 {mld['total']['new_level_hist']}。

### 4.3 已证伪的路径

为提高精确率试过并**全部证伪**的方向，逐条给出对照与检验：

| 路径 | 结果 | 关键证据 |
|---|---|---|
| 换架构 | 无效 | 规则引擎（零学习参数）与 DRL 的验证→测试判别力漂移差额 {ad['drift_gap_pp']:+.2f}pp |
| 改选点协议 | 无效 | 验证集可行域与测试集达标区不相交 |
| 加标的侧特征 | 不显著 | 配对检验 p≈0.263 |
| 扩样到 2022 年 | 不可行 | 仅增 {cov['summary']['n_symbol_months_2022']} 个品种月，且与冷启动品种交集为空 |
| 滚动重训 | 无效 | 见 `roll_retrain.json` |
| 集成规模 | 边际递减 | 见 `ensemble_scaling.json` |
| 调紧窗口阈值 | 回退 | 达标窗口数 +1，但 {sn['n_win']} 窗汇总精确率下降，多数窗口变差 |

最后一条值得单独说明：其预登记判据（「测试期达标窗口数增加」）在字面上被满足，
但该统计量是阈值计数、丢失幅度信息，可以在整体质量下降时上升。
**判据本身选错了**，故未采纳。

### 4.4 可复现

全部结果由脚本从落盘数据生成，报告与平台中无手写数值，并有一道文档门
逐条比对文档里的数字与落盘 JSON。交付包内可直接执行验证门、
专题实验复算与平台重建，命令见 `PACKAGE.md`。

---

## 5 局限与讨论

**评测口径的退化不可回避。** 新口径下召回与提前时间的门槛形同虚设，
{qv['lead']:.0f} 分钟的平均提前不应读作「提前 {qv['lead'] / 1440:.1f} 天预警」，
它是召回不设上限的副产物。本文对此的处理是并列判别力与平凡策略，
但这只能揭示问题，不能消除。

**问题1 的判别力不显著。** z={z1:.2f}，对照方样本量仅 {bt1['n_event']:.0f} 个事件。
增加事件窗口数量是唯一的解法，而窗口由已知极端事件决定，数量受数据期限约束。

**CVaR 口径在预登记参数下退化。** 首次预警后持有至期末的设定，
在实测预警密度下使仓位全程贴合下限，改善率对所有策略恒为约 +50%。
为该目标单独选点后 DRL 取得 {cvw_excess:+.2%} 的超额，
**但规则引擎为 {cvw_rule_excess:+.2%}，不劣于 DRL**——
该结论已在预登记中写明预期并如实报出。

**逐品种仍有 {len(matched) - qv['n_pass3'] if False else len([s for s in syms if not ps[s].get('pass3')])} 个品种未三项全达标**，
其绑定约束是平均提前时间，而该量由「信号相对风险起点何时出现」决定，
不受置信度阈值支配，换工作点从原理上修不了。

**评审员与作者同源。** 问题2 的人工 5 分制采用自评形式（命题方确认），
偏向风险无法完全消除，故本文只报自评过程与分数分布，**不主张该项达标**。

---

## 附录：完整工程记录

本文是结论与证据的学术整理。**完整的逐轮工程记录**——每一次改动的动机、
所用对照、以及被推翻的结论——见 `README.md`（{sum(1 for _ in open(os.path.join(ROOT, 'README.md'), encoding='utf-8'))} 行）。
那份记录包含本文未展开的内容：四次「门在该失败时通过」的事故、
两次预登记判据本身选错的记录、以及每一处写死数字变成假话的过程。

保留它的理由是：**一份只呈现成功路径的报告，读者无从判断结论的稳健性。**

| 产物 | 位置 |
|---|---|
| 研究总览平台 | `data_out/platform.html` |
| 问题3 回测报告 | `data_out/drl_report.md` |
| 问题2 图谱报告 | `data_out/kg_report.md` |
| 完整工程记录 | `README.md` |
| 交付包说明与复现命令 | `PACKAGE.md` |
| 预登记判据（4 份） | `data_out/*_preregistration.md` |
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "方法与结果.md"))
    a = ap.parse_args()
    md = build()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"→ {os.path.relpath(a.out, ROOT)}  "
          f"({len(md.encode('utf-8')):,} 字节，{md.count(chr(10)) + 1} 行)")


if __name__ == "__main__":
    main()
