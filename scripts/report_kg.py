"""report_kg — 生成问题2 的技术报告、解释样本与人工评分材料。

用法::

    python scripts/report_kg.py                  # 生成全部产物
    python scripts/report_kg.py --n-samples 30   # 抽样条数

产物::

    data_out/kg_report.md                 技术报告（数字全部来自 kg_edges.json / gates.json）
    data_out/kg/explanation_samples.md    随机抽样的解释文本，供人工评分
    data_out/kg/scoring_sheet.csv         空白评分表（三维度 × 5 分制）
    data_out/kg/self_check.json           机器自查结果（**不是**人工评分）

关于赛题指标3（解释能力人工评分 ≥4 分）
--------------------------------------
该指标要求**人工**按 5 分制评分。本脚本只负责：
  (1) 把评分细则与锚点写清楚；
  (2) 随机抽样导出待评样本；
  (3) 生成空白评分表。
**脚本不会给出「已达标」的结论**——自评分不能替代人工评分。报告中该项一律标注
「待人工评审」，并附上可机器检查的客观项（无交易建议、证据可回溯）作为下限保证。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.dataset import FEATURES, load_episodes, split_episodes
from kg.explain import DISCLAIMER, METHOD_NOTE, explain
from kg.graph import DiGraph
from kg.reason import reason
from kg.schema import ANOM_BY_ID, ANOMS, CONSEQ_BY_ID, CONSEQS, HORIZON, MECH_MAP
from kg.tests import ADVICE_WORDS, TRADE_ACTIONS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_DIR = os.path.join(ROOT, "data_out", "kg")
REPORT = os.path.join(ROOT, "data_out", "kg_report.md")

RUBRIC = [
    ("语义清晰", "文本是否易读、术语是否准确、数字是否带明确口径",
     {5: "完全清楚，非本领域同事也能读懂结论与依据",
      4: "清楚，个别术语需要背景知识",
      3: "基本可读，但有歧义或冗余",
      2: "需要反复阅读才能理解",
      1: "难以理解"}),
    ("因果逻辑合理", "根因→传导→后果的链条是否成立，统计与机制是否分得清",
     {5: "链条合理，且明确区分了数据结论与业务先验",
      4: "链条合理，个别环节略牵强",
      3: "方向大致对，但缺乏支撑或有跳跃",
      2: "存在明显逻辑问题",
      1: "因果关系错误或臆造"}),
    ("不含交易建议", "是否只提示风险与分析，不出现任何操作指令",
     {5: "完全没有操作暗示，且有免责声明",
      4: "无操作指令，个别措辞略有倾向",
      3: "有轻微暗示性表述",
      2: "出现类似操作倾向的表述",
      1: "包含明确交易建议"}),
]


def _pct(x):
    return "—" if x is None or not np.isfinite(x) else f"{x * 100:.1f}%"


def _num(x, d=2):
    return "—" if x is None or not np.isfinite(x) else f"{x:,.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default=KG_DIR)
    ap.add_argument("--out", default=REPORT)
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    def _load(name):
        p = os.path.join(args.kg_dir, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

    E = _load("kg_edges.json")
    X = _load("kg_cross.json")
    G = _load("gates.json")
    SESS = _load("kg_session.json")      # 交易时段画像，支撑第 5 节的日内季节性论断
    if E is None:
        raise SystemExit("缺少 kg_edges.json，请先跑 python scripts/build_kg.py")
    g = DiGraph.load(os.path.join(args.kg_dir, "kg_graph.json"))
    meta = E["meta"]

    # ---------------------------------------------------------------- 抽样
    eps = load_episodes(os.path.join(ROOT, "data_out", "anchor"), verify=True)
    tr, va, te = split_episodes(eps)
    rng = np.random.default_rng(args.seed)
    samples = []
    tries = 0
    while len(samples) < args.n_samples and tries < args.n_samples * 60:
        tries += 1
        e = te[int(rng.integers(0, len(te)))]
        i = int(rng.integers(0, len(e)))
        feats = {k: float(e.X[i, j]) for j, k in enumerate(FEATURES)}
        res = reason(g, feats, symbol=e.symbol, timestamp=e.ts[i])
        if res["n_active"] == 0:       # 只抽有异常的截面供评分
            continue
        samples.append({"symbol": e.symbol, "ym": e.ym, "ts": e.ts[i],
                        "res": res, "exp": explain(res)})

    # ---------------------------------------------------------------- 自查
    def _selfcheck(s):
        body = s["exp"]["text"].replace(DISCLAIMER, "")
        return {
            "has_disclaimer": DISCLAIMER in s["exp"]["text"],
            "no_advice_words": not any(w in body for w in ADVICE_WORDS + TRADE_ACTIONS),
            "n_sections": len(s["exp"]["sections"]),
            "n_evidence_refs": len(s["exp"]["evidence_refs"]),
            "chars": len(s["exp"]["text"]),
            "has_cause": "**根因" in s["exp"]["text"],
            "has_path": "**传导" in s["exp"]["text"],
            "labels_prior": "业务先验" in s["exp"]["text"],
        }
    checks = [_selfcheck(s) for s in samples]
    self_check = {
        "n_samples": len(samples),
        "note": "以下为机器可检查的客观项，**不是**赛题要求的人工 5 分制评分",
        "all_have_disclaimer": all(c["has_disclaimer"] for c in checks),
        "all_no_advice": all(c["no_advice_words"] for c in checks),
        "all_have_cause_and_path": all(c["has_cause"] and c["has_path"] for c in checks),
        "pct_labels_prior": float(np.mean([c["labels_prior"] for c in checks])),
        "median_chars": float(np.median([c["chars"] for c in checks])),
        "median_evidence_refs": float(np.median([c["n_evidence_refs"] for c in checks])),
    }
    with open(os.path.join(args.kg_dir, "self_check.json"), "w", encoding="utf-8") as f:
        json.dump(self_check, f, ensure_ascii=False, indent=2, default=float)

    # ---------------------------------------------------------------- 样本文件
    #
    # 文档层的无损去重（第 10 轮）
    # ----------------------------
    # 实测：30 条样本的 121 条传导语句只来自 **18 条边**，逐字不同的仅 19 种，
    # 重复率 84.3%；**贪心集覆盖显示 3 条样本就能覆盖全部 18 条边**，
    # 其余 27 条一条新边都不带。第 8 位评审把「81% 逐字重复」列为头号缺陷。
    #
    # 但重复的根源不在产品，而在**这份文档的抽样方式**：传导语句里的统计量
    # （条件概率/基础率/提升度/下界/置换检验/测试期复现）全部是**边级**的，
    # 与所处截面无关，因此同一条边在 30 条样本里必然渲染得逐字相同。
    # 真实用户一次只看一条预警，这种重复对他根本不存在。
    #
    # 所以去重放在**文档层**而不是 `kg/explain.py`：单条预警必须保持自足完整，
    # 否则会为了这份评分材料的观感去损害真正的产品。
    # 做法是无损的——第一次出现印全文并编号 [T*]，之后引用编号；
    # 因为字符串逐字相同，引用与全文承载的信息完全等价，读者可自行核对。
    # 逐截面变化的内容（触发依据、触发量读数、方向提示、跨品种）一律**不**去重。
    seen_seg: dict = {}          # 传导语句原文 -> 编号
    seg_home: dict = {}          # 编号 -> 首次出现的样本号
    # 去重前/后的传导正文字数，**在去重发生的那一刻如实累计**。
    # 早前想事后从成品文档反推「去重前」，但编号标记 [T*] 的增删无法精确还原，
    # 反推值 54,317 与直接测得的 53,281 差了 1,036 字——**近似的审计数字等于没有**。
    dedup_stat = {"before": 0, "after": 0, "n_full": 0, "n_ref": 0}

    def _dedup(text: str, k: int) -> str:
        out = []
        for ln in text.split("\n"):
            s = ln.strip()
            if not s.startswith("**传导｜") or "：" not in s:
                out.append(ln)
                continue
            dedup_stat["before"] += len(s)
            if s in seen_seg:
                dedup_stat["n_ref"] += 1
                tag = seen_seg[s]
                head = s.split("：", 1)[0]
                # 早前每行引用都跟一句 53 字的说明「——该条统计为边级，与所处截面
                # 无关…」，重复 102 遍约 5,400 字，占评分视图正文 10%。
                # **去重机制自己长出了一层新的、可去重的样板**（第 9 位评审指出）。
                # 该说明对全体引用一致成立，故提到文首说一次，此处只留指针。
                out.append(f"{head}：同 **[{tag}]**（首见于样本 {seg_home[tag]}，"
                           f"逐字相同；见文首去重说明）。")
            else:
                dedup_stat["n_full"] += 1
                tag = f"T{len(seen_seg) + 1}"
                seen_seg[s] = tag
                seg_home[tag] = k
                head, body = s.split("：", 1)
                out.append(f"{head}**[{tag}]**：{body}")
        for ln in out:
            if ln.strip().startswith("**传导｜"):
                dedup_stat["after"] += len(ln.strip())
        return "\n".join(out)

    lines = ["# 风险传导解释样本（供人工评分）\n",
             f"> 从**测试期**（2025-07..2026-04）随机抽取 {len(samples)} 条含异常的截面，"
             f"seed={args.seed}，由 `scripts/report_kg.py` 生成。\n",
             "> 评分细则见 `data_out/kg_report.md` 第 6 节；"
             "空白评分表见 `scoring_sheet.csv`。\n",
             "> **关于本文档的去重**：传导语句里的统计量是**边级**的"
             "（条件概率、基础率、提升度、下界、置换检验、测试期复现均与所处截面无关），"
             "因此同一条边在不同样本里渲染得**逐字相同**。本文档对完全相同的传导语句"
             "只印一次并编号 `[T*]`，其余位置引用该编号。**这是无损的**——"
             "被替换的字符串与首次出现处逐字一致，可自行核对。"
             "逐截面变化的内容（触发依据、触发量读数、方向提示、跨品种共现）**一律不去重**，"
             "**故每条样本的实例特有信息全部保留在其上方的触发依据与读数中**。\n",
             "> 需要说明的是，**系统本身不做 `[T*]` 去重**：真实场景下用户一次只看一条预警，"
             "每条预警的传导语句都是完整的（见 `kg/explain.py`，产品原文另存于 "
             "`explanation_samples_full.md`）。此处的 `[T*]` 去重只针对"
             "「把 30 条预警并排放进同一份文档」这一评分场景。\n",
             "> 但**有一层因子化是产品级的、须如实说明**：`[†]` / `[‡]` 标记的统计口径"
             "由**容器**统一给出一次（见下），单条预警正文只留回指标记，不逐条重复"
             "（该做法把正文里重复 119/121 遍、占 18.5% 篇幅的方法学注释消掉了）。"
             "因此**单条预警并非在任何容器之外都自足**——凡展示本系统解释文本的界面，"
             "都必须同时渲染这段口径。平台与图谱可视化均已照此处理，"
             "并由 `scripts/check_doc_numbers.py` 断言 `[†]` 不得悬空。\n",
             METHOD_NOTE + "\n", "---\n"]
    for k, s in enumerate(samples, 1):
        lines.append(f"## 样本 {k}｜{s['symbol']} {s['ts']}\n")
        lines.append(_dedup(s["exp"]["text"], k) + "\n")
        lines.append("---\n")
    with open(os.path.join(args.kg_dir, "explanation_samples.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines))

    # 未去重的**产品原文**。评分文档里的 [T*] 引用是为「30 条预警并排评分」这一
    # 场景做的**视图**，下游消费者（平台、可视化）必须拿原文，否则用户会看到
    # 「同 [T3]（首见于样本 1）」——而产品里根本没有「样本」这个概念。
    # 这个坑是实测踩到的：文档层去重上线后，platform.html 里立刻出现 36 处引用占位。
    full = ["# 风险传导解释样本（产品原文，未去重）\n",
            "> 与 `explanation_samples.md` 同源，但**不做 [T*] 去重**——"
            "后者是为人工评分场景生成的视图。平台与可视化一律读本文件。\n",
            METHOD_NOTE + "\n", "---\n"]
    for k, s_ in enumerate(samples, 1):
        full.append(f"## 样本 {k}｜{s_['symbol']} {s_['ts']}\n")
        full.append(s_["exp"]["text"] + "\n")
        full.append("---\n")
    with open(os.path.join(args.kg_dir, "explanation_samples_full.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(full))

    # 去重统计落盘，供 `scripts/check_doc_numbers.py` 核对 README 里的数字
    self_check["doc_dedup"] = dedup_stat
    with open(os.path.join(args.kg_dir, "self_check.json"), "w", encoding="utf-8") as f:
        json.dump(self_check, f, ensure_ascii=False, indent=2, default=float)

    with open(os.path.join(args.kg_dir, "scoring_sheet.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["样本编号", "品种", "时间戳", "语义清晰(1-5)",
                    "因果逻辑合理(1-5)", "不含交易建议(1-5)", "评分人", "备注"])
        for k, s in enumerate(samples, 1):
            w.writerow([k, s["symbol"], s["ts"], "", "", "", "", ""])

    # ---------------------------------------------------------------- 报告
    edges = E["edges"]
    strong = [e for e in edges if e["strong"]]
    rep = [e for e in strong if e.get("replicated")]
    selfp = [e for e in strong if e["self_feature"]]
    cross = (X or {}).get("cross_symbol", [])
    cs_strong = [c for c in cross if c["strong"]]

    L, A = [], None
    A = L.append
    A("# 问题2：基于知识图谱的风险传导路径推理与可解释性 —— 技术报告\n")
    A("> 由 `scripts/report_kg.py` 生成。全部统计数字来自 `data_out/kg/` 下的")
    A("> `kg_edges.json` / `kg_cross.json` / `gates.json`，重跑 `scripts/build_kg.py` 后")
    A("> 重新执行本脚本即可刷新。\n")

    A("## 1. 这版要解决的核心问题\n")
    A("知识图谱最容易滑向的失败模式，是把作者拍脑袋写的因果关系包装成「推理」。"
      "本项目的 v1 骨架（已归档到 `kg/_legacy/`）正落在这里：主推理骨架 AMPLIFIES 的"
      "边权是「启发式 0.6–0.8」的先验，跨品种回退权重直接写死 `1.0`（等于断言"
      "「同板块必然传导」），其 schema 文档自己也注明「后续可用 anchor 数据估计」——"
      "而这一步一直没做。\n")
    A("因此 v2 的设计原则是**把实测层与诠释层分开，并在图里显式标注**：\n")
    A("| 层 | 内容 | 权重来源 | 是否参与推理打分 |")
    A("|----|------|---------|----------------|")
    A("| measured 实测层 | `ANOM→CONSEQ` 主干边、`XSYM→XSYM` 跨品种边 | "
      f"{meta.get('n_train_slices', 0):,} 个**训练期**真实截面实测的提升度 lift | 是 |")
    A("| interpretive 诠释层 | `MECH` 机制节点、`CONSTRAINT` 无套利约束、边上的 `how` 文本 | "
      "业务先验，**无权重** | 否 |")
    A("\n评审因此可以清楚区分：哪些是数据说的，哪些是人说的。\n")

    A("## 2. 图谱规模\n")
    from collections import Counter
    nt = Counter(g.node(n).get("type") for n in g.nodes()
                 if g.node(n).get("type") != "META")
    et = Counter(a.get("kind") for _, _, a in g.edges())
    A(f"- 节点 {len(g) - 1} 个：{dict(nt)}")
    A(f"- 边 {len(g.edges())} 条：{dict(et)}")
    A(f"- 主干边候选 {meta['n_candidate']} 条 → **通过检验 {len(strong)} 条**"
      f"（{len(strong) / meta['n_candidate']:.0%}），其中测试期可复现 {len(rep)} 条")
    A(f"- 跨品种候选 {len(cross)} 条 → 通过 {len(cs_strong)} 条")
    n_mech = sum(1 for e in strong if MECH_MAP.get((e["anom"], e["conseq"])))
    A(f"- 诠释层覆盖率：{len(strong)} 条实测主干边中只有 **{n_mech} 条**配有机制说明，"
      f"其余 {len(strong) - n_mech} 条是「数据观测到、但尚未归纳出机制」。"
      f"这类边在解释文本中会明说这一点，不硬凑机制。\n")

    A("## 3. 边是怎么测出来的\n")
    A(f"对每个「异常 A → 后果 C」组合，在训练期（2023-01..2024-12，"
      f"{meta['n_train_episodes']} 幕 / {meta['n_train_slices']:,} 截面）上统计：\n")
    A("- **提升度** `lift = P(C 在未来 8 个截面内发生 | A 当前发生) / P(C)`，"
      "lift≈1 即这条边没有信息量；")
    A("- **95% Wilson 置信下界**：用保守下界而非点估计作为边权，避免小样本虚高；")
    A(f"- **分块置换检验**（{meta['n_perm']} 次）：金融时序自相关强，朴素检验会把 p 值"
      "压到几乎条条边都「显著」。这里按**整日为块做循环平移**，只打乱 A 与 C 的时间对齐、"
      "保留 A 自身的自相关结构；")
    A("- **测试期独立复算**（2025-07..2026-04）：只在训练期显著、测试期塌掉的边会被标出。\n")
    c = meta["criteria"]
    A(f"保留标准（三条同时满足）：支撑度 ≥ {c['min_support']}、"
      f"置换检验 p ≤ {c['max_p']}、lift 置信下界 > {c['min_lift_ci_low']}。"
      f"未通过的候选边**不进入推理图**，但全部保留在 `kg_edges.json` 中供审计。\n")

    A("### 通过检验的主干边\n")
    A("| 根因 | 后果 | lift | 95%下界 | P(C\\|A) | 基础率 | 支撑度 | p 值 | 测试期复现 | 备注 |")
    A("|------|------|------|--------|--------|--------|--------|------|-----------|------|")
    for e in sorted(strong, key=lambda x: -x["lift"]):
        an = ANOM_BY_ID[e["anom"]].name
        cn = CONSEQ_BY_ID[e["conseq"]].name
        note = "同源特征" if e["self_feature"] else ""
        if not MECH_MAP.get((e["anom"], e["conseq"])):
            note = (note + "，无机制说明").lstrip("，")
        A(f"| {an} | {cn} | {_num(e['lift'])} | {_num(e['lift_ci_low'])} | "
          f"{_pct(e['p_cond'])} | {_pct(e['p_conseq'])} | {e['n_anom']:,} | "
          f"{e['p_value']:.4f} | {'✓' if e.get('replicated') else '✗'} | {note or '—'} |")
    A("")
    A(f"> **「同源特征」提示**：{len(selfp)} 条边的根因与后果判定共用同一底层指标"
      "（例如「曲面凸性违反」与「曲面凸性破坏」都基于 `convexity_violation`）。"
      "这类边的高 lift 有相当部分来自同一指标的时间自相关，**不是独立证据**。"
      "图中已打 `self_feature` 标记，解释文本也会逐条提示，不让读者误读。\n")

    A("## 4. 被数据否掉的先验\n")
    A("这部分比通过的边更值得看——它们是「如果只靠人写，很可能会被写进图里」的关系。\n")
    A("| 先验关系 | 机制说法 | 实测 lift | p 值 | 结论 |")
    A("|---------|---------|----------|------|------|")
    rejected = [e for e in edges if not e["strong"] and MECH_MAP.get((e["anom"], e["conseq"]))]
    for e in sorted(rejected, key=lambda x: x["lift"])[:8]:
        an = ANOM_BY_ID[e["anom"]].name
        cn = CONSEQ_BY_ID[e["conseq"]].name
        mech = MECH_MAP.get((e["anom"], e["conseq"]), "")
        from kg.schema import MECH_BY_ID
        mn = MECH_BY_ID[mech].name if mech in MECH_BY_ID else "—"
        A(f"| {an} → {cn} | {mn} | {_num(e['lift'])} | {e['p_value']:.4f} | 未通过，已剔除 |")
    A("")
    r4 = next((e for e in edges if e["anom"] == "ANOM-R4_conc"
               and e["conseq"] == "CONSEQ-VolSpike"), None)
    if r4:
        A(f"> 其中最值得一提的是「**Gamma/Vega 截面集中 → 波动率水平跳升**」——"
          f"即教科书式的「做市商 Gamma 对冲压力推高波动」。这是几乎所有期权风险图谱都会"
          f"写进去的一条边，但在本数据集上实测 lift 仅 {_num(r4['lift'])}"
          f"（p={r4['p_value']:.4f}），**不足以支持**，已从推理图中剔除。\n")

    A("## 5. 跨品种：一个差点写进报告的假结论\n")
    r6 = [c for c in cross if c["rule"] == "R6_liquidity"]
    if r6:
        naive = max((c["lift"] for c in r6), default=float("nan"))
        A("用全时段平均作基线时，「流动性退化」在 ag/au/sc 之间的跨品种提升度高达 "
          f"{_num(naive)} 倍，看起来是一条极强的传导链。但进一步核查发现：\n")
        if SESS and SESS.get("by_symbol"):
            bys = SESS["by_symbol"]
            night = {k: v["night_share"] for k, v in bys.items()}
            hi = [k for k, v in night.items() if v > 0.5]
            lo = [k for k, v in night.items() if v <= 0.5]
            if hi:
                A(f"- {'/'.join(sorted(hi))} 的流动性异常有 "
                  f"**{min(night[k] for k in hi) * 100:.0f}% 以上集中在夜盘时段**；")
            for k in sorted(lo):
                tops = "、".join(f"{h} 点（{v:.0%}）" for h, v in bys[k]["top_hours"][:3])
                A(f"- {k} 不交易夜盘（夜盘占比 {night[k]:.1%}），其异常集中在 {tops}；")
            jac = SESS.get("jaccard", {})
            # **按「是否交易夜盘」分组，而不是按「含不含 si」。**
            #
            # 原实现取 `max(含 si 的对)` 与 `min(不含 si 的对)`，输出
            # 「含 si 的品种对 Jaccard **仅 1.00**，而不含 si 的为 0.50」——
            # 与它要论证的结论**完全相反**：Jaccard=1.00 意味着交易小时完全相同。
            # 根因是 si 与 lc **都不做夜盘**，故 si|lc = 1.00 被 max() 取中；
            # 而 ag|lc = 0.50 被 min() 取中。「含不含 si」根本不是数据的真实切面。
            #
            # 真实结构是二分：ag/au/sc/cu/rb 交易夜盘，si/lc 不交易。
            # 组内高度相似、跨组显著更低——这才是「日内时段必须匹配后再比」的依据。
            night = {s: v.get("night_share", 0.0) for s, v in bys.items()}
            is_night = {s: v > 0.2 for s, v in night.items()}
            # 变量名带 _j 后缀：`cross` 在本函数后段是跨品种**边列表**，
            # 撞名会让它变成 float 列表并在几十行后炸掉（实测踩到）。
            within_j, cross_j = [], []
            for k, v in jac.items():
                a, b = k.split("|")
                if a in is_night and b in is_night:
                    (within_j if is_night[a] == is_night[b] else cross_j).append(v)
            if within_j and cross_j:
                _ng = sorted(s for s, x in is_night.items() if x)
                _dg = sorted(s for s, x in is_night.items() if not x)
                A(f"- 交易时段呈**二分结构**：{'/'.join(_ng)} 交易夜盘，"
                  f"{'/'.join(_dg)} 不交易。**组内**品种对的交易小时集合 Jaccard "
                  f"相似度为 {min(within_j):.2f}~{max(within_j):.2f}（均值 "
                  f"{sum(within_j) / len(within_j):.2f}），而**跨组**仅 "
                  f"{min(cross_j):.2f}~{max(cross_j):.2f}（均值 "
                  f"{sum(cross_j) / len(cross_j):.2f}）。"
                  f"这正是跨品种共现必须**先按日内时段匹配**的原因——"
                  f"不匹配的话，跨组品种会因为「根本不在同一时间交易」而被算成不共现，"
                  f"同组品种则会因为「总在同一时间交易」而被算成强共现。\n")
            A("> 以上数字来自 `data_out/kg/kg_session.json`（由 `scripts/build_kg.py` "
              "在标定时一并落盘），非人工填写。\n")
        A("也就是说，这条「传导」几乎完全是**「大家都在夜盘、夜盘流动性本来就薄」**"
          "造成的日内季节性。因此改用**按小时匹配的基线**（用源品种触发时刻的小时分布"
          "重新加权目标品种的基础率）后：\n")
        A("| 关系 | 朴素 lift | 小时匹配后 lift | 结论 |")
        A("|------|----------|---------------|------|")
        for cc in sorted(r6, key=lambda x: -x["lift"])[:4]:
            A(f"| {cc['src']}→{cc['dst']} 流动性退化 | {_num(cc['lift'])} | "
              f"{_num(cc.get('lift_matched'))} | "
              f"{'保留' if cc['strong'] else '**剔除**'} |")
        si = [c for c in r6 if c["src"] == "si"]
        if si:
            s0 = si[0]
            A(f"\n> 反过来也一样：`si→{s0['dst']}` 的朴素 lift 只有 {_num(s0['lift'])}"
              f"（看似强烈负相关），小时匹配后是 {_num(s0.get('lift_matched'))}——"
              f"朴素口径在这里把方向都搞反了。\n")
        A(f"经此修正，跨品种边从朴素口径的一大片收敛到 {len(cs_strong)} 条。"
          "即便如此，**留下来的边也只按「同步共现」表述，不写「A 导致 B」**——"
          "统计上无法区分传导与共同暴露于同一宏观冲击。\n")

    # ---- 跨板块传导（扩样后才具备验证条件）
    from kg.schema import SECTORS as _SEC, SYMS as _SY, ANOM_BY_RULE as _ABR
    _sec = lambda x: _SY[x]["sector"]
    same = [c for c in cs_strong if _sec(c["src"]) == _sec(c["dst"])]
    diff = [c for c in cs_strong if _sec(c["src"]) != _sec(c["dst"])]
    n_sym = len({s for c in cross for s in (c["src"], c["dst"])})
    n_sec = len({_sec(s) for s in _SY})
    A("### 跨板块传导\n")
    A(f"本版覆盖 {n_sym} 个品种、{n_sec} 个板块。"
      f"通过检验的 {len(cs_strong)} 条跨品种边中，**跨板块 {len(diff)} 条**、"
      f"同板块 {len(same)} 条。\n")
    A("> 早期版本只有 4 个品种 3 个板块（且 ag/au 同属贵金属、sc 与 si 各自单独），"
      "跨板块传导实际上没有验证条件——通过的边几乎都是贵金属内部的 ag↔au。"
      "扩样到 7 个品种后这一块才真正可测。\n")
    if diff:
        A("跨板块边中提升度最高的若干条（均已做日内时段匹配）：\n")
        A("| 源 | 目标 | 异常类型 | 朴素 lift | 时段匹配后 | 支撑度 | p 值 |")
        A("|----|------|---------|----------|-----------|--------|------|")
        for c in sorted(diff, key=lambda x: -x["lift_matched"])[:10]:
            A(f"| {_SEC[_sec(c['src'])]}·{_SY[c['src']]['title']} "
              f"| {_SEC[_sec(c['dst'])]}·{_SY[c['dst']]['title']} "
              f"| {_ABR[c['rule']].name} | {_num(c['lift'])} | "
              f"**{_num(c['lift_matched'])}** | {c['n_src']:,} | {c['p_value']:.4f} |")
        A("")
        from collections import Counter as _C
        pc = _C(tuple(sorted((_SEC[_sec(c['src'])], _SEC[_sec(c['dst'])]))) for c in cs_strong)
        A("板块对通过的边数：\n")
        A("| 板块对 | 边数 |")
        A("|--------|------|")
        for k, v in pc.most_common():
            A(f"| {k[0]} ↔ {k[1]} | {v} |")
        A("")
    A("> 仍需强调：这里衡量的是**超出日内节律的同步共现**，不是因果传导。"
      "统计上无法区分「风险由 A 传导到 B」与「A、B 共同暴露于同一宏观冲击」——"
      "对同属中国宏观定价的品种（如螺纹与白银），后者可能是更自然的解释。\n")

    A("## 6. 解释生成与人工评分材料\n")
    A("解释文本采用固定的「根因 → 传导环节 → 可能后果」三段式，每段都附证据："
      "根因段给当前实测读数与触发阈值，传导段给机制说明（标注为业务先验）与"
      "统计支撑（lift、条件概率、样本量、p 值、是否测试期复现）。\n")
    A("赛题指标3 要求对解释文本做**人工** 5 分制评分，平均分 ≥4 分。"
      "本项目提供评分材料，但**不自评达标**——自评分不能替代人工评分：\n")
    A(f"- `data_out/kg/explanation_samples.md`：测试期随机抽取的 {len(samples)} 条解释")
    A("- `data_out/kg/scoring_sheet.csv`：空白评分表（三维度 × 5 分制）")
    A("- `data_out/kg/self_check.json`：机器可检查的客观项（**非**人工评分）\n")
    A("### 评分细则\n")
    for name, desc, anchors in RUBRIC:
        A(f"**{name}**——{desc}\n")
        for sc in (5, 4, 3, 2, 1):
            A(f"- {sc} 分：{anchors[sc]}")
        A("")
    A("### 机器自查（客观项，非人工评分）\n")
    A(f"- 全部 {self_check['n_samples']} 条样本均带免责声明：{self_check['all_have_disclaimer']}")
    A(f"- 全部样本正文无劝导词/交易动作词：{self_check['all_no_advice']}")
    A(f"- 全部样本均含「根因」与「传导」两段：{self_check['all_have_cause_and_path']}")
    A(f"- 显式标注「业务先验」的样本比例：{_pct(self_check['pct_labels_prior'])}")
    A(f"- 解释文本中位长度：{self_check['median_chars']:.0f} 字；"
      f"中位证据引用数：{self_check['median_evidence_refs']:.0f} 条\n")
    A("> 这些客观项只能保证「不会明显违规」，**不能替代**对语义清晰度与因果合理性的人工判断。"
      "指标3 的达标与否，以人工评分结果为准。\n")

    A("## 7. 验证门\n")
    if G:
        A(f"`python -m kg.tests --emit` 的实际执行结果：**{G['n_pass']}/{G['n_total']} 通过**"
          f"（在 {G.get('n_samples', '—')} 条测试期样本上，结果存档于 `data_out/kg/gates.json`）。\n")
        A("| 门 | 结果 | 实测结论 |")
        A("|----|------|---------|")
        for x in G["gates"]:
            A(f"| {x['name']} | {'✓' if x['passed'] else '✗'} | {x['message']} |")
        A("")
    else:
        A("（未找到 gates.json，请运行 `python -m kg.tests --emit`）\n")
    A("> 三道门在开发中真的拦下了问题：K1 初版把免责声明里的「操作建议」四字"
      "误判为违规；K2 的数字提取正则把日期 `2026-02-12` 里的 `-02` 当成负数；"
      "K5 初版只在前 24 幕上复算 lift，与全训练期标定的结果对不上（3.04 vs 2.45），"
      "看着像图谱有 bug，实为测试自己取错了样本。\n")

    A("## 8. 已知局限\n")
    A("1. **lift 衡量的是条件共现，不是因果**。分块置换只能排除「纯属偶然」，"
      "排除不了「同源」与「共同暴露于第三方冲击」。图谱的定位是"
      "**可解释的风险提示**，不是因果推断结论。")
    A(f"2. **{len(selfp)} 条边存在同源特征问题**（根因与后果共用同一底层指标），"
      "已标注但无法根除——要根除需要引入曲面之外的独立观测（如标的已实现波动、"
      "成交持仓明细），当前数据集不包含。")
    A("3. **后果集合只有 5 类**，且都定义在曲面特征上。像「保证金压力」「跨市场传染」"
      "这类业务上更关心的后果，缺乏可观测代理变量，本版未纳入。")
    A(f"4. **机制层未经检验且覆盖不全**。`MECH` 的机制说明是业务先验，图里已标注为"
      f"诠释层、不参与打分，但其本身正确性没有独立验证手段；且 {len(strong)} 条实测边中"
      f"只有 {sum(1 for e in strong if MECH_MAP.get((e['anom'], e['conseq'])))} 条配有机制说明。")
    A("5. **指标3 未经人工评审**，本报告不对该项作达标声明。\n")

    A("---\n")
    A("## 复现\n")
    A("```bash\n"
      "python scripts/build_kg.py          # 标定 + 装配图谱（约 20s）\n"
      f"python -m kg.tests --emit           # {(G or {}).get('n_total', 10)} 道验证门\n"
      "python scripts/report_kg.py         # 重新生成本报告与评分材料\n"
      "python scripts/viz_kg.py            # 生成传导路径可视化 HTML\n"
      "```\n")

    # ---------------------------------------------------------------- Schema 文档
    # 赛题成果形式点名要求「知识图谱 Schema 文档」。这里从 kg/schema.py 直接生成，
    # 避免文档与代码各说各话（v1 的 schema_v1.md 就与实现脱节了）。
    S = []
    B = S.append
    B("# 期权风险知识图谱 Schema\n")
    B("> 本文件由 `scripts/report_kg.py` 从 `kg/schema.py` 自动生成，请勿手工编辑。\n")
    B("## 分层\n")
    B("| layer | 含义 | 权重来源 | 参与推理打分 |")
    B("|-------|------|---------|------------|")
    B(f"| `measured` | 实测层 | {meta.get('n_train_slices', 0):,} 个**训练期**真实截面标定的提升度 lift | 是 |")
    B("| `interpretive` | 诠释层 | 业务先验，无权重 | 否 |")
    B("| `meta` | 品种/板块等元数据 | — | 否 |\n")
    B("## 节点类型\n")
    B("| type | layer | 数量 | 说明 |")
    B("|------|-------|-----|------|")
    for t, lay, desc in (("ANOM", "measured", "曲面异常模式，与问题1 的 8 条预警规则一一对应"),
                         ("CONSEQ", "measured", "可测后果，由未来 8 个截面的特征客观判定"),
                         ("MECH", "interpretive", "传导机制说明，业务先验"),
                         ("CONSTRAINT", "interpretive", "无套利约束，说明异常为何属定价结构问题"),
                         ("SYM", "meta", "品种"), ("SECTOR", "meta", "板块"),
                         ("XSYM", "measured", "品种级异常节点，承载跨品种共现边")):
        B(f"| `{t}` | `{lay}` | {len(g.nodes(t))} | {desc} |")
    B("")
    B("## 边类型\n")
    B("| kind | 起点 → 终点 | layer | 权重含义 |")
    B("|------|------------|-------|---------|")
    B("| `AMPLIFIES` | ANOM → CONSEQ | measured | lift 的 95% 保守下界（推理主干）|")
    B("| `PROPAGATES_TO` | XSYM → XSYM | measured | 小时匹配后 lift 的保守下界 |")
    B("| `EXPLAINS` | ANOM → MECH → CONSEQ | interpretive | 恒 0，仅供可视化 |")
    B("| `VIOLATES` | ANOM → CONSTRAINT | interpretive | 恒 0 |")
    B("| `MEMBER_OF` | SYM → SECTOR | meta | 恒 1 |\n")
    B("## 异常节点（ANOM）\n")
    B("| id | 名称 | 对应规则 | 主特征 | 含义 |")
    B("|----|------|---------|--------|------|")
    for a in ANOMS:
        B(f"| `{a.id}` | {a.name} | `{a.rule}` | `{a.feature or '复合'}` | {a.desc} |")
    B("")
    B("## 后果节点（CONSEQ）——判定口径\n")
    B("| id | 名称 | 判定式 | 业务含义 |")
    B("|----|------|--------|---------|")
    for c in CONSEQS:
        B(f"| `{c.id}` | {c.name} | {c.rule_text()} | {c.desc} |")
    B("")
    B("## 机制节点（MECH，诠释层）\n")
    from kg.schema import MECHS as _MECHS
    B("| id | 名称 | 机制说明 |")
    B("|----|------|---------|")
    for m in _MECHS:
        B(f"| `{m.id}` | {m.name} | {m.how} |")
    B("")
    B("> 机制说明是**业务先验**，不参与权重计算，也未经独立验证。"
      "它只回答「为什么会这样传导」，其正确性不由本项目的数据背书。\n")
    B("## 边属性\n")
    B("`AMPLIFIES` 边携带以下可审计字段：\n")
    B("| 字段 | 含义 |")
    B("|------|------|")
    for k, v in (("lift", "提升度 P(C|A)/P(C)"), ("lift_ci_low", "lift 的 95% Wilson 保守下界，即边权"),
                 ("p_cond", "条件概率 P(C|A)"), ("p_conseq", "无条件基础率 P(C)"),
                 ("n_anom", "支撑度（异常出现次数）"), ("p_value", "分块置换检验 p 值"),
                 ("self_feature", "根因与后果是否共用同一底层特征"),
                 ("replicated", "测试期是否可复现"), ("mech / how", "机制注解（诠释层）")):
        B(f"| `{k}` | {v} |")
    B("")
    with open(os.path.join(ROOT, "kg", "schema.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(S))
    print(f"→ {os.path.join(ROOT, 'kg', 'schema.md')}")

    txt = "\n".join(L)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"→ {args.out}  ({len(txt.splitlines())} 行)")
    print(f"→ {args.kg_dir}/explanation_samples.md  ({len(samples)} 条样本)")
    print(f"→ {args.kg_dir}/scoring_sheet.csv")
    print(f"→ {args.kg_dir}/self_check.json")


if __name__ == "__main__":
    main()
