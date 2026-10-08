"""kg.tests — 知识图谱验证门。不通过就不许把结果写进报告。

延续 `drl/tests.py` 的做法：把「知识图谱最容易出的假」逐个变成可执行断言。
知识图谱的典型翻车方式与 DRL 不同，主要是三类：
  1. 图里塞满没有数据支撑的边，靠数量唬人；
  2. 解释文本里的数字对不上图里的统计（写作时手滑或凭印象）；
  3. 遇到图谱覆盖不到的情况仍硬凑一条解释（幻觉）。

运行::
    python -m kg.tests                 # 全部
    python -m kg.tests --emit          # 结果落盘为 data_out/kg/gates.json

门列表
------
K1 无交易建议    生成的解释不得出现面向读者的操作指令（劝导词 + 交易动作组合）
K2 证据可回溯    解释里出现的每个数字都能对到特征读数或图中边统计，无孤儿数字
K3 图结构合法    实测层只含 ANOM→CONSEQ / XSYM→XSYM；无自环；诠释层不参与打分
K4 确定性        同一输入两次调用得到逐字相同的解释
K5 边统计可复算  从原始 anchor 数据重算若干条边的 lift，须与图中记录一致
K6 拒绝编造      对「触发了但图中无通过检验出边」的异常，必须明说不作推断
K7 无未检验边    图中实测层不得包含 strong=False 的边
K8 文案未失效    各项解释文案功能仍在样本中出现（防「修 A 静默废掉 B」）
K9 不等式为真    打印的判定不等式必须字面成立
K10 解读不抵消   通俗解读句的「平时 a 次 → 形态后 b 次」必须 b > a
K11 强调成对    每行的 `**` 个数须为偶数（防嵌套/悬空导致强调错位）
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.dataset import FEATURES, load_episodes, split_episodes
from kg.calibrate import anom_matrix, conseq_matrix, valid_mask
from kg.explain import DISCLAIMER, explain
from kg.graph import DiGraph
from kg.reason import active_anomalies, reason
from kg.schema import ANOMS, CONSEQS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "data_out", "anchor")
KG_DIR = os.path.join(ROOT, "data_out", "kg")

_RESULTS, _DETAILS = [], {}


def gate(name: str):
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                msg = fn(*a, **kw)
                _RESULTS.append((name, True, msg or ""))
                print(f"  [PASS] {name}  {msg or ''}")
                return True
            except AssertionError as e:
                _RESULTS.append((name, False, str(e)))
                print(f"  [FAIL] {name}  {e}")
                return False
            except Exception as e:                        # noqa: BLE001
                _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  [ERROR] {name}  {type(e).__name__}: {e}")
                return False
        return wrapped
    return deco


# ---------------------------------------------------------------- 样本

def sample_explanations(g: DiGraph, episodes, n: int = 200, seed: int = 0) -> list:
    """从测试期截面里采样，生成解释文本，供各门检查。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        e = episodes[int(rng.integers(0, len(episodes)))]
        i = int(rng.integers(0, len(e)))
        feats = {k: float(e.X[i, j]) for j, k in enumerate(FEATURES)}
        res = reason(g, feats, symbol=e.symbol, timestamp=e.ts[i])
        out.append({"feats": feats, "res": res, "exp": explain(res),
                    "symbol": e.symbol, "ts": e.ts[i]})
    return out


# ---------------------------------------------------------------- K1

# 劝导词：面向读者的建议口吻
ADVICE_WORDS = ["建议", "推荐", "应当", "应该", "宜于", "不妨", "务必", "请考虑",
                "可考虑", "最好", "建议您", "我们认为应"]
# 交易动作：具体操作指令
TRADE_ACTIONS = ["买入", "卖出", "做多", "做空", "加仓", "减仓", "平仓", "开仓",
                 "止损", "止盈", "建仓", "抄底", "追多", "追空", "清仓", "补仓"]


@gate("K1 解释文本不含交易建议")
def test_no_trading_advice(samples) -> str:
    bad = []
    for s in samples:
        t = s["exp"]["text"]
        assert DISCLAIMER in t, f"{s['ts']} 缺少免责声明"
        # 免责声明本身含「操作建议」四字，须先剥离再扫正文，否则会自己把自己判死
        body = t.replace(DISCLAIMER, "")
        # 正文从严：出现任何劝导词或交易动作词即判失败
        for w in ADVICE_WORDS + TRADE_ACTIONS:
            if w in body:
                bad.append((s["symbol"], s["ts"], w))
    assert not bad, f"检出交易建议措辞 {bad[:5]}（共 {len(bad)} 处）"
    n_nonempty = sum(1 for s in samples if s["res"]["n_active"] > 0)
    return (f"{len(samples)} 条解释（其中 {n_nonempty} 条含异常）均无劝导词/交易动作词，"
            f"且都带免责声明")


# ---------------------------------------------------------------- K2

# 前面若紧邻数字或连字符，则该 "-" 属于日期分隔符而非负号（如 2026-02-12）
_NUM = re.compile(r"(?<![\d\-])-?\d+(?:\.\d+)?")


@gate("K2 解释中的数字均可回溯")
def test_evidence_traceable(samples, g: DiGraph) -> str:
    """解释里出现的每个数字，都必须能对到特征读数、边统计或已知常量。"""
    consts = {8, 120, 15, 2.0, 1.0, 0.45, -1.5, 0.5, 95, 0, 100,
              # 「平时每 10 次随机时点里约 a 次」的分母——由文案自行选定的表述基数，
              # 不是任何统计量。它暴露了本门的一个盲区：早前 10 未登记，却只在
              # 2 个时间戳上报错，因为其余样本的 ts[8:10] 恰好是 "10"（10 点档），
              # 把时间戳数字当成了它的来源。**同值不同源的巧合会掩盖真缺陷**，
              # 故此处显式登记，而不是依赖某个碰巧同值的字段放行。
              10}
    orphans = []
    checked = 0
    for s in samples[:60]:
        allowed = set(consts)
        # 方法学常数：置换检验重排次数与由它决定的 p 值分辨率下限。
        # 解释文本要引用它们来说明「p=0.0050 是地板值不是测得值」。
        from kg.calibrate import N_PERM_DEFAULT
        allowed.update({float(N_PERM_DEFAULT), float(N_PERM_DEFAULT + 1),
                        round(1.0 / (N_PERM_DEFAULT + 1), 4)})
        # 入选门槛常数：解释文本要引用它们来说明「严格复现 vs 宽松判据」的差别
        allowed.update({1.1, 1.10, 1.0, 0.05})
        # 「弱效应」判据里印的阈值 = **有机制边的最小 lift**。
        # 它来自图中**其他**边，不在当前 path 里，故必须单独登记——
        # 本门在加入「为什么没有机制」那段后立刻 FAIL 27 个，抓得对。
        # **从 schema.weak_lift_threshold 现算，不写死**：图一重标定它就会变。
        from kg.schema import weak_lift_threshold
        allowed.add(round(float(weak_lift_threshold(g.edges("AMPLIFIES"))), 2))
        # R1/R5 双重条件里打印的 z 阈值（「本级要求 z > X」）。
        # **从 DEFAULT_PARAMS 动态取，不写死**——写死的话改了参数这门就会误判。
        # 补这一条是因为它真的抓到过：新印 z 阈值后 K2 立刻 FAIL，说明门是活的。
        from vol_surface.alert_rules import DEFAULT_PARAMS
        allowed.update({round(float(v), 4) for k, v in DEFAULT_PARAMS.items()
                        if k.endswith(("_watch_z", "_warn_z", "_ser_z"))})
        for v in s["feats"].values():
            allowed.update({round(v, 4), round(v, 2), round(v, 1)})
        for r in s["res"]["anomalies"]:
            tr = r["trigger"]
            # 证据 detail 里的数值（触发量、原始值、胜出分量读数、阈值）都要放行
            for _e in r.get("evidence", []):
                for _k in ("trigger_value", "raw_value", "winning_value",
                           "threshold", "value"):
                    _v = _e.get("detail", {}).get(_k)
                    if isinstance(_v, (int, float)):
                        allowed.update({round(float(_v), 4), round(float(_v), 2)})
            allowed.update({round(tr["value"], 4), round(tr["value"], 3),
                            round(tr["value"], 2), round(tr["threshold"], 4),
                            round(tr["threshold"], 2), float(tr["level"])})
            # 截断披露里的数字：被隐去传导的条数与其提升度上限。
            # 它们来自**被丢弃**的 path，不在 r["paths"] 里，故必须单独登记——
            # 本门在加入截断披露的当次就 FAIL 了 15 个，抓得对。
            _tc = r.get("truncated")
            if _tc:
                allowed.update({float(_tc["n_hidden"]),
                                round(float(_tc["max_hidden_lift"]), 2)})
            if r.get("n_paths_all") is not None:
                allowed.add(float(r["n_paths_all"]))
            for p in r["paths"]:
                for k in ("lift", "weight", "p_value"):
                    allowed.update({round(p[k], 4), round(p[k], 2)})
                # 「这意味着什么」那句里的派生量（见 explain._plain_reading）：
                # 它不引入任何新统计量，只是把已有的 p_cond / p_conseq / lift
                # 换成读者能感知的说法。三类派生都必须显式登记，否则本门会
                # 把它们判为"编造的数字"——实测补上解读句后 K2 立刻 FAIL 251 个，
                # 门抓得对，是可回溯集合漏了派生规则。
                # **分母从 explain.reading_denominator 取，不写死**——它是自适应的
                # （见该函数：低 lift 边会自动降到 1/100 刻度以免自我抵消）。
                # 写死 10 的话，实现一改本门就会误判；实测改自适应后 FAIL 39 个。
                from kg.explain import reading_denominator
                _den = reading_denominator(p)
                allowed.update({
                    float(_den),                             # 分母本身
                    float(round(p["p_conseq"] * _den)),      # 「平时每 N 次里约 a 次」
                    float(round(p["p_cond"] * _den)),        # 「本形态后变成约 b 次」
                    round(p["lift"], 1),                     # 「概率提高到 x.x 倍」
                    float(round((1 - p["p_cond"]) * 100)),   # 「仍有 xx% 不发生」
                })
                # 测试期 lift 也是解释文本会引用的数字（用于披露样本外衰减），
                # 必须纳入可回溯集合，否则本门会把它误判为"编造的数字"。
                for k in ("test_lift", "test_p", "test_ci_low"):
                    v = p.get(k)
                    if v is not None and v == v:
                        allowed.update({round(v, 4), round(v, 2)})
                allowed.update({round(p["p_cond"] * 100, 1),
                                round(p["p_conseq"] * 100, 1), float(p["n_anom"])})
        for c in s["res"]["cross_symbol"]:
            allowed.update({round(c["p_cond"] * 100, 1), round(c["lift"], 2),
                            float(c["n_src"])})
        allowed.update({float(s["res"]["n_active"]),
                        float(len(s["res"]["anomalies"]))})
        # 时间戳里的数字
        ts = s["ts"]
        allowed.update({float(ts[:4]), float(ts[4:6]), float(ts[6:8]),
                        float(ts[8:10]), float(ts[10:12])})

        for tok in _NUM.findall(s["exp"]["text"]):
            try:
                v = float(tok)
            except ValueError:
                continue
            if not any(abs(v - a) < 1e-6 for a in allowed):
                orphans.append((s["ts"], tok))
        checked += 1
    assert not orphans, f"检出无法回溯的数字 {orphans[:8]}（共 {len(orphans)} 个）"
    return f"抽查 {checked} 条解释，全部数字均可回溯到特征读数或图中边统计"


# ---------------------------------------------------------------- K3

@gate("K3 图结构合法")
def test_graph_structure(g: DiGraph) -> str:
    anom_ids = {a.id for a in ANOMS}
    conseq_ids = {c.id for c in CONSEQS}
    for u, v, a in g.edges("AMPLIFIES"):
        assert u in anom_ids, f"AMPLIFIES 起点 {u} 不是异常节点"
        assert v in conseq_ids, f"AMPLIFIES 终点 {v} 不是后果节点"
        assert a.get("layer") == "measured", f"{u}→{v} 未标为实测层"
    for u, v, a in g.edges("PROPAGATES_TO"):
        assert u.startswith("XSYM-") and v.startswith("XSYM-"), f"{u}→{v} 类型错误"
        assert a["src"] != a["dst"], f"{u}→{v} 跨品种边的源与目标相同"
    for u, v, _ in g.edges():
        assert u != v, f"存在自环 {u}"
    for u, v, a in g.edges("EXPLAINS"):
        assert a.get("layer") == "interpretive" and a.get("weight", 0) == 0.0, \
            f"诠释层边 {u}→{v} 不应带非零权重"
    return (f"{len(g.edges('AMPLIFIES'))} 条主干边方向与类型正确、"
            f"{len(g.edges('PROPAGATES_TO'))} 条跨品种边合法、无自环、"
            f"诠释层边权重恒为 0")


# ---------------------------------------------------------------- K4

@gate("K4 解释生成确定性")
def test_deterministic(g: DiGraph, samples) -> str:
    for s in samples[:40]:
        r2 = reason(g, s["feats"], symbol=s["symbol"], timestamp=s["ts"])
        t2 = explain(r2)["text"]
        assert t2 == s["exp"]["text"], f"{s['ts']} 两次生成结果不同"
    return "抽查 40 条，两次调用输出逐字相同"


# ---------------------------------------------------------------- K5

@gate("K5 边统计可从原始数据复算")
def test_edge_stats_recomputable(g: DiGraph, train_eps) -> str:
    """抽几条主干边，从 anchor 原始特征重算 lift，与图中记录比对。

    注意必须在**与标定时完全相同**的 episode 集合（全部训练期）上复算。
    初版只取了前 24 幕，重算出 3.04 而图中是 2.45，看着像 bug，实为样本不同——
    这类"测试自己算错了"的假阳性，比漏测更容易浪费时间。"""
    edges = sorted(g.edges("AMPLIFIES"), key=lambda e: (e[0], e[1]))[:4]
    assert edges, "图中没有主干边"
    a_idx = {a.id: j for j, a in enumerate(ANOMS)}
    c_idx = {c.id: j for j, c in enumerate(CONSEQS)}
    checked = []
    for u, v, attrs in edges:
        n_a = n_ac = n_c = n_t = 0
        for e in train_eps:
            A = anom_matrix(e.X, FEATURES)
            C = conseq_matrix(e.X, FEATURES)
            m = valid_mask(len(e))
            a, c = A[m, a_idx[u]], C[m, c_idx[v]]
            n_a += int(a.sum()); n_ac += int((a & c).sum())
            n_c += int(c.sum()); n_t += int(m.sum())
        lift = (n_ac / n_a) / (n_c / n_t)
        assert abs(lift - attrs["lift"]) < 0.02, \
            f"{u}→{v} 重算 lift={lift:.4f} 与图中 {attrs['lift']:.4f} 不符"
        assert n_a == attrs["n_anom"], \
            f"{u}→{v} 重算支撑度 {n_a} 与图中 {attrs['n_anom']} 不符"
        checked.append(f"{u.split('-')[1]}→{v.split('-')[1]}:{lift:.2f}")
    return f"复算 {len(checked)} 条边全部一致（{', '.join(checked)}）"


# ---------------------------------------------------------------- K6

@gate("K6 无证据时明确拒绝推断")
def test_refuses_without_evidence(g: DiGraph, samples) -> str:
    """触发了但图中没有通过检验出边的异常，必须明说不作推断（防幻觉）。

    分两步：
      (a) 真实样本里若自然出现这种情形，逐条核验；
      (b) **无论如何都做一次构造性检验**——人为摘掉某个异常的全部出边，
          再喂入一个确实触发该异常的截面，要求解释文本如实拒绝推断。

    为什么要有 (b)：扩样到 7 品种后，样本量变大使得每个异常都至少有一条通过检验的
    出边，(a) 再也遇不到这种情形，该门一度只能报「未被真正检验」而 FAIL。
    删掉断言让它「通过」是自欺——拒答能力仍然是防幻觉的核心保证，必须主动构造场景来测。
    """
    import copy
    n_nat = 0
    for s in samples:
        if not s["res"]["unresolved"]:
            continue
        n_nat += 1
        txt = s["exp"]["text"]
        assert "不对后果作推断" in txt or "不对其下游影响作推断" in txt, \
            f"{s['ts']} 存在无出边异常却未声明拒绝推断"
        for u in s["res"]["unresolved"]:
            assert u["name"] in txt, f"{s['ts']} 未列出无法推断的异常 {u['name']}"

    # ---- (b) 构造性检验：挑一个最常触发的异常，摘掉其全部出边
    from collections import Counter
    cnt = Counter(r["anom"] for s in samples for r in s["res"]["anomalies"])
    assert cnt, "样本中没有任何异常触发，无法做构造性检验"
    target = cnt.most_common(1)[0][0]

    g2 = DiGraph.from_dict(copy.deepcopy(g.to_dict()))
    kept = [(u, v, a) for u, v, a in g2.edges("AMPLIFIES") if u != target]
    g3 = DiGraph()
    for n in g2.nodes():
        g3.add_node(n, **g2.node(n))
    for u, v, a in g2.edges():
        if not (a.get("kind") == "AMPLIFIES" and u == target):
            g3.add_edge(u, v, **a)
    assert not g3.successors(target, kind="AMPLIFIES"), "构造失败：出边未摘干净"

    hit = None
    for s in samples:
        if any(r["anom"] == target for r in s["res"]["anomalies"]):
            hit = s
            break
    assert hit is not None, f"找不到触发 {target} 的样本"
    res2 = reason(g3, hit["feats"], symbol=hit["symbol"], timestamp=hit["ts"])
    txt2 = explain(res2)["text"]
    names = {u["anom"] for u in res2["unresolved"]}
    assert target in names, f"摘掉出边后 {target} 未被列入 unresolved"
    assert "不对后果作推断" in txt2 or "不对其下游影响作推断" in txt2, \
        "摘掉出边后解释文本仍未声明拒绝推断——存在编造风险"
    tname = next(r["name"] for r in hit["res"]["anomalies"] if r["anom"] == target)
    assert tname in txt2, f"未在文本中列出无法推断的异常 {tname}"

    _DETAILS["K6"] = {"natural_cases": n_nat, "constructed_target": target}
    return (f"自然样本中出现 {n_nat} 例；另做构造性检验：摘掉「{tname}」的全部出边后，"
            f"解释如实声明不作推断（未编造路径）")


# ---------------------------------------------------------------- K7

@gate("K7 图中不含未通过检验的边")
def test_no_weak_edges(g: DiGraph) -> str:
    bad = [(u, v) for u, v, a in g.edges("AMPLIFIES") if not a.get("strong")]
    assert not bad, f"图中混入未通过检验的边 {bad[:5]}"
    n_rep = sum(1 for _, _, a in g.edges("AMPLIFIES") if a.get("replicated"))
    n_loose = sum(1 for _, _, a in g.edges("AMPLIFIES")
                  if a.get("replicated_loose") and not a.get("replicated"))
    n_all = len(g.edges("AMPLIFIES"))
    _DETAILS["K7"] = {"n_measured": n_all, "n_replicated": n_rep,
                      "n_loose_only": n_loose}
    # 「复现」必须按与训练期同一标准计数。早前的判据不含 p 值检验，于是测试期
    # p=0.154 的边也被算进「可复现」，本门据此对外宣称「21 条全部可复现」——
    # 是一处假陈述（第 5 位独立评审点名）。现分两档如实报出。
    return (f"{n_all} 条主干边全部通过支撑度+置换检验+lift 下界三项；"
            f"测试期**按同一标准**复现 {n_rep} 条，"
            f"另 {n_loose} 条仅满足宽松判据（下界>1.0 但未达 p≤0.05 或下界>1.1），"
            f"其样本外证据偏弱、已在解释文本中就地标注")


# ---------------------------------------------------------------- K8

# 「文案功能仍然接得上」的哨兵。
#
# 为什么需要这道门
# ----------------
# K1~K7 检查的是**内容合规与统计正确**，没有一道会因为「某段文案再也不出现」而失败。
# 实测踩过：第 7 轮为消除「atm_iv_z = −5.3294，判定为 > 3.5000」这类字面为假的
# 不等式，把触发量改印 |z|；而方向提示的判据恰好读同一字段，取绝对值后恒非负，
# 于是提示**永远不触发**。两处改动各自都对，门全绿，评审读的又是改动前生成的
# 旧样本文档——三重遮蔽，直到全链路复现时核对输出计数才暴露。
#
# 门的形态是「下限」而非「等于」：样本随 seed 与数据变动，写死等号会变成噪声门。
# 关键是**不许归零**——归零几乎总意味着触发条件被某次改动切断了。
_FEATURE_SENTINELS = {
    "方向提示（双向触发规则的负向偏离警示）": ("**方向提示：", 1),
    "R7 方向盲区说明": ("本规则名为「短期 IV 急升」", 1),
    "R8 机制分量错配提示": ("**分量提示**：", 1),
    "双重条件的 z 阈值": ("本级要求 z > ", 1),
    "业务语义解读句": ("**这意味着什么**：", 50),
    # 抬头句有三个分支（n==1 / 全部列出 / 有截断），故哨兵匹配**三者共有的
    # 那句排序说明**，而不是某一个分支的措辞——按分支写会在样本分布变化时误报。
    # n==1 分支不谈排序（那 45 字在只有一项时不可能适用），故用「全部触发项均已列出」
    # 单独守它。
    "抬头句·排序说明": ("按触发等级从高到低排序", 20),
    "抬头句·单项分支不谈排序": ("全部触发项均已列出", 3),
    "机制↔后果落点提示": ("落点提示：该机制的叙事收尾在", 3),
    "同源边警示进通俗层": ("**但这条要打折看**", 20),
    "未展开根因具名": ("项因等级", 1),
    "分量提示块内传播": ("本根因下的各条传导均适用此提示", 1),
    "触发等级可见": ("（触发等级 **L", 50),
    "p 值根因内上提": ("条传导的**置换检验 p 值相同**", 10),
    "后果口径预警内回指": ("后果的判定口径同上", 5),
    "跨品种无边时如实说明": ("这是「查了但没有」，不是「没查」", 1),
    "主干路径截断披露": ("**另有 ", 5),
    "跨品种截断披露": ("通过检验的共现关系未列出", 1),
    "测试期 lift 披露": ("**测试期**该提升度为", 50),
    "样本外结论已平实化": ("换一段**没参与建模**的行情重测", 50),
    "宽松判据就地标注": ("未达训练期入选门槛 1.10", 1),
    "自相关警示": ("同一指标的时间自相关", 1),
    "因果免责": ("不是已证实的因果关系", 1),
    "停用规则双重身份说明": ("在本项目的**预警引擎**里已默认停用", 1),
    "统一口径回指": ("训练期统计[†]", 50),
}


@gate("K8 解释文案的各项功能仍在生效")
def test_feature_sentinels(samples) -> str:
    text = "\n".join(s["exp"]["text"] for s in samples)
    counts = {name: text.count(pat) for name, (pat, _) in _FEATURE_SENTINELS.items()}
    dead = [f"{n}（{_FEATURE_SENTINELS[n][0]!r} 出现 {c} 次，至少应 "
            f"{_FEATURE_SENTINELS[n][1]} 次）"
            for n, c in counts.items() if c < _FEATURE_SENTINELS[n][1]]
    assert not dead, ("下列文案功能已失效——多半是某次改动切断了触发条件，"
                      "而非样本恰好没覆盖：\n  " + "\n  ".join(dead))
    _DETAILS["K8"] = counts
    return (f"{len(counts)} 项文案功能全部在 {len(samples)} 条样本中出现："
            + "、".join(f"{n.split('（')[0]} {c}" for n, c in counts.items()))


@gate("K9 打印的判定不等式必须为真")
def test_printed_inequalities(samples) -> str:
    """凡是印在解释文本里的不等式，都必须字面成立。

    本项目在这上面栽过两次：
      - 「`atm_iv_z` = −5.3294，判定为 > 3.5000」——比较的其实是 |z|，字面为假；
      - 「lift 下界 1.07 > 1.10」——判据改严前的残留。
    这一次补的是 R1/R5 的双重条件（原值超阈 **且** 滚动 z 超阈）：
    文本现在会印「本级要求 z > X，实测 z = Y」，那么 **Y > X 必须成立**，
    且 X 必须等于 `DEFAULT_PARAMS` 里该等级对应的阈值——写死或取错等级都会被此门抓住。
    """
    import re
    from vol_surface.alert_rules import DEFAULT_PARAMS
    pat = re.compile(r"本级要求 z > ([\d.]+)，实测 z = ([-\d.]+)")
    n, bad = 0, []
    valid = {round(float(DEFAULT_PARAMS[k]), 4)
             for k in DEFAULT_PARAMS if k.endswith(("_watch_z", "_warn_z", "_ser_z"))}
    for s in samples:
        for thr, got in pat.findall(s["exp"]["text"]):
            n += 1
            t, g = float(thr), float(got)
            if g <= t:
                bad.append(f"实测 z={g} 未超过所印阈值 {t}")
            if round(t, 4) not in valid:
                bad.append(f"所印阈值 {t} 不在 DEFAULT_PARAMS 的 z 阈集合内")
    assert not bad, "打印的不等式不成立：" + "；".join(bad[:5])
    _DETAILS["K9"] = {"n_checked": n, "valid_thresholds": sorted(valid)}
    return (f"抽查 {n} 处「本级要求 z > X，实测 z = Y」，"
            f"全部满足 Y > X 且 X 取自 DEFAULT_PARAMS")


@gate("K10 通俗解读句不得自我抵消")
def test_plain_reading_not_degenerate(samples) -> str:
    """「平时约 a 次 → 本形态后约 b 次」必须满足 **b > a**。

    为什么单设一门：这句是为**非专业读者**加的，它的失败方式与其他门都不同——
    数字全部可回溯（K2 过）、文案确实出现（K8 过）、两次生成一致（K4 过），
    但读者读到的是「约 2 次 → 约 2 次，概率提高到 1.2 倍」，等于告诉他「没变化」。
    **一个句子可以每个字都为真，却传达出与证据相反的意思**，这是前九轮的门
    完全覆盖不到的一类缺陷（第 10 位评审实测 19/121 = 15.7%）。

    同时断言 b/a 与所印 lift 方向一致，防止把分母调细后引入新的不一致。
    """
    import re
    # 句式：平时每 N 次…出现「后果」；「异常」出现后变成约 b 次——概率提高到 x.x 倍
    # 名字部分用非贪婪、且禁止跨越分号/换行，避免一条正则吃掉相邻两句。
    pat = re.compile(r"平时每 (\d+) 次随机时点里约 (\d+) 次会在 \d+ 分钟内出现[^；\n]+；"
                     r"[^；\n]*?出现后变成约 (\d+) 次——概率提高到 ([\d.]+) 倍")
    n, bad = 0, []
    for s in samples:
        for den, a, b, lift in pat.findall(s["exp"]["text"]):
            n += 1
            a, b, lift = int(a), int(b), float(lift)
            if b <= a:
                bad.append(f"{s['ts']} 分母 {den}：约 {a} 次 → 约 {b} 次"
                           f"（lift {lift}），读者会读成「没变化」")
            elif lift > 1.0 and b < a:
                bad.append(f"{s['ts']} 计数方向与 lift={lift} 相反")
    assert n > 0, "样本中没有任何通俗解读句——该功能可能已被切断（另见 K8）"
    assert not bad, f"检出自我抵消的解读句 {bad[:5]}（共 {len(bad)} 条）"
    _DETAILS["K10"] = {"n_checked": n}
    return f"抽查 {n} 条解读句，全部满足「本形态后的次数 > 平时次数」，无自我抵消"


@gate("K11 强调标记必须成对")
def test_emphasis_balanced(samples) -> str:
    """每一行里的 `**` 个数必须是偶数。

    为什么单设一门
    --------------
    这类错**犯了两次**，且第二次就发生在「刚修完第一次并写下注释」之后：
      ① 落点提示写成 `**…收尾在**隐含波动率水平**上…**`——嵌套导致
         **要对比的两个术语反而没加粗**、连接词全被加粗（6 处全中）；
      ② 修完 ① 之后，我在分量提示的 `。**` 后又接了一句 `**…**`，
         得到 `**A**B**` 三个未配对标记，**整句失去加粗**（4 处）。
    两次都是「语义正确、渲染错误」——K2 查数字可回溯、K8 查文案是否出现、
    K10 查解读句不自我抵消，**没有一道门会看渲染**。
    评审逐条读才发现，而这本该是机器一秒钟的事。
    """
    bad = []
    for s in samples:
        for ln in s["exp"]["text"].split("\n"):
            n = ln.count("**")
            if n % 2:
                bad.append((s["ts"], n, ln.strip()[:60]))
    assert not bad, (f"检出 {len(bad)} 行 `**` 未配对（首个：{bad[0][0]} "
                     f"共 {bad[0][1]} 个 · 「{bad[0][2]}…」）")
    n_line = sum(len(s["exp"]["text"].split("\n")) for s in samples)
    return f"{len(samples)} 条解释、{n_line} 行，`**` 全部成对"


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--kg", default=os.path.join(KG_DIR, "kg_graph.json"))
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--emit", nargs="?", const=os.path.join(KG_DIR, "gates.json"))
    args = ap.parse_args()

    if not os.path.exists(args.kg):
        raise SystemExit(f"缺少 {args.kg}，请先运行 python scripts/build_kg.py")
    g = DiGraph.load(args.kg)
    print(f"载入图谱：{len(g)} 节点 / {len(g.edges())} 边")
    eps = load_episodes(args.anchor, verify=True)
    tr, va, te = split_episodes(eps)
    samples = sample_explanations(g, te, n=args.n_samples)
    print(f"采样 {len(samples)} 条测试期截面生成解释\n")

    print("验证门:")
    test_no_trading_advice(samples)
    test_evidence_traceable(samples, g)
    test_graph_structure(g)
    test_deterministic(g, samples)
    test_edge_stats_recomputable(g, tr)   # 必须用全部训练期，与标定口径一致
    test_refuses_without_evidence(g, samples)
    test_no_weak_edges(g)
    test_feature_sentinels(samples)
    test_printed_inequalities(samples)
    test_plain_reading_not_degenerate(samples)
    test_emphasis_balanced(samples)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{n_pass}/{len(_RESULTS)} 通过")

    if args.emit:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.emit)), exist_ok=True)
        with open(args.emit, "w", encoding="utf-8") as f:
            json.dump({"n_pass": n_pass, "n_total": len(_RESULTS),
                       "n_samples": len(samples),
                       "gates": [{"name": n, "passed": bool(ok), "message": m}
                                 for n, ok, m in _RESULTS],
                       "details": _DETAILS}, f, ensure_ascii=False, indent=2,
                      default=float)
        print(f"→ {args.emit}")
    sys.exit(0 if n_pass == len(_RESULTS) else 1)


if __name__ == "__main__":
    main()
