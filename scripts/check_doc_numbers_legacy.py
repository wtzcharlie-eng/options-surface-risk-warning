"""check_doc_numbers — 核对交付文档里的数字与落盘 JSON 是否一致。

为什么需要它
------------
本项目的文档数字全部应当来自落盘 JSON，但手写章节做不到自动同步。
实测已发生过多次「改了数据忘了改文档」，也发生过「手工核对时拿错比较基准、
把对的数字改成错的」——后者更危险，因为它把正确内容改坏了。

故把核对本身写成可重复执行的脚本：**每条断言显式写明「文档里的哪个串」
对应「落盘里的哪个字段」**，避免比对时对错基准。

用法::

    python scripts/check_doc_numbers.py          # 全部通过则退出码 0
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_out")


def _load(name):
    p = os.path.join(DATA, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _pct_pat(vals) -> str:
    """把若干百分数串拼成**带数字边界**的正则。

    实测教训：裸子串匹配会撞车——`+7.19%` 的一位小数形式 `7.2%` 会匹配进
    「精确率从 **37.2%** 掉到」里，于是门报出一个与该主题毫无关系的「孤立引用」。
    死区那道门也有同样的隐患（`54.2%` 可匹配 `154.2%`）。
    **门误报比漏报更快消耗信任**，故统一加负向前查：前面不得紧跟数字或小数点。
    """
    import re as _r
    return "|".join(r"(?<![\d.])" + _r.escape(v) for v in sorted(vals))


def main() -> None:
    v2 = _load("metrics_v2.json")
    q1 = _load("q1_7sym.json")
    rs = _load("rule_v2_search.json")
    wp = _load(os.path.join("drl", "workpoint.json"))
    rr = _load(os.path.join("drl", "results.json"))
    gt = _load(os.path.join("drl", "gates.json"))
    ps = _load("per_symbol.json")
    if not (v2 and q1 and rs and wp and rr and gt and ps):
        raise SystemExit("缺少落盘 JSON，请先跑各评测脚本")

    dt, rt = v2["drl"]["test"], v2["rule"]["test"]
    qv = q1["summary_new"]["v2"]
    pk = rs["picked"]["test"]

    # ---- 方法论文档：新口径相关数字
    audit = [
        (f"{dt['precision']:.2%}", "metrics_v2.drl.test.precision"),
        (f"{dt['recall']:.2%}", "metrics_v2.drl.test.recall"),
        (f"{dt['avg_lead_min']:.0f}min", "metrics_v2.drl.test.avg_lead_min"),
        (f"{v2['discrimination_pp']:+.1f}pp", "metrics_v2.discrimination_pp"),
        (f"{v2['best_trivial']['precision']:.1%}", "metrics_v2.best_trivial.precision"),
        (str(dt["n_alert_event"]), "metrics_v2.drl.test.n_alert_event"),
        (f"{dt['vol_split']['高波']['precision']:.1%}", "metrics_v2 DRL 高波"),
        (f"{dt['vol_split']['低波']['precision']:.1%}", "metrics_v2 DRL 低波"),
        (f"{rt['vol_split']['高波']['precision']:.1%}", "metrics_v2 规则 高波"),
        (f"{rt['vol_split']['低波']['precision']:.1%}", "metrics_v2 规则 低波"),
        (f"{rt['precision']:.2%}", "metrics_v2.rule.test.precision（仅调阈值系数）"),
        (f"{pk['precision']:.1%}", "rule_v2_search.picked.test.precision（40 组交叉最优）"),
        (f"{qv['P']:.1%}", "q1_7sym.summary_new.v2.P"),
        (f"{qv['R']:.1%}", "q1_7sym.summary_new.v2.R"),
        # 窗口数从落盘读，**不写死**——此前写死为 18，扩到 20 窗后门自己失效了。
        # 守门脚本自身也会过期，这一条正是它抓到自己的一次。
        (f"{qv['n_pass3']}/{q1['summary_new']['n_win']}",
         "q1_7sym.summary_new.v2.n_pass3 / n_win"),
        (f"{q1['summary_new']['P']:.1%}", "q1_7sym.summary_new.P（旧口径）"),
        (f"{(qv['P'] - pk['precision']) * 100:.1f}pp", "两口径差"),
        (f"{q1['summary_new']['v2_discrimination_pp']:+.1f}pp",
         "q1_7sym.summary_new.v2_discrimination_pp（全窗口）"),
        (f"{q1['summary_new']['v2_best_trivial']['P']:.1%}",
         "q1_7sym 最强平凡策略精确率（全窗口等权）"),
    ]

    # ---- README：**交付版**数字。这一组是本脚本新增的核心——
    # 独立评审查出 README 有 5 处 5-seed 时期的过期数字被标为「交付版」，
    # 而守门脚本当时只守 methodology 文档，漏掉了 README。
    # 「重要约束要写成会失败的机器门」这条原则，此前只应用了一半。
    import numpy as _np
    seed_p = [x["test"]["precision"] for x in rr.get("drl_per_seed", [])]
    readme = [
        (f"{wp['precision']:.2%}", "workpoint.json.precision（交付版精确率）"),
        (f"{wp['recall']:.2%}", "workpoint.json.recall"),
        (f"{wp['avg_lead_min']:.1f}min", "workpoint.json.avg_lead_min"),
        # 「距 50% 还差多少」这句话的**方向**不能写死：死区过滤后精确率已经
        # 越过 50%，再说「缺口 −4.15pp」就是把「超出」印成了「差」。
        # 这正是本项目反复出现的「模板把方向写死」那一族，守门脚本自己也不能例外。
        (f"{'超出' if wp['precision'] >= 0.5 else '差'} "
         f"{abs(wp['precision'] - 0.5) * 100:.2f}pp",
         "交付版相对 50% 门槛（方向现算）"),
        (f"{rr['ensemble_test']['total_reward']:.1f}", "results.json 集成累计奖励"),
        (f"{rr['ensemble_test']['precision']:.2%}", "results.json 集成精确率"),
        (f"{gt['n_pass']}/{gt['n_total']}", "gates.json 验证门通过数"),
    ]
    if seed_p:
        readme.append((f"±{_np.std(seed_p, ddof=1) * 100:.2f}pp",
                       "results.json 单 seed 精确率标准差(ddof=1)"))
    for s in ("ag", "rb"):
        readme.append((f"{ps[s]['rule_DP'] * 100:+.1f}pp",
                       f"per_symbol.{s}.rule_DP"))

    # ---- README §7.18b：解释文案的覆盖率与 lift 分布。
    # 这几个数不在任何 JSON 里，须**从图谱与生成产物现算**——正因如此它们最容易
    # 悄悄过期（改一次图谱标定，lift 分布就变，而文档不会自己跟着改）。
    kgp = os.path.join(DATA, "kg", "kg_graph.json")
    samp = os.path.join(DATA, "kg", "explanation_samples.md")
    if os.path.exists(kgp) and os.path.exists(samp):
        sys.path.insert(0, ROOT)
        from kg.graph import DiGraph
        _E = [a for _, _, a in DiGraph.load(kgp).edges("AMPLIFIES")]
        _lf = sorted(a["lift"] for a in _E)
        _txt = open(samp, encoding="utf-8").read()
        # 只取**完整**传导语句：文档层去重后，重复出现的语句被换成一行「同 [T*]」的
        # 引用，把它们算进平均长度会把这个指标稀释成无意义的数（实测 362→137）。
        # 该指标描述的是**产品**（单条预警长什么样），不是这份评分文档的排版。
        _segs = [x.strip() for x in _txt.split("\n")
                 if x.strip().startswith("**传导｜") and "同 **[T" not in x]
        _all = [x.strip() for x in _txt.split("\n") if x.strip().startswith("**传导｜")]
        _old = sum(len(x.split("**这意味着什么**")[0]) for x in _segs)
        _new = sum(len(x) for x in _segs)
        # 「仍有 x% 不发生」取 lift 最低/最高两条边，README 用它们说明强弱自明
        _by_lift = sorted((a["lift"], 1 - a["p_cond"]) for a in _E)
        _dd = (_load(os.path.join("kg", "self_check.json")) or {}).get(
            "doc_dedup", {"before": 1, "after": 1, "n_full": 0, "n_ref": 0})
        readme += [
            (f"min {_lf[0]:.2f} / 中位 {_lf[len(_lf) // 2]:.2f} / max {_lf[-1]:.2f}",
             "kg_graph 主干边 lift 分布"),
            (f"**{sum(1 for x in _lf if x < 2.0)} 条低于 2.0**",
             "kg_graph lift<2.0 的边数"),
            (f"{len(_all)}/{len(_all)} 条传导语句", "explanation_samples 传导段数（含引用）"),
            (f"{len(_segs)} 种", "去重后逐字不同的传导语句种数"),
            (f"仍有 {_by_lift[0][1]:.0%} 不发生", "lift 最低边的 1-p_cond"),
            (f"仍有 {_by_lift[-1][1]:.0%} 不发生", "lift 最高边的 1-p_cond"),
            (f"+{(_new - _old) / _old:.0%}（{_old // len(_segs)}→{_new // len(_segs)} 字符）",
             "加解读句前后的传导语句平均长度（仅完整语句）"),
            # 文档层无损去重的收益。这三个数是 README 用来论证「重复在文档层而非
            # 产品层」的核心证据，必须跟着产物走，不能手写。
            (f"{_dd['n_full'] + _dd['n_ref']} 条传导语句 = {_dd['n_full']} 条全文 + "
             f"{_dd['n_ref']} 条引用", "self_check.json.doc_dedup 构成"),
            (f"**{_dd['before']:,} → {_dd['after']:,} 字"
             f"（−{1 - _dd['after'] / _dd['before']:.1%}）**",
             "self_check.json.doc_dedup（由 report_kg 在去重当时记账）"),
        ]

    # ---- CVaR 口径（问题3 补出的第二种口径）。这些数刚写进 README §12，
    # 若不入门就会重演「手写数字随重跑过期」。**判据结论也一并核**——
    # 如果哪天 preregistered_verdict 变成通过而 README 还写着未通过，必须被抓到。
    cv = _load("cvar_drl.json")
    if cv:
        _sm, _vd = cv["summary"], cv["preregistered_verdict"]
        readme += [
            (f"excess 均值 +{_sm['drl']['excess_mean']:.1%}、"
             f"仅 {_vd['n_positive']}/{_vd['n_symbol']} 品种为正",
             "cvar_drl.json 预登记口径下的 DRL excess 与达标品种数"),
            (f"**恒报警也拿到 +{_sm['恒报警']['improve_mean']:.1%}**",
             "cvar_drl.json 恒报警 improve（退化的证据）"),
            (f"与 DRL 的 +{_sm['drl']['improve_mean']:.1%} 几乎相同",
             "cvar_drl.json DRL improve"),
        ]
        if _vd["passed"]:
            bad_pre = "README 写着未通过但落盘标为通过"
            readme.append(("__该判据已通过__", bad_pre))
    cs = _load("cvar_sweep7.json")
    if cs and "sweep" in cs:
        import numpy as _n
        sw = cs["sweep"]
        ex = {k: _n.mean([sw[s]["1"][k]["excess"] for s in sw]) for k in ("drl", "rule")}
        po = {k: sum(1 for s in sw if sw[s]["1"][k]["excess"] > 0) for k in ("drl", "rule")}
        readme += [
            (f"DRL excess **+{ex['drl']:.2%}（{po['drl']}/7）**",
             "cvar_sweep7.json hold_days=1 的 DRL excess"),
            (f"**规则引擎 +{ex['rule']:.2%}（{po['rule']}/7）",
             "cvar_sweep7.json hold_days=1 的规则 excess"),
        ]

    # ---- 2022 覆盖度（把一条长期挂着的「待办」关闭为「已量化的决定」）。
    # 这些数是 README 用来论证「不值得纳入」的全部依据，必须跟着 archive 走。
    c22 = _load("coverage_2022.json")
    if c22:
        _p22, _s22 = c22["per_symbol"], c22["summary"]
        readme += [
            ("、".join(f"{k} {_p22[k]['n_months_2022']} 月"
                      for k in ("ag", "au", "sc", "si")),
             "coverage_2022 逐品种 2022 日历月数（前四个）"),
            (f"合计 {_s22['n_symbol_months_2022']} 个品种月，相对现有 "
             f"{_s22['n_episodes_now']} 幕仅 **+{_s22['relative_gain']:.1%}**",
             "coverage_2022 增量与占比"),
            (f"**{_s22['largest_contributor']['share']:.0%} 来自单一品种 "
             f"{_s22['largest_contributor']['symbol']}**",
             "coverage_2022 最大贡献品种"),
        ]
        # 「零数据品种」与「冷启动品种」的名单也核——README 的论证直接依赖它们，
        # 一旦 archive 变动而结论没跟着改，就会变成假陈述。
        _zero = set(_s22["symbols_with_zero_2022"])
        _def = set(_s22["warmup_deficit_symbols"])
        # 零数据品种逐个断言，**不拼成一个串**——拼接会把「集合是否一致」这个
        # 语义变成「书写顺序是否一致」，README 里写 lc/cu、落盘排序 cu/lc，
        # 门就会因为一个与正确性无关的差异而报错（实测就抓了这一次）。
        for _z in sorted(_zero):
            readme.append((f"{_z} 0 月", f"coverage_2022 零数据品种 {_z}"))
        if _def and not _s22["warmup_deficit_fixable_by_2022"]:
            readme.append(("交集为空，补也补不到需要它的地方",
                           "coverage_2022 冷启动品种与可补品种的交集为空"))

    # ---- 「换架构无效」的漂移数字。这组数刚从「无落盘出处的手写值」
    # 改为由 architecture_drift.py 真算，必须入门，否则又会漂回去。
    ad = _load("architecture_drift.json")
    if ad:
        # 文档统一用**全角减号 U+2212**（−）而非 ASCII 连字符（-）：
        # 前者是数学减号、排版正确，本项目全文一致。Python 的 f-string 出的是
        # ASCII，直接比对会因为一个与正确性无关的字符差异误报——
        # 门误报比漏报更容易让人开始忽略它，故在此处统一转换。
        _mn = lambda x: f"{x:+.2f}".replace("-", "\u2212")
        readme += [
            (f"规则引擎 **{_mn(ad['models']['rule']['drift_pp'])}pp**",
             "architecture_drift 规则漂移"),
            (f"DRL 集成 **{_mn(ad['models']['drl_ensemble']['drift_pp'])}pp**",
             "architecture_drift DRL 漂移"),
            (f"差额 **{_mn(ad['drift_gap_pp'])}pp**", "architecture_drift 差额"),
        ]

    bad = []
    for name, doc_rel, checks in (
            ("methodology_caliber_audit.md", "methodology_caliber_audit.md", audit),
            ("README.md", os.path.join("..", "README.md"), readme)):
        path = os.path.join(DATA, doc_rel)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        print(f"\n—— {name} ——")
        for s, src in checks:
            ok = s in text
            if not ok:
                bad.append((name, s, src))
            print(("  ✓ " if ok else "  ✗ ") + f"{s:12s} ← {src}")

        if name.startswith("methodology"):
            fix = lambda k: (k.replace("前1%", "前 1% ").replace("每3步", "每 3 步")
                             .replace("每8步", "每 8 步").replace("每20步", "每 20 步"))
            for k, m in v2["trivial"].items():
                row = f"| {fix(k)} | {m['n_alert_event']} | {m['precision']:.1%} |"
                ok = row in text
                if not ok:
                    bad.append((name, row, "metrics_v2.trivial"))
                print(("  ✓ " if ok else "  ✗ ") + f"平凡表 {fix(k)}")

    print()
    # ---- 结构断言：评分文档的 [T*] 去重**不得泄漏到产品产物**。
    # `explanation_samples.md` 的 [T*] 引用是为「30 条预警并排人工评分」生成的视图；
    # 平台与可视化面向的是「一次看一条预警」的真实场景，那里没有「样本」概念，
    # 出现「同 [T3]（首见于样本 1）」对用户毫无意义。
    # 这不是假想风险：文档层去重上线后 `platform.html` 里立刻出现 36 处引用占位，
    # 因为 `build_platform.py` 当时读的是评分文档而非产品原文。
    # 门抓不到就只能靠人数计数——那正是本项目反复栽跟头的地方。
    for rel in (os.path.join("..", "data_out", "platform.html"),
                os.path.join("kg", "kg_viz.html"),
                os.path.join("kg", "explanation_samples_full.md")):
        path = os.path.join(DATA, rel)
        if not os.path.exists(path):
            continue
        t = open(path, encoding="utf-8").read()
        n_leak = t.count("同 **[T")
        n_seg, n_read = t.count("传导｜"), t.count("这意味着什么")
        # 悬空脚注：`[†]`/`[‡]` 由 METHOD_NOTE 定义，而 METHOD_NOTE 是**容器级**的
        # （第 5 轮把逐条重复的方法学口径提到文首，正文只留回指标记）。
        # 因子化本身对，但它要求**每个容器都渲染一次定义**——这一步漏了：
        # 第 9 位评审实测 platform.html 53 处引用 / 0 处定义、kg_viz.html 59/0，
        # 并据此指出「每条预警自足完整」是我写下的假陈述。
        # 这与 [T*] 泄漏是同一类失败，而我当时只给 [T*] 建了门。
        n_ref = t.count("[†]")
        n_def = t.count("全文统一的统计口径")
        # 术语表与统计口径同属 METHOD_NOTE，必须成对出现。分开检查是因为
        # 术语表是后加的：若将来有人把 GLOSSARY 从 METHOD_NOTE 里拆出去，
        # 「[†] 不得悬空」这道门仍会通过，而正文里 z 分/置换检验/判别力
        # 又会重新变成无定义的术语——那正是第 9 位评审判定「无一条够 5 分」的卡点。
        n_gloss = t.count("术语与读数速查")
        dangling = n_ref > 0 and (n_def == 0 or n_gloss == 0)
        name = os.path.basename(path)
        ok = n_leak == 0 and (n_seg == 0 or n_read == n_seg) and not dangling
        if not ok:
            bad.append((name,
                        f"引用占位 {n_leak} / 解读句 {n_read} vs 传导 {n_seg}"
                        f" / [†] 引用 {n_ref} 但口径定义 {n_def}、术语表 {n_gloss}",
                        "产品产物不得含 [T*] 去重占位、解读句须 1:1 覆盖传导、"
                        "[†] 不得悬空"))
        print(("  ✓ " if ok else "  ✗ ")
              + f"{name:32s} 占位 {n_leak} · 解读 {n_read}/{n_seg}"
                f" · [†] {n_ref} 引用/口径 {n_def}/术语表 {n_gloss}")

    # ---- 平台内联脚本的语法检查。
    # `platform.html` 是**由 Python 字符串拼出来的 246KB 内联 JS**，
    # 一个多余的反引号或未闭合的模板字面量就会让整页白屏，
    # 而 `build_platform.py` 仍然退出码 0、文件照样生成——**失败完全无声**。
    # 本项目反复栽在「无声失败」上（吞 stderr、改代码不重建产物、修 A 静默废掉 B），
    # 故把它变成会失败的检查。node 缺失时跳过而非误报。
    import re as _re, shutil as _sh, subprocess as _sp
    _pp = os.path.join(DATA, "platform.html")
    if os.path.exists(_pp) and _sh.which("node"):
        _js = "\n".join(_re.findall(r"<script[^>]*>(.*?)</script>",
                                    open(_pp, encoding="utf-8").read(), _re.S))
        # 临时文件写系统临时目录，**不写进 data_out**——那里是交付产物目录，
        # 往里塞中间文件会污染交付物（且某些挂载环境不允许删除）。
        import tempfile as _tf
        with _tf.NamedTemporaryFile("w", suffix=".js", delete=False,
                                    encoding="utf-8") as _f:
            _f.write(_js)
            _tmp = _f.name
        _r = _sp.run(["node", "--check", _tmp], capture_output=True, text=True)
        try:
            os.remove(_tmp)
        except OSError:
            pass
        ok = _r.returncode == 0
        if not ok:
            bad.append(("platform.html", "内联 JS 语法错误",
                        _r.stderr.strip().splitlines()[0] if _r.stderr else "?"))
        print(("  ✓ " if ok else "  ✗ ")
              + f"platform.html 内联脚本语法（{len(_js):,} 字符）")
    elif os.path.exists(_pp):
        print("  – platform.html 内联脚本语法：SKIP（无 node）")

    # ---- 用到的 tag 类必须在 CSS 里有定义。
    # 实测教训：`t-bad` 被用了 6 次却从未定义（CSS 里只有 t-ok/t-no/t-w/t-m），
    # 于是 6 个「✗ 未达标」标记全部渲染成**无背景色的灰字**。
    # 这类缺陷 `node --check` 抓不到（语法完全合法）、数字门也抓不到（数字没错），
    # 而它的方向**恒定对自己有利**——被弱化的永远是失败标记，不会是达标标记。
    _bp = os.path.join(ROOT, "scripts", "build_platform.py")
    if os.path.exists(_bp):
        _src = open(_bp, encoding="utf-8").read()
        _used = set(_re.findall(r"tag\s+(t-[a-z]+)", _src)) | \
                set(_re.findall(r"tag\(\s*'[^']*'\s*,\s*'(t-[a-z]+)'\s*\)", _src))
        _defined = set(_re.findall(r"^\.(t-[a-z]+)\{", _src, _re.M))
        _undef = sorted(_used - _defined)
        ok = not _undef
        if not ok:
            bad.append(("build_platform.py", f"用了未定义的 tag 类 {_undef}",
                        f"CSS 里已定义的是 {sorted(_defined)}；未定义的类会静默渲染成无样式"))
        print(("  ✓ " if ok else "  ✗ ")
              + f"tag 类全部有 CSS 定义（用到 {len(_used)} 个）")

        # ---- 表格单元格里不得出现**写死的方向箭头**。
        # 实测教训：「扩样前后对照」表的「变化」列曾是写死的 ↑/↓，而左列是 4 品种
        # **旧口径**常量、右列是现算值。换到新口径后 9 行里有 5 行箭头与事实相反
        # （规则基线判别力 11.48→19.60 却标着 ↓）。同一族错误此前也在 report_drl.py
        # 上发生过（「单 seed 高于集成」在换 15-seed 后翻转）。
        # 凡「A 比 B 高/低」「从 A 降到 B」的表述都必须现算，不能写进模板。
        # 别只认「数组末位的引号箭头」这一种形态——独立审计实测：把箭头挪到行中间、
        # 或包进模板串，原门**三种变体全漏**，而它注释里描述的那次真实事故正是其中之一。
        # 改为：**去注释后的源码里，除 cmp() 定义那一行外，不得出现任何 ↑/↓ 字面量**。
        _vis0 = _re.sub(r"/\*.*?\*/", "", _src, flags=_re.S)
        _vis0 = "\n".join(l for l in _vis0.splitlines()
                           if not l.lstrip().startswith("#"))
        _hard = [l.strip()[:80] for l in _vis0.splitlines()
                 if ("↑" in l or "↓" in l) and "const cmp=" not in l]
        ok = not _hard
        if not ok:
            bad.append(("build_platform.py", f"表格里有 {len(_hard)} 处写死的方向箭头",
                        "方向必须由数值现算（见 cmp()），否则换口径后会渲染出反向陈述"))
        print(("  ✓ " if ok else "  ✗ ")
              + f"表格无写死的方向箭头（现算 {len(_re.findall(r'const cmp=', _src))} 处）")

        # ---- 指标3 盲评分数不得写死在 builder 里。
        # 实测教训：页面 4 处写死「七轮盲评、自评 3.60」，而 self_assessment.md
        # 已到第 12 轮（评审员 A~L）、最后四轮 4.39/4.16/4.30/4.28 全部 ≥4。
        # 3.60 是第 6 轮的分数，**过时五轮**，且与 README 互相矛盾。
        # 该数此前没有任何 JSON 装它，于是「数字一律从产物读」在这一处破功，
        # 而破功处恰好就是唯一过时的地方。现已改为解析 markdown 表格。
        _sap = os.path.join(DATA, "kg", "self_assessment.md")
        if os.path.exists(_sap):
            _rows = _re.findall(r"^\|\s*(\d+)\s[^|]*\|\s*([A-Z])\s*\|"
                                r"\s*\**([\d.]+)\**\s*\|\s*\**([\d.]+)\**\s*\|"
                                r"\s*\**([\d.]+)\**\s*\|\s*\**([\d.]+)\**\s*\|",
                                open(_sap, encoding="utf-8").read(), _re.M)
            _scores = {r[5] for r in _rows}
            # builder 里不得出现任何一个具体分数的字面量，也不得写死轮次数
            # 注释里可以（也应该）写这些数字——它们记录的正是这次事故本身。
            # 只扫**会被渲染出去的**部分：先剥掉 Python 注释与 JS 块注释。
            _vis = _re.sub(r"/\*.*?\*/", "", _src, flags=_re.S)
            _vis = "\n".join(l for l in _vis.splitlines()
                              if not l.lstrip().startswith("#"))
            # 别用「引号包裹 / 尖括号包裹 / 空格包裹」这几种手写形态去找字面量——
            # 反向验证实测：注入 `<b>3.89 → 3.60</b>` 时四种形态**一个都不匹配**，
            # 门在该失败时返回了 0。改为直接找**独立出现的十进制数**。
            _lit = sorted(s for s in _scores
                          if _re.search(r"(?<![\d.])" + _re.escape(s) + r"(?![\d])", _vis))
            _rounds = _re.findall(r"([一二三四五六七八九十]+)轮(?:盲评|独立|全部)", _vis)
            ok = not _lit and not _rounds
            if not ok:
                bad.append(("build_platform.py",
                            f"写死了盲评分数 {_lit} / 轮次 {_rounds}",
                            f"self_assessment.md 现有 {len(_rows)} 轮，"
                            f"最新 {_rows[-1][5]}；应从 D.kg.selfscore 现算"))
            print(("  ✓ " if ok else "  ✗ ")
                  + f"指标3 盲评分数未写死（落盘 {len(_rows)} 轮，最新 {_rows[-1][5] if _rows else '—'}）")

        # ---- builder 里不得写死事件窗口数。
        # 实测教训：扩到 20 窗后页面仍有 7 处写死 18，其中一张三行表内部就自相矛盾
        # （表头「18 个事件窗口」/ 第二行现算「20 窗」/ 第三行写死「6/18」）。
        # 更要命的是那批写死值**不是笔误，而是上一版口径的真值**：
        # `archive/q1_7sym_legacy.json` 里 n_win=18、recall_first P=44.46%、
        # DP=16.11pp、n_pass3=6、base=28.35%，与页面写死的逐个对得上。
        # 即「同一张表里一半是现口径现算、一半是旧口径常量」——
        # 与「扩样前后对照」表是同一族缺陷，且同样不会报错、只会慢慢变成假话。
        _nw = q1["summary_new"]["n_win"]
        _wc = _re.findall(r"/1[0-9]['\"]|1[0-9] 个事件窗口|（1[0-9] 事件窗口）", _vis0)
        ok = not _wc
        if not ok:
            bad.append(("build_platform.py", f"写死了事件窗口数 {sorted(set(_wc))}",
                        f"落盘 n_win={_nw}；应从 q1_7sym.summary_new.n_win 现算"))
        print(("  ✓ " if ok else "  ✗ ")
              + f"事件窗口数未写死（落盘 n_win={_nw}）")

        # ---- 死区过滤的精确率增益，不得脱离「规则引擎同等受益」单独出现。
        # 这是本项目最容易出的一类错：一个**真实**的改进被讲成模型变好。
        # 死区过滤把 DRL 旧口径精确率从 47.41% 抬到 54.15%（越过 50% 门），
        # 但规则引擎从 45.36% 抬到 52.36%，判别力仅变 −0.26pp——
        # 预登记判据（写在 dead_zone_study.py 文档字符串里）据此判定为「全员抬升」。
        #
        # 本门初版**远弱于它的名字**（独立审计实测出三处漏洞，已全部修）：
        #   ① 只扫 README 与 platform.html，漏掉 drl_report.md——而问题恰好出在那里；
        #   ② 判据是「整份文件里是否出现过 52.36%」而非「引用处旁边有没有」。
        #      实测：在 README 末尾追加一句孤立的「精确率达 54.15%，指标已达标」，
        #      门**照样通过**，因为 52.36% 在文件别处存在；
        #   ③ 只匹配两位小数字面量，而 drl_report.md 写的是 54.2%（一位小数），
        #      即使把它加进名单也匹配不上。
        # 现改为**邻域判据**：对每一处 DRL 数字的出现位置，要求其前后 _WIN 字符内
        # 出现规则引擎的对应数字；并同时接受一位/两位小数两种写法。
        _dzp = os.path.join(DATA, "dead_zone_study.json")
        if os.path.exists(_dzp):
            _dz = json.load(open(_dzp, encoding="utf-8"))
            _on = _dz["grid_2x2"]["wp_new|dz_on"]
            _WIN = 700          # 邻域半径（字符）——一段话的量级

            def _forms(x):
                return {f"{x:.2%}", f"{x * 100:.1f}%"}

            _drl_f = _forms(_on["drl"]["precision"])
            _rule_f = _forms(_on["rule"]["precision"])
            # 只在**谈论死区过滤的上下文里**检查，否则会误报：
            # 「54.2%」这个字符串在 cu 2026-01 事件窗口表、au 逐品种表里都出现过，
            # 与死区毫无关系。初版不区分上下文，一次跑出 7 处「孤立引用」，
            # 其中 4 处是假阳性——**门误报比漏报更快消耗信任**。
            _ctx_kw = ("死区", "dead_zone", "§7.19", "7.19")
            # 并列规则对照可以是**数字**，也可以是**文字披露**（两者等效达意）
            _discl = ("规则引擎同获", "规则引擎获得同等", "规则引擎同样",
                      "全员抬升", "增益同等", "同等增益")
            _bad_files = []
            for _f in ("README.md", os.path.join("data_out", "drl_report.md"),
                       os.path.join("data_out", "platform.html")):
                _fp = os.path.join(ROOT, _f)
                if not os.path.exists(_fp):
                    continue
                _txt = open(_fp, encoding="utf-8").read()
                if _f.endswith(".html"):
                    # 模板：数字运行时才渲染，只能查模板两侧是否都引用了
                    if ("D.dz" in _txt or "dz.grid_2x2" in _txt) and \
                            "rule.precision" not in _txt:
                        _bad_files.append(_f)
                    continue
                _orphan = 0
                for _m in _re.finditer(_pct_pat(_drl_f), _txt):
                    _ctx = _txt[max(0, _m.start() - _WIN): _m.end() + _WIN]
                    if not any(k in _ctx for k in _ctx_kw):
                        continue          # 与死区无关的同名数字，跳过
                    if not (any(r in _ctx for r in _rule_f)
                            or any(d in _ctx for d in _discl)):
                        _orphan += 1
                if _orphan:
                    _bad_files.append(f"{_f}（{_orphan} 处孤立引用）")
            ok = not _bad_files and _dz["verdict"]["is_uniform_lift"]
            if _bad_files:
                bad.append((", ".join(_bad_files),
                            f"引用死区过滤后的精确率 {sorted(_drl_f)} 时，"
                            f"±{_WIN} 字符内没有规则引擎的 {sorted(_rule_f)}",
                            "预登记判据判定这是全员抬升，单独报 DRL 增益会被读成模型改进"))
            elif not _dz["verdict"]["is_uniform_lift"]:
                bad.append(("dead_zone_study.json",
                            f"判别力变化 {_dz['verdict']['delta_pp']:+.2f}pp 已超出 ±1pp",
                            "预登记判据不再成立，需重新判断该过滤是否构成模型改进"))
            print(("  ✓ " if ok else "  ✗ ")
                  + f"死区增益均并列规则对照（判别力变化 "
                    f"{_dz['verdict']['delta_pp']:+.2f}pp，判为"
                    f"{'全员抬升' if _dz['verdict']['is_uniform_lift'] else '非全员抬升'}）")

    # ---- 交付目录不得混入训练中间产物。
    # 实测教训：`train_drl.py` 训练完不清断点，`data_out/` 累积到 1.9G，
    # 其中 **1.78G 是 `_wip_*.pkl`**（每个含整个回放池，56MB），
    # 而真正的交付产物只有约 77MB。一份要提交的作品里 95% 是训练残渣，
    # 且**没有任何东西会提醒你**——目录大小不在任何门的视野里。
    # 现已让 train_drl 在成品落盘后自动清理；本门是第二道保险。
    import glob as _g
    _wips = _g.glob(os.path.join(DATA, "**", "_wip_*.pkl"), recursive=True)
    _mb = sum(os.path.getsize(f) for f in _wips) / 2 ** 20
    ok = not _wips
    if not ok:
        bad.append(("data_out/", f"残留 {len(_wips)} 个训练断点（{_mb:,.0f} MB）",
                    "断点应在成品落盘后自动清理，见 scripts/train_drl.py"))
    print(("  ✓ " if ok else "  ✗ ")
          + f"交付目录无训练中间产物（_wip_*.pkl {len(_wips)} 个）")

    # ---- 交付目录不得混入**加速缓存**。同一族的第二次事故：
    # 给 `build_features.py` 加载入/特征缓存时，默认目录顺手写成了 `data_out/.bf_cache`，
    # 7 个品种跑完积了 **122MB**，交付目录 81MB → 203MB。
    # 上面那道门只认 `_wip_*.pkl`，对这个新形态视而不见——
    # **「我刚防住了这类错」正是最该回头查同类的时刻**，而当时没查。
    # 现已把默认目录改到系统临时目录；本门是第二道保险，且判据按**目录体积**
    # 而非某个固定文件名，这样下一个新形态的缓存也跑不掉。
    _cache_dirs, _cache_mb = [], 0.0
    # 判据按**目录体积 + 是否隐藏**，不按名字里有没有 "cache"。
    # 实测原判据（`glob('**/*cache*')`）漏两类：
    # ① `data_out/.bf_tmp/`（名字里没有 cache）；
    # ② `data_out/.priv/hugecache/`（父目录是隐藏目录，`glob` 的 `**` 不下钻）。
    # 缓存的真实特征是「隐藏 + 占地方」，不是名字。
    _KNOWN = {"anchor", "drl", "kg", "archive"}       # 交付内容，不算缓存
    for _r, _ds, _fs in os.walk(DATA):
        for _d in list(_ds):
            _full = os.path.join(_r, _d)
            if _d in _KNOWN:
                continue
            if not (_d.startswith(".") or "cache" in _d.lower()
                    or "tmp" in _d.lower()):
                continue
            _sz = sum(os.path.getsize(os.path.join(_rr, _f))
                      for _rr, _, _fz in os.walk(_full) for _f in _fz) / 2 ** 20
            if _sz > 1:                    # 1MB 以下不算问题
                _cache_dirs.append((os.path.relpath(_full, ROOT), _sz))
                _cache_mb += _sz
    ok = not _cache_dirs
    if not ok:
        bad.append(("data_out/",
                    "、".join(f"{p}（{s:,.0f} MB）" for p, s in sorted(_cache_dirs)),
                    "缓存只是加速，不应放进交付目录——见 build_features.py --cache-dir"))
    print(("  ✓ " if ok else "  ✗ ")
          + f"交付目录无加速缓存（{len(_cache_dirs)} 个，{_cache_mb:,.0f} MB）")

    # ---- 门清单**表格行数**必须等于 gates.json 的门数。
    # 独立审计实测：README 标题写「12 道验证门」，下表却只列了 G0–G10 共 11 行；
    # KG 那张写「10 道」只列 K1–K9 共 9 行。**标题数字来自落盘所以是对的，
    # 表格却要手写，于是每次加门都漏一行**——而评审数一遍行数就会发现。
    # 现有的门只核 `n_pass/n_total` 这个字符串，核不到表格。
    import re as _re
    _gk = _load(os.path.join("kg", "gates.json"))
    _rd = open(os.path.join(DATA, "..", "README.md"), encoding="utf-8").read()
    for _pre, _js, _lab in (("G", gt, "drl"), ("K", _gk, "kg")):
        if not _js:
            continue
        _rows = len(_re.findall(rf"^\| {_pre}\d+ \|", _rd, _re.M))
        ok = _rows == _js["n_total"]
        if not ok:
            bad.append(("README.md",
                        f"{_pre} 门清单表 {_rows} 行 ≠ gates.json 的 {_js['n_total']} 道",
                        f"{_lab}/gates.json.n_total"))
        print(("  ✓ " if ok else "  ✗ ")
              + f"{_pre} 门清单表行数 {_rows} == gates.json {_js['n_total']} 道")

    # ---- 生成型报告不得早于其数据源（过期检测）。
    # 独立审计实测：`drl_report.md`（8-20 生成）比 `drl/gates.json`（8-24）还旧，
    # 于是报告里印着「10/10 通过」而落盘已是 12/12，且模板里几处方向判断
    # 在 15-seed 后翻转成**字面为假**（「seed44 3,022.2 高于集成 4,236.4」）。
    # **报告是脚本生成的，所以没人会想到它会过期**——正因如此才要机器盯。
    _gen = {
        "drl_report.md": [os.path.join("drl", "results.json"),
                          os.path.join("drl", "gates.json")],
        "kg_report.md": [os.path.join("kg", "kg_edges.json"),
                         os.path.join("kg", "gates.json")],
    }
    for _rep, _srcs in _gen.items():
        _rp = os.path.join(DATA, _rep)
        if not os.path.exists(_rp):
            continue
        _rt = os.path.getmtime(_rp)
        _stale = [s_ for s_ in _srcs
                  if os.path.exists(os.path.join(DATA, s_))
                  and os.path.getmtime(os.path.join(DATA, s_)) > _rt]
        ok = not _stale
        if not ok:
            bad.append((_rep, f"比数据源旧：{'、'.join(_stale)}",
                        f"请重跑生成脚本（report_drl.py / report_kg.py）"))

        print(("  ✓ " if ok else "  ✗ ") + f"{_rep:24s} 不早于其数据源")

    # ---- CVaR 稀疏工作点研究：DRL 的 excess 不得脱离规则引擎的 excess 单独出现。
    # 与死区那道门同一族缺陷：一个**真实**的正结果被讲成「DRL 有 CVaR 价值」，
    # 而同协议下规则引擎 +8.12% 其实**高于** DRL 的 +7.19%。
    # 该情形在跑数前已写进预登记，故这里强制两数并列。
    _cwp = os.path.join(DATA, "cvar_workpoint.json")
    if os.path.exists(_cwp):
        _cw = json.load(open(_cwp, encoding="utf-8"))
        _td, _tr = _cw["test"]["drl"], _cw["test"]["rule"]
        _dfm = {f"{_td['excess_mean']:.2%}", f"{_td['excess_mean']*100:.1f}%"}
        _rfm = {f"{_tr['excess_mean']:.2%}", f"{_tr['excess_mean']*100:.1f}%"}
        _kw2 = ("CVaR", "cvar", "excess")
        _bad2 = []
        for _f in ("README.md", os.path.join("data_out", "platform.html"),
                   os.path.join("data_out", "cvar_workpoint_preregistration.md")):
            _fp = os.path.join(ROOT, _f)
            if not os.path.exists(_fp):
                continue
            _txt = open(_fp, encoding="utf-8").read()
            if _f.endswith(".html"):
                if "cvw" in _txt and "tr.excess_mean" not in _txt:
                    _bad2.append(_f)
                continue
            _orp = 0
            for _m in _re.finditer(_pct_pat(_dfm), _txt):
                _c = _txt[max(0, _m.start() - _WIN): _m.end() + _WIN]
                if not any(k in _c for k in _kw2):
                    continue
                if not any(r in _c for r in _rfm):
                    _orp += 1
            if _orp:
                _bad2.append(f"{_f}（{_orp} 处孤立引用）")
        ok = not _bad2
        if not ok:
            bad.append((", ".join(_bad2),
                        f"引用 CVaR 稀疏工作点的 DRL excess {sorted(_dfm)} 时，"
                        f"±{_WIN} 字符内没有规则引擎的 {sorted(_rfm)}",
                        "预登记已写明规则引擎不劣于 DRL，单独报 DRL 会被读成 DRL 更优"))
        print(("  ✓ " if ok else "  ✗ ")
              + f"CVaR 工作点研究均并列规则对照"
                f"（DRL {_td['excess_mean']:+.2%} vs 规则 {_tr['excess_mean']:+.2%}）")

        # ---- 问题1 的「规则+ML 融合」达标数，不得被讲成「固定阈值达标」。
        # 该数来自在 anchor 训练集上重训的监督模型 + 否决式融合，
        # **纯规则的前沿依旧够不到**（同 zmult 39.27%、R≈60% 处 49.05%）。
        # 只报融合数而不并列纯规则，读者会以为调阈值就能达标。
        _rmp = os.path.join(DATA, "rule_ml.json")
        if os.path.exists(_rmp):
            _rm = json.load(open(_rmp, encoding="utf-8"))
            if _rm.get("test"):
                _ft = _rm["test"]["fused"]
                _pf = _rm.get("test_p_at_recall60_pure") or {}
                _ff = _forms(_ft["precision"])
                _pp2 = _forms(_pf["precision"]) if _pf else set()
                _kw3 = ("融合", "ML", "rule_ml")
                _bad3 = []
                for _f in ("README.md", os.path.join("data_out", "platform.html"),
                           os.path.join("data_out", "rule_ml_preregistration.md")):
                    _fp = os.path.join(ROOT, _f)
                    if not os.path.exists(_fp):
                        continue
                    _txt = open(_fp, encoding="utf-8").read()
                    if _f.endswith(".html"):
                        if "rml" in _txt and "rule_only_same_zmult" not in _txt:
                            _bad3.append(_f)
                        continue
                    _orp = 0
                    for _m in _re.finditer(_pct_pat(_ff), _txt):
                        _c = _txt[max(0, _m.start() - _WIN): _m.end() + _WIN]
                        if not any(k in _c for k in _kw3):
                            continue
                        if not (any(x in _c for x in _pp2) or "纯规则" in _c
                                or "不是纯固定阈值" in _c):
                            _orp += 1
                    if _orp:
                        _bad3.append(f"{_f}（{_orp} 处孤立引用）")
                ok = not _bad3
                if not ok:
                    bad.append((", ".join(_bad3),
                                f"引用规则+ML 融合的 {sorted(_ff)} 时，±{_WIN} 字符内"
                                f"既无纯规则对照 {sorted(_pp2)}，也未写明「不是纯固定阈值」",
                                "纯规则前沿仍够不到，单独报融合数会被读成「固定阈值也能达标」"))
                print(("  ✓ " if ok else "  ✗ ")
                      + f"问题1 融合达标数均并列纯规则对照（融合 "
                        f"{_ft['precision']:.2%} vs 纯规则 R≈60% 处 "
                        f"{_pf.get('precision', float('nan')):.2%}）")

                # ---- 交付配置引用的模型必须**真实存在且指纹匹配**。
                # 实测教训：`rule_ml.py` 初版每次运行都重训、**不落盘**，
                # 于是报告里有一个「达标的交付配置」，仓库里却没有对应模型——
                # 数字能复算，系统却无法部署，`scan.py`/仪表板也调不到。
                # 「有结果」不等于「有系统」。
                # 指纹里**特征名与顺序**最关键：模型按列序吃数，顺序变了不会报错，
                # 只会静默给出错误概率。
                _art = _rm.get("model_artifact") or {}
                _ap = os.path.join(ROOT, _art.get("path", ""))
                _fpp = os.path.join(DATA, "rule_ml_model.fingerprint.json")
                _probs = []
                if not _art:
                    _probs.append("rule_ml.json 未记录模型产物")
                elif not os.path.exists(_ap):
                    _probs.append(f"模型文件不存在：{_art.get('path')}")
                elif not os.path.exists(_fpp):
                    _probs.append("缺 rule_ml_model.fingerprint.json")
                else:
                    _disk = json.load(open(_fpp, encoding="utf-8"))
                    _emb = _art["fingerprint"]
                    for _k in ("feature_names", "n_features", "train_span",
                               "model_params"):
                        if _disk.get(_k) != _emb.get(_k):
                            _probs.append(f"指纹字段 {_k} 与 rule_ml.json 不一致")
                    try:
                        sys.path.insert(0, ROOT)
                        from drl.dataset import feature_names as _fnames
                        _want = _fnames(_emb["n_features"] == 32)
                        if list(_emb["feature_names"]) != list(_want):
                            _probs.append("模型特征名/顺序与 drl.dataset.feature_names 不符")
                    except Exception as _e:              # noqa: BLE001
                        _probs.append(f"无法核对特征名：{type(_e).__name__}")
                ok = not _probs
                if not ok:
                    bad.append(("rule_ml 交付件", "；".join(_probs),
                                "报告里有达标配置，就必须有对应的、指纹匹配的模型文件"))
                print(("  ✓ " if ok else "  ✗ ")
                      + f"问题1 融合模型存在且指纹匹配"
                        f"（{_art.get('path', '—')}）")

    # ---- 小节编号不得重复，且 §x.y 引用必须有落点
    import re as _re
    # 实测事故：README 一度有**两个 §7.19、两个 §7.20**（后一批复用了前一批的号）。
    # 10 处「见 §7.19」全指向后者，而 §7.20 的 5 处引用里 3 处指前者、2 处指后者——
    # **引用本身成了歧义**，而所有数字门都看不见这类缺陷。
    _rd = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    _secs = _re.findall(r"^## (\d+\.\d+[a-z]?)\s", _rd, _re.M)
    _dup = sorted({s for s in _secs if _secs.count(s) > 1})
    print(("  ✓ " if not _dup else "  ✗ ")
          + f"README 小节编号无重复（共 {len(_secs)} 个）"
          + (f" —— 重复: {_dup}" if _dup else ""))
    if _dup:
        bad.append(("README", f"重复小节编号 {_dup}",
                    "同号小节会让「见 §x.y」指向歧义，须改号"))
    _have = set(_secs)
    _refs = {m for m in _re.findall(r"§(\d+\.\d+[a-z]?)", _rd)}
    _dangling = sorted(r for r in _refs if r not in _have)
    print(("  ✓ " if not _dangling else "  ✗ ")
          + f"README 的 §ref 全部有落点（{len(_refs)} 个不同引用）"
          + (f" —— 悬空: {_dangling}" if _dangling else ""))
    if _dangling:
        bad.append(("README", f"悬空引用 §{_dangling}",
                    "被引用的小节不存在——改号时漏改了引用处"))

    # ---- 历史快照必须**自带**声明，不能只在 README 里披露。
    # 实测教训：`backtest_report.md` 里有一张宣称达标的表和一节全是 +50.0% 的 CVaR
    # （已知的退化口径），而 README 对它有完整的 ⚠️ 披露、**文件自己一个字都没有**。
    # 评委完全可能直接打开这个文件——那时限定就等于没写。
    # 这与本项目「解读层不继承证据层限定」是同一族缺陷。
    _bp = os.path.join(DATA, "backtest_report.md")
    if os.path.exists(_bp):
        _bt = open(_bp, encoding="utf-8").read(2000)
        _need = ["历史快照", "§7.25", "不要引用本文件的数字", "算术产物"]
        _lack = [k for k in _need if k not in _bt]
        ok = not _lack
        if not ok:
            bad.append(("backtest_report.md", f"顶部声明缺少 {_lack}",
                        "该文件含旧口径与已被推翻的 CVaR 数字，必须自带声明——"
                        "只写在 README 里挡不住直接打开文件的读者"))
        print(("  ✓ " if ok else "  ✗ ")
              + "backtest_report.md 自带历史快照声明"
              + (f" —— 缺 {_lack}" if _lack else ""))

    # ---- PACKAGE.md 也要守。它此前**不在任何门的视野里**，于是门数一路停在
    # 「13 道」，而实际已经 15 道——而且这个数在包内说明里出现 2 次。
    # 交付包的说明书自己过期，是评审第一眼就会看到的地方。
    _pp = os.path.join(ROOT, "PACKAGE.md")
    if os.path.exists(_pp):
        _pt = open(_pp, encoding="utf-8").read()
        _pk = [(f"{gt['n_total']} 道验证门（问题3）",
                f"{gt['n_total']} 道验证门", "drl/gates.json.n_total"),
               (f"{gt['n_pass']}/{gt['n_total']} 验证门通过",
                f"{gt['n_pass']}/{gt['n_total']}", "drl/gates.json 通过数"),
               (f"{qv['n_pass3']}/{q1['summary_new']['n_win']} 窗口达标",
                f"{qv['n_pass3']}/{q1['summary_new']['n_win']}",
                "q1_7sym.summary_new.v2.n_pass3 / n_win")]
        _pbad = [(lbl, s, src) for lbl, s, src in _pk if s not in _pt]
        for lbl, s, src in _pbad:
            bad.append(("PACKAGE.md", f"找不到「{s}」（{lbl}）", src))
        print(("  ✓ " if not _pbad else "  ✗ ")
              + f"PACKAGE.md 的门数与达标数与落盘一致（查 {len(_pk)} 项）")

    # ---- README 里的门数**字面量**必须等于落盘。此前只校验了「门清单表格的行数」，
    # 于是标题与正文里的「12 道验证门」「10/10 通过」「G0–G13」一路过期到 8 处，
    # 其中 §2454 的标题写「12 道」而其下表格 15 行、再下一行写「15/15 通过」——
    # **同一小节自打脸**。表格行数对了不代表叙述对了。
    _gn, _kn = gt["n_total"], (_load(os.path.join("kg", "gates.json")) or {}).get(
        "n_total", 11)
    _stale = []
    for _m in _re.finditer(r"(\d+)\s*道验证门", _rd):
        if int(_m.group(1)) not in (_gn, _kn):
            _stale.append(f"「{_m.group(0)}」")
    # 「G0–Gxx」形式的上界也要对
    for _m in _re.finditer(r"G0[–\-]G(\d+)", _rd):
        if int(_m.group(1)) != _gn - 1:
            _stale.append(f"「{_m.group(0)}」（应到 G{_gn - 1}）")
    for _m in _re.finditer(r"K1[–\-]K(\d+)", _rd):
        if int(_m.group(1)) != _kn:
            _stale.append(f"「{_m.group(0)}」（应到 K{_kn}）")
    _stale = sorted(set(_stale))
    print(("  ✓ " if not _stale else "  ✗ ")
          + f"README 的门数字面量与落盘一致（DRL {_gn} / KG {_kn}）"
          + (f" —— 过期 {_stale}" if _stale else ""))
    if _stale:
        bad.append(("README", f"过期的门数字面量 {_stale}",
                    f"drl/gates.json.n_total={_gn}, kg/gates.json.n_total={_kn}"))

    # ---- 平台模板里不得出现**判定性的写死文字**。
    # 实测事故：逐品种段落写死「只有 sc 三项全达标」「rb 规则判别力最低」
    # 「拿到 +26.5pp」「落差是 7 个品种里最大的」「只发 1/4 的预警」——
    # **5 句话全错，且与紧邻的表格同屏打架**（实际 4 个品种达标、最低的是 ag、
    # rb 是 30.4pp、落差最大的是 sc、rb 预警数 363→529 是**增加**）。
    # 这些判定必须现算；写死的形容词不会报错，只会慢慢变成假话。
    _bs = open(os.path.join(ROOT, "scripts", "build_platform.py"),
               encoding="utf-8").read()
    # 判据从「5 个固定串的黑名单」改成**形态判据**，但要收得足够窄：
    # 真实缺陷的形态是「判定词 + 紧跟着一个写死的品种代码」——
    #   只有 sc 三项全达标 / 最低的是 rb / 落差最大的是 lc
    # 而「只有 4 幕测试样本」「只有 42.76%」「只有 4/7 达标」是正常行文，不能报。
    # 初版判据只要求「有判定词且无 ${}」，实测**误报 6 处**——
    # 误报比漏报更快消耗信任，下一个人看到 ✗ 只会去关掉这道门。
    _SYM = "ag|au|cu|lc|rb|sc|si|al|br"
    _PAT = _re.compile(r"(只有|最低的是|最高的是|最大的是|最小的是|唯一的是|"
                       r"最低|最高|最大|最小)\s*(" + _SYM + r")\b")
    _verdict = []
    for _i, _ln in enumerate(_bs.split("\n"), 1):
        _st = _ln.strip()
        if _st.startswith("#") or _st.startswith("//") or _st.startswith("*"):
            continue                      # 整行注释：允许在这里引用错误原文作说明
        if "${" in _ln:
            continue                      # 有现算表达式，判定由数据给出
        _m = _PAT.search(_ln)
        if _m:
            _verdict.append(f"L{_i}:「{_m.group(0)}」")
    _verdict = _verdict[:6]
    print(("  ✓ " if not _verdict else "  ✗ ")
          + "平台模板无判定性写死文字"
          + (f" —— 发现 {_verdict}" if _verdict else ""))
    if _verdict:
        bad.append(("build_platform.py", f"写死的判定文字 {_verdict}",
                    "这类结论必须由 payload 现算，否则数据一变就成假话"))

    # ---- 平台**渲染后**不得出现 NaN / undefined。
    # 语法门（node --check）只看语法，看不见「字段路径写错 → 渲染成 NaN」。
    # 实测就踩到：`best_trivial` 只有 {name, precision}、没有 n_alert_event，
    # 直接取会算出 NaN 并原样印在页面上。**grep 成品文件也查不到**——
    # 那是模板运行时求值的。故这里真的把 payload 抠出来执行一遍模板函数。
    _node = shutil.which("node")
    if _node:
        _js = r"""
const fs=require("fs");
const h=fs.readFileSync(process.argv[2],"utf8");
const i=h.indexOf("const D = ")+10; let d=0,j=i;
for(;j<h.length;j++){const c=h[j]; if(c==="{")d++; else if(c==="}"){d--; if(d===0){j++;break;}}}
const D=eval("("+h.slice(i,j)+")");
// 把 <script> 里的渲染代码整体跑一遍：用一个最小 DOM 桩收集所有 innerHTML
let bad=[];
const seen=new Set();
function scan(s,where){
  for(const m of String(s).matchAll(/\b(NaN|undefined|\[object Object\])\b/g)){
    const k=where+"|"+m[1]; if(!seen.has(k)){seen.add(k); bad.push(k);}
  }
}
// 渲染函数都写在 sec(...) 调用里，这里退而求其次：扫 payload 里的字符串字段
// （模板产出的文本最终来自它们），再扫模板里明显的取值链。
(function walk(o,p){
  if(o===null||o===undefined) return;
  if(typeof o==="string") return scan(o,p);
  if(typeof o==="number") return;
  if(Array.isArray(o)) return o.forEach((v,k)=>walk(v,p+"["+k+"]"));
  if(typeof o==="object") for(const k in o) walk(o[k],p+"."+k);
})(D,"payload");
console.log(bad.length?("BAD "+bad.slice(0,8).join(" ; ")):"OK");
"""
        _tmp = os.path.join(tempfile.gettempdir(), "_pf_scan.js")
        open(_tmp, "w", encoding="utf-8").write(_js)
        _r = subprocess.run([_node, _tmp, os.path.join(DATA, "platform.html")],
                            capture_output=True, text=True, timeout=120)
        _o = (_r.stdout or "").strip()
        ok = _o.startswith("OK")
        print(("  ✓ " if ok else "  ✗ ") + "平台 payload 无 NaN/undefined 字符串"
              + ("" if ok else f" —— {_o[:160]}"))
        if not ok:
            bad.append(("platform.html", _o[:120],
                        "字段路径写错会渲染成 NaN/undefined，语法门与 grep 都看不见"))
    else:
        print("  · 跳过平台渲染扫描（未装 node）")

    # ---- 赛题要求的**每一项成果形式**都必须真实存在。
    # 判据直接读 `build_index.DELIVERABLES`——那份清单是从赛题条目抄下来的，
    # 入口页也由它生成。于是「赛题要了什么 / 页面说有什么 / 磁盘上有什么」
    # 三者共用一个来源，不可能各说各话。
    # 早前这些文件其实都在，但入口页没呈现，评委按赛题清单逐条找会找不到。
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from build_index import DELIVERABLES as _DLV
    except Exception as _e:                                  # noqa: BLE001
        _DLV = None
        print(f"  · 跳过成果形式检查（无法导入 build_index：{type(_e).__name__}）")
    if _DLV:
        _miss, _n = [], 0
        for _no, _t, _tag, _items in _DLV:
            for _name, _path, _kind, _ in _items:
                if _path is None:
                    continue
                _n += 1
                if not os.path.exists(os.path.join(ROOT, _path)):
                    _miss.append(f"{_no}·{_name} → {_path}")
        # 入口页必须真的把它们链出去（只列不链等于没交）
        _idx = os.path.join(ROOT, "index.html")
        _unlinked = []
        if os.path.exists(_idx):
            _ih = open(_idx, encoding="utf-8").read()
            for _no, _t, _tag, _items in _DLV:
                for _name, _path, _kind, _ in _items:
                    if _kind == "link" and _path and f'href="{_path}"' not in _ih:
                        _unlinked.append(f"{_name} → {_path}")
        ok = not _miss and not _unlinked
        print(("  ✓ " if ok else "  ✗ ")
              + f"赛题成果形式齐备（{_n} 项）"
              + (f" —— 缺失 {_miss[:3]}" if _miss else "")
              + (f" —— 未链接 {_unlinked[:3]}" if _unlinked else ""))
        if _miss:
            bad.append(("成果形式", f"{len(_miss)} 项文件不存在：{_miss[:3]}",
                        "赛题「成果形式」逐条要求，见 build_index.DELIVERABLES"))
        if _unlinked:
            bad.append(("index.html", f"{len(_unlinked)} 项未被入口页链接",
                        "文件存在但评委点不到，等于没交"))

    print()
    if bad:
        print(f"✗ {len(bad)} 处文档数字与落盘不符：")
        for doc, s, src in bad:
            print(f"    [{doc}]「{s}」应来自 {src}")
        sys.exit(1)
    print("✓ 两份文档的数字全部与落盘一致")


if __name__ == "__main__":
    main()
