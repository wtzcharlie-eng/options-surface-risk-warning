"""问题2 · 风险传导路径推理（交互演示）

赛题成果形式：「风险传导路径可视化界面」。

**直接调用 `kg.reason`**——推理逻辑、边权、证据链全部来自交付本体，
本页只负责把结果摆出来。若哪天二者不一致，说明图谱真的变了，
而不是演示页没跟上。

输入用的是**真实截面**，不是手编的特征值
------------------------------------------
`reason()` 吃的是完整特征字典，内部由 `active_anomalies()` 判定哪些异常被触发。
让用户手填 30 多个特征数只会逼他瞎填，填出来的路径也没有意义。
故这里从交付产物 `alerts_<品种>.parquet` 里挑真实截面——
优先列出**实际触发过异常**的那些，选中即用它的特征跑推理。

另有一份静态交互图 `data_out/kg/kg_viz.html`（零依赖、双击即开），数据同源。
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data_out")

st.set_page_config(page_title="传导推理", layout="wide")
st.title("风险传导路径推理")
st.caption("选一个真实截面 → 检索传播路径 → 输出「根因 → 中间环节 → 可能后果」"
           "与证据链。推理引擎为 `kg/reason.py`，边权全部实测标定。")

try:
    from kg.graph import DiGraph
    from kg.reason import reason, rank_root_causes
    from kg.schema import CONSEQ_BY_ID
    from vol_surface.features import FEATURE_COLUMNS
except Exception as e:                                       # noqa: BLE001
    st.error(f"无法导入模块：{type(e).__name__}: {e}")
    st.stop()

GP = os.path.join(DATA, "kg", "kg_graph.json")
if not os.path.exists(GP):
    st.error("缺 `data_out/kg/kg_graph.json` —— 请先跑 `python scripts/build_kg.py`")
    st.stop()
g = DiGraph.load(GP)


@st.cache_data
def _load(sym: str) -> pd.DataFrame:
    p = os.path.join(DATA, f"alerts_{sym}.parquet")
    return pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()


syms = sorted(f[len("alerts_"):-len(".parquet")] for f in os.listdir(DATA)
              if f.startswith("alerts_") and f.endswith(".parquet"))
if not syms:
    st.error("`data_out/` 下没有 alerts_*.parquet")
    st.stop()

left, right = st.columns([1, 2])

with left:
    st.subheader("选一个截面")
    sym = st.selectbox("品种", syms, index=0)
    df = _load(sym)
    if df.empty:
        st.error(f"{sym} 的数据为空")
        st.stop()
    # 优先给触发过异常的截面——全是「正常」的话推理结果必然为空，
    # 演示看不出东西。这里按预警等级降序排，等级高的在前。
    d = df.sort_values(["level", "n_triggers"], ascending=False)
    only_alert = st.checkbox("只看有触发的截面", value=True)
    if only_alert:
        d = d[d["n_triggers"] > 0]
    if d.empty:
        st.warning("该品种没有触发过异常的截面。取消勾选可看全部。")
        st.stop()
    labels = [f"{int(r.level)}级 · {r.timestamp} · 触发{int(r.n_triggers)}项"
              for r in d.head(200).itertuples()]
    idx = st.selectbox("截面", range(len(labels)),
                       format_func=lambda i: labels[i])
    row = d.head(200).iloc[idx]
    st.caption(f"该截面的规则层触发原因：\n\n{row.get('triggers') or '（无）'}")

with right:
    feats = {c: (0.0 if pd.isna(row[c]) else float(row[c]))
             for c in FEATURE_COLUMNS if c in row.index}
    res = reason(g, feats, symbol=sym, timestamp=str(row["timestamp"]))
    ranked = rank_root_causes(res)

    if not ranked:
        st.info("该截面未触发任何图谱节点，故无传导路径。\n\n"
                "**这本身是结果**：没有触发就不该编出路径来。")
    else:
        n_path = sum(len(r["paths"]) for r in ranked)
        st.subheader(f"{len(ranked)} 个根因 · {n_path} 条传导路径")
        st.caption("根因按「触发等级 × 最强下游权重」排序。"
                   "**权重是 lift 的 95% 保守下界**，不是点估计。")
        for r in ranked:
            trg = r["trigger"]
            with st.expander(
                    f"根因：{r.get('name', r.get('anom'))}　"
                    f"（{trg['level']} 级）",
                    expanded=(r is ranked[0])):
                st.markdown(f"**触发读数**　{trg.get('reason', '—')}")
                for ev in r.get("evidence", []):
                    st.caption(f"· {ev['text'] if isinstance(ev, dict) else ev}")
                if not r["paths"]:
                    st.info("该根因下无显著传导边——候选边未通过显著性检验就不进图谱，"
                            "空结果好过编一条弱关联。")
                    continue
                st.markdown("**传导路径**")
                for p in r["paths"]:
                    mech = p.get("mech") or p.get("no_mech_why") or ""
                    cname = (CONSEQ_BY_ID[p["conseq"]].name
                             if p["conseq"] in CONSEQ_BY_ID else p["conseq"])
                    st.markdown(
                        f"- **{cname}**　"
                        f"提升度 {p['lift']:.2f}×（95% 下界 {p['weight']:.2f}）"
                        f"　条件概率 {p['p_cond']:.1%}　基础率 {p['p_conseq']:.1%}"
                        f"　n={p['n_anom']}")
                    if mech:
                        st.caption(f"　{mech}")
                    if p.get("replicated") is False:
                        st.caption("　⚠ 该边在测试期未复现，读时须打折")
                if r.get("truncated"):
                    n_all = r.get("n_paths_all", len(r["paths"]))
                    st.caption(
                        f"　另有 {n_all - len(r['paths'])} 条未列出——"
                        f"**它们的权重不高于已列出的**（按权重降序，未做筛选）。"
                        f"静默截断且截掉的总是最弱那条，会让图谱显得比实际更强，"
                        f"故此处显式说明。")

st.divider()
st.caption("静态交互图（零依赖、双击即开）：`data_out/kg/kg_viz.html`　·　"
           "Schema 文档：`data_out/doc_kg_schema.html`　·　"
           "图谱给出的是**统计关联**，不是因果断言")
