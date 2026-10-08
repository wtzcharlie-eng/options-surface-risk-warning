"""期权波动率曲面风控预警系统 · 交互演示

多页结构（Streamlit `pages/` 约定）：

    app.py                     首页 · 系统总览与导航
    pages/1_曲面预警.py         问题1 · 曲面热力图 / Greeks / 预警时间轴
    pages/2_传导推理.py         问题2 · 输入异常 → 根因/路径/后果 + 证据链
    pages/3_DRL推理.py          问题3 · 现场调特征看四级输出（走交付 API）

运行::

    cd dashboard && streamlit run app.py

主题在 `../.streamlit/config.toml`，与交付入口页 `index.html` 同一套色值。

设计约定
--------
**三个子页各自直接调用交付本体**——`vol_surface.alert_engine`、`kg.reason`、
`drl.api.AlertAgentAPI`，不复制任何逻辑。这样演示看到的行为就是交付系统的行为；
若哪天二者不一致，说明交付本体真的变了，而不是演示页没跟上。
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data_out")

st.set_page_config(page_title="期权曲面风控预警系统", layout="wide", page_icon="📈")


def _j(rel):
    p = os.path.join(DATA, rel)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


st.title("期权波动率曲面风控预警系统")
st.caption("宁证期货赛题 · 交互演示。三个子页分别对应赛题的三个问题，"
           "各自直接调用交付本体，不复制逻辑。")

q1 = _j("q1_7sym.json")
v2 = _j("metrics_v2.json")
gd = _j(os.path.join("drl", "gates.json"))
gk = _j(os.path.join("kg", "gates.json"))

c1, c2, c3 = st.columns(3)
if q1:
    sn = q1["summary_new"]
    with c1:
        st.subheader("问题1 · 分级预警")
        st.metric("事件级精确率", f"{sn['v2']['P']:.2%}",
                  f"判别力 {sn['v2_discrimination_pp']:+.1f}pp")
        st.caption(f"{sn['v2']['n_pass3']}/{sn['n_win']} 个事件窗口三项全达标。"
                   f"该口径下最强平凡策略精确率 "
                   f"{sn['v2_best_trivial']['P']:.1%}，**恰好等于达标线**——"
                   f"故只有判别力算数。")
if gk:
    with c2:
        st.subheader("问题2 · 知识图谱")
        st.metric("验证门", f"{gk['n_pass']}/{gk['n_total']}")
        st.caption("边权全部实测标定，未通过显著性的候选一并落盘。"
                   "人工 5 分制为自评形式，**不宣称达标**。")
if v2 and gd:
    with c3:
        st.subheader("问题3 · DRL 策略")
        st.metric("事件级精确率", f"{v2['drl']['test']['precision']:.2%}",
                  f"判别力 {v2['discrimination_pp']:+.1f}pp")
        st.caption(f"{gd['n_pass']}/{gd['n_total']} 道验证门通过。"
                   f"对照最强平凡策略「{v2['best_trivial']['name']}」"
                   f"{v2['best_trivial']['precision']:.1%}。")

st.divider()

st.markdown("""
### 从左侧选择子页

| 子页 | 对应赛题要求 | 能做什么 |
|---|---|---|
| **曲面预警** | 问题1 成果形式「预警可视化仪表板」 | 曲面热力图、Greeks 曲线、预警时间轴、历史回放 |
| **传导推理** | 问题2 成果形式「风险传导路径可视化」 | 勾选异常 → 检索传播路径 → 根因/中间环节/后果 + 证据链 |
| **DRL 推理** | 问题3 成果形式「模型推理 API 接口」 | 现场调特征向量，看交付模型输出的四级预警与 Q 值 |

### 读数字前必看

本系统在**两套评测口径**下会给出差很多的结论，页面上每个数字都标了口径。
新口径（命题方 2026-08 答复）因召回不设时间上限而**自带退化**——
不看任何特征的平凡策略也能拿到很高召回，三条门槛实际塌缩成只剩精确率。
故此处每个「达标」都并列了**判别力**（精确率减最强平凡策略精确率）。

完整证据链见交付入口页 `index.html` 与研究总览 `data_out/platform.html`。
""")

st.divider()
st.caption("主题配置 `.streamlit/config.toml`　·　"
           "各子页直接调用 `vol_surface` / `kg` / `drl` 交付本体")
