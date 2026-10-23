"""问题3 · DRL 推理接口（交互演示）

赛题成果形式：「模型推理 API 接口」。

**本页就是 `drl/api.py::AlertAgentAPI` 的一层壳**——加载交付的 15 个智能体，
逐截面调 `predict()`，把返回原样摆出来。看到的行为就是线上行为。

两件必须写在页面上的事
----------------------
1. **Q 值不是概率**。它是各动作的期望累计奖励估计，可正可负，不能读成置信度。
2. **一致度（agreement）是集成内部的分歧程度**，不是「预测有多准」——
   15 个种子都投同一级只说明它们看法一致，不说明那一级是对的。

这两条不写清楚，读者很容易把 margin/agreement 当成「模型有多确定」。
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
DATA = os.path.join(ROOT, "data_out")

st.set_page_config(page_title="DRL 推理", layout="wide")
st.title("DRL 自适应预警 · 推理接口")
st.caption("加载交付的智能体集成，对真实截面调用 `AlertAgentAPI.predict()`。"
           "本页不复制任何决策逻辑——使用完整锚点特征进行单截面假设推演。")

try:
    from drl.api import AlertAgentAPI
    from vol_surface.features import FEATURE_COLUMNS
except Exception as e:                                       # noqa: BLE001
    st.error(f"无法导入模块：{type(e).__name__}: {e}")
    st.stop()


@st.cache_resource
def _api():
    d = os.path.join(DATA, "drl")
    if not os.path.isdir(d):
        return None
    try:
        return AlertAgentAPI.from_dir(d)
    except Exception as e:                                   # noqa: BLE001
        st.error(f"加载智能体失败：{type(e).__name__}: {e}")
        return None


@st.cache_data
def _load(sym: str) -> pd.DataFrame:
    frames=[]
    for path in sorted(Path(DATA, 'anchor', sym).glob('*/features.parquet')):
        f=pd.read_parquet(path).sort_values('_timestamp').reset_index(drop=True)
        u=pd.read_parquet(path.with_name('underlying.parquet')).sort_values('_timestamp').reset_index(drop=True)
        if f['_timestamp'].astype(str).tolist()!=u['_timestamp'].astype(str).tolist():
            raise ValueError('曲面与标的特征时间不对齐')
        from drl.dataset import UNDERLYING
        f=pd.concat([f, u[UNDERLYING]],axis=1)
        f['timestamp']=f['_timestamp'].astype(str)
        frames.append(f)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


api = _api()
if api is None:
    st.error("缺 `data_out/drl/agent_seed*.npz` —— 交付包内应当自带，请检查完整性")
    st.stop()

syms = sorted(p.name for p in Path(DATA,'anchor').iterdir() if p.is_dir())

left, right = st.columns([1, 2])

with left:
    st.subheader("输入截面")
    sym = st.selectbox("品种", syms, index=0)
    df = _load(sym)
    if df.empty:
        st.error(f"{sym} 数据为空")
        st.stop()
    d = df
    if d.empty:
        st.warning("无符合条件的截面，取消勾选可看全部")
        st.stop()
    d = d.sort_values("timestamp")
    d = d.tail(300).reset_index(drop=True)
    labels = [str(r.timestamp) for r in d.itertuples()]
    idx = st.selectbox("截面", range(len(labels)), format_func=lambda i: labels[i])
    row = d.iloc[idx]

    st.divider()
    lean = st.slider("预警 Q 值差阈值 lean", 0.0, 8.0, 0.3, 0.1,
                     help="提高则更保守（少报），降至 0 时使用原贪心策略。"
                          "交付工作点由验证集选出，见 workpoint.json。")
    api.set_confidence_threshold(lean)

with right:
    feats = {c: (0.0 if pd.isna(row[c]) else float(row[c]))
             for c in api.features if c in row.index}
    missing=set(api.features)-set(feats)
    if missing:
        st.error('缺少必需特征：'+', '.join(sorted(missing)));st.stop()
    from vol_surface.alert_engine import evaluate as evaluate_rule
    rule_level=evaluate_rule(feats, model=None, state=None)['level']
    st.caption('完整 32 维输入；当前为单截面推演，上下文重置为无近期预警，未加序列回测的休市过滤。结果不能直接解释为回测工作点成绩。')
    # update_state=False：本页是逐截面的假设推演，不该让上一次查询污染
    # 「距上次预警多久」这个上下文维度。
    r = api.predict(feats, timestamp=str(row["timestamp"]),
                    context=(1.0,0.0), update_state=False, explain=True)

    COLORS = {0: "#4fb79c", 1: "#d8a24a", 2: "#d8874f", 3: "#e8695e"}
    c = COLORS.get(r["level"], "#8fa0b2")
    st.markdown(
        f"<div style='border:1px solid #22303f;background:#151f2b;padding:18px 22px'>"
        f"<div style='font-size:12px;color:#8fa0b2;letter-spacing:.1em'>DRL 输出</div>"
        f"<div style='font-size:30px;font-weight:700;color:{c};margin:6px 0'>"
        f"{r['level']} 级 · {r['level_name']}</div>"
        f"<div style='font-size:13px;color:#c2d0dd'>"
        f"默认固定规则参照为 {int(rule_level)} 级"
        f"{'（一致）' if int(rule_level) == r['level'] else '（<b>不一致</b>）'}"
        f"</div></div>", unsafe_allow_html=True)

    a, b = st.columns(2)
    a.metric("最优与次优的 Q 值差", f"{r['margin']:.3f}")
    b.metric("集成一致度", f"{r['agreement']:.0%}",
             help=f"{r['n_models']} 个种子中投给该等级的比例")

    st.markdown("**各等级 Q 值**（期望累计奖励估计）")
    qdf = pd.DataFrame({
        "等级": [f"{i} 级" for i in range(len(r["q_values"]))],
        "Q 值": [float(x) for x in r["q_values"]],
    })
    st.bar_chart(qdf.set_index("等级"), height=200)

    st.warning(
        "**Q 值不是概率**——它是各动作的期望累计奖励估计，可正可负，"
        "不能读成「有多大把握」。**一致度也不是准确率**："
        f"{r['n_models']} 个种子投同一级只说明它们看法一致，不说明那一级是对的。",
        icon="⚠️")

    trg = r.get("triggers")
    if trg:
        st.markdown("**规则层触发原因**（可解释性来自规则层，非 DRL）")
        for t in (trg if isinstance(trg, list) else [trg]):
            if isinstance(t, dict):
                st.caption(f"· [{t.get('rule')}] {t.get('reason')}")
            else:
                st.caption(f"· {t}")
    else:
        st.caption("规则层无触发。")

st.divider()
st.caption("接口：`drl.api.AlertAgentAPI.predict(feats, timestamp=…) "
           "→ {level, level_name, q_values, margin, agreement, is_alert, triggers}`"
           "　·　另有 `predict_sequence()` 供批量推理")
