"""Competition overview: current audit, explicit protocols and evidence gaps."""
from pathlib import Path
import json
import sys
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
st.set_page_config(page_title='比赛交付总览',layout='wide')
st.title('期权曲面风控 · 比赛交付总览')
st.caption('分钟级分级预警、图谱解释与 DRL 决策原型；V3 多日预测另有独立工作台。')
path=ROOT/'data_out/submission/audit.json'
if not path.exists():
    st.error('缺少当前交付审计，请先运行 python scripts/audit_submission.py。');st.stop()
a=json.loads(path.read_text())
st.warning('核心业务 CVaR 的真实开盘与交易日证据、可核验人工评分仍待补齐。运行检查通过不代表全部比赛指标达标。')
cols=st.columns(3)
for col,key,label in [(cols[0],'rule_ml','Q1 规则与轻量模型'),(cols[1],'drl','Q3 DRL')]:
    m=a['metrics'][key]['finite_strict']
    with col:
        st.subheader(label)
        st.metric('有限窗严格提前 · 精确率',f'{m["precision"]:.2%}')
        st.caption(f'召回 {m["recall"]:.2%} · 平均提前 {m["avg_lead_min"]:.1f} 分钟。')
with cols[2]:
    st.subheader('运行核验')
    st.metric('DRL / KG', ' · '.join(f'{c["n_passed"]}/{c["n_total"]}' for c in a['checks']))
    st.caption('当前环境的机器检查；人工评分与业务效果另行核验。')
st.markdown('### 先选择匹配口径，再读数字')
st.info('有限窗参考按风险起点匹配并排除同刻预警，是额外从严的敏感性分析。赛题未明确召回上限与所有区间重叠细节；该参考不代表唯一官方算法。')
labels={'finite_strict':'有限窗严格提前','legacy_closed':'闭区间历史实现','unlimited_strict':'严格提前、召回无上限'}
rows=[]
for key,name in [('rule_ml','Q1 规则+ML'),('drl','Q3 DRL')]:
    for protocol,label in labels.items():
        m=a['metrics'][key][protocol]
        rows.append({'模型':name,'口径':label,'精确率':f'{m["precision"]:.2%}',
                     '召回率':f'{m["recall"]:.2%}','平均提前/分钟':round(m['avg_lead_min'],2),
                     '预警截面数':m['n_alert']})
st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
st.markdown('### 查看功能与证据')
st.markdown('''
- 曲面预警：首页提供热力图、历史时间轴和曲面特征。精简包未提供全部原始合约 Greeks，可查阅历史静态预览。
- 传导推理：选择真实截面，查看图谱路径和统计证据；相关关系不作为确定因果。
- DRL 推理：使用完整 32 维锚点特征，展示单截面假设推演。单截面重置上下文，不能代替序列回测。
- 当前方法结果、比赛要求对照与部署说明：打开项目根目录 `index.html`。
- 1–5 个交易日预测与证据问答：按 README 启动 `dashboard/predictive_app.py`。
''')
with st.expander('复核限制与待补材料'):
    for text in a['caveats']:st.write(text)
    st.write(a['business_cvar']['reason'])
    st.write('原材料记载的后续口径答复此次未提供原件，历史事件合并成绩单列保留。')
