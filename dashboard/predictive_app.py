"""Primary application: observed trading-day forecasts and evidence-grounded chat."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / '.env', override=False)

from vol_surface.predictive import load_anchor_data
from vol_surface.predictive_service import build_evidence, load_report, default_model_dir, load_model_artifact, forecast_model
from vol_surface.risk_assistant import LLMConfig, answer, evidence_fields, sanitize_evidence

st.set_page_config(page_title='曲面风险 · 预测与对话', page_icon='📈', layout='wide')
st.markdown('''<style>
.block-container {padding-top:2rem; max-width:1450px;}
h1,h2,h3 {font-family: -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;}
div[data-testid="stMetric"] {border:1px solid #283a49; border-radius:12px; padding:18px;}
.eyebrow {color:#6bd1b3; letter-spacing:3px; font-size:12px; font-weight:600;}
.subtle {color:#94aabe; line-height:1.8;}
</style>''', unsafe_allow_html=True)

SYMBOL_NAMES = {'ag':'白银', 'au':'黄金', 'sc':'原油', 'si':'工业硅', 'lc':'碳酸锂', 'cu':'铜', 'rb':'螺纹钢'}
ANCHOR_DIR = Path(os.environ.get('RISK_ANCHOR_DIR', str(ROOT / 'data_out' / 'anchor')))


@st.cache_data(show_spinner='读取历史曲面特征…')
def read_history(path: str, fingerprint: tuple):
    return load_anchor_data(path)


@st.cache_data
def read_model(path: str, modified: int):
    return load_model_artifact(path)


@st.cache_data(show_spinner='计算日级风险预测…')
def predict_cached(_artifact, _history, symbol: str, as_of: str, model_modified: int, fingerprint: tuple, algorithm_modified: tuple):
    return forecast_model(_artifact, _history, symbol, as_of=as_of)


def num(value, pattern='.1%'):
    if value is not None and pattern == '.1%' and 0 < float(value) < .001:
        return '<0.1%'
    return format(float(value), pattern) if value is not None else '不可用'


def horizon_days(row):
    return int(row.get('days', row.get('horizon_days', 0)))


st.markdown('<div class="eyebrow">SURFACE RISK / PREDICTIVE INTELLIGENCE</div>', unsafe_allow_html=True)
st.title('看见风险，再追问原因')
st.markdown('<p class="subtle">未来 1、3、5 个交易日的曲面风险预测 · 有限视窗评估 · 可追溯的对话解读</p>', unsafe_allow_html=True)
st.info('当前为历史数据回放。尚未连接实时行情；预测对象是曲面风险代理事件，未包含账户持仓和组合损益。', icon='🕘')

available_versions = ['v2', 'v1'] if (ROOT / 'data_out' / 'predictive_v2' / 'model.json').exists() else ['v1']
with st.sidebar:
    selected_version = st.radio('预测版本', available_versions, format_func=lambda v: {'v2':'联合时序版 V2', 'v1':'独立期限版 V1（对照）'}[v])
MODEL_DIR = default_model_dir(selected_version)
model_path = MODEL_DIR / 'model.json'
if not model_path.exists():
    st.warning('预测模型尚未生成。请先在项目目录运行：python scripts/train_predictive.py')
    st.stop()
try:
    files = tuple(sorted((str(p), p.stat().st_mtime_ns) for p in ANCHOR_DIR.glob('*/*/features.parquet')))
    history = read_history(str(ANCHOR_DIR), files)
    artifact = read_model(str(model_path), model_path.stat().st_mtime_ns)
    report = load_report(MODEL_DIR)
except (OSError, ValueError, KeyError) as exc:
    st.error(f'模型或数据读取失败（{type(exc).__name__}）。请检查本地数据路径并重新生成模型。')
    st.stop()

symbols = sorted(history['symbol'].unique().tolist())
freshest_symbol = history.groupby('symbol')['_dt'].max().idxmax()
with st.sidebar:
    st.markdown('### 预测工作台')
    symbol = st.selectbox('品种', symbols, index=symbols.index(freshest_symbol), format_func=lambda s: f'{SYMBOL_NAMES.get(s, s)} · {s.upper()}')
    st.caption('默认选择缓存日期最新的品种。')
    days = st.select_slider('预测周期（交易日）', options=[1, 2, 3, 4, 5], value=3)
    sym_history = history[history['symbol'] == symbol]
    dates = sorted(sym_history['_dt'].dt.date.unique().tolist())
    replay_date = st.selectbox('历史观察日', dates, index=len(dates)-1, format_func=str)
    st.caption('按所选日 23:59:59 的可用信息回放；未结束的交易日不用于预测。')
    st.divider()
    st.markdown('### 对话设置')
    try:
        env_config = LLMConfig.from_env()
    except ValueError:
        env_config = LLMConfig()
        st.warning('环境中的 LLM 配置无效，请重新填写。')
    providers = ['local', 'openai-compatible', 'local-openai', 'openai']
    provider_labels = {'local':'本地证据解读', 'openai-compatible':'OpenAI 兼容 API', 'local-openai':'本机大模型服务', 'openai':'OpenAI 官方 API'}
    provider = st.selectbox('对话服务', providers, index=providers.index(env_config.provider) if env_config.provider in providers else 0, format_func=provider_labels.get)
    model_name, base_url, api_key = env_config.model, env_config.base_url, env_config.api_key
    if provider != 'local':
        base_url = st.text_input('服务地址 / Base URL', value=('http://127.0.0.1:11434/v1' if provider == 'local-openai' and env_config.provider != provider else base_url), help='填写服务商提供的兼容接口地址，包含 /v1（若服务商要求）。')
        model_name = st.text_input('模型名称', value=model_name, placeholder='填写服务商的模型 ID')
        api_key = st.text_input('API Key', value=api_key, type='password', help='仅当前会话使用，不写入报告和聊天记录。')
        st.caption('发送问题时，会将当前风险快照和近期对话发送到所填服务地址。')
    config = LLMConfig(provider=provider, model=model_name, base_url=base_url, api_key=api_key, timeout_seconds=env_config.timeout_seconds)
    if st.button('检查对话配置', width='stretch'):
        try:
            config.validate()
            st.success('配置格式有效。联网连通性将在发送问题时检验。')
        except ValueError as exc:
            st.error(str(exc))
    st.caption('未配置 API 时，本地解读始终可用。LLM 负责解释已计算结果。')

as_of = pd.Timestamp(replay_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
try:
    algorithm_signature = tuple((name, (ROOT / 'vol_surface' / name).stat().st_mtime_ns) for name in ['predictive.py', 'predictive_service.py', 'predictive_v2.py'] if (ROOT / 'vol_surface' / name).exists())
    snapshot = predict_cached(artifact, history, symbol, as_of.isoformat(), model_path.stat().st_mtime_ns, files, algorithm_signature)
except (ValueError, KeyError) as exc:
    st.error(f'此时点无法预测（{type(exc).__name__}）。请检查输入数据是否完整。')
    st.stop()

st.subheader(f'{SYMBOL_NAMES.get(symbol, symbol)} · 预测快照')
status_names = {'ready':'预测可用', 'degraded':'降级输出', 'stale':'数据陈旧', 'unavailable':'数据不足', 'out_of_distribution':'超出训练分布'}
st.caption(f"观察时点：{snapshot.get('as_of', as_of)}　｜　数据截止：{snapshot.get('data_timestamp', '未知')}　｜　{status_names.get(snapshot.get('status'), snapshot.get('status', '未知'))}")
st.caption('预测起点为“数据截止”对应的已完成日盘收盘；未来交易日从该日之后计算。夜盘进行中仍展示前一完整交易日的收盘预测。')
if 'currently_exceeding' in snapshot.get('signals', {}):
    current_state = '已超过历史阈值' if snapshot['signals']['currently_exceeding'] else '当前未超过历史阈值'
    task_labels = {'onset':'新异常出现', 'persistence':'当前异常后的再次超阈值', 'mixed_exceedance':'新异常与延续混合'}
    task_name = task_labels.get(snapshot.get('prediction_task', 'mixed_exceedance'), snapshot.get('prediction_task', '混合事件'))
    st.caption(f'当前曲面状态：{current_state}。本次预测任务：{task_name}。')
    if snapshot.get('prediction_task_definition'):
        st.caption(snapshot['prediction_task_definition'])
if snapshot.get('model_version') == 'v2':
    st.caption('联合模型共享一条累计风险曲线；1–5 日概率在结构上保持一致。')
quality_warnings = [str(w) for w in snapshot.get('warnings', []) if '保留测试段' not in str(w)]
evaluation_warnings = [str(w) for w in snapshot.get('warnings', []) if '保留测试段' in str(w)]
if quality_warnings:
    st.warning(' '.join(quality_warnings))
if evaluation_warnings:
    with st.expander('部分期限的时间外表现未超过历史基线，请查看评估提示'):
        for warning in evaluation_warnings:
            st.write(warning)

horizons = snapshot.get('horizons', [])
for col, item in zip(st.columns(3), [h for h in horizons if horizon_days(h) in (1, 3, 5)]):
    d = horizon_days(item)
    with col:
        model_status = item.get('model_status', 'unknown')
        label = '历史基础率' if model_status == 'baseline_only' else '状态条件概率' if model_status in {'state_baseline', 'hazard_baseline'} else '风险概率'
        st.metric(f'未来 {d} 个交易日 · {label}', num(item.get('probability')))
        if item.get('alert_suppressed_reason'):
            st.caption('期限间估计不一致，仅供研究；预测预警已暂停。')
        elif item.get('alert'):
            st.warning('▲ 达到预警阈值')
        elif model_status == 'hazard_baseline':
            st.caption('统计参考 · 预测预警关闭')
        elif model_status == 'baseline_only':
            st.caption('基线参考 · 预测模型未获验证支持')
        else:
            st.caption('未达到预警阈值' if item.get('probability') is not None else '暂不可用')
        if model_status == 'state_baseline':
            st.caption('简单统计模型 · 按品种和当前异常状态估计，不代表复杂模型带来增量。')
        elif model_status == 'hazard_baseline':
            st.caption('条件风险率基准 · 与当前任务一致的历史统计，不宣称复杂模型增量。')
        baseline_label = '已校准条件基准' if snapshot.get('model_version') == 'v2' else '历史基线'
        threshold_label = '' if model_status in {'baseline_only', 'hazard_baseline'} else f"阈值 {num(item.get('threshold'))} · "
        st.caption(f"{threshold_label}{baseline_label} {num(item.get('baseline_probability'))}")

selected = next((h for h in horizons if horizon_days(h) == days), {})
evidence = build_evidence(snapshot, days, report)
context_key = f"{selected_version}|{model_path.stat().st_mtime_ns}|{symbol}|{replay_date}|{days}|{snapshot.get('prediction_task')}"
if st.session_state.get('risk_context_key') != context_key:
    st.session_state['risk_context_key'] = context_key
    st.session_state['risk_chat_history'] = []

tab_chat, tab_model, tab_data = st.tabs(['与风险助手对话', '预测依据与评估', '数据与快照'])

with tab_chat:
    st.caption(f'当前问题上下文：{SYMBOL_NAMES.get(symbol, symbol)}，未来 {days} 个交易日。切换品种、日期或周期会开启新对话。')
    buttons = st.columns(3)
    suggested = None
    for col, label in zip(buttons, ['解读当前风险', '预测效果可靠吗？', '为什么会有这个信号？']):
        if col.button(label, width='stretch'):
            suggested = label
    for entry in st.session_state['risk_chat_history']:
        with st.chat_message(entry['role']):
            st.markdown(entry['content'])
    question = st.chat_input('例如：未来 3 天的风险概率是多少？模型是否优于基线？')
    question = (suggested or question or '').strip()
    if len(question) > 4000:
        st.error('问题请控制在 4000 字以内。')
        question = ''
    if question:
        with st.chat_message('user'):
            st.write(question)
        with st.chat_message('assistant'):
            with st.spinner('正在核对预测证据…'):
                response = answer(question, evidence, history=st.session_state['risk_chat_history'], config=config)
            if response.warning:
                st.warning(response.warning)
            st.caption('本地证据解读（未调用 LLM）' if response.mode == 'local' else f'LLM · {response.model}')
            st.markdown(response.text)
        st.session_state['risk_chat_history'] = response.history
    with st.expander('查阅回答引用的字段'):
        fields = evidence_fields(sanitize_evidence(evidence))
        st.dataframe(pd.DataFrame([{'字段':k, '值':str(v)} for k,v in fields.items()]), hide_index=True, width='stretch')

with tab_model:
    st.markdown('#### 预测概率与预警阈值')
    if horizons:
        fig = go.Figure()
        fig.add_bar(x=[f'{horizon_days(h)} 个交易日' for h in horizons], y=[h.get('probability') for h in horizons], name='预测概率', marker_color='#4fb79c')
        fig.add_scatter(x=[f'{horizon_days(h)} 个交易日' for h in horizons], y=[h.get('threshold') if h.get('model_status') not in {'baseline_only', 'hazard_baseline'} else None for h in horizons], mode='markers', name='预警阈值', marker=dict(color='#e8af5f', size=13, symbol='line-ew'))
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=10,b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#c2d0dd', yaxis=dict(range=[0,1], tickformat='.0%', title='风险概率'), legend=dict(orientation='h'))
        st.plotly_chart(fig, width='stretch')
    contributions = selected.get('top_contributions', [])
    st.markdown('#### 模型使用的信号')
    if contributions:
        st.caption('V2 贡献解释对应条件风险率的指定步骤，不是累计概率的直接加总；V1 对应其逻辑回归对数几率。贡献不代表因果关系。')
        st.dataframe(pd.DataFrame(contributions), hide_index=True, width='stretch')
    else:
        st.caption('当前结果未提供特征贡献。统计基准使用训练期条件发生率；任务定义、模型状态与输入证据可在快照中核对。')
    st.markdown('#### 时间外验证')
    st.caption('训练、概率校准、阈值选择和测试按时间分开；标签跨越边界的样本会被剔除。Brier 越低越好，技巧分数大于零才优于报告所用基准。当前属于已查看历史上的开发评估。')
    if snapshot.get('model_version') == 'v2':
        ev = snapshot.get('evaluation', {}).get(str(days), {})
        op = snapshot.get('evaluation_details', {}).get(str(days), {}).get('operational', {})
        if ev:
            st.caption('下表对应当前品种、当前任务和所选期限；运行统计保留全部预测原点，弃权按未发出预警计入召回分母。')
            st.dataframe(pd.DataFrame([{
                '测试原点数':ev.get('n'), '模型 Brier':ev.get('brier'),
                '条件基准 Brier':ev.get('baseline_brier'),
                '相对基准技巧':num(ev.get('brier_skill')),
                '概率可用率（含统计基准）':num(op.get('probability_coverage')),
                '增量模型覆盖率':num(op.get('predictive_signal_coverage')),
                '运行误报数':op.get('false_alarms'),
                '运行召回率（含弃权）':num(op.get('recall_all_origins')),
            }]), hide_index=True, width='stretch')
    report_md = MODEL_DIR / 'report.md'
    if report_md.exists():
        st.markdown(report_md.read_text(encoding='utf-8'))
    else:
        st.json(report)

with tab_data:
    st.markdown('#### 数据证据')
    st.json(snapshot)
    st.download_button('下载当前预测快照', json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False), file_name=f'risk_{symbol}_{replay_date}.json', mime='application/json')
    st.download_button('下载对话证据', json.dumps(sanitize_evidence(evidence), ensure_ascii=False, indent=2, allow_nan=False), file_name=f'evidence_{symbol}_{replay_date}.json', mime='application/json')
    st.caption('预测以附带的 anchor 历史特征为输入。接入新行情后，需按同一字段与时间规范持续更新数据，再运行预测。')
