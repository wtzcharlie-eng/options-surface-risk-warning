"""预警可视化仪表板 — 曲面热力图 / Greeks 曲线 / 预警时间轴 / 特征仪表 / 历史回放。

基于 dataviz 规范：
- 序贯热力图用 teal→vermilion 发散刻度（与 index.html 主视觉、platform.html 同一条）
- 发散用 teal↔vermilion，深灰中点
- 状态色（0-3 预警等级）配图标+标签，不靠色独传
- 深色主题（与交付入口页同一套色值，见 .streamlit/config.toml），recessive 网格
- 单一轴原则、分类色固定顺序、文本用 ink 不用系列色

数据来源：data_out/{alerts,features,surfaces}_<symbol>.parquet
运行：streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vol_surface.alert_engine import LEVEL_NAMES
from vol_surface.backtest import load_events
from vol_surface.cleaning import clean_slice
from vol_surface.io_loader import load_symbol_family, slice_at_timestamp

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_out")
ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "archive")

# ---- 调色板（dataviz 规范，深色主题）----
# **与 index.html / platform.html / doc_*.html 同一套**：深墨蓝底 +
# teal↔vermilion 发散刻度。早前这里是暖灰底 + 蓝色标，与其余三处不是一套色语，
# 评委从入口页点进来会觉得换了个产品。色值改动只影响观感，不改任何数据。
BG = "#0e1621"            # 页面底（与 index.html 的 --ink 一致）
SURFACE = "#151f2b"       # 图表面板（= --panel）
INK_PRIMARY = "#e3eaf1"
INK_SECONDARY = "#c2d0dd"
INK_MUTED = "#8fa0b2"
GRID = "#22303f"
# 状态色（0-3 级预警）— 配图标+标签，不靠色独传
# 与主题一致：正常=teal、严重=vermilion，中间两级用琥珀过渡
LEVEL_COLORS = {0: "#4fb79c", 1: "#d8a24a", 2: "#d8874f", 3: "#e8695e"}
LEVEL_ICONS = {0: "✓", 1: "▲", 2: "■", 3: "✖"}
# 序贯 ramp（热力图）：与 platform.html 的 RAMP、index.html 主视觉同一条发散刻度
SEQ_BLUE = ["#1e3a4a", "#245a63", "#2f8b7e", "#4fb79c", "#a8b26c", "#d8874f", "#e8695e"]
# 发散 teal↔vermilion（残差/变化）
DIV_COLORS = ["#2f8b7e", "#4fb79c", "#a9cfc6", "#2b3a4a", "#e6b3ad", "#e8918a", "#e8695e"]
# 分类色（Greeks 系列，固定顺序）
CAT = {"call": "#6fb2e8", "put": "#d8874f"}


def _common_layout(title: str = "", height: int = 380, hovermode: str = "closest") -> dict:
    return dict(
        title=dict(text=title, font=dict(color=INK_PRIMARY, size=14)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY,
                  family="Times New Roman, Times, PingFang SC, Hiragino Sans GB, "
                         "Microsoft YaHei, sans-serif"),
        margin=dict(l=60, r=20, t=50, b=40),
        height=height,
        legend=dict(orientation="h", y=1.08, font=dict(color=INK_SECONDARY)),
        hovermode=hovermode,
    )


def _style_axes(fig: go.Figure) -> None:
    fig.update_xaxes(gridcolor=GRID, linecolor=INK_MUTED, zerolinecolor=GRID, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=INK_MUTED, zerolinecolor=GRID, tickfont=dict(color=INK_MUTED))


@st.cache_data
def load_symbol_data(symbol: str) -> dict:
    base = os.path.join(DATA_OUT, f"alerts_{symbol}.parquet")
    if not os.path.exists(base):
        return None
    alerts = pd.read_parquet(base)
    features = pd.read_parquet(os.path.join(DATA_OUT, f"features_{symbol}.parquet"))
    surf_path = os.path.join(DATA_OUT, f"surfaces_{symbol}.parquet")
    surfaces = pd.read_parquet(surf_path) if os.path.exists(surf_path) else pd.DataFrame()
    alerts["dt"] = pd.to_datetime(alerts["timestamp"], format="%Y%m%d%H%M%S", errors="coerce")
    return {"alerts": alerts, "features": features, "surfaces": surfaces}


@st.cache_data
def load_raw_slice(symbol: str, timestamp: str) -> pd.DataFrame:
    """从 archive 按需加载某时间截面的原始数据并清洗，供 Greeks 曲线用。

    只读取包含该时间戳的数据月文件，避免全量加载。
    """
    # 时间戳 YYYYMMDDHHMMSS -> 年月
    yyyy, mm = timestamp[:4], timestamp[4:6]
    from vol_surface.io_loader import _files_for_root
    files = _files_for_root(symbol, year=int(yyyy), month=int(mm))
    # 也加载前一月（时间戳可能跨月初）
    import datetime as _dt
    d = _dt.date(int(yyyy), int(mm), 1)
    prev = (d.replace(day=1) - _dt.timedelta(days=1))
    files += _files_for_root(symbol, year=prev.year, month=prev.month)
    files = list(dict.fromkeys(files))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sl = df[df["timestamp"] == timestamp]
    if sl.empty:
        return pd.DataFrame()
    clean, _ = clean_slice(sl)
    return clean


def available_symbols() -> list[str]:
    syms = []
    for f in os.listdir(DATA_OUT):
        if f.startswith("alerts_") and f.endswith(".parquet"):
            syms.append(f[len("alerts_"):-len(".parquet")])
    return sorted(syms)


def surface_heatmap(surfaces: pd.DataFrame, timestamp: str, option_type: str = "call") -> go.Figure | None:
    sub = surfaces[(surfaces["timestamp"] == timestamp) & (surfaces["option_type"] == option_type)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    dte = np.array(row["dte_grid"], float)
    money = np.array(row["moneyness_grid"], float)
    surf = np.stack([np.asarray(x, float) for x in row["surface"]])
    finite = surf[np.isfinite(surf)]
    # zmax 用 95 分位封顶，避免单一极端单元冲淡整个 ramp；并保底 0.4 让常态曲面也有色阶
    zmax = max(0.4, float(np.percentile(finite, 95))) if finite.size else 0.4
    fig = go.Figure(data=go.Heatmap(
        z=surf, x=money, y=dte,
        colorscale=[[i / (len(SEQ_BLUE) - 1), c] for i, c in enumerate(SEQ_BLUE)],
        colorbar=dict(title="IV", tickfont=dict(color=INK_MUTED), title_font=dict(color=INK_SECONDARY),
                      outlinecolor=GRID, outlinewidth=0.5),
        hovertemplate="moneyness=%{x:.3f}<br>dte=%{y:.0f}<br>IV=%{z:.4f}<extra></extra>",
        zmin=0, zmax=zmax,
    ))
    fig.update_layout(**_common_layout(f"IV 曲面热力图 · {option_type} · {timestamp}", 360))
    fig.update_xaxes(title_text="moneyness")
    fig.update_yaxes(title_text="距到期天数 dte")
    _style_axes(fig)
    return fig


def greeks_curves(raw_slice: pd.DataFrame, timestamp: str) -> go.Figure | None:
    """从原始截面数据画 Greeks 曲线（Delta/Gamma/Vega vs moneyness，call/put 双系列）。

    取近月（dte 最小的到期月）切片，按 moneyness 排序。遵循 dataviz：分类色固定
    顺序（call=blue, put=magenta），单一 y 轴，recessive 网格，tooltip+crosshair。
    """
    if raw_slice is None or len(raw_slice) == 0:
        return None
    # 取 dte 最小的到期月
    near_dte = raw_slice["dte"].min()
    g = raw_slice[raw_slice["dte"] == near_dte].sort_values(["option_type", "moneyness"])
    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=(f"Delta (近月 dte={int(near_dte)})", "Gamma",
                                       "Vega", "Theta"))
    for ot, col in CAT.items():
        sub = g[g["option_type"] == ot]
        if sub.empty:
            continue
        for i, (key, label) in enumerate([("delta", "Delta"), ("gamma", "Gamma"), ("vega", "Vega"), ("theta", "Theta")]):
            r, c = i // 2 + 1, i % 2 + 1
            fig.add_trace(go.Scatter(
                x=sub["moneyness"], y=sub[key], name=f"{ot} {label}",
                line=dict(color=col, width=2), mode="lines+markers", marker=dict(size=5),
                # legendgroup 按 option_type 分组，跨子图联动显隐
                legendgroup=ot, showlegend=(i == 0),
                hovertemplate=f"{ot} {label}<br>moneyness=%{{x:.3f}}<br>{label}=%{{y:.4f}}<extra></extra>",
            ), row=r, col=c)
    # 线图配 crosshair（x unified），跨四个子图同 moneyness 联动
    fig.update_layout(**_common_layout(f"Greeks 截面曲线 · 近月 · {timestamp}", 460, hovermode="x unified"))
    for i in range(1, 5):
        fig.update_xaxes(gridcolor=GRID, linecolor=INK_MUTED, title_text="moneyness",
                         row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
        fig.update_yaxes(gridcolor=GRID, linecolor=INK_MUTED,
                         row=(i - 1) // 2 + 1, col=(i - 1) % 2 + 1)
    return fig


def alert_timeline(alerts: pd.DataFrame, events: pd.DataFrame | None, symbol: str) -> go.Figure:
    """预警时间轴：散点按等级着色，hover 显示触发原因；事件期阴影叠加。"""
    fig = go.Figure()
    # 事件期阴影
    if events is not None:
        for _, ev in events.iterrows():
            if ev.start_date is pd.NaT:
                continue
            fig.add_vrect(x0=ev.start_date, x1=ev.end_date, fillcolor="#d8a24a", opacity=0.08,
                          line_width=0)
    # 各等级散点
    for lv in range(4):
        sub = alerts[alerts["level"] == lv]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["dt"], y=sub["level"], name=f"{LEVEL_ICONS[lv]} L{lv} {LEVEL_NAMES[lv]}",
            mode="markers", marker=dict(color=LEVEL_COLORS[lv], size=6 + lv * 2,
                                        line=dict(width=0)),
            customdata=sub["triggers"].to_numpy(),
            hovertemplate="<b>L%{y}</b><br>%{x}<br>%{customdata}<extra></extra>",
        ))
    fig.update_layout(**_common_layout(f"预警时间轴 · {symbol}", 320))
    fig.update_yaxes(title="预警等级", range=[-0.5, 3.5], dtick=1,
                     tickvals=[0, 1, 2, 3], ticktext=[f"{LEVEL_ICONS[i]} L{i}" for i in range(4)])
    fig.update_xaxes(title="时间")
    _style_axes(fig)
    return fig


def feature_gauges(features: pd.DataFrame, alerts: pd.DataFrame, timestamp: str) -> go.Figure:
    """关键特征实时值 + 滚动分位仪表。"""
    f = features[features["timestamp"] == timestamp]
    a = alerts[alerts["timestamp"] == timestamp]
    if f.empty:
        return None
    keys = [("convexity_violation", "凸性违反"), ("term_slope_roc", "期限斜率Δ"),
            ("atm_iv_z", "ATM IV z"), ("gamma_concentration", "Gamma集中"),
            ("vega_concentration", "Vega集中"), ("arb_score", "无套利分")]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[k[1] for k in keys],
                        specs=[[{"type": "indicator"}] * 3, [{"type": "indicator"}] * 3])
    for i, (key, label) in enumerate(keys):
        hist = features[key].astype(float).dropna().to_numpy()
        val = float(f[key].iloc[0])
        pct = float(np.searchsorted(np.sort(hist), val) / max(len(hist), 1)) if hist.size else 0.5
        r, c = i // 3 + 1, i % 3 + 1
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=pct * 100,
            number=dict(suffix="%", font=dict(color=INK_PRIMARY)),
            gauge=dict(axis=dict(range=[0, 100], tickcolor=INK_MUTED),
                       bar=dict(color=INK_SECONDARY),
                       steps=[dict(range=[0, 50], color="#1a2532"),
                              dict(range=[50, 80], color="#31280f"),
                              dict(range=[80, 100], color="#331b1a")],
                       threshold=dict(value=pct * 100, line=dict(color=LEVEL_COLORS[3], width=2))),
            title=dict(text=f"{label}={val:.3f}", font=dict(color=INK_SECONDARY, size=11)),
        ), row=r, col=c)
    fig.update_layout(**_common_layout("关键特征滚动分位仪表", 380), showlegend=False)
    return fig


def main():
    st.set_page_config(page_title="期权曲面风控预警", layout="wide", page_icon="📊")
    st.markdown("""
    <style>
    .stApp { background-color: #0e1621; }
    .stMarkdown, .stText, h1, h2, h3 { color: #e3eaf1 !important; }
    .stMetric label, .stMetric div { color: #c2d0dd !important; }
    </style>
    """, unsafe_allow_html=True)
    st.title("📊 期权波动率曲面风控预警系统")

    syms = available_symbols()
    if not syms:
        st.warning("未找到数据。请先运行 `python scripts/build_features.py` 生成 data_out/*.parquet")
        return
    syms_disp = {"ag": "白银 ag", "si": "工业硅 si", "au": "黄金 au", "sc": "原油 sc",
                 "cu": "铜 cu", "lc": "碳酸锂 lc", "rb": "螺纹 rb"}.get
    with st.sidebar:
        st.header("控制面板")
        symbol = st.selectbox("品种", syms, format_func=lambda s: syms_disp(s) or s)
        data = load_symbol_data(symbol)
        if data is None:
            st.error(f"{symbol} 数据加载失败")
            return
        alerts = data["alerts"]
        surfaces = data["surfaces"]
        features = data["features"]
        events = load_events()
        # 时间回放
        st.subheader("历史回放")
        ts_list = sorted(surfaces["timestamp"].unique().tolist()) if not surfaces.empty else sorted(alerts["timestamp"].tolist())
        ts_idx = st.slider("时间截面", 0, max(len(ts_list) - 1, 0), len(ts_list) // 2)
        play = st.button("▶ 自动播放")
        if play and ts_idx < len(ts_list) - 1:
            ts_idx = min(ts_idx + 1, len(ts_list) - 1)
        ts = ts_list[ts_idx]
        st.caption(f"当前: {ts}")
        # 等级过滤
        st.subheader("过滤")
        show_levels = st.multiselect("显示预警等级", [0, 1, 2, 3], default=[1, 2, 3],
                                     format_func=lambda l: f"{LEVEL_ICONS[l]} L{l} {LEVEL_NAMES[l]}")

    # 顶部指标条
    cur = alerts[alerts["timestamp"] == ts]
    cur_lv = int(cur["level"].iloc[0]) if not cur.empty else 0
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("当前预警等级", f"{LEVEL_ICONS[cur_lv]} L{cur_lv}", LEVEL_NAMES[cur_lv])
    with col2:
        st.metric("ATM IV", f"{float(cur['atm_iv'].iloc[0]):.4f}" if not cur.empty else "-")
    with col3:
        st.metric("凸性违反", f"{float(cur['convexity_violation'].iloc[0]):.3f}" if not cur.empty else "-")
    with col4:
        st.metric("无套利分", f"{float(cur['arb_score'].iloc[0]):.1f}" if not cur.empty else "-")

    # 触发原因
    if not cur.empty and cur["triggers"].iloc[0]:
        with st.expander(f"🚨 触发原因 (L{cur_lv})", expanded=cur_lv >= 2):
            for t in cur["triggers"].iloc[0].split("; "):
                st.markdown(f"- {t}")

    # 曲面热力图 + Greeks
    left, right = st.columns([1, 1])
    with left:
        if not surfaces.empty:
            ot = st.radio("期权类型", ["call", "put"], horizontal=True)
            hm = surface_heatmap(surfaces, ts, ot)
            if hm:
                st.plotly_chart(hm, use_container_width=True)
                with st.expander("曲面数据表（IV × moneyness × dte）"):
                    sub = surfaces[(surfaces["timestamp"] == ts) & (surfaces["option_type"] == ot)]
                    if not sub.empty:
                        r = sub.iloc[0]
                        surf = np.stack([np.asarray(x, float) for x in r["surface"]])
                        tbl = pd.DataFrame(surf,
                            columns=[f"{m:.2f}" for m in np.asarray(r["moneyness_grid"], float)],
                            index=[f"dte={int(d)}" for d in np.asarray(r["dte_grid"], float)])
                        st.dataframe(tbl.style.format("{:.4f}"), use_container_width=True)
            else:
                st.info("该截面无曲面抽样数据，尝试邻近时间点")
    with right:
        raw = load_raw_slice(symbol, ts)
        gc = greeks_curves(raw, ts)
        if gc:
            st.plotly_chart(gc, use_container_width=True)
            with st.expander("Greeks 数据表（近月截面）"):
                if raw is not None and len(raw):
                    near = raw["dte"].min()
                    g = raw[raw["dte"] == near][["option_type", "strike", "moneyness",
                        "delta", "gamma", "vega", "theta"]].sort_values(["option_type", "moneyness"])
                    st.dataframe(g.style.format({"delta": "{:.4f}", "gamma": "{:.5f}",
                        "vega": "{:.3f}", "theta": "{:.3f}"}), use_container_width=True)
        else:
            st.info("该截面原始数据加载中或无数据")

    # 预警时间轴 + 表
    st.plotly_chart(alert_timeline(alerts, events, symbol), use_container_width=True)
    with st.expander("预警明细表"):
        cols = ["timestamp", "date", "level", "ml_score", "convexity_violation",
                "term_slope_roc", "atm_iv_z", "arb_score", "triggers"]
        show = alerts[alerts["level"].isin(show_levels)][cols].sort_values("timestamp", ascending=False)
        st.dataframe(show.head(50).style.format({"ml_score": "{:.2f}", "convexity_violation": "{:.2f}",
            "term_slope_roc": "{:.2f}", "atm_iv_z": "{:.2f}", "arb_score": "{:.1f}"}),
            use_container_width=True, height=320)

    # 特征仪表 + 表
    fg = feature_gauges(features, alerts, ts)
    if fg:
        st.plotly_chart(fg, use_container_width=True)
    with st.expander("本截面特征值表"):
        if not cur.empty:
            kv = {k: float(cur[k].iloc[0]) for k in
                  ["convexity_violation", "term_slope_roc", "atm_iv_z", "gamma_concentration",
                   "vega_concentration", "arb_score", "iv_spike", "liquidity_z", "atm_iv_vel_z", "convexity_vel_z"]
                  if k in cur.columns}
            st.dataframe(pd.DataFrame({"特征": list(kv), "值": list(kv.values())}).style.format({"值": "{:.4f}"}),
                         use_container_width=True)

    # 回测摘要
    st.subheader("回测摘要")
    from vol_surface.backtest import backtest, format_result
    bt = backtest(alerts[["timestamp", "date", "level"]].assign(symbol=symbol), events, symbol=symbol)
    st.code(format_result(bt))


if __name__ == "__main__":
    main()
