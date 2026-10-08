"""build_dashboard_preview — 生成预警仪表板的**静态预览页**。

为什么需要它
------------
`dashboard/app.py` 是 Streamlit 应用，**双击打不开**——浏览器会把 Python 源码
原样显示出来。而入口页早前给它加了 `href="dashboard/app.py"`，评委点进去看到的
就是一屏源代码。那是个死链，且比没有链接更糟：它让人以为交付物坏了。

更根本的问题是：**评委不一定装 streamlit**。「预警可视化仪表板」是赛题问题1
明确要求的成果形式，若必须装依赖、起服务才能看见，等于把一项成果交在了门外。

故本脚本用**真实交付数据**渲染一份静态预览：四级预警分布、预警时间轴、
IV 曲面热力图（平静/承压对照）、触发规则频次。原生 SVG、零依赖、双击即开，
配色字体与 `index.html` 完全一致。

**预览不是替代品**：交互功能（Greeks 曲线联动、历史回放、参数调节）仍需
`streamlit run`，页面顶部写明了这一点与启动命令。

用法::

    python scripts/build_dashboard_preview.py     # → data_out/doc_dashboard.html
"""

from __future__ import annotations

import argparse
import html as _html
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_out")

# 与 index.html / platform.html / dashboard 同一套
LEVEL_COLORS = {0: "#4fb79c", 1: "#d8a24a", 2: "#d8874f", 3: "#e8695e"}
LEVEL_NAMES = {0: "正常", 1: "关注", 2: "预警", 3: "严重"}
RAMP = ["#1e3a4a", "#245a63", "#2f8b7e", "#4fb79c", "#a8b26c", "#d8874f", "#e8695e"]


def _esc(s):
    return _html.escape(str(s))


def _ramp(t: float) -> str:
    t = max(0.0, min(1.0, t))
    n = len(RAMP) - 1
    i = min(int(t * n), n - 1)
    f = t * n - i

    def mix(a, b):
        A = [int(a[1:][k:k + 2], 16) for k in (0, 2, 4)]
        B = [int(b[1:][k:k + 2], 16) for k in (0, 2, 4)]
        return "#%02x%02x%02x" % tuple(round(x + (y - x) * f) for x, y in zip(A, B))
    return mix(RAMP[i], RAMP[i + 1])


def _bars(counts: dict, w=520, h=150) -> str:
    """四级分布柱状图。**四级都画**，哪怕某级为 0——
    等级缺失本身是信息（本项目曾因缺陷让 0/2 级不可达）。"""
    keys = [0, 1, 2, 3]
    mx = max(counts.get(k, 0) for k in keys) or 1
    bw, gap = (w - 60) / 4, 14
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for i, k in enumerate(keys):
        v = counts.get(k, 0)
        bh = (h - 46) * v / mx
        x, y = 40 + i * bw, h - 26 - bh
        out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - gap:.1f}" '
                   f'height="{max(bh, 1):.1f}" fill="{LEVEL_COLORS[k]}"/>')
        out.append(f'<text x="{x + (bw - gap) / 2:.1f}" y="{y - 6:.1f}" '
                   f'text-anchor="middle" font-size="12" fill="#c2d0dd">{v}</text>')
        out.append(f'<text x="{x + (bw - gap) / 2:.1f}" y="{h - 8}" '
                   f'text-anchor="middle" font-size="11" fill="#8fa0b2">'
                   f'{k} {LEVEL_NAMES[k]}</text>')
    out.append("</svg>")
    return "".join(out)


def _timeline(df: pd.DataFrame, w=980, h=170) -> str:
    """预警时间轴：横轴时间，纵轴等级，点色随等级。"""
    d = df.sort_values("timestamp").reset_index(drop=True)
    n = len(d)
    if not n:
        return ""
    pad_l, pad_b = 34, 26
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for lv in range(4):
        y = pad_b + (h - pad_b - 14) * (1 - lv / 3)
        out.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - 8}" y2="{y:.1f}" '
                   f'stroke="#1b2734" stroke-width="1"/>')
        out.append(f'<text x="6" y="{y + 4:.1f}" font-size="10" fill="#8fa0b2">'
                   f'{lv}</text>')
    for i, r in enumerate(d.itertuples()):
        x = pad_l + (w - pad_l - 12) * i / max(n - 1, 1)
        lv = int(r.level)
        y = pad_b + (h - pad_b - 14) * (1 - lv / 3)
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2 + lv * .7:.1f}" '
                   f'fill="{LEVEL_COLORS[lv]}" opacity="{.45 + lv * .18:.2f}"/>')
    t0, t1 = str(d.timestamp.iloc[0]), str(d.timestamp.iloc[-1])
    fmt = lambda t: f"{t[:4]}-{t[4:6]}-{t[6:8]}"
    out.append(f'<text x="{pad_l}" y="{h - 6}" font-size="10" fill="#5d6f82">'
               f'{fmt(t0)}</text>')
    out.append(f'<text x="{w - 12}" y="{h - 6}" font-size="10" fill="#5d6f82" '
               f'text-anchor="end">{fmt(t1)}</text>')
    out.append("</svg>")
    return "".join(out)


def _heat(row, lo, hi, w=300, h=78) -> str:
    g = row["surface"]
    nr, nc = len(g), len(g[0])
    cw, ch = w / nc, h / nr
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart">']
    for r in range(nr):
        for c in range(nc):
            v = g[r][c]
            if v is None or v != v:
                continue
            t = (v - lo) / (hi - lo) if hi > lo else .5
            out.append(f'<rect x="{c * cw:.2f}" y="{r * ch:.2f}" '
                       f'width="{cw + .6:.2f}" height="{ch + .6:.2f}" '
                       f'fill="{_ramp(t)}"/>')
    out.append("</svg>")
    return "".join(out)


def build(sym: str) -> str:
    ap = os.path.join(DATA, f"alerts_{sym}.parquet")
    if not os.path.exists(ap):
        raise SystemExit(f"缺 {ap}")
    df = pd.read_parquet(ap)
    counts = df["level"].value_counts().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}

    # 触发规则频次
    rules = {}
    for s in df["trigger_rules"].fillna(""):
        for r in str(s).split(","):
            r = r.strip()
            if r:
                rules[r] = rules.get(r, 0) + 1
    rule_rows = "".join(
        f"<tr><td><code>{_esc(k)}</code></td><td>{v}</td>"
        f"<td>{v / len(df):.1%}</td></tr>"
        for k, v in sorted(rules.items(), key=lambda x: -x[1])[:10])

    # 曲面对照
    heat = ""
    sp = os.path.join(DATA, f"surfaces_{sym}.parquet")
    if os.path.exists(sp):
        s = pd.read_parquet(sp)
        s = s[s["option_type"] == "call"]
        if len(s) >= 2:
            ts_lv = df.set_index("timestamp")["level"].to_dict()
            s = s.assign(lv=[ts_lv.get(t, 0) for t in s["timestamp"]])
            calm = s[s.lv == s.lv.min()].iloc[0]
            stress = s[s.lv == s.lv.max()].iloc[0]
            vals = [v for row in (calm, stress) for r in row["surface"]
                    for v in r if v is not None and v == v]
            lo, hi = min(vals), max(vals)
            fmt = lambda t: f"{str(t)[:4]}-{str(t)[4:6]}-{str(t)[6:8]} {str(t)[8:10]}:{str(t)[10:12]}"
            heat = f"""
  <div class="two">
    <div class="cell"><div class="cap">低等级截面（{int(calm.lv)} 级）</div>
      {_heat(calm, lo, hi)}
      <div class="meta">{_esc(fmt(calm.timestamp))}</div></div>
    <div class="cell"><div class="cap">高等级截面（{int(stress.lv)} 级）</div>
      {_heat(stress, lo, hi)}
      <div class="meta">{_esc(fmt(stress.timestamp))}</div></div>
  </div>
  <p class="note">看涨侧隐含波动率曲面，纵轴到期、横轴 moneyness，两图同一色标。
    色标与入口页主视觉、总览平台是同一条 teal→vermilion 发散刻度。</p>"""

    syms = sorted(f[len("alerts_"):-len(".parquet")] for f in os.listdir(DATA)
                  if f.startswith("alerts_") and f.endswith(".parquet"))
    n_zero = [k for k in range(4) if counts.get(k, 0) == 0]
    zero_note = (f"<p class='warn'>⚠ 等级 {n_zero} 一次都没出现——"
                 f"四级预警未全部可达，须查证。</p>" if n_zero else
                 "<p class='note'>四个等级均有出现。这一点值得单独确认："
                 "本项目曾因异常分缺陷让 0 级与 2 级<b>结构上不可达</b>，"
                 "四级预警塌成两级（见 README §7.25）。</p>")

    return f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>预警仪表板 · 静态预览</title>
<style>
:root{{color-scheme:dark;--ink:#0e1621;--panel:#151f2b;--rule:#22303f;
 --fg:#e3eaf1;--fg2:#8fa0b2;--fg3:#5d6f82;--calm:#4fb79c;--caveat:#d8a24a;
 --sans:"Times New Roman",Times,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;
 --mono:"Times New Roman",Times,ui-monospace,Menlo,Consolas,monospace}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ink);color:var(--fg);font-family:var(--sans);
 font-size:15px;line-height:1.8}}
a{{color:var(--calm);text-decoration:none}} a:hover{{text-decoration:underline}}
.bar{{position:sticky;top:0;z-index:9;background:rgba(14,22,33,.94);
 backdrop-filter:blur(8px);border-bottom:1px solid var(--rule)}}
.bar .in{{max-width:1080px;margin:0 auto;padding:13px 28px;display:flex;
 align-items:baseline;gap:16px}}
.bar h1{{font-size:16px;margin:0;font-weight:700}}
.bar .sub{{color:var(--fg3);font-size:12.5px;margin-left:auto}}
.wrap{{max-width:1080px;margin:0 auto;padding:30px 28px 90px}}
h2{{font-size:19px;margin:38px 0 12px;font-weight:700;padding-bottom:8px;
 border-bottom:1px solid var(--rule)}}
.launch{{border:1px solid var(--caveat);background:#1d1712;padding:15px 19px;margin:0 0 26px}}
.launch b{{color:var(--caveat)}}
.launch pre{{margin:9px 0 0;font-family:var(--mono);font-size:13px;color:#cfe0ee}}
.panel{{border:1px solid var(--rule);background:var(--panel);padding:16px 20px;margin:14px 0}}
.chart{{display:block;width:100%;height:auto}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.cell{{border:1px solid var(--rule);background:var(--panel);padding:13px}}
.cap{{font-family:var(--mono);font-size:11.5px;color:var(--fg2);margin-bottom:9px}}
.meta{{font-family:var(--mono);font-size:11px;color:var(--fg3);margin-top:8px}}
.note{{font-size:13px;color:var(--fg2);margin:12px 0}}
.note b{{color:var(--caveat)}}
.warn{{font-size:13px;color:#e8695e;margin:12px 0}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}}
th,td{{border:1px solid var(--rule);padding:7px 11px;text-align:left}}
th{{background:#1a2532;font-weight:700}} td{{color:var(--fg2)}}
code{{font-family:var(--mono);font-size:12.5px;color:#cfe0ee}}
.foot{{margin-top:44px;padding-top:18px;border-top:1px solid var(--rule);
 font-size:12px;color:var(--fg3);line-height:1.9}}
@media(max-width:760px){{.two{{grid-template-columns:1fr}}}}
</style></head><body>
<div class="bar"><div class="in"><a href="../index.html">← 交付入口</a>
  <h1>预警可视化仪表板 · 静态预览</h1>
  <span class="sub">品种 {_esc(sym)}　·　{len(df):,} 个截面</span></div></div>
<div class="wrap">

<div class="launch">
  <b>这是静态预览，不是仪表板本身。</b>
  完整仪表板是 Streamlit 应用（曲面热力图、Greeks 曲线联动、预警时间轴、
  特征仪表、历史回放），<b>双击打不开</b>，需要起服务：
  <pre>cd dashboard &amp;&amp; streamlit run app.py</pre>
  <p class="note" style="margin-bottom:0">本页用<b>同一批交付数据</b>把其中几块静态渲染出来，
  供不便安装依赖时查看。交互功能仍需上面的命令。</p>
</div>

<h2>四级预警分布</h2>
<div class="panel">{_bars(counts)}</div>
{zero_note}

<h2>预警时间轴</h2>
<div class="panel">{_timeline(df)}</div>
<p class="note">横轴为采样截面的时间顺序，纵轴为预警等级，点的大小与不透明度随等级递增。
  仪表板里这张图可按事件区间高亮、悬停查看触发原因。</p>

<h2>IV 曲面热力图</h2>
{heat or '<p class="note">该品种缺 surfaces parquet，无法渲染曲面。</p>'}

<h2>触发规则频次</h2>
<div class="panel"><table>
<thead><tr><th>规则</th><th>触发截面数</th><th>占比</th></tr></thead>
<tbody>{rule_rows or '<tr><td colspan="3">无触发记录</td></tr>'}</tbody>
</table></div>
<p class="note">每条预警的触发原因都来自<b>规则层</b>并可读回溯——
  可解释性由规则层提供，ML 层只贡献异常分。</p>

<p class="foot">
  数据源　<code>data_out/alerts_{_esc(sym)}.parquet</code>、
  <code>surfaces_{_esc(sym)}.parquet</code>（共 {len(syms)} 个品种可选）<br>
  完整仪表板　<code>cd dashboard &amp;&amp; streamlit run app.py</code>　·
  主题 <code>.streamlit/config.toml</code><br>
  本页由 <code>scripts/build_dashboard_preview.py</code> 生成
</p>
</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="lc")
    ap.add_argument("--out", default=os.path.join(DATA, "doc_dashboard.html"))
    a = ap.parse_args()
    h = build(a.symbol)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(h)
    print(f"  → {os.path.relpath(a.out, ROOT)}  ({len(h.encode('utf-8')):,} 字节)")


if __name__ == "__main__":
    main()
