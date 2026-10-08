"""viz_kg — 生成风险传导路径可视化界面（自包含 HTML）。

赛题成果形式要求「风险传导路径可视化界面」。这里输出一个**零依赖、可离线打开**的
单文件 HTML：图数据内联为 JSON，绘制用原生 SVG + JS，不引任何 CDN。
这样评审双击即可查看，不需要起服务、不需要装包。

用法::

    python scripts/viz_kg.py                       # → data_out/kg/kg_viz.html
    python scripts/viz_kg.py --n-cases 8           # 内嵌的真实案例条数
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.dataset import FEATURES, load_episodes, split_episodes
from kg.explain import METHOD_NOTE, explain
from kg.graph import DiGraph
from kg.reason import reason
from kg.schema import ANOM_BY_ID, CONSEQ_BY_ID, MECH_BY_ID

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_DIR = os.path.join(ROOT, "data_out", "kg")

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>期权风险传导图谱</title>
<style>
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#f7f7f5;color:#1b1b18;
 font:15px/1.65 "Noto Sans SC","PingFang SC","Microsoft YaHei",-apple-system,sans-serif}
header{padding:22px 30px 14px;border-bottom:1px solid #e3e2dd;background:#fff}
h1{margin:0 0 6px;font-size:20px;font-weight:650;letter-spacing:.01em}
.sub{color:#6b6a63;font-size:13px}
.wrap{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,1fr);
 gap:18px;padding:18px 30px 40px;max-width:1500px;margin:0 auto}
@media(max-width:1040px){.wrap{grid-template-columns:1fr}}
.card{background:#fff;border:1px solid #e3e2dd;border-radius:10px;padding:16px 18px}
.card h2{margin:0 0 4px;font-size:14px;font-weight:650;letter-spacing:.02em}
.card .hint{color:#6b6a63;font-size:12px;margin-bottom:12px}
svg{width:100%;height:auto;display:block}
.node rect{rx:7;stroke-width:1.2}
.node text{font-size:12.5px;dominant-baseline:middle}
.node .sub{font-size:10.5px;fill:#7a7970}
.edge{fill:none;stroke:#b9b7ad;opacity:.5;transition:.15s}
.edge.rep{stroke:#4a7c59}
.edge.norep{stroke:#c08a3e;stroke-dasharray:4 3}
.edge.on{opacity:1;stroke-width:3.2}
.edge.dim{opacity:.08}
.node.dim{opacity:.25}
.node{cursor:pointer}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:11.5px;color:#6b6a63;
 margin-top:10px;padding-top:10px;border-top:1px solid #eeede8}
.legend i{display:inline-block;width:16px;height:0;border-top-width:2.5px;
 border-top-style:solid;vertical-align:middle;margin-right:5px}
#detail{font-size:13px}
#detail .ttl{font-weight:650;margin:12px 0 4px;font-size:13.5px}
#detail table{border-collapse:collapse;width:100%;margin-top:8px;font-size:12px}
#detail td{padding:3px 6px;border-bottom:1px solid #f0efea;vertical-align:top}
#detail td:first-child{color:#6b6a63;white-space:nowrap;width:40%}
.tag{display:inline-block;padding:1px 7px;border-radius:20px;font-size:10.5px;
 margin-left:6px;vertical-align:middle}
.tag.m{background:#e6efe8;color:#3d6b4c}
.tag.i{background:#f2ece0;color:#8a6d35}
.tag.w{background:#f7e6e6;color:#9a4b4b}
select{width:100%;padding:7px 9px;border:1px solid #dad8d1;border-radius:6px;
 background:#fff;font:inherit;font-size:13px;margin-bottom:12px}
.exp{white-space:pre-wrap;font-size:12.5px;line-height:1.7;max-height:520px;
 overflow:auto;padding-right:6px}
.exp b{font-weight:650}
.foot{padding:0 30px 34px;max-width:1500px;margin:0 auto;color:#6b6a63;font-size:12px}
</style></head><body>
<header>
  <h1>期权风险传导知识图谱</h1>
  <div class="sub">左侧为曲面异常（根因），右侧为可测后果；连线粗细对应实测提升度 lift。
  点击任一节点可高亮其传导路径。<b>实线＝测试期可复现，虚线＝仅训练期显著。</b></div>
</header>
<div class="wrap">
  <div class="card">
    <h2>传导路径图</h2>
    <div class="hint">仅显示通过「支撑度 + 分块置换检验 + lift 置信下界」三项检验的边；
    未通过的候选边不进入本图。</div>
    <div id="graph"></div>
    <div class="legend">
      <span><i style="border-color:#4a7c59"></i>测试期可复现</span>
      <span><i style="border-color:#c08a3e;border-top-style:dashed"></i>仅训练期显著</span>
      <span><span class="tag m">实测层</span>权重来自数据</span>
      <span><span class="tag i">诠释层</span>业务先验，不参与打分</span>
      <span><span class="tag w">同源特征</span>与后果共用底层指标</span>
    </div>
  </div>
  <div class="card">
    <h2>边详情 / 证据</h2>
    <div class="hint">点击左图节点或连线查看。</div>
    <div id="detail">选择一个节点或连线以查看其统计证据。</div>
  </div>
  <div class="card" style="grid-column:1/-1">
    <h2>真实截面案例</h2>
    <div class="hint">下列案例取自测试期（2025-07..2026-04）真实数据，
    解释文本由 <code>kg/explain.py</code> 自动生成。</div>
    <details class="lede"><summary><b>[†] / [‡] 全文统一的统计口径</b>（点开）</summary>
    <div class="exp" style="white-space:pre-wrap">{METHOD_NOTE_HTML}</div></details>
    <select id="pick"></select>
    <div class="exp" id="exp"></div>
  </div>
</div>
<div class="foot">
  本界面仅用于风险提示与成因分析，不构成任何交易指令或操作建议。<br>
  由 <code>scripts/viz_kg.py</code> 生成，数据内联、零外部依赖。
</div>
<script>
const DATA = __DATA__;
const W=760, PAD=26, BOXW=210, BOXH=44;
const A=DATA.anoms, C=DATA.conseqs, E=DATA.edges;
const rowsA=A.length, rowsC=C.length;
const GAP=16, H=Math.max(rowsA,rowsC)*(BOXH+GAP)+PAD*2;
const yA=i=>PAD+i*(BOXH+GAP)+BOXH/2+(Math.max(rowsA,rowsC)-rowsA)*(BOXH+GAP)/2;
const yC=i=>PAD+i*(BOXH+GAP)+BOXH/2+(Math.max(rowsA,rowsC)-rowsC)*(BOXH+GAP)/2;
const xA=PAD, xC=W-PAD-BOXW;
const lifts=E.map(e=>e.lift), lmin=Math.min(...lifts), lmax=Math.max(...lifts);
const sw=l=>1.1+4.6*(lmax>lmin?(l-lmin)/(lmax-lmin):.5);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="风险传导路径图">`;
E.forEach((e,i)=>{
  const y1=yA(A.findIndex(a=>a.id===e.source)), y2=yC(C.findIndex(c=>c.id===e.target));
  const x1=xA+BOXW, x2=xC, mx=(x1+x2)/2;
  svg+=`<path class="edge ${e.replicated?'rep':'norep'}" data-e="${i}"
    d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}"
    stroke-width="${sw(e.lift).toFixed(2)}"><title>${esc(e.stitle)}</title></path>`;
});
A.forEach((a,i)=>{
  svg+=`<g class="node" data-n="${a.id}"><rect x="${xA}" y="${yA(i)-BOXH/2}"
    width="${BOXW}" height="${BOXH}" fill="#fbfaf7" stroke="#d6d4cb"/>
    <text x="${xA+12}" y="${yA(i)-7}">${esc(a.title)}</text>
    <text class="sub" x="${xA+12}" y="${yA(i)+11}">${esc(a.rule)}</text></g>`;
});
C.forEach((c,i)=>{
  svg+=`<g class="node" data-n="${c.id}"><rect x="${xC}" y="${yC(i)-BOXH/2}"
    width="${BOXW}" height="${BOXH}" fill="#f5f7f5" stroke="#cdd6ce"/>
    <text x="${xC+12}" y="${yC(i)-7}">${esc(c.title)}</text>
    <text class="sub" x="${xC+12}" y="${yC(i)+11}">${esc(c.feature)}</text></g>`;
});
svg+=`</svg>`;
document.getElementById('graph').innerHTML=svg;

const det=document.getElementById('detail');
function clear(){document.querySelectorAll('.edge,.node').forEach(x=>
  x.classList.remove('on','dim'));}
function showEdge(i){
  const e=E[i]; clear();
  document.querySelectorAll('.edge').forEach((p,k)=>
    p.classList.add(k===i?'on':'dim'));
  document.querySelectorAll('.node').forEach(n=>{
    if(n.dataset.n!==e.source&&n.dataset.n!==e.target) n.classList.add('dim');});
  det.innerHTML=`<div class="ttl">${esc(e.stitle)}
    <span class="tag m">实测层</span>${e.self_feature?'<span class="tag w">同源特征</span>':''}</div>
    <table>
    <tr><td>提升度 lift</td><td><b>${e.lift.toFixed(2)}</b>（95% 保守下界 ${e.weight.toFixed(2)}）</td></tr>
    <tr><td>P(后果|异常)</td><td>${(e.p_cond*100).toFixed(1)}%</td></tr>
    <tr><td>无条件基础率</td><td>${(e.p_conseq*100).toFixed(1)}%</td></tr>
    <tr><td>支撑度</td><td>${e.n_anom.toLocaleString()} 次</td></tr>
    <tr><td>分块置换 p 值</td><td>${e.p_value.toFixed(4)}</td></tr>
    <tr><td>测试期复现</td><td>${e.replicated?'✓ 是':'✗ 否（仅供参考）'}</td></tr>
    <tr><td>后果判定口径</td><td>${esc(e.detect)}</td></tr>
    </table>
    ${e.mech?`<div class="ttl">传导机制 <span class="tag i">诠释层</span></div>
      <div style="font-size:12.5px;color:#4a4942">${esc(e.mech_name)}——${esc(e.how)}<br>
      <em style="color:#8a897f">机制为业务先验说明，不参与权重计算。</em></div>`:
      `<div class="ttl">传导机制</div><div style="font-size:12.5px;color:#8a897f">
      该关联由数据观测到，但尚未归纳出对应的机制说明。</div>`}
    ${e.self_feature?`<div style="margin-top:10px;font-size:12px;color:#9a4b4b">
      注意：该关联与后果判定共用同一底层指标，部分来自同一指标的时间自相关，
      并非完全独立的证据。</div>`:''}`;
}
function showNode(id){
  clear();
  const rel=[];
  document.querySelectorAll('.edge').forEach((p,k)=>{
    const e=E[k];
    if(e.source===id||e.target===id){p.classList.add('on');rel.push(e);}
    else p.classList.add('dim');});
  const keep=new Set([id]); rel.forEach(e=>{keep.add(e.source);keep.add(e.target);});
  document.querySelectorAll('.node').forEach(n=>{
    if(!keep.has(n.dataset.n)) n.classList.add('dim');});
  const meta=[...A,...C].find(x=>x.id===id)||{};
  det.innerHTML=`<div class="ttl">${esc(meta.title||id)}</div>
    <div style="font-size:12.5px;color:#4a4942">${esc(meta.desc||'')}</div>
    <div class="ttl">相关传导边（${rel.length} 条）</div>
    <table>${rel.sort((a,b)=>b.lift-a.lift).map(e=>
      `<tr><td>${esc(e.stitle)}</td><td>lift <b>${e.lift.toFixed(2)}</b>
       ${e.replicated?'':'<span class="tag w">未复现</span>'}</td></tr>`).join('')
      ||'<tr><td colspan="2">该节点没有通过检验的传导边。</td></tr>'}</table>`;
}
document.querySelectorAll('.edge').forEach((p,i)=>p.addEventListener('click',
  ev=>{ev.stopPropagation();showEdge(i);}));
document.querySelectorAll('.node').forEach(n=>n.addEventListener('click',
  ev=>{ev.stopPropagation();showNode(n.dataset.n);}));

const pick=document.getElementById('pick'), expEl=document.getElementById('exp');
DATA.cases.forEach((c,i)=>{const o=document.createElement('option');
  o.value=i;o.textContent=`${c.label}`;pick.appendChild(o);});
function render(i){
  expEl.innerHTML=esc(DATA.cases[i].text).replace(/\\*\\*(.+?)\\*\\*/g,'<b>$1</b>');}
pick.addEventListener('change',e=>render(+e.target.value));
if(DATA.cases.length) render(0);
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg-dir", default=KG_DIR)
    ap.add_argument("--out", default=os.path.join(KG_DIR, "kg_viz.html"))
    ap.add_argument("--n-cases", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    g = DiGraph.load(os.path.join(args.kg_dir, "kg_graph.json"))
    anoms = [{"id": n, "title": g.node(n)["title"], "rule": g.node(n)["rule"],
              "desc": g.node(n).get("desc", "")} for n in sorted(g.nodes("ANOM"))]
    conseqs = [{"id": n, "title": g.node(n)["title"],
                "feature": g.node(n).get("feature", ""),
                "desc": g.node(n).get("desc", "")} for n in sorted(g.nodes("CONSEQ"))]
    edges = []
    for u, v, a in sorted(g.edges("AMPLIFIES"), key=lambda e: (e[0], e[1])):
        mech = a.get("mech")
        edges.append({
            "source": u, "target": v,
            "stitle": f"{ANOM_BY_ID[u].name} → {CONSEQ_BY_ID[v].name}",
            "lift": a["lift"], "weight": a["weight"], "p_cond": a["p_cond"],
            "p_conseq": a["p_conseq"], "n_anom": a["n_anom"],
            "p_value": a["p_value"], "replicated": bool(a.get("replicated")),
            "self_feature": bool(a.get("self_feature")),
            "detect": CONSEQ_BY_ID[v].rule_text(),
            "mech": mech or "", "mech_name": MECH_BY_ID[mech].name if mech else "",
            "how": a.get("how", ""),
        })

    # 真实案例
    eps = load_episodes(os.path.join(ROOT, "data_out", "anchor"), verify=True)
    _, _, te = split_episodes(eps)
    rng = np.random.default_rng(args.seed)
    cases, tries = [], 0
    while len(cases) < args.n_cases and tries < args.n_cases * 80:
        tries += 1
        e = te[int(rng.integers(0, len(te)))]
        i = int(rng.integers(0, len(e)))
        feats = {k: float(e.X[i, j]) for j, k in enumerate(FEATURES)}
        res = reason(g, feats, symbol=e.symbol, timestamp=e.ts[i])
        if res["n_active"] < 2:
            continue
        cases.append({"label": f"{e.symbol} {e.ts[i]}（{res['n_active']} 项异常）",
                      "text": explain(res)["text"]})

    payload = {"anoms": anoms, "conseqs": conseqs, "edges": edges, "cases": cases}
    # [†]/[‡] 的定义必须随解释文本进入本页面，否则页内会出现 59 处无定义的悬空标记
    # （第 9 位评审实测）。方法学口径因子化到容器是设计，但**每个容器都要渲染一次**。
    import html as _html
    doc = (TEMPLATE
           .replace("{METHOD_NOTE_HTML}",
                    _html.escape(METHOD_NOTE).replace("**", ""))
           .replace("__DATA__", json.dumps(payload, ensure_ascii=False)))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"→ {args.out}  ({len(doc):,} 字节, {len(edges)} 条边, {len(cases)} 个案例)")


if __name__ == "__main__":
    main()
