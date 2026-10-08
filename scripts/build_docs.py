"""build_docs — 把交付文档渲染成与入口页同主题的深色 HTML。

为什么要有它
------------
`README.md` / `PACKAGE.md` / `drl_report.md` / `kg_report.md` 都是 Markdown，
评委点开看到的是**没有任何样式的纯文本**——与 `index.html` 的深色封面风格断裂，
表格是一堆竖线，代码块和正文分不开。

本脚本用一个**极小的 Markdown 子集渲染器**（无第三方依赖，与项目「零新增依赖」
一致）把它们转成 HTML，配色/字体与 `index.html` 完全一致，并自动生成右侧目录。

**只做渲染，不改内容。** 原 `.md` 仍然留在包里——它是可 grep、可 diff 的正本，
HTML 只是给人看的那一份。

支持的 Markdown 子集（够这四份文档用，不做通用实现）::

    # ~ ###### 标题        ``` 代码块        | 表格 |
    > 引用块              - / 1. 列表        --- 分隔线
    **粗体** `行内代码` [链接](url)

用法::

    python scripts/build_docs.py            # → data_out/doc_*.html
"""

from __future__ import annotations

import argparse
import html as _html
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_out")

# (源文件, 输出名, 页面标题, 一句话说明)
DOCS = [
    (os.path.join(ROOT, "方法与结果.md"), "doc_paper.html",
     "方法与结果",
     "学术结构正文：摘要 / 数据与口径 / 方法 / 实验与结果 / 验证与可复现 / 局限与讨论。"),
    (os.path.join(DATA, "drl_report.md"), "doc_drl_report.html",
     "问题3 · 离线回测报告",
     "基线对照、各 seed、收敛曲线、奖励消融。全部数字由 report_drl.py 从 JSON 读出。"),
    (os.path.join(DATA, "kg_report.md"), "doc_kg_report.html",
     "问题2 · 知识图谱报告",
     "边权标定、解释样本、验证门与自评记录。由 report_kg.py 生成。"),
    (os.path.join(ROOT, "README.md"), "doc_readme.html",
     "完整工程记录（附录）",
     "逐轮修订：每一次改动的动机、对照与结论——包括推翻自己的那些。"),
    (os.path.join(ROOT, "kg", "schema.md"), "doc_kg_schema.html",
     "问题2 · 知识图谱 Schema",
     "节点类型（曲面特征 / Greeks / 标的）、边语义与标定口径。赛题要求的 Schema 文档。"),
    (os.path.join(ROOT, "PACKAGE.md"), "doc_package.html",
     "交付包说明",
     "包里有什么、缺什么、什么能直接跑，以及容易踩的坑。"),
]


# ---------------------------------------------------------------- 行内

def _inline(s: str) -> str:
    """行内标记。**先转义再替换**——否则文档里的 `<` 会被当成标签。"""
    s = _html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    # ⚠ 粗体不能写成 `\*\*([^*]+)\*\*`：本项目文档里的粗体**经常包含星号**
    # （`data_out/*.joblib`、`features_*.parquet` 这类 glob），
    # `[^*]+` 会整段匹配失败，于是 `**` 原样漏到页面上。实测漏了 10+ 处。
    # 用非贪婪 `.+?` 即可，且此时行内代码已替换成 <code>，不会被吃掉。
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def _slug(t: str, seen: dict) -> str:
    b = re.sub(r"[^\w一-鿿]+", "-", t).strip("-").lower() or "h"
    seen[b] = seen.get(b, 0) + 1
    return b if seen[b] == 1 else f"{b}-{seen[b]}"


def md_to_html(md: str) -> tuple:
    """返回 (正文 HTML, 目录条目)。目录只收 ## 与 ###。"""
    out, toc, seen = [], [], {}
    lines = md.split("\n")
    i, n = 0, len(lines)
    in_code = False
    code_buf = []
    para = []
    lst = None            # None | 'ul' | 'ol'

    def flush_para():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    def flush_list():
        nonlocal lst
        if lst:
            out.append(f"</{lst}>")
            lst = None

    while i < n:
        ln = lines[i]

        # 代码块
        if ln.lstrip().startswith("```"):
            if in_code:
                out.append("<pre><code>" + _html.escape("\n".join(code_buf))
                           + "</code></pre>")
                code_buf, in_code = [], False
            else:
                flush_para(); flush_list(); in_code = True
            i += 1
            continue
        if in_code:
            code_buf.append(ln)
            i += 1
            continue

        # 表格：当前行有 | 且下一行是分隔行
        if "|" in ln and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$",
                                                lines[i + 1]):
            flush_para(); flush_list()
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
            head = cells(ln)
            i += 2
            body = []
            while i < n and "|" in lines[i] and lines[i].strip():
                body.append(cells(lines[i]))
                i += 1
            th = "".join(f"<th>{_inline(c)}</th>" for c in head)
            tb = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r)
                         + "</tr>" for r in body)
            out.append(f"<div class='tw'><table><thead><tr>{th}</tr></thead>"
                       f"<tbody>{tb}</tbody></table></div>")
            continue

        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            flush_para(); flush_list()
            lv, txt = len(m.group(1)), m.group(2).strip()
            sid = _slug(txt, seen)
            if lv in (2, 3):
                toc.append((lv, txt, sid))
            out.append(f'<h{lv} id="{sid}">{_inline(txt)}</h{lv}>')
            i += 1
            continue

        # 引用块（连续多行合并成一个）
        if ln.lstrip().startswith(">"):
            flush_para(); flush_list()
            buf = []
            while i < n and lines[i].lstrip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner, _ = md_to_html("\n".join(buf))
            out.append(f"<blockquote>{inner}</blockquote>")
            continue

        # 分隔线
        if re.match(r"^\s*(-{3,}|\*{3,})\s*$", ln):
            flush_para(); flush_list()
            out.append("<hr>")
            i += 1
            continue

        # 列表
        m = re.match(r"^\s*([-*+]|\d+\.)\s+(.*)$", ln)
        if m:
            flush_para()
            want = "ol" if m.group(1)[0].isdigit() else "ul"
            if lst != want:
                flush_list()
                out.append(f"<{want}>")
                lst = want
            out.append(f"<li>{_inline(m.group(2))}</li>")
            i += 1
            continue

        # 空行 / 正文
        if not ln.strip():
            flush_para(); flush_list()
        else:
            para.append(ln.strip())
        i += 1

    flush_para(); flush_list()
    if in_code and code_buf:                    # 未闭合的代码块也要吐出来
        out.append("<pre><code>" + _html.escape("\n".join(code_buf)) + "</code></pre>")
    return "\n".join(out), toc


# ---------------------------------------------------------------- 页面

CSS = """
:root{
  color-scheme:dark;
  --ink:#0e1621; --panel:#151f2b; --rule:#22303f;
  --fg:#e3eaf1; --fg2:#8fa0b2; --fg3:#5d6f82;
  --calm:#4fb79c; --stress:#e8695e; --caveat:#d8a24a;
  /* 与 index.html 完全一致：西文族在前走 Times，汉字回退到黑体 */
  --sans:"Times New Roman",Times,"PingFang SC","Hiragino Sans GB",
         "Microsoft YaHei","Source Han Sans SC",sans-serif;
  --mono:"Times New Roman",Times,ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ink);color:var(--fg);font-family:var(--sans);
  font-size:15px;line-height:1.8;-webkit-font-smoothing:antialiased}
a{color:var(--calm);text-decoration:none}
a:hover,a:focus-visible{text-decoration:underline}
:focus-visible{outline:2px solid var(--calm);outline-offset:2px}

.bar{position:sticky;top:0;z-index:9;background:rgba(14,22,33,.94);
  backdrop-filter:blur(8px);border-bottom:1px solid var(--rule)}
.bar .in{max-width:1180px;margin:0 auto;padding:13px 28px;display:flex;
  align-items:baseline;gap:16px}
.bar .home{color:var(--fg2);font-size:13px}
.bar h1{font-size:16px;margin:0;font-weight:700;letter-spacing:.01em}
.bar .sub{color:var(--fg3);font-size:12.5px;margin-left:auto}

.shell{max-width:1180px;margin:0 auto;padding:0 28px;display:grid;
  grid-template-columns:1fr 236px;gap:44px;align-items:start}
main{min-width:0;padding:34px 0 96px}

nav.toc{position:sticky;top:66px;padding:34px 0;max-height:calc(100vh - 76px);
  overflow:auto}
nav.toc .h{font-size:11px;letter-spacing:.14em;color:var(--fg3);
  margin-bottom:10px;font-family:var(--mono)}
nav.toc a{display:block;color:var(--fg2);font-size:12.5px;line-height:1.55;
  padding:3px 0 3px 10px;border-left:1px solid var(--rule);text-decoration:none}
nav.toc a:hover{color:var(--fg);border-left-color:var(--calm)}
nav.toc a.l3{padding-left:22px;color:var(--fg3);font-size:12px}

h1,h2,h3,h4,h5,h6{font-weight:700;line-height:1.35;letter-spacing:.01em}
main h1{font-size:27px;margin:6px 0 22px}
main h2{font-size:21px;margin:44px 0 14px;padding-bottom:9px;
  border-bottom:1px solid var(--rule)}
main h3{font-size:17px;margin:30px 0 10px;color:#f0f5fa}
main h4{font-size:15px;margin:22px 0 8px;color:var(--fg2)}
p{margin:13px 0}
strong{color:#f4f8fc;font-weight:700}
code{font-family:var(--mono);font-size:13.5px;background:#1b2735;
  border:1px solid var(--rule);border-radius:3px;padding:1px 5px;color:#cfe0ee}
pre{background:var(--panel);border:1px solid var(--rule);padding:15px 17px;
  overflow:auto;margin:16px 0}
pre code{background:none;border:0;padding:0;font-size:13px;line-height:1.65;
  color:#cfe0ee}
blockquote{margin:16px 0;padding:2px 18px;border-left:3px solid var(--caveat);
  background:#141d29;color:var(--fg2)}
blockquote strong{color:var(--caveat)}
blockquote p{margin:10px 0}
ul,ol{margin:13px 0;padding-left:24px}
li{margin:5px 0}
hr{border:0;border-top:1px solid var(--rule);margin:34px 0}
.tw{overflow-x:auto;margin:17px 0}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{border:1px solid var(--rule);padding:8px 11px;text-align:left;
  vertical-align:top}
th{background:#1a2532;color:var(--fg);font-weight:700;white-space:nowrap}
td{color:var(--fg2)}
tbody tr:nth-child(even) td{background:#121b26}
td code,th code{font-size:12.5px}

.foot{margin-top:52px;padding-top:18px;border-top:1px solid var(--rule);
  font-size:12px;color:var(--fg3);line-height:1.9}
@media (max-width:900px){
  .shell{grid-template-columns:1fr;gap:0}
  nav.toc{position:static;max-height:none;padding:0 0 20px}
}
"""


def page(title: str, sub: str, body: str, toc: list, src: str) -> str:
    tl = "".join(f'<a href="#{sid}" class="l{lv}">{_html.escape(t)}</a>'
                 for lv, t, sid in toc)
    return f"""<!doctype html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_html.escape(title)} · 期权波动率曲面风控预警系统</title>
<style>{CSS}</style></head><body>
<div class="bar"><div class="in">
  <a class="home" href="../index.html">← 交付入口</a>
  <h1>{_html.escape(title)}</h1>
  <span class="sub">{_html.escape(sub)}</span>
</div></div>
<div class="shell">
  <main>{body}
    <p class="foot">源文件 <code>{_html.escape(src)}</code>　·
      本页由 <code>scripts/build_docs.py</code> 渲染，<strong>不改内容</strong>；
      正本仍是那份 Markdown。</p>
  </main>
  <nav class="toc"><div class="h">目录</div>{tl}</nav>
</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=DATA)
    a = ap.parse_args()
    n = 0
    for src, name, title, sub in DOCS:
        if not os.path.exists(src):
            print(f"  · 跳过（缺）{os.path.relpath(src, ROOT)}")
            continue
        body, toc = md_to_html(open(src, encoding="utf-8").read())
        html = page(title, sub, body, toc, os.path.relpath(src, ROOT))
        p = os.path.join(a.out_dir, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  → {os.path.relpath(p, ROOT)}  "
              f"({len(html.encode('utf-8')):,} 字节，目录 {len(toc)} 条)")
        n += 1
    if not n:
        raise SystemExit("一份文档都没渲染出来——检查源文件路径")


if __name__ == "__main__":
    main()
