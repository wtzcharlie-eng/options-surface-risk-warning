"""Export the public HTML entry and its reachable, tracked dependencies only."""
from pathlib import Path
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote,quote
import argparse,subprocess,sys,re,html,json,hashlib,os
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
root=a.source.resolve();dest=a.out.resolve();dest.mkdir(parents=True,exist_ok=True)
assert not any(dest.iterdir()),'export requires a new empty output directory'
sys.path.insert(0,str(root))
from scripts.build_docs import md_to_html,page
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=root,text=True).split('\0'))
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
repo='https://github.com/wtzcharlie-eng/options-surface-risk-warning'
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('src','href') and v:self.links.append((tag,k,v))
allowed={'.html','.css','.js','.png','.jpg','.jpeg','.svg','.webp','.ico','.woff','.woff2','.json','.csv','.parquet','.pdf','.md'}
pending=deque(['index.html']);seen=set();missing=[];source_links=set();external_assets=set()
def output_name(name):return str(Path(name).with_suffix('.html')) if name.endswith('.md') else name
def rewrite_refs(text,name):
 parser=Links();parser.feed(text)
 outpath=dest/output_name(name)
 for tag,attr,url in parser.links:
  u=urlsplit(url)
  if u.scheme or u.netloc:
   if u.scheme in {'file'} or u.hostname in {'localhost','127.0.0.1','0.0.0.0'}:
    replacement=repo
   else:
    if tag!='a':external_assets.add(url)
    continue
  else:
   if not u.path:continue
   target=(root/name).parent/unquote(u.path)
   if u.path.startswith('/'):target=root/unquote(u.path).lstrip('/')
   target=target.resolve()
   if not target.is_relative_to(root):raise ValueError('outside-root dependency: '+name)
   rel=target.relative_to(root).as_posix()
   if rel not in tracked or not target.is_file():missing.append((name,rel));continue
   if target.suffix.lower() in allowed and not target.name.startswith('.env'):
    pending.append(rel)
    replacement=quote(os.path.relpath(dest/output_name(rel),outpath.parent).replace(os.sep,'/'),safe='/.-_')
   else:
    source_links.add(rel);replacement=repo+'/blob/'+revision+'/'+quote(rel,safe='/.-_')
   if u.fragment:replacement+='#'+u.fragment
   if u.query:replacement+='?'+u.query
  for q in ('"',"'"):
   text=text.replace(f'{attr}={q}{html.escape(url,quote=True)}{q}',f'{attr}={q}{html.escape(replacement,quote=True)}{q}')
   text=text.replace(f'{attr}={q}{url}{q}',f'{attr}={q}{html.escape(replacement,quote=True)}{q}')
 return text
while pending:
 name=pending.popleft()
 if name in seen:continue
 if name not in tracked:raise ValueError('untracked dependency: '+name)
 p=root/name;assert p.is_file() and not p.is_symlink(),name
 seen.add(name);out=dest/output_name(name);out.parent.mkdir(parents=True,exist_ok=True)
 if p.suffix in {'.html','.md'}:
  text=p.read_text()
  if p.suffix=='.md':
   body,toc=md_to_html(text)
   title=next((ln.lstrip('# ').strip() for ln in text.splitlines() if ln.startswith('# ')),p.stem)
   text=page(title,'公开报告 · 保留原始实验口径与限制说明',body,toc,name)
   home=quote(os.path.relpath(dest/'index.html',out.parent).replace(os.sep,'/'),safe='/.-_')
   # Generated wrappers keep navigation relative to the exported file.
   text=text.replace('href="../index.html"',f'href="{home}"')
  text=rewrite_refs(text,name)
  if name=='index.html':
   text=text.replace('<title>期权曲面风控 · 比赛交付审阅版</title>','<title>期权曲面风控 · 公开展示</title>')
   text=text.replace('</head>','<meta name="description" content="期权曲面风控项目公开展示：方法报告、曲面预警、图谱解释和预测评估。"></head>')
   text=text.replace('<main>','<main><p style="font-size:14px;border-bottom:1px solid var(--line);padding-bottom:18px">公开展示页 · 可直接分享此网址。报告、图表和评估可在线阅读；完整预测与大模型问答需启动项目服务。<a href="'+repo+'" target="_blank" rel="noopener">查看公开源码 ↗</a></p>',1)
   text=text.replace('原始行情和真实 API Key 不包含在交付包中。','本公网展示不包含原始行情、真实 API Key 或比赛私有提交包。')
  out.write_text(text)
 else:out.write_bytes(p.read_bytes())
if missing:raise ValueError('missing dependencies: '+repr(missing))
(dest/'.nojekyll').touch()
for public_file in dest.rglob('*'):
 if public_file.is_file():
  if re.search(rb'(?:sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',public_file.read_bytes()):
   raise ValueError('credential-like content in public export: '+public_file.relative_to(dest).as_posix())
manifest={p.relative_to(dest).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(dest.rglob('*')) if p.is_file()}
report=dict(source_revision=revision,files=len(manifest),size_bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),source_links=sorted(source_links),external_assets=sorted(external_assets),manifest_sha256=manifest)
(dest.parent/'publication-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['source_revision','files','size_bytes','source_links','external_assets']},ensure_ascii=False))
