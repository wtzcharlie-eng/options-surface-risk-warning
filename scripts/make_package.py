"""Build and verify a portable zip, including hashes and checks in the extracted copy."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
NAME="期权曲面风控预警系统"
INCLUDE_DIRS=["vol_surface","kg","drl","scripts","dashboard","tests","data_out","docs","deployment",".streamlit",".github"]
INCLUDE_FILES=["README.md","README.legacy.md","PACKAGE.md","requirements.txt","requirements-predictive.lock.txt","extreme_event.csv","index.html","方法与结果.md","升级说明.md","启动系统.command",".env.example","Dockerfile",".dockerignore","compose.yaml","部署说明.md"]
REQUIRED = [
    # 门与结论依赖的模型 —— 上一版包整类漏掉了
    ("data_out/rule_ml_model.joblib", "问题1 融合达标配置的模型；check_doc_numbers 校验其指纹"),
    ("data_out/rule_ml_model.fingerprint.json", "上者的特征名与顺序、切分边界"),
    *[(f"data_out/model_{s}.joblib", f"{s} 仪表板 ML 层；G14 逐个检查其分数未退化")
      for s in ("ag", "au", "cu", "lc", "rb", "sc", "si")],
    # 门的外部参照
    ("data_out/alert_engine_golden.json", "G13 的金标准夹具；缺了该门无法证明公共路径未被改动"),
    # 门本体与其落盘
    ("data_out/drl/gates.json", "问题3 验证门结果"),
    ("data_out/kg/gates.json", "问题2 验证门结果"),
    # 交付结论的落盘
    ("data_out/q1_7sym.json", "问题1 事件窗口交付数字"),
    ("data_out/metrics_v2.json", "新口径交付数字"),
    ("data_out/per_symbol.json", "逐品种对照（含 rule_R，用于同召回比较）"),
    ("data_out/drl/workpoint.json", "问题3 交付工作点"),
    ("data_out/drl/results.json", "问题3 全部 seed 结果"),
    # 本轮新增
    ("data_out/ml_score_defect.json", "异常分缺陷的量化（README §7.25）"),
    ("scripts/ml_score_defect.py", "上者的复算脚本"),
    ("scripts/preflight.py", "交付前一键验收；PACKAGE.md 第 0 条命令就是它"),
    ("index.html", "交付入口页：评委解压后第一个打开的文件"),
    ("scripts/build_index.py", "上者的生成脚本（页面无手写数字，全部现算）"),
    ("scripts/build_docs.py", "把 Markdown 渲染成与入口页同主题的深色 HTML"),
    ("方法与结果.md", "学术结构正文；入口页「方法与结果」卡片指向它的 HTML 版"),
    ("scripts/build_paper.py", "上者的生成脚本（正文无手写数值，全部现算）"),
    (".streamlit/config.toml", "仪表板主题：与入口页同一套色值，缺了外壳会变浅色"),
    ("dashboard/app.py", "问题1 成果形式「预警可视化仪表板」，同时是多页应用的首页"),
    ("data_out/doc_dashboard.html", "仪表板静态预览：不装 streamlit 也能看见这项成果"),
    ("scripts/build_dashboard_preview.py", "上者的生成脚本"),
    *[(f"dashboard/pages/{n}", f"多页交互系统：{n[2:-3]}")
      for n in ("1_传导推理.py", "2_DRL推理.py", "3_系统总览.py")],
    ("data_out/kg/kg_viz.html", "问题2 成果形式「风险传导路径可视化界面」（静态零依赖版）"),
    ("kg/schema.md", "问题2 成果形式「知识图谱 Schema 文档」"),
    ("drl/api.py", "问题3 成果形式「模型推理 API 接口」"),
    ("vol_surface/replay.py", "问题1「历史数据回放」"),
    ("scripts/scan.py", "问题1「定时扫描」"),
    *[(f"data_out/doc_{n}.html", f"{n} 的深色 HTML 版；入口页卡片指向它")
      for n in ("paper", "drl_report", "kg_report", "kg_schema", "readme", "package")],
    ("data_out/q1_window_zmult.json", "窗口阈值扫描（README §7.24）"),
    ("data_out/rule_ml.json", "规则+ML 融合（README §7.23）"),
    ("data_out/dead_zone_study.json", "结构性死区 2×2（README §7.19）"),
    ("data_out/cvar_workpoint.json", "CVaR 工作点（README §7.20）"),
    ("data_out/rule_frontier.json", "纯规则前沿（README §7.21）"),
    ("data_out/per_symbol_frontier.json", "逐品种前沿（README §7.22）"),
    # 预登记（判据写在跑数之前的证据）
    ("data_out/cvar_workpoint_preregistration.md", "预登记判据"),
    ("data_out/rule_frontier_preregistration.md", "预登记判据"),
    ("data_out/rule_ml_preregistration.md", "预登记判据"),
    ("data_out/q1_window_zmult_preregistration.md", "预登记判据"),
    # 仪表板数据（7 品种齐全，schema 一致）
    *[(f"data_out/alerts_{s}.parquet", f"{s} 仪表板预警数据") for s in
      ("ag", "au", "cu", "lc", "rb", "sc", "si")],
    # 文档
    ("README.md", "主文档"),
    ("PACKAGE.md", "包内说明"),
    ("extreme_event.csv", "极端事件表：dashboard/app.py 在 main() 里无条件读它，"
                          "缺了仪表板首屏即崩；validate.py / generate_report.py 同样依赖"),
    ("data_out/platform.html", "总览平台（评委入口）"),
    ("data_out/backtest_report.md", "历史快照（须自带声明，见 check_doc_numbers）"),
]
REQUIRED += [(p,"当前交付及 V3 扩展依赖") for p in [
    "docs/competition_requirements.json","docs/比赛要求对照.md","docs/交付审阅与修订.md",
    "docs/V2实验设计.md","docs/V3实验设计.md","README.legacy.md","requirements-predictive.lock.txt",
    "启动系统.command",".env.example","data_out/submission/audit.json",
    "data_out/rule_ml_portable.json","data_out/rule_ml_portable.validation.json",
    "scripts/build_submission.py","scripts/audit_submission.py","scripts/evaluate_submission_cvar.py",
    "vol_surface/time_utils.py","vol_surface/submission_metrics.py","vol_surface/portable_rule_ml.py",
    "dashboard/predictive_app.py","data_out/predictive/model.json","data_out/predictive_v2/model.json",
    "data_out/predictive_v3/model.json","data_out/predictive_v3/report.json",
    "data_out/predictive_v3/test_predictions.parquet",
    "docs/人工解释评分表.csv","docs/cvar_prices.schema.csv","docs/cvar_alerts.schema.csv",
    "scripts/smoke_competition.py","scripts/smoke_dashboard.py",
    "部署说明.md","Dockerfile",".dockerignore","compose.yaml","deployment/competition.env",
    "deployment/runtime.py","dashboard/submission_app.py","scripts/serve_submission.py","scripts/healthcheck.py"]]


def submission_env(include_model_key=False):
    from dotenv import dotenv_values
    template=dotenv_values(ROOT/'deployment/competition.env')
    allowed=set(template)
    if include_model_key:
        from deployment.runtime import RuntimeSettings
        private=dotenv_values(ROOT/'.env')
        for name in allowed:
            if name in private:template[name]=private[name]
        settings=RuntimeSettings.from_env(template)
        if not settings.model_ready:raise ValueError('private submission requires complete MODEL_* settings')
    else:
        template['MODEL_API_KEY']=''
        template['EMBEDDING_API_KEY']=''
    # Docker CLI --env-file preserves quotes, while python-dotenv removes them.
    # Emit their common unquoted subset so source, Compose and Docker agree.
    lines=[]
    for name,raw in template.items():
        value=str(raw or '')
        if (value!=value.strip() or any(c in value for c in ('\r','\n','\0'))
                or value.startswith(("'",'"')) or value.endswith(("'",'"'))
                or ' #' in value or '${' in value):
            raise ValueError('environment value cannot be represented consistently: '+name)
        lines.append(f'{name}={value}\n')
    return '# Generated submission configuration; keep private if keys are present.\n'+''.join(lines)


def _excluded(rel):
    parts=PurePosixPath(rel).parts
    if any(x in {".git","__pycache__",".pytest_cache",".venv","node_modules",".DS_Store"} for x in parts): return True
    name=parts[-1]
    if name.startswith(".env") and name!=".env.example":return True
    if name in {"secrets.toml",".DS_Store"} or name.endswith((".pyc",".pyo",".log",".pem",".key")):return True
    if len(parts)>2 and parts[0]=="data_out" and any(x.startswith('.') for x in parts[1:-1]):return True
    return any(rel.startswith("data_out/"+x+"/") for x in ["drl_more","drl_legacy_monthly","drl_evrew","drl_ctrl_old","roll","drl_c26","drl_5seed_archive"])


def files_to_include():
    files=set()
    for folder in INCLUDE_DIRS:
        files.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file())
    files.update(p for p in INCLUDE_FILES if (ROOT/p).is_file())
    result=sorted(p for p in files if not _excluded(p))
    for name in result:
        if (ROOT/name).is_symlink() or not (ROOT/name).resolve().is_relative_to(ROOT):
            raise ValueError('refuse symlink or external file: '+name)
    missing=[p for p,_ in REQUIRED if p not in result]
    if missing:raise ValueError('package dependencies missing: '+str(missing))
    return result


def build(out, competition=False, include_model_key=False):
    if include_model_key and not competition:
        raise ValueError('model keys may only be included in an explicit private competition package')
    files=files_to_include()
    manifest={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}
    generated={}
    if competition:
        generated['.env']=submission_env(include_model_key)
        generated['PACKAGE_PROFILE.json']=json.dumps(dict(profile='competition',contains_model_key=include_model_key,
            architecture='linux/amd64',image_tag='options-risk-submission:2026.10.24',
            note='仅用于比赛平台提交；含密钥的包禁止公开分发。' if include_model_key else '无真实密钥，由部署方配置。'),ensure_ascii=False,indent=2)+'\n'
        manifest.update({p:hashlib.sha256(content.encode()).hexdigest() for p,content in generated.items()})
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(ROOT/p,NAME+'/'+p)
        for p,content in generated.items():z.writestr(NAME+'/'+p,content)
        z.writestr(NAME+'/MANIFEST.sha256.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'已收录 {len(files)} 个文件，{out.stat().st_size/2**20:.1f} MB；清单 {len(REQUIRED)} 个必要依赖。')
    if include_model_key:
        out.chmod(0o600)
        print('此比赛专用包包含平台模型凭据，仅提交比赛平台，不得上传公开仓库或公开网盘。')


def verify(out):
    with zipfile.ZipFile(out) as z, tempfile.TemporaryDirectory(prefix='risk-delivery-') as temp:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate archive members')
        for name in names:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or not name.startswith(NAME+'/') or name.endswith('/'):
                raise ValueError('unsafe archive member: '+name)
        manifest=json.loads(z.read(NAME+'/MANIFEST.sha256.json'))
        profile=json.loads(z.read(NAME+'/PACKAGE_PROFILE.json')) if NAME+'/PACKAGE_PROFILE.json' in names else None
        if profile:
            from dotenv import dotenv_values
            from io import StringIO
            from deployment.runtime import RuntimeSettings
            config=dotenv_values(stream=StringIO(z.read(NAME+'/.env').decode()))
            allowed=set(dotenv_values(ROOT/'deployment/competition.env'))
            if set(config)!=allowed:raise ValueError('unexpected submission environment fields')
            RuntimeSettings.from_env(config)
            if bool(config.get('MODEL_API_KEY'))!=bool(profile['contains_model_key']):
                raise ValueError('credential inclusion flag disagrees with private environment')
        wanted={NAME+'/'+p for p in manifest}|{NAME+'/MANIFEST.sha256.json'}
        if set(names)!=wanted:raise ValueError('archive and manifest disagree')
        missing=[p for p,_ in REQUIRED if p not in manifest]
        if missing:raise ValueError('missing required files: '+str(missing))
        for p,expected in manifest.items():
            if _excluded(p) and not (profile and p=='.env'):raise ValueError('local/secret file in archive: '+p)
            if hashlib.sha256(z.read(NAME+'/'+p)).hexdigest()!=expected:raise ValueError('corrupt archive file: '+p)
        z.extractall(temp)
        extracted=Path(temp)/NAME
        if profile:(extracted/'.env').chmod(0o600)
        run=subprocess.run([sys.executable,'scripts/build_submission.py','--check'],cwd=extracted,
                           capture_output=True,text=True,timeout=300)
        if run.returncode:raise ValueError('extracted package check failed: '+run.stdout+run.stderr)
        print(f'解压副本逐文件 SHA-256 与当前材料复核通过（{len(manifest)} 个文件）。')
    return 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT.parent/'期权曲面预测风控系统.zip');ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--competition',action='store_true');ap.add_argument('--include-model-key',action='store_true')
    a=ap.parse_args()
    if a.out.suffix.lower()!='.zip':raise SystemExit('当前打包器输出 .zip；旧 tar 打包器保存在 make_package_legacy.py。')
    if not a.verify_only:build(a.out,a.competition,a.include_model_key)
    verify(a.out)


if __name__=='__main__':main()
