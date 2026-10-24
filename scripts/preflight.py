"""Read-only runtime and evidence verification; never implies competition pass."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fast',action='store_true')
    ap.add_argument('--json',type=Path)
    a=ap.parse_args(); checks=[]
    tasks=[('当前交付一致性',['scripts/build_submission.py','--check']),
           ('部署入口配置与文件',['scripts/serve_submission.py','--check']),
           ('V3 分数与冻结来源',['scripts/audit_predictive_v3.py'])]
    if not a.fast:
        tasks += [('DRL 运行门',['-W','ignore','-m','drl.tests','--underlying']),
                  ('KG 运行门',['-W','ignore','-m','kg.tests']),
                  ('比赛原型页面内容',['scripts/smoke_competition.py']),
                  ('统一部署入口与导航',['scripts/smoke_submission.py']),
                  ('V3 交互流程',['scripts/smoke_dashboard.py'])]
    for name,args in tasks:
        p=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True,timeout=900)
        checks.append(dict(name=name,passed=p.returncode==0,returncode=p.returncode))
        print(f'{"通过" if p.returncode==0 else "失败"}：{name}',flush=True)
        if p.returncode:print((p.stdout+p.stderr)[-5000:])
    audit=json.loads((ROOT/'data_out/submission/audit.json').read_text())
    reference={k:m['finite_strict']['three_targets_met'] for k,m in audit['metrics'].items()}
    result=dict(runtime_checks='passed' if all(r['passed'] for r in checks) else 'failed',
                scope='fast' if a.fast else 'full', checks=checks,
                skipped=['DRL 运行门','KG 运行门','比赛原型页面内容','统一部署入口与导航','V3 交互流程'] if a.fast else [],
                finite_strict_reference_targets_met=reference,
                competition_business_evidence='pending',
                pending=['原赛题 CVaR 真实开盘及交易日/合约证据','可核验人工评分','后续口径答复原件（若采用）'],
                note='机器检查通过仅代表所执行范围内的运行与材料一致，不代表全部比赛指标达标。')
    if a.fast: print('未执行：DRL、KG 运行门及页面流程。')
    print('严格有限窗三项参考：'+'；'.join(k+('达到' if v else '未同时达到') for k,v in reference.items()))
    print(result['note']);print('待补：'+'；'.join(result['pending']))
    if a.json:a.json.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if result['runtime_checks']=='failed':raise SystemExit(1)


if __name__=='__main__':main()
