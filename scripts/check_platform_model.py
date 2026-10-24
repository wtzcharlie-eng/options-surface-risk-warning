"""Opt-in real platform chat check using current frozen V3 evidence. No keys in report."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deployment.runtime import load_settings
from vol_surface.risk_assistant import LLMConfig, answer
from vol_surface.predictive import load_anchor_data
from vol_surface.predictive_service import load_model_artifact, forecast_model, load_report, build_evidence


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'data_out/deployment/platform_model_check.json')
    a=ap.parse_args()
    s=load_settings(ROOT)
    if not s.model_ready:raise SystemExit('平台模型参数未齐，请由部署方填写 .env。')
    s.apply()
    history=load_anchor_data(ROOT/'data_out/anchor')
    symbol=history.groupby('symbol')['_dt'].max().idxmax()
    last=history.loc[history.symbol==symbol,'_dt'].max()
    when=last.normalize().replace(hour=23,minute=59,second=59).isoformat()
    folder=ROOT/'data_out/predictive_v3'
    snapshot=forecast_model(load_model_artifact(folder/'model.json'),history,symbol,as_of=when)
    evidence=build_evidence(snapshot,3,load_report(folder))
    questions=['请简要解释当前曲面风险预测的任务和主要限制。只描述证据支持的事实，不要给交易建议；不要添加任何数值，每项事实附字段引用。',
               '继续说明这些预测能否代表账户实际损失，以及应该如何理解模型可靠性。每项事实附字段引用，避免添加数值。']
    history_chat=[];checks=[]
    for question in questions:
        r=answer(question,evidence,history_chat,LLMConfig.from_env())
        checks.append(dict(mode=r.mode,passed=r.mode=='llm' and r.warning is None,
                           cited_fields=r.cited_fields,warning=r.warning,response=r.text))
        history_chat=r.history
    result=dict(checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                real_service=True,model=s.model_name,model_provider=s.model_provider,
                protocol='HTTPS compatible chat; enable_thinking=false',
                symbol=symbol,as_of=when,model_version='v3',horizon_days=3,
                n_turns=len(checks),passed=all(c['passed'] for c in checks),checks=checks,
                note='此记录证明当次平台真实调用、证据引用与多轮对话，不证明预测有效性、平台部署状态或未来配额。')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['checked_at','model','real_service','n_turns','passed']},ensure_ascii=False))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
