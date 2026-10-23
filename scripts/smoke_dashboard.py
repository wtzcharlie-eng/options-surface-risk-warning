import sys
import json
from pathlib import Path
import os
os.environ["RISK_LLM_PROVIDER"] = "local"
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
at=AppTest.from_file(str(root/'dashboard/predictive_app.py'), default_timeout=90).run()
assert not at.exception, [x.message for x in at.exception]
assert len(at.metric)==3, [(m.label,m.value) for m in at.metric]
expected_version = 'v3' if (root/'data_out/predictive_v3/model.json').exists() else 'v2' if (root/'data_out/predictive_v2/model.json').exists() else 'v1'
assert at.sidebar.radio[0].value == expected_version
print('metrics',[(m.label,m.value) for m in at.metric])
at.button[0].click().run()
assert not at.exception, [x.message for x in at.exception]
assert at.session_state['risk_chat_history'][-1]['role']=='assistant'
assert '未调用大模型' in at.session_state['risk_chat_history'][-1]['content']
at.chat_input[0].set_value('预测效果可靠吗？').run()
assert not at.exception, [x.message for x in at.exception]
assert len(at.session_state['risk_chat_history'])==4
assert 'Brier' in at.session_state['risk_chat_history'][-1]['content']
at.sidebar.selectbox[0].select('au').run()
assert not at.exception, [x.message for x in at.exception]
assert len(at.session_state['risk_chat_history'])==0
at.sidebar.select_slider[0].set_value(5).run()
assert not at.exception, [x.message for x in at.exception]
print('flow complete: render, suggested question, follow-up, symbol reset, 5day selection')

at.chat_input[0].set_value('x'*4001).run()
assert not at.exception, [x.message for x in at.exception]
assert any('4000' in e.value for e in at.error)
at.chat_input[0].set_value('   ').run()
assert not at.exception, [x.message for x in at.exception]
print('long and blank questions handled without crashing')

if (root/'data_out/predictive_v2/model.json').exists():
    at.sidebar.radio[0].set_value('v1').run()
    assert not at.exception, [x.message for x in at.exception]
    assert at.session_state['risk_chat_history'] == []
    assert len(at.metric) == 3
    at.sidebar.radio[0].set_value('v2').run()
    assert not at.exception, [x.message for x in at.exception]
    assert at.session_state['risk_chat_history'] == []
    assert any('联合模型共享一条' in c.value for c in at.caption)
    print('V2/V1 version switching and evidence context reset verified')

if expected_version == 'v3':
    at.sidebar.radio[0].set_value('v3').run()
    assert not at.exception, [x.message for x in at.exception]
    assert at.session_state['risk_chat_history'] == []
    shown=json.loads(at.json[0].value)
    assert shown['model_version']=='v3'
    if shown.get('model_context'):
        assert any('V3 在独立历史段' in c.value for c in at.caption)
    else:
        assert shown['status'] in {'unavailable','out_of_distribution','stale'}
    at.chat_input[0].set_value('概率可用率和运行召回率是多少？').run()
    assert not at.exception, [x.message for x in at.exception]
    reply=at.session_state['risk_chat_history'][-1]['content']
    assert '[operational_evaluation.probability_coverage]' in reply
    assert '[operational_evaluation.recall_all_origins]' in reply
    print('V3 version switching and operational-evidence conversation verified')
