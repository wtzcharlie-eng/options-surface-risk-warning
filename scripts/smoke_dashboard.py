import sys
from pathlib import Path
import os
os.environ["RISK_LLM_PROVIDER"] = "local"
from streamlit.testing.v1 import AppTest
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
at=AppTest.from_file(str(root/'dashboard/predictive_app.py'), default_timeout=90).run()
assert not at.exception, [x.message for x in at.exception]
assert len(at.metric)==3, [(m.label,m.value) for m in at.metric]
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
