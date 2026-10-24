"""Content-level check for the supported one-port entry, always without model calls."""
from pathlib import Path
import os,sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest


def main():
    with patch.dict(os.environ,{'MODEL_API_KEY':'','MODEL_BASE_URL':''}):
        at=AppTest.from_file(str(ROOT/'dashboard/submission_app.py'),default_timeout=120).run()
        assert not at.exception and not at.error
        assert len(at._registered_pages)==5 and len(at.metric)==3
        assert len(at.text_input)==0,'server credentials must not be editable in the public UI'
        at.button[0].click().run()
        assert not at.exception
        assert '未调用大模型' in at.session_state['risk_chat_history'][-1]['content']
        for path,metrics in [('app.py',None),('pages/1_传导推理.py',None),('pages/2_DRL推理.py',2),('pages/3_系统总览.py',3)]:
            at.switch_page(path).run()
            assert not at.exception,[e.message for e in at.exception]
            assert not at.error,[e.value for e in at.error]
            if metrics is not None:assert len(at.metric)==metrics
            print('统一入口可访问：'+path,flush=True)
    print('五页统一导航、服务端模型配置保护与本地问答通过。')


if __name__=='__main__':main()
