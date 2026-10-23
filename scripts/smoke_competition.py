"""Assert visible content of competition pages, including silent-empty regressions."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from streamlit.testing.v1 import AppTest


def main():
    os.chdir(ROOT)
    for name in ['dashboard/app.py','dashboard/pages/1_传导推理.py',
                 'dashboard/pages/2_DRL推理.py','dashboard/pages/3_系统总览.py']:
        at=AppTest.from_file(str(ROOT/name),default_timeout=120).run(timeout=120)
        assert not at.exception, [e.message for e in at.exception]
        assert not at.error, [e.value for e in at.error]
        if name.endswith('app.py'):
            assert len(at.get('plotly_chart'))>=2,'surface/time-series content must render'
        elif '1_传导' in name:
            assert len(at.selectbox)>=2 and len(at.markdown)>2
        elif '2_DRL' in name:
            assert len(at.metric)==2 and len(at.get('vega_lite_chart'))>=1
            assert at.slider[0].value==.3
            assert any('完整 32 维输入' in m.value for m in at.caption)
        else:
            assert len(at.metric)==3,'overview silently failed to load root data'
            assert [m.value for m in at.metric[:2]]==['40.44%','34.73%']
            assert len(at.dataframe[0].value)==6
        print('已验证页面内容：'+name,flush=True)
    print('比赛原型四个页面内容核验通过。')


if __name__=='__main__':main()
