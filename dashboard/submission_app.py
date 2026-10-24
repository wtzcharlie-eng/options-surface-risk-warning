"""Single-port competition entry with explicit, reachable navigation."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import streamlit as st
from deployment.runtime import load_settings

settings = load_settings(ROOT)
settings.apply()
st.set_page_config(page_title="期权曲面预测风控", page_icon="📈", layout="wide")
pages = [st.Page("predictive_app.py", title="预测与问答", default=True),
         st.Page("app.py", title="曲面预警", url_path="surface"),
         st.Page("pages/1_传导推理.py", title="图谱解释", url_path="graph"),
         st.Page("pages/2_DRL推理.py", title="DRL 推演", url_path="drl"),
         st.Page("pages/3_系统总览.py", title="比赛对照", url_path="review")]
page = st.navigation(pages, position="top")
page.run()
