"""kg — 问题2：期权风险知识图谱与传导路径推理。

模块划分
--------
graph     : 极简有向图核（纯 Python，零依赖，JSON 序列化）
schema    : 节点/边定义。**实测层与诠释层分离**是本模块的核心设计
calibrate : 在 90k 真实截面上标定边权（lift + Wilson 下界 + 分块置换检验）
build     : 由标定结果装配图谱，只放通过检验的边
reason    : 传导路径推理与证据链；无证据时明确拒绝推断
explain   : 中文解释生成，严格不含交易建议
tests     : 8 道验证门（K1–K8）

Schema 文档见 `kg/schema.md`（由 `scripts/report_kg.py` 从 `schema.py` 自动生成）。
v1 骨架已归档到 `kg/_legacy/`，归档原因见该目录的 README。
"""

from .build import build, summary
from .explain import explain, explain_brief
from .graph import DiGraph
from .reason import active_anomalies, reason
from .schema import ANOMS, CONSEQS, HORIZON, MECHS

__all__ = ["DiGraph", "build", "summary", "reason", "active_anomalies",
           "explain", "explain_brief", "ANOMS", "CONSEQS", "MECHS", "HORIZON"]
