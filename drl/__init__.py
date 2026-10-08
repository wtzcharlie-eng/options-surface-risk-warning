"""drl — 问题3：基于深度强化学习的自适应预警策略。

模块划分
--------
dataset  : anchor 数据集 → episode，含风险起点复现与标签自校验
env      : 预警决策 MDP（状态/动作/奖励）
metrics  : 赛题口径精确率/召回率/提前时间
baseline : 固定规则基线（复用 vol_surface.alert_rules 本体）与平凡策略
dqn      : 纯 NumPy 实现的 Double DQN
api      : 模型推理接口
"""

from .dataset import Episode, FEATURES, Normalizer, load_episodes, split_episodes
from .env import AlertEnv, RewardSpec
from .metrics import aggregate, evaluate_alerts, format_metrics

__all__ = [
    "Episode", "FEATURES", "Normalizer", "load_episodes", "split_episodes",
    "AlertEnv", "RewardSpec", "aggregate", "evaluate_alerts", "format_metrics",
]
