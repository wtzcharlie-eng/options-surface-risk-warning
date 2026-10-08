"""drl.api — DRL 预警模型的推理接口。

赛题「成果形式」要求提供模型推理 API。本模块把训练好的 checkpoint 包成一个可以
直接接进在线扫描（scripts/scan.py）或看板的对象。

两个关键设计点
--------------
1. **状态是有上下文的**。状态向量里除曲面特征外，还有 `since_alert`
   （距上次 level≥2 预警的分钟数）与 `last_level`。因此推理不是纯函数——
   必须按时间顺序调用，对象内部维护这两个量。提供 `reset()` 在切换品种/回放
   起点时清空；也支持显式传入上下文做无状态调用。

2. **默认用集成**。单 seed 的测试集表现方差不小（累计奖励 −2201 ~ +12），
   多 seed 平均 Q 值能显著降低这种随机性，是更合理的生产选择。
   `AlertAgentAPI.from_dir()` 会自动把目录下所有 `agent_seed*.npz` 组成集成。

特征表长度有两种：26 维（仅曲面）与 32 维（曲面 + 标的已实现波动，README §7.10）。
以 checkpoint 内存的 `features` 为准，不看模块常量；两种维度的模型不允许混进同一个集成。
`predict` 的 `feats` 字典需含对应的键，32 维模型还要 `vol_surface.underlying`
产出的 6 个键（缺失按 0 处理，但那等于喂了错误的输入，务必补齐）。

用法::

    from drl.api import AlertAgentAPI
    api = AlertAgentAPI.from_dir("data_out/drl")

    api.reset()
    for ts, feats in stream:                     # feats 来自 compute_features
        out = api.predict(feats, timestamp=ts)
        print(out["level"], out["q_values"], out["margin"])

与规则引擎的关系：本接口只输出等级与 Q 值，**不产生中文触发原因**。生产上建议
与 `vol_surface.alert_rules.evaluate_rules` 并用——规则给可解释的触发理由，
DRL 给自适应的等级判定，二者在 `explain=True` 时会一并返回。
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from .dataset import FEATURES, Normalizer
from .dqn import DQNAgent
from .env import ALERT_LEVEL, CONTEXT_DIM

LEVEL_NAMES = {0: "正常 NORMAL", 1: "关注 WATCH", 2: "预警 WARN", 3: "严重 SERIOUS"}


class AlertAgentAPI:
    """DRL 预警智能体推理接口（支持单模型与多 seed 集成）。"""

    def __init__(self, agents: list, normalizer: Normalizer, meta: dict | None = None):
        if not agents:
            raise ValueError("至少需要一个 agent")
        dims = {ag.state_dim for ag in agents}
        if len(dims) > 1:
            raise ValueError(f"集成里混入了不同状态维度的 checkpoint: {sorted(dims)}——"
                             f"26 维与 32 维模型不能混用，请清理目录后重训")
        self.agents = agents
        self.norm = normalizer
        self.meta = meta or {}
        # 特征表以 checkpoint 里存的为准，而非模块常量：训练后若有人改了
        # dataset.FEATURES，用模块常量会静默地把特征喂错位置。
        self.features = list(self.meta.get("features") or FEATURES)
        exp = len(self.features) + CONTEXT_DIM
        if agents[0].state_dim != exp:
            raise ValueError(f"checkpoint 状态维度 {agents[0].state_dim} 与特征表长度 "
                             f"{len(self.features)}+{CONTEXT_DIM} 不符")
        self.reset()

    # ---------------------------------------------------------------- 构造

    @classmethod
    def from_file(cls, path: str) -> "AlertAgentAPI":
        ag, extra = DQNAgent.load(path)
        return cls([ag], Normalizer.from_dict(extra["normalizer"]), extra)

    @classmethod
    def from_dir(cls, d: str, pattern: str = "agent_seed*.npz") -> "AlertAgentAPI":
        """把目录下所有 checkpoint 组成集成（Q 值平均）。"""
        files = sorted(f for f in glob.glob(os.path.join(d, pattern))
                       if "shuf_" not in os.path.basename(f))
        if not files:
            raise FileNotFoundError(f"{d} 下没有找到 {pattern}")
        agents, extra = [], None
        for f in files:
            ag, ex = DQNAgent.load(f)
            agents.append(ag)
            extra = extra or ex
        api = cls(agents, Normalizer.from_dict(extra["normalizer"]), extra)
        api.meta["n_models"] = len(files)
        api.meta["files"] = [os.path.basename(f) for f in files]
        return api

    # ---------------------------------------------------------------- 状态

    def reset(self) -> None:
        """清空预警上下文。切换品种或重新开始回放时必须调用。"""
        self._last_alert_ts = None
        self._last_level = 0

    def _context(self, timestamp: str | None) -> np.ndarray:
        if self._last_alert_ts is None or timestamp is None:
            since = 1.0
        else:
            dt = (_to_epoch(timestamp) - _to_epoch(self._last_alert_ts)) / 60.0
            since = min(max(dt, 0.0), 240.0) / 240.0
        return np.array([since, self._last_level / 3.0])

    # ---------------------------------------------------------------- 推理

    def _vector(self, feats: dict, timestamp: str | None,
                context: tuple | None) -> np.ndarray:
        x = np.array([float(feats.get(k, 0.0)) for k in self.features], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        ctx = np.array(context, dtype=float) if context is not None \
            else self._context(timestamp)
        if len(ctx) != CONTEXT_DIM:
            raise ValueError(f"context 需 {CONTEXT_DIM} 维，收到 {len(ctx)}")
        return np.concatenate([self.norm(x.reshape(1, -1))[0], ctx])

    def predict(self, feats: dict, timestamp: str | None = None,
                context: tuple | None = None, update_state: bool = True,
                explain: bool = False, rule_params: dict | None = None) -> dict:
        """对单个截面特征输出预警等级。

        Parameters
        ----------
        feats : compute_features 产出的特征字典（缺失键按 0 处理）
        timestamp : "YYYYMMDDHHMMSS"，用于计算距上次预警的分钟数
        context : 显式传入 (since_alert_norm, last_level_norm) 做无状态调用
        update_state : False 则不更新内部上下文（用于假设推演）
        explain : True 时附带规则引擎的中文触发原因

        Returns
        -------
        {level, level_name, q_values, margin, agreement, is_alert, [triggers]}
        """
        s = self._vector(feats, timestamp, context)
        qs = np.stack([ag.q_values(s)[0] for ag in self.agents])
        q = qs.mean(axis=0)
        level = self._apply_threshold(q)
        order = np.sort(q)[::-1]

        out = {
            "level": level,
            "level_name": LEVEL_NAMES[level],
            "q_values": [round(float(v), 4) for v in q],
            # margin：最优动作与次优动作的 Q 值差，越大越确定
            "margin": round(float(order[0] - order[1]), 4),
            # agreement：集成中投票给最终等级的模型比例
            "agreement": round(float((qs.argmax(axis=1) == level).mean()), 3),
            "is_alert": bool(level >= ALERT_LEVEL),
            "n_models": len(self.agents),
        }
        if explain:
            from vol_surface.alert_rules import evaluate_rules
            r = evaluate_rules(feats, params=rule_params)
            out["triggers"] = r["triggers"]
            out["rule_level"] = r["level"]

        if update_state:
            if level >= ALERT_LEVEL and timestamp is not None:
                self._last_alert_ts = timestamp
            self._last_level = level
        return out

    def predict_sequence(self, rows: list, explain: bool = False) -> list:
        """按时间顺序批量推理。`rows` = [(timestamp, feats), ...]，内部维护上下文。"""
        self.reset()
        return [self.predict(f, timestamp=ts, explain=explain) for ts, f in rows]

    def policy(self):
        """返回可直接喂给 `AlertEnv.rollout` 的策略（离线回测用）。"""
        qs_cache = {}

        def _p(state, i):
            q = np.mean([ag.q_values(state)[0] for ag in self.agents], axis=0)
            return self._apply_threshold(q)
        return _p


    # ---------------------------------------------------------------- 置信度阈值

    def set_confidence_threshold(self, lean: float = 0.0) -> None:
        """调整发出预警所需的置信度门槛（沿 P-R 前沿移动工作点）。

        `lean = max(Q₂,Q₃) − max(Q₀,Q₁)`，即「倾向预警」相对「倾向不报」的 Q 值优势。
        默认 0.0 等价于纯贪心（argmax），提高它会更保守。

        **默认保持 0.0。** 实测（测试集 59 幕，**旧逐月标签口径**）把门槛提到验证集选出的
        0.34 后：精确率 41.18% → 43.85%（+2.67pp），召回率 63.04% → 54.24%（−8.80pp），
        每损失 1pp 召回只换到 0.30pp 精确率，且召回跌破赛题 60% 门槛。
        详见 README §7.8 的完整实验（5 类决策规则、约 30 组配置）。

        **口径更新（README §7.10）**：修正标签缺陷并重训后，该结论变了——在连续标签
        口径下，验证集选出的门槛是 **0.35**，测试集给出精确率 47.61% / 召回 71.93%，
        比 0.0 处（45.5% / 78.6%）更靠近赛题要求。交付配置 `data_out/drl`
        因此**建议设为 0.35**；此处默认值仍为 0.0，以免影响既有调用方。
        """
        self._lean_thr = float(lean)

    def _apply_threshold(self, q: np.ndarray) -> int:
        a = int(np.argmax(q))
        thr = getattr(self, "_lean_thr", 0.0)
        if thr > 0 and a >= ALERT_LEVEL:
            lean = float(max(q[2], q[3]) - max(q[0], q[1]))
            if lean <= thr:
                return min(a, 1)
        return a

    def info(self) -> dict:
        return {"n_models": len(self.agents), "state_dim": self.agents[0].state_dim,
                "features": self.features, "meta": self.meta}


def _to_epoch(ts: str) -> float:
    import pandas as pd
    return pd.Timestamp(f"{ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                        f"{ts[8:10]}:{ts[10:12]}:{ts[12:14] or '00'}").timestamp()
