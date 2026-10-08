"""drl.baseline — 对照策略。

`RulePolicy` 直接调用 `vol_surface.alert_rules.evaluate_rules`——即问题1 交付的
那套规则引擎本体，不是复刻。这样 DRL 与固定规则跑在同一批状态、同一个奖励函数、
同一套指标上，比较才成立。

`alert_rules` 的纯函数契约（不碰 FeatureHistory、不解析 timestamp、无内部状态）
正是它能被这样复用的原因：anchor 数据集里 26 维特征已包含规则需要的全部 z-score
与速度项，喂进去即可复现原始判定。

同时提供若干平凡策略作为下界，用于回答"DRL 是不是只是学会了乱报"：
never（恒 0）/ always_warn（恒 2）/ always_serious（恒 3）/ random。
"""

from __future__ import annotations

import json
import os

import numpy as np

from vol_surface.alert_rules import evaluate_rules

from .dataset import FEATURES


def load_best_params(path: str | None = None) -> dict:
    """读 scripts/grid_search.py 校准出的最优阈值（data_out/best_params.json）。"""
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "data_out", "best_params.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("params", {})


class RulePolicy:
    """固定阈值规则基线（问题1 的交付版本）。

    params=None 时用 alert_rules.DEFAULT_PARAMS；传入 best_params.json 的参数即为
    网格搜索校准后的版本。best_params.json 里 `state: null`，说明校准最优配置**不启用**
    状态机，故这里也不接 AlertState，与 README 记录的最优配置一致。
    """

    name = "rule"

    def __init__(self, episode, params: dict | None = None):
        self.params = params or None
        rows = episode.X
        self.levels = np.zeros(len(rows), dtype=np.int64)
        for i in range(len(rows)):
            feats = {k: float(rows[i, j]) for j, k in enumerate(FEATURES)}
            self.levels[i] = int(evaluate_rules(feats, params=self.params)["level"])

    def __call__(self, state, i: int) -> int:
        return int(self.levels[i])


class ConstantPolicy:
    def __init__(self, level: int):
        self.level = int(level)
        self.name = f"const{level}"

    def __call__(self, state, i: int) -> int:
        return self.level


class RandomPolicy:
    """按给定分布随机出等级，用作"无信息"下界。"""

    name = "random"

    def __init__(self, seed: int = 0, p=(0.7, 0.15, 0.10, 0.05)):
        self.rng = np.random.default_rng(seed)
        self.p = np.array(p, dtype=float)
        self.p /= self.p.sum()

    def __call__(self, state, i: int) -> int:
        return int(self.rng.choice(4, p=self.p))


class PeriodicPolicy:
    """**盲节奏**策略：完全不看特征，每 `period` 个截面机械地报一次 2 级。

    这是本项目最重要的一条对照线。原因：本任务的奖励（以及赛题指标本身）对召回
    的权重远高于误报，因此一个只会「按固定节奏刷预警」的策略就能拿到相当高的
    召回率与不低的累计奖励——实测在测试集上它甚至**优于**校准后的固定规则引擎。

    若不摆出这条线，"DRL 累计奖励高于规则基线"会被误读成"DRL 学到了曲面信号"，
    而实际上一部分增益只是预警节奏的优化。真正能区分"有无判别力"的是
    **精确率相对 hit 基础率的提升**：盲节奏策略的精确率恒等于基础率，
    而有判别力的策略必须显著高于它。
    """

    def __init__(self, period: int = 3, level: int = 2):
        self.period, self.level = int(period), int(level)
        self.name = f"periodic{period}"

    def __call__(self, state, i: int) -> int:
        return self.level if i % self.period == 0 else 0


class QualityGate:
    """数据质量门：在流动性过低的截面上抑制预警（等级归 0）。

    实测依据（测试集，59 幕）
    ------------------------
    误报明显集中在低流动性截面。按 `liquidity_ratio` 五等分看规则基线的精确率，
    呈单调关系：16.5% → 28.9% → 33.0% → 32.5% → 36.3%。门槛 0.55 是在**验证集**
    上选的（更高的门槛会让召回快速塌陷）。

    效果与代价（测试集，只评估一次；**旧逐月标签口径**，见 README §7.10）
    ----------------------------------------------------------------
    | 被包装的策略 | 精确率 | 召回率 | 换取比 |
    |------------|--------|--------|-------|
    | 规则基线 | 29.58% → **32.68%**（+3.10pp）| 58.08% → 55.74%（−2.35pp）| 1.32 pp/pp（划算）|
    | DRL 集成 | 41.18% → 42.46%（+1.29pp）| 63.04% → 61.41%（−1.63pp）| 0.79 pp/pp（**不划算**）|

    **因此默认不启用。** 对固定阈值规则它是划算的买卖；但 DRL 已经从
    `liquidity_ratio` / `liquidity_z` 这两维特征里自己学到了同样的信息，
    再加一道硬门只是重复抑制、白白损失召回。这也从侧面说明 DRL 确实在用曲面信号，
    而不是靠预警节奏。

    用法::

        gated = QualityGate(RulePolicy(ep, params=bp), ep, min_liquidity=0.55)
        res = AlertEnv(ep, norm, spec).rollout(gated)
    """

    def __init__(self, policy, episode, min_liquidity: float = 0.55,
                 min_contracts: int = 0):
        from .dataset import FEATURES
        self.policy = policy
        self.name = f"gated({getattr(policy, 'name', 'policy')})"
        lr = episode.X[:, FEATURES.index("liquidity_ratio")]
        nc = episode.X[:, FEATURES.index("n_contracts")]
        self.blocked = (lr < min_liquidity) | (nc < min_contracts)

    def __call__(self, state, i: int) -> int:
        return 0 if self.blocked[i] else int(self.policy(state, i))
