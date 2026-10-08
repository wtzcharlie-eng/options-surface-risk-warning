"""alert_engine — 规则 + 轻量ML 融合的四级预警引擎。

融合策略：
- level = max(rule_level, ml_level)，取两者更严的判定
- 触发原因必须来自规则层（可解释）；ML 层只贡献等级与异常分
- 输出每截面 {level, triggers, top_features, scores, ml_score}

四级定义
--------
0 正常 NORMAL   无触发，ML 分低
1 关注 WATCH    单一规则 watch 触发，或 ML 分中等
2 预警 WARN     多规则触发 / 严重规则触发，或 ML 分高
3 严重 SERIOUS  严重规则触发，或 ML 极高 + 规则同步触发
"""

from __future__ import annotations

import numpy as np

from .alert_model import AlertModel, MODEL_FEATURES
from .alert_rules import evaluate_rules

# ML 异常分阈值（0-1）：> ml_high 视为 ML 严重，> ml_mid 视为 ML 预警
ML_MID = 0.6
ML_HIGH = 0.8


class AlertState:
    """预警状态机：抑制单次抖动，要求持续触发才升级，降低固定阈值的误报。

    - 每条规则维护最近 `persistence` 个截面的触发计数
    - 单次触发给 watch(1)；连续 >= confirm 触发才升到 warn(2)；持续更久升 serious(3)
    - 不持久化，进程内维护；离线重放时按时间顺序逐截面更新即可

    用法：state = AlertState(); 在每个截面调用 evaluate(..., state=state)。
    """

    def __init__(self, persistence: int = 8, confirm: int = 2, escalate: int = 5,
                 cooldown: int = 8):
        """
        persistence: 触发计数的衰减窗口
        confirm: 升到 warn 所需的连续触发数
        escalate: 升到 serious 所需的连续触发数
        cooldown: 一条规则触发 level>=2 后进入冷却的截面数，冷却期内不再重复升级
                  （除非 base_level>=3 即强信号）。把密集预警簇折叠为单次有效预警，
                  显著提升赛题口径精确率。
        """
        self.persistence = persistence
        self.confirm = confirm
        self.escalate = escalate
        self.cooldown = cooldown
        self._counts: dict = {}      # rule -> 触发计数
        self._cd: dict = {}          # rule -> 剩余冷却截面数

    def update(self, triggers: list) -> None:
        fired = {t["rule"] for t in triggers}
        # 衰减未触发的计数
        for r in list(self._counts):
            if r not in fired:
                self._counts[r] = max(0, self._counts[r] - 1)
        # 累加触发的
        for r in fired:
            self._counts[r] = self._counts.get(r, 0) + 1
        # 推进冷却计时
        for r in list(self._cd):
            self._cd[r] -= 1
            if self._cd[r] <= 0:
                del self._cd[r]

    def rule_level_boost(self, rule: str, base_level: int) -> int:
        """根据持续触发与冷却调整单条规则等级。"""
        c = self._counts.get(rule, 0)
        on_cd = rule in self._cd
        # 强信号 (base_level>=3) 不受冷却限制
        if base_level >= 3 and c >= self.escalate:
            self._cd[rule] = self.cooldown  # 触发后进入冷却
            return 3
        # 冷却期内的弱/中信号降级为 0（不重复计入精确率分母）
        if on_cd and base_level < 3:
            return 0
        if base_level >= 2 and c >= self.confirm:
            self._cd[rule] = self.cooldown
            return min(3, base_level)
        if base_level >= 1:
            return 1 if c < self.confirm else min(3, base_level)
        return base_level


def _ml_level(ml_score: float, rule_level: int) -> int:
    """ML 分转等级（钳制 0-3）。单独 ML 触发不超过 WATCH；与规则同步才升级。"""
    if ml_score > ML_HIGH:
        # ML 极高但无规则触发，仍限 WATCH（避免纯黑箱升级）
        cand = rule_level + 1 if rule_level >= 2 else 1
    elif ml_score > ML_MID:
        cand = max(1, rule_level)
    else:
        cand = rule_level
    return int(min(max(cand, 0), 3))


def evaluate(
    feats: dict,
    model: AlertModel | None = None,
    rule_result: dict | None = None,
    state: "AlertState | None" = None,
    params: dict | None = None,
    composite: bool = False,
    composite_min_rules: int = 2,
    veto_prob: float | None = None,
    veto_thr: float | None = None,
) -> dict:
    """对单截面输出最终预警。

    Parameters
    ----------
    state : 可选 AlertState，传入则按持续触发对单条规则等级做状态化调整，
        抑制单次抖动误报。离线连续评测时传入；单次扫描可不传。
    params : 可选规则阈值参数，覆盖 alert_rules.DEFAULT_PARAMS，供网格搜索。
    composite : 复合投票模式。True 时 level≥2 需 ≥composite_min_rules 条规则同时
        触发 warn+，或一条 serious+ 且 ML 异常分>ML_MID。把孤立单规则触发降为
        watch，减少不 preceding 风险区间的误报，提升赛题口径精确率。
    veto_prob, veto_thr : **否决式融合**（问题1 连续口径交付配置，见 README §7.23）。
        `veto_prob` 是外部监督模型给出的「该截面属于风险前兆」的概率，
        `veto_thr` 是阈值；`veto_prob < veto_thr` 时**把等级压到 0**。

        为什么另设一条路径、而不复用上面的 `model`
        ------------------------------------------
        上面 `model` 那条走的是 `level = max(rule_level, ml_level)`，
        **只能加预警、不能减** —— 对「缺精确率」的场景方向恰好相反。
        实测：纯规则在 anchor 连续口径上整条 P-R 前沿都够不到目标角点，
        而否决式融合在同召回量级上把精确率推高 **+4.81pp**，三项全达标。

        **两条路径互不影响**：`veto_prob=None`（默认）时本函数行为与改动前**逐位一致**，
        由 `drl/tests.py::G13` 守。
    """
    if rule_result is None:
        rule_result = evaluate_rules(feats, params=params)
    triggers = rule_result["triggers"]

    # 状态化：对每条规则按持续触发调整等级
    if state is not None:
        state.update(triggers)
        adj_triggers = []
        max_level = 0
        for t in triggers:
            lv = state.rule_level_boost(t["rule"], t["level"])
            if lv >= 1:
                t2 = dict(t)
                t2["level"] = lv
                adj_triggers.append(t2)
                max_level = max(max_level, lv)
        triggers = adj_triggers
        rule_level = max_level
    else:
        rule_level = rule_result["level"]

    ml_score = 0.0
    ml_level = rule_level
    if model is not None and model.fitted:
        x = model.featurize(feats).reshape(1, -1)
        ml_score = float(model.score_samples(x)[0])
        ml_level = _ml_level(ml_score, rule_level)

    level = max(rule_level, ml_level)

    # 复合投票模式：level>=2 需多规则协同，或 serious 强信号，或 ML 强同意；孤立单 warn 规则降为 watch
    if composite and level >= 2:
        n_warn_rules = len({t["rule"] for t in triggers if t["level"] >= 2})
        has_serious = any(t["level"] >= 3 for t in triggers)
        ml_strong = ml_score > ML_HIGH
        if not (n_warn_rules >= composite_min_rules or has_serious or ml_strong):
            level = 1  # 孤立单 warn 规则降为 watch，不计入精确率分母（min_level=2）

    level = int(min(max(level, 0), 3))
    # 等级衰减保护：仅 ML 触发而无规则时不超过 2
    if not triggers and level > 2:
        level = 2

    # 否决式融合：监督模型认为「不像风险前兆」就压掉这条预警。
    # **放在最后**——它是对最终输出的否决，不参与上面任何一步的计算，
    # 这样 veto_prob=None 时整条链路与改动前逐位一致。
    vetoed = False
    if veto_prob is not None and veto_thr is not None and veto_prob < veto_thr:
        level = 0
        vetoed = True

    # top features：按 |z-score| 或归一化幅度排序
    top_features = _top_features(feats)

    return {
        "level": int(level),
        "triggers": triggers,
        "ml_score": ml_score,
        "rule_level": int(rule_level),
        "ml_level": int(ml_level),
        "top_features": top_features,
        "vetoed": vetoed,
    }


_TOP_KEYS = [
    "convexity_violation", "term_slope_roc", "atm_iv_z", "gamma_concentration",
    "vega_concentration", "arb_score", "iv_spike", "liquidity_z", "fit_degradation",
]


def _top_features(feats: dict) -> list:
    items = []
    for k in _TOP_KEYS:
        v = feats.get(k, 0.0)
        z = feats.get(k + "_z", feats.get(k, 0.0))  # 部分 z 已在 feats
        items.append((k, float(v), float(z) if np.isfinite(z) else 0.0))
    items.sort(key=lambda t: abs(t[1]), reverse=True)
    return items[:5]


LEVEL_NAMES = {0: "正常 NORMAL", 1: "关注 WATCH", 2: "预警 WARN", 3: "严重 SERIOUS"}
LEVEL_COLORS = {0: "#3FA66A", 1: "#E8B43A", 2: "#E87A3A", 3: "#D94B4B"}
