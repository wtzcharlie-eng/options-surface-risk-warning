"""drl.env — 预警决策的 MDP 环境。

状态 / 动作 / 奖励设计
---------------------
**状态 s_t（28 维）**：26 维曲面风险特征（用**训练集**统计量 z-score 归一、截断 ±5）
  + 2 维决策上下文：
    - `since_alert`：距上次发出 level≥2 预警的分钟数 / 240（截断到 1）
    - `last_level` ：上一步动作 / 3
  上下文这两维让智能体感知"我刚报过了"，从而学会抑制冗余预警——这正是无状态的
  固定阈值规则做不到的。**状态不含任何未来信息**（`hit` / `lead_min` / 风险起点
  只进奖励，不进状态），由 `drl/tests.py::test_state_is_causal` 守住。

**动作 a_t**：0=无预警 / 1=关注 / 2=警告 / 3=危险，与赛题四级定义一致。

**奖励 r_t —— 事件级会计，直接映射赛题指标**

第一版曾用「每步下注」式奖励（命中 +1 / 误报 −1）。实测发现它是坏设计：
风险起点的基础率只有 14.6%，导致「恒不预警」的累计奖励（−3260）反而**高于**
规则基线（−6113）。在那种奖励下，DRL 只要学会闭嘴就能"赢"，而召回率为 0——
这是典型的指标可被套利。故改为按**事件**计价，与赛题精确率/召回率的分母口径对齐：

  1. `cover` 首次覆盖：某次 level≥2 预警是第一条落在某风险起点前 120min 内的预警
     → `+W_COVER · (1 + lead_frac) · gain[level]`
     这是召回的业务价值所在，也是唯一的大额正奖励。
  2. `redundant` 冗余正确：预警正确但该风险起点已被更早的预警覆盖
     → `+W_TRUE · gain[level]`（很小；不算错，但边际价值低）
  3. `false` 误报：预警后 120min 内无风险起点 → `−W_FP · cost[level]`
  4. `miss` 漏报：走到某个风险起点时它从未被覆盖 → `−W_MISS`
  5. `watch` 关注档：level==1 不计入赛题预警集合，只给极小整形量，
     使四级输出保有梯度而不污染指标。

  等级 2/3 通过 `gain`/`cost` 系数区分为**置信度下注**：3 级正确时多得 15%，
  错误时多罚 60%。赛题指标只认 level≥2，故这一层不影响评分口径，只让四级语义成立。

  奖励量级校准（测试集 923 个风险起点）：预言机策略 ≈ +11k，规则基线 ≈ −2.7k，
  恒预警 ≈ −6.5k，恒不预警 ≈ −9.2k。**累计奖励的正负大致分隔"有用"与"无用"**，
  且平凡策略全部劣于规则基线——这是奖励没有被套利的必要条件。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

N_ACTIONS = 4
CONTEXT_DIM = 2
LEAD_WINDOW_MIN = 120.0     # 赛题：预警后 2 小时内
ALERT_LEVEL = 2             # 赛题：等级≥2 才算"发出预警"


@dataclass
class RewardSpec:
    """奖励权重。事件级计价，见模块 docstring。

    `mode` 保留了第一版「每步下注」奖励作为可选项，用于复现「该设计可被闭嘴套利」
    这一结论（见 scripts/train_drl.py --stage ablation）。生产与主结果一律用
    默认的 event 模式。
    """
    cover: float = 6.0        # 首次覆盖一个风险起点
    miss: float = 10.0        # 风险起点未被任何预警覆盖
    true: float = 0.2         # 冗余但正确的预警
    fp: float = 1.0           # 误报
    lead: float = 1.0         # 提前时间加权系数（乘在 cover 上）
    watch_hit: float = 0.05   # level==1 且未来有风险
    watch_miss: float = -0.02 # level==1 但未来无风险
    gain: tuple = (0.0, 0.0, 1.0, 1.15)   # 各等级正向系数
    cost: tuple = (0.0, 0.0, 1.0, 1.60)   # 各等级负向系数
    mode: str = "event"       # "event"（默认）或 "perstep_v1"（第一版，仅供消融）
    # ---- 事件级误报计价（默认关闭，见 data_out/event_reward_preregistration.md）
    #
    # 默认口径下每个误报**截面**都罚 `fp`；而命题方澄清的新评测口径按
    # **合并后的预警事件**计精确率——同一段风险内连报十条与只报一条代价相同。
    # 奖励与评测口径因此不一致：模型被训练去避免"连续多报"，但评测并不惩罚它。
    #
    # `event_level_fp=True` 时改为**只罚每段连续误报的第一条**，后续同段重复
    # 按 `fp_repeat` 计价（默认 0，即完全不罚），与新口径对齐。
    # `event_gap_min` 为判定"同一段"的间隔阈值，与 metrics_v2.MERGE_GAP_MIN 一致。
    #
    # **默认关闭**：开启会改变全部既有结果与 G0–G10 验证门的基线，
    # 故作为对照实验的开关，不动交付链路。
    event_level_fp: bool = False
    fp_repeat: float = 0.0
    event_gap_min: float = 120.0
    # 第一版「每步下注」参数
    v1_tp: tuple = (0.0, 0.3, 1.0, 2.0)
    v1_fp: tuple = (0.0, -0.1, -1.0, -3.0)
    v1_fn: float = -1.0
    v1_lead: float = 0.5

    def describe(self) -> str:
        if self.mode == "perstep_v1":
            return (f"[perstep_v1] tp={self.v1_tp} fp={self.v1_fp} "
                    f"fn={self.v1_fn} lead={self.v1_lead}")
        return (f"cover={self.cover} miss={self.miss} true={self.true} fp={self.fp} "
                f"lead={self.lead} gain={self.gain} cost={self.cost}")


class AlertEnv:
    """单幕 = 一个 (品种, 月份) 的连续截面序列。

    用法（训练）::
        env = AlertEnv(episode, normalizer); s = env.reset()
        while not done: s, r, done, info = env.step(agent.act(s))

    用法（评估任意策略，含规则基线）::
        res = env.rollout(policy)      # policy(state, i) -> action
    """

    def __init__(self, episode, normalizer, spec: RewardSpec | None = None):
        self.ep = episode
        self.norm = normalizer
        self.spec = spec or RewardSpec()
        self.Xn = normalizer(episode.X)
        self.lead = episode.lead_min
        # 命中判定与赛题精确率口径一致：未来 120min 内存在风险起点
        self.hit = (self.lead <= LEAD_WINDOW_MIN)
        self.dt_sec = episode.dt.astype("datetime64[s]").astype(np.int64)
        self.n = len(episode)
        self.risk_idx = np.asarray(episode.risk_idx, dtype=np.int64)
        self._is_risk = np.zeros(self.n, dtype=bool)
        if len(self.risk_idx):
            self._is_risk[self.risk_idx] = True
        # next_risk[i] = i 之后（含 i 自身）最近的风险起点索引；无则 -1。
        # 「含自身」与 metrics 的闭区间窗口一致，见 dataset._lead_minutes 的说明。
        self.next_risk = np.full(self.n, -1, dtype=np.int64)
        ptr = 0
        rs = self.risk_idx.tolist()
        for i in range(self.n):
            while ptr < len(rs) and rs[ptr] < i:
                ptr += 1
            if ptr < len(rs):
                self.next_risk[i] = rs[ptr]
        self.reset()

    # ---------------------------------------------------------------- core

    def reset(self) -> np.ndarray:
        self.i = 0
        self.last_alert_sec = None
        self.last_level = 0
        self._covered = set()      # 已被覆盖的风险起点索引
        self._last_fp_sec = None   # 上次误报时刻（事件级计价用）；**必须在此清空**，
                                   # 否则跨幕泄漏，上一幕末尾的误报会让下一幕开头免罚
        return self._state(0)

    def _state(self, i: int) -> np.ndarray:
        if self.last_alert_sec is None:
            since = 1.0
        else:
            since = min((self.dt_sec[i] - self.last_alert_sec) / 60.0, 240.0) / 240.0
        return np.concatenate([self.Xn[i], [since, self.last_level / 3.0]])

    @property
    def state_dim(self) -> int:
        return self.Xn.shape[1] + CONTEXT_DIM

    def _reward(self, i: int, a: int) -> tuple:
        """返回 (reward, event)，event 用于诊断统计。"""
        if self.spec.mode == "perstep_v1":
            return self._reward_v1(i, a)
        sp = self.spec
        r, ev = 0.0, "none"

        if a >= ALERT_LEVEL:
            nr = self.next_risk[i]
            if self.hit[i] and nr >= 0:
                if nr not in self._covered:
                    self._covered.add(nr)
                    lead_frac = min(self.lead[i], LEAD_WINDOW_MIN) / LEAD_WINDOW_MIN
                    r += sp.cover * (1.0 + sp.lead * lead_frac) * sp.gain[a]
                    ev = "cover"
                else:
                    r += sp.true * sp.gain[a]
                    ev = "redundant"
            else:
                # 事件级计价：只罚每段连续误报的**第一条**，同段后续按 fp_repeat 计。
                # `_last_fp_sec` 记上一次误报的时刻，间隔 ≤ event_gap_min 即视为同一段
                # （与 metrics_v2.merge_alerts 的合并规则一致）。
                w = sp.fp
                if sp.event_level_fp:
                    lf = getattr(self, "_last_fp_sec", None)
                    same = (lf is not None
                            and (self.dt_sec[i] - lf) <= sp.event_gap_min * 60.0)
                    w = sp.fp_repeat if same else sp.fp
                    self._last_fp_sec = self.dt_sec[i]
                r -= w * sp.cost[a]
                ev = "false"
        elif a == 1:
            r += sp.watch_hit if self.hit[i] else sp.watch_miss
            ev = "watch"

        # 走到风险起点时结算漏报
        if self._is_risk[i] and i not in self._covered:
            r -= sp.miss
            ev = "miss" if ev == "none" else ev + "+miss"
        return float(r), ev

    def _reward_v1(self, i: int, a: int) -> tuple:
        """第一版「每步下注」奖励——**已弃用**，仅供消融复现。

        问题：风险起点基础率偏低时，「恒不预警」的累计奖励会高于规则基线，
        智能体只要学会闭嘴就能拿高分而召回率为 0。量化证据见
        `scripts/train_drl.py --stage ablation`。
        """
        sp = self.spec
        if self.hit[i]:
            r = sp.v1_tp[a]
            if a >= ALERT_LEVEL:
                r += sp.v1_lead * min(self.lead[i], LEAD_WINDOW_MIN) / LEAD_WINDOW_MIN
                nr = self.next_risk[i]
                if nr >= 0:
                    self._covered.add(nr)
            elif self.next_risk[i] not in self._covered:
                r += sp.v1_fn
            ev = "v1_hit"
        else:
            r = sp.v1_fp[a]
            ev = "v1_miss"
        return float(r), ev

    def step(self, a: int):
        i = self.i
        r, ev = self._reward(i, a)
        if a >= ALERT_LEVEL:
            self.last_alert_sec = self.dt_sec[i]
        self.last_level = a
        self.i += 1
        done = self.i >= self.n
        s2 = self._state(self.i) if not done else np.zeros(self.state_dim)
        return s2, r, done, {"hit": bool(self.hit[i]), "event": ev}

    # ---------------------------------------------------------------- rollout

    def rollout(self, policy) -> dict:
        """跑完整一幕。`policy(state, i) -> action`。"""
        s = self.reset()
        acts = np.zeros(self.n, dtype=np.int64)
        rews = np.zeros(self.n)
        events = []
        for i in range(self.n):
            a = int(policy(s, i))
            s, r, done, info = self.step(a)
            acts[i], rews[i] = a, r
            events.append(info["event"])
            if done:
                break
        return {
            "symbol": self.ep.symbol, "ym": self.ep.ym,
            "ts": self.ep.ts, "actions": acts, "rewards": rews,
            "events": events,
            "total_reward": float(rews.sum()),
            "n_alert": int((acts >= ALERT_LEVEL).sum()),
            "risk_idx": self.ep.risk_idx,
        }


def oracle_reward(episode, normalizer, spec: RewardSpec | None = None) -> float:
    """预言机上界：在每个风险起点前恰好 120min 处报一次 3 级，其余不报。

    用来给累计奖励一把尺子——没有这把尺子，"DRL 比基线高多少"是没有量纲的。
    """
    spec = spec or RewardSpec()
    env = AlertEnv(episode, normalizer, spec)
    want = set()
    for i in range(len(episode)):
        if env.hit[i] and env.next_risk[i] >= 0 and env.next_risk[i] not in want:
            want.add(env.next_risk[i])
    # 对每个风险起点，选其覆盖窗口内最早的截面
    best = {}
    for i in range(len(episode)):
        nr = env.next_risk[i]
        if env.hit[i] and nr >= 0 and nr not in best:
            best[nr] = i
    fire = set(best.values())
    ro = env.rollout(lambda s, i: 3 if i in fire else 0)
    return ro["total_reward"]
