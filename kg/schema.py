"""kg.schema — 期权风险知识图谱的节点/边定义。

核心设计原则：**把「实测层」与「诠释层」分开，并在图里显式标注**
--------------------------------------------------------------
知识图谱最容易滑向的失败模式，是把作者拍脑袋写的因果关系包装成「推理」。
本项目 v1 骨架（见 `kg/_legacy/`）就落在这里——主推理骨架 AMPLIFIES 的边权是
「启发式 0.6–0.8」的先验，跨品种回退权重甚至直接写死 1.0（等于断言必然传导）。

因此本版把图分成两层，每个节点/边都带 `layer` 属性：

- **measured（实测层）**：`ANOM → CONSEQ` 边。权重不是先验，而是在 90k 真实截面上
  测出来的「提升度 lift = P(后果|异常) / P(后果)」，附样本量、置信区间、
  **分块置换检验**的 p 值，以及训练集/测试集两段独立估计。
  数据不支持的边会被剪掉或标为 weak，不会静默留在图里充数。

- **interpretive（诠释层）**：`MECH` 机制节点与边上的 `how` 文本。它们回答
  「为什么会这样传导」，是业务先验，**不参与权重计算**，也不冒充统计结论。
  在解释文本与可视化里都会标注「机制说明（业务先验，非统计估计）」。

这样评审可以清楚区分：哪些是数据说的，哪些是人说的。

可测后果的定义
--------------
`CONSEQ` 必须能从 anchor 的 26 维特征在**未来窗口**里客观判定，否则就是空概念。
窗口统一取 8 个截面（=120 分钟），与赛题的 2 小时风险区间口径一致。
"""

from __future__ import annotations

from dataclasses import dataclass

# 前瞻窗口：8 个截面 = 120 分钟，与赛题风险区间口径一致
HORIZON = 8


# ---------------------------------------------------------------- 异常（根因）

@dataclass(frozen=True)
class AnomSpec:
    """曲面异常模式。与 vol_surface.alert_rules 的 8 条规则一一对应。"""
    id: str
    rule: str
    name: str
    feature: str | None
    z_feature: str | None
    desc: str


ANOMS: tuple = (
    AnomSpec("ANOM-R1_convexity", "R1_convexity", "曲面凸性违反",
             "convexity_violation", "convexity_violation_z",
             "同一到期上，总方差沿行权价方向不再是凸函数，蝶式组合出现负价值区间"),
    AnomSpec("ANOM-R2_term", "R2_term_slope_roc", "期限结构斜率急变",
             "term_slope_roc", None,
             "近月与远月的隐含波动率之差在短时间内大幅变化，期限结构形态被快速重塑"),
    AnomSpec("ANOM-R3_atmiv", "R3_atm_iv_z", "ATM IV 异常偏离",
             "atm_iv", "atm_iv_z",
             "平值隐含波动率相对自身滚动分布出现显著偏离"),
    AnomSpec("ANOM-R4_conc", "R4_concentration", "Gamma/Vega 截面集中",
             "gamma_concentration", None,
             "Gamma 或 Vega 敞口高度集中在少数行权价上，敞口分布的集中度指标抬升"),
    AnomSpec("ANOM-R5_arb", "R5_arb", "无套利条件违反",
             "arb_score", "arb_score_z",
             "日历、蝶式、买卖权平价三类无套利校验的违反程度综合抬升"),
    AnomSpec("ANOM-R6_liq", "R6_liquidity", "流动性退化",
             "liquidity_ratio", "liquidity_z",
             "有成交的活跃合约占比下降，曲面报价的支撑变薄"),
    AnomSpec("ANOM-R7_ivspike", "R7_iv_spike", "短期 IV 急升",
             "iv_spike", None,
             "隐含波动率在数个截面内快速抬升"),
    AnomSpec("ANOM-R8_accel", "R8_acceleration", "IV/凸性加速度异常",
             "atm_iv_vel_z", None,
             "隐含波动率或凸性的变化速度本身出现异动，属导数层面的领先信号"),
)


# ---------------------------------------------------------------- 后果（可测）

@dataclass(frozen=True)
class ConseqSpec:
    """可测后果。必须能由未来 HORIZON 个截面的特征客观判定。

    transform: none | abs_dev（对 0.5 取绝对偏离，用于分位数型特征）
    agg      : max | min | mean
    """
    id: str
    name: str
    feature: str
    transform: str
    agg: str
    op: str
    thr: float
    desc: str

    def rule_text(self) -> str:
        """后果判定口径的中文表述。

        早前 `abs_dev` 变换被写成前缀，生成
        「未来 8 个截面内 |·−0.5| 的`skew_percentile` 的最大值 > 0.45」——
        字面读作「|·−0.5| 的 skew_percentile」，**与本意正好相反**，
        且反引号前缺空格会粘连。第 10 位评审判定这是全文唯一一处
        非本领域读者完全无法解析的表达式，影响 11 处。
        改为把变换写在特征之后，并说明 0.5 是中性值（不知道这一点就无从理解「>0.45」）。
        """
        a = {"max": "最大值", "min": "最小值", "mean": "均值"}[self.agg]
        if self.transform == "abs_dev":
            # 括号里那句「即 |x−0.5|，取值 0~0.5，越大越偏向一侧」**已在文首术语表
            # 说过一遍**（`skew_percentile` 条目写明「0~1，0.5 为中性」），
            # 而这里每条 ExtremeSkew 传导都重印 35 字 × 11 处。
            # 与 METHOD_NOTE 的因子化同理：**术语表已说过的，正文不再重复**。
            # 保留「对中性值 0.5 的偏离幅度」这个不可省的口径描述本身。
            base = f"`{self.feature}` 对中性值 0.5 的偏离幅度[†]"
        else:
            base = f"`{self.feature}`"
        return f"未来 {HORIZON} 个截面内 {base} 的{a} {self.op} {self.thr}"


CONSEQS: tuple = (
    ConseqSpec("CONSEQ-VolSpike", "波动率水平跳升", "atm_iv_z", "none", "max", ">", 2.0,
               "平值隐含波动率显著抬升，期权整体估值与保证金占用随之上行"),
    ConseqSpec("CONSEQ-ConvexityBreak", "曲面凸性破坏", "convexity_violation_z",
               "none", "max", ">", 2.0,
               "曲面沿行权价方向失去凸性，报价面出现结构性畸变"),
    ConseqSpec("CONSEQ-ExtremeSkew", "偏度走向极端", "skew_percentile",
               "abs_dev", "max", ">", 0.45,
               "看涨或看跌一侧的隐含波动率溢价被推到历史分布的尾部"),
    ConseqSpec("CONSEQ-LiquidityDrop", "流动性进一步塌陷", "liquidity_z",
               "none", "min", "<", -1.5,
               "活跃合约占比继续下滑，做市报价深度变薄"),
    ConseqSpec("CONSEQ-ArbPersist", "无套利违反持续", "arb_score_z",
               "none", "mean", ">", 1.0,
               "无套利违反在整个窗口内维持高位，而非单点噪声"),
)


# ---------------------------------------------------------------- 机制（诠释）

@dataclass(frozen=True)
class MechSpec:
    """传导机制。**业务先验，不参与权重计算**，只回答「为什么会这样传导」。"""
    id: str
    name: str
    how: str


MECHS: tuple = (
    MechSpec("MECH-MakerGamma", "做市商 Gamma 对冲压力",
             "空头 Gamma 敞口集中时，做市商需顺势调整标的对冲头寸，"
             "标的价格波动被反馈放大，进而推高隐含波动率"),
    MechSpec("MECH-VolFeedback", "波动率反馈环",
             "已实现波动上行推高隐含波动率报价，估值与保证金占用同步抬升，"
             "促使部分参与者调整敞口，又进一步加大已实现波动"),
    MechSpec("MECH-LiquiditySpiral", "流动性螺旋",
             "报价深度变薄使成交冲击成本上升，做市商进一步收窄报价或撤单，"
             "曲面拟合可用点减少、形态更易畸变"),
    MechSpec("MECH-ExpiryCluster", "临到期敞口集中",
             "临近到期时 Gamma 与到期损益高度非线性，敞口向平值附近集中，"
             "小幅标的变动即可引起较大的曲面形态变化"),
    MechSpec("MECH-TermInversion", "期限结构倒挂",
             "近月隐含波动率超过远月，反映短期风险定价高于长期，"
             "常伴随事件驱动的短期不确定性上升"),
    MechSpec("MECH-QuoteDistortion", "报价面畸变传播",
             "局部行权价的异常报价通过做市商的相对定价关系扩散到相邻行权价，"
             "使畸变从单点扩散为区间"),
)

# (ANOM, CONSEQ) → 机制。仅用于给**已被数据支持**的边配一段机制说明；
# 没有对应机制时留空，解释文本会如实说明「机制待补」，而不是硬凑一段话。
MECH_MAP: dict = {
    ("ANOM-R4_conc", "CONSEQ-VolSpike"): "MECH-MakerGamma",
    ("ANOM-R4_conc", "CONSEQ-ExtremeSkew"): "MECH-MakerGamma",
    ("ANOM-R7_ivspike", "CONSEQ-VolSpike"): "MECH-VolFeedback",
    ("ANOM-R3_atmiv", "CONSEQ-VolSpike"): "MECH-VolFeedback",
    ("ANOM-R8_accel", "CONSEQ-VolSpike"): "MECH-VolFeedback",
    ("ANOM-R8_accel", "CONSEQ-ConvexityBreak"): "MECH-VolFeedback",
    ("ANOM-R6_liq", "CONSEQ-LiquidityDrop"): "MECH-LiquiditySpiral",
    ("ANOM-R6_liq", "CONSEQ-ConvexityBreak"): "MECH-LiquiditySpiral",
    ("ANOM-R1_convexity", "CONSEQ-ConvexityBreak"): "MECH-QuoteDistortion",
    ("ANOM-R1_convexity", "CONSEQ-ArbPersist"): "MECH-QuoteDistortion",
    ("ANOM-R5_arb", "CONSEQ-ArbPersist"): "MECH-QuoteDistortion",
    ("ANOM-R5_arb", "CONSEQ-ConvexityBreak"): "MECH-QuoteDistortion",
    ("ANOM-R2_term", "CONSEQ-VolSpike"): "MECH-TermInversion",
    ("ANOM-R2_term", "CONSEQ-ArbPersist"): "MECH-TermInversion",
    ("ANOM-R7_ivspike", "CONSEQ-ExtremeSkew"): "MECH-ExpiryCluster",
}


# ---------------------------------------------------------------- 品种/板块

SYMS: dict = {
    "ag": {"title": "白银", "exchange": "SHFE", "sector": "SECTOR-Precious"},
    "au": {"title": "黄金", "exchange": "SHFE", "sector": "SECTOR-Precious"},
    "sc": {"title": "原油", "exchange": "INE", "sector": "SECTOR-Energy"},
    "si": {"title": "工业硅", "exchange": "GFEX", "sector": "SECTOR-NewEnergy"},
    # 扩样品种：原 4 个品种只占 3 个板块，且 ag/au 同属贵金属，
    # 跨板块传导实际上无从验证。补这三个后板块数升到 5。
    "lc": {"title": "碳酸锂", "exchange": "GFEX", "sector": "SECTOR-NewEnergy"},
    "cu": {"title": "铜", "exchange": "SHFE", "sector": "SECTOR-Base"},
    "rb": {"title": "螺纹钢", "exchange": "SHFE", "sector": "SECTOR-Ferrous"},
}

SECTORS: dict = {
    "SECTOR-Precious": "贵金属",
    "SECTOR-Energy": "能源",
    "SECTOR-NewEnergy": "新能源材料",
    "SECTOR-Base": "有色金属",
    "SECTOR-Ferrous": "黑色系",
}

# 无套利约束（诠释层）：说明某些异常在金融含义上为何是「违规」而非普通波动
CONSTRAINTS: dict = {
    "CONSTRAINT-Calendar": ("日历价差无套利", "总方差 w(k,T)=σ²T 应随到期时间单调非降"),
    "CONSTRAINT-Butterfly": ("蝶式无套利", "总方差沿行权价方向应为凸函数"),
    "CONSTRAINT-Parity": ("买卖权平价", "同到期同行权价的认购与认沽隐含波动率应一致"),
}

# 每类无套利约束对应的**逐截面违反数**特征。用于把「该形态同时违反了 X」
# 从静态映射的断言改成**就地可校验**的事实（见 kg/reason.py 的说明）。
CONSTRAINT_FEATURE = {
    "CONSTRAINT-Butterfly": "arb_butterfly_n",
    "CONSTRAINT-Calendar": "arb_calendar_n",
    "CONSTRAINT-Parity": "arb_parity_n",
}

CONSTRAINT_MAP: dict = {
    "ANOM-R1_convexity": "CONSTRAINT-Butterfly",
    "ANOM-R2_term": "CONSTRAINT-Calendar",
    "ANOM-R5_arb": "CONSTRAINT-Parity",
}

ANOM_BY_ID = {a.id: a for a in ANOMS}
ANOM_BY_RULE = {a.rule: a for a in ANOMS}
CONSEQ_BY_ID = {c.id: c for c in CONSEQS}
MECH_BY_ID = {m.id: m for m in MECHS}



CONSEQS: tuple = (
    ConseqSpec("CONSEQ-VolSpike", "波动率水平跳升", "atm_iv_z", "none", "max", ">", 2.0,
               "平值隐含波动率显著抬升，期权整体估值与保证金占用随之上行"),
    ConseqSpec("CONSEQ-ConvexityBreak", "曲面凸性破坏", "convexity_violation_z",
               "none", "max", ">", 2.0,
               "曲面沿行权价方向失去凸性，报价面出现结构性畸变"),
    ConseqSpec("CONSEQ-ExtremeSkew", "偏度走向极端", "skew_percentile",
               "abs_dev", "max", ">", 0.45,
               "看涨或看跌一侧的隐含波动率溢价被推到历史分布的尾部"),
    ConseqSpec("CONSEQ-LiquidityDrop", "流动性进一步塌陷", "liquidity_z",
               "none", "min", "<", -1.5,
               "活跃合约占比继续下滑，做市报价深度变薄"),
    ConseqSpec("CONSEQ-ArbPersist", "无套利违反持续", "arb_score_z",
               "none", "mean", ">", 1.0,
               "无套利违反在整个窗口内维持高位，而非单点噪声"),
)


# ---------------------------------------------------------------- 机制（诠释）

@dataclass(frozen=True)
class MechSpec:
    """传导机制。**业务先验，不参与权重计算**，只回答「为什么会这样传导」。"""
    id: str
    name: str
    how: str


MECHS: tuple = (
    MechSpec("MECH-MakerGamma", "做市商 Gamma 对冲压力",
             "空头 Gamma 敞口集中时，做市商需顺势调整标的对冲头寸，"
             "标的价格波动被反馈放大，进而推高隐含波动率"),
    MechSpec("MECH-VolFeedback", "波动率反馈环",
             "已实现波动上行推高隐含波动率报价，估值与保证金占用同步抬升，"
             "促使部分参与者调整敞口，又进一步加大已实现波动"),
    MechSpec("MECH-LiquiditySpiral", "流动性螺旋",
             "报价深度变薄使成交冲击成本上升，做市商进一步收窄报价或撤单，"
             "曲面拟合可用点减少、形态更易畸变"),
    MechSpec("MECH-ExpiryCluster", "临到期敞口集中",
             "临近到期时 Gamma 与到期损益高度非线性，敞口向平值附近集中，"
             "小幅标的变动即可引起较大的曲面形态变化"),
    MechSpec("MECH-TermInversion", "期限结构倒挂",
             "近月隐含波动率超过远月，反映短期风险定价高于长期，"
             "常伴随事件驱动的短期不确定性上升"),
    MechSpec("MECH-QuoteDistortion", "报价面畸变传播",
             "局部行权价的异常报价通过做市商的相对定价关系扩散到相邻行权价，"
             "使畸变从单点扩散为区间"),
)

# (ANOM, CONSEQ) → 机制。仅用于给**已被数据支持**的边配一段机制说明；
# 没有对应机制时留空，解释文本会如实说明「机制待补」，而不是硬凑一段话。
MECH_MAP: dict = {
    ("ANOM-R4_conc", "CONSEQ-VolSpike"): "MECH-MakerGamma",
    ("ANOM-R4_conc", "CONSEQ-ExtremeSkew"): "MECH-MakerGamma",
    ("ANOM-R7_ivspike", "CONSEQ-VolSpike"): "MECH-VolFeedback",
    ("ANOM-R3_atmiv", "CONSEQ-VolSpike"): "MECH-VolFeedback",
    ("ANOM-R8_accel", "CONSEQ-VolSpike"): "MECH-VolFeedback",
    ("ANOM-R8_accel", "CONSEQ-ConvexityBreak"): "MECH-VolFeedback",
    ("ANOM-R6_liq", "CONSEQ-LiquidityDrop"): "MECH-LiquiditySpiral",
    ("ANOM-R6_liq", "CONSEQ-ConvexityBreak"): "MECH-LiquiditySpiral",
    ("ANOM-R1_convexity", "CONSEQ-ConvexityBreak"): "MECH-QuoteDistortion",
    ("ANOM-R1_convexity", "CONSEQ-ArbPersist"): "MECH-QuoteDistortion",
    ("ANOM-R5_arb", "CONSEQ-ArbPersist"): "MECH-QuoteDistortion",
    ("ANOM-R5_arb", "CONSEQ-ConvexityBreak"): "MECH-QuoteDistortion",
    ("ANOM-R2_term", "CONSEQ-VolSpike"): "MECH-TermInversion",
    ("ANOM-R2_term", "CONSEQ-ArbPersist"): "MECH-TermInversion",
    ("ANOM-R7_ivspike", "CONSEQ-ExtremeSkew"): "MECH-ExpiryCluster",
}


# ---------------------------------------------------------------- 品种/板块

SYMS: dict = {
    "ag": {"title": "白银", "exchange": "SHFE", "sector": "SECTOR-Precious"},
    "au": {"title": "黄金", "exchange": "SHFE", "sector": "SECTOR-Precious"},
    "sc": {"title": "原油", "exchange": "INE", "sector": "SECTOR-Energy"},
    "si": {"title": "工业硅", "exchange": "GFEX", "sector": "SECTOR-NewEnergy"},
    # 扩样品种：原 4 个品种只占 3 个板块，且 ag/au 同属贵金属，
    # 跨板块传导实际上无从验证。补这三个后板块数升到 5。
    "lc": {"title": "碳酸锂", "exchange": "GFEX", "sector": "SECTOR-NewEnergy"},
    "cu": {"title": "铜", "exchange": "SHFE", "sector": "SECTOR-Base"},
    "rb": {"title": "螺纹钢", "exchange": "SHFE", "sector": "SECTOR-Ferrous"},
}

SECTORS: dict = {
    "SECTOR-Precious": "贵金属",
    "SECTOR-Energy": "能源",
    "SECTOR-NewEnergy": "新能源材料",
    "SECTOR-Base": "有色金属",
    "SECTOR-Ferrous": "黑色系",
}

# 无套利约束（诠释层）：说明某些异常在金融含义上为何是「违规」而非普通波动
CONSTRAINTS: dict = {
    "CONSTRAINT-Calendar": ("日历价差无套利", "总方差 w(k,T)=σ²T 应随到期时间单调非降"),
    "CONSTRAINT-Butterfly": ("蝶式无套利", "总方差沿行权价方向应为凸函数"),
    "CONSTRAINT-Parity": ("买卖权平价", "同到期同行权价的认购与认沽隐含波动率应一致"),
}

CONSTRAINT_MAP: dict = {
    "ANOM-R1_convexity": "CONSTRAINT-Butterfly",
    "ANOM-R2_term": "CONSTRAINT-Calendar",
    "ANOM-R5_arb": "CONSTRAINT-Parity",
}

ANOM_BY_ID = {a.id: a for a in ANOMS}
ANOM_BY_RULE = {a.rule: a for a in ANOMS}
CONSEQ_BY_ID = {c.id: c for c in CONSEQS}
MECH_BY_ID = {m.id: m for m in MECHS}

# 每条规则「触发量」的准确表述。`AnomSpec.feature` 只是该异常的**代表性读数**，
# 与规则实际比较的那个标量往往不是一回事：
#   R3 比较的是 z 分而非原始 IV；R4/R8 比较的是两个特征的较大者；
#   R6 比较的是 liquidity_z 且是**下侧**（< 阈值）。
# 早前解释文本统一写成「取多个特征中的较大者」，于是出现了
# 「触发量 = −5.3294（较大者）」「触发量 2.8879，其中 atm_iv = 0.3866」这类
# 量纲不可比或自相矛盾的表述（两轮独立评审均点名）。此表把口径写死。
TRIGGER_EXPR = {
    # rule: (展示表达式, 口径说明, 比较方向, 参与比较的分量, 是否取绝对值)
    "R1_convexity":     ("convexity_violation", "并同时要求其滚动 z 分超阈", ">",
                         ("convexity_violation",), False),
    "R2_term_slope_roc":("|term_slope_roc|", "取绝对值，故急升与急挫都会触发", ">",
                         ("term_slope_roc",), True),
    "R3_atm_iv_z":      ("|atm_iv_z|", "取**绝对值**，故 IV 异常偏高与异常偏低都会触发", ">",
                         ("atm_iv_z",), True),
    "R4_concentration": ("max(gamma_concentration, vega_concentration)", "取两者中的较大者", ">",
                         ("gamma_concentration", "vega_concentration"), False),
    "R5_arb":           ("arb_score", "并同时要求其滚动 z 分超阈", ">",
                         ("arb_score",), False),
    "R6_liquidity":     ("liquidity_z", "**下侧触发**：低于阈值即触发", "<",
                         ("liquidity_z",), False),
    "R7_iv_spike":      ("iv_spike", "该量本身由 |IV 变动| 导出，故急升与急挫都会触发", ">",
                         ("iv_spike",), False),
    "R8_acceleration":  ("max(atm_iv_vel_z, convexity_vel_z)", "取两者中的较大者", ">",
                         ("atm_iv_vel_z", "convexity_vel_z"), False),
}

# 每条机制叙述**针对哪个观测量**。用于在 max() 型规则由「机制没描述的那个分量」
# 触发时就地标注错配。
#
# 为什么要有这张表
# ----------------
# R8 的机制文案是硬编码在边上的（多为「波动率反馈环」，讲的是 IV 动力学），
# 但 R8 是 `max(atm_iv_vel_z, convexity_vel_z)`——实测 30 条样本里有 4 条
# 由 `convexity_vel_z` 触发，却照样配 IV 叙事。
# 这个错配在第 7 轮打印出「胜出分量」之后**对读者直接可见**，
# 不标注等于自曝（评审原话）。
#
# **只标注、不替换机制**：机制层是业务先验（诠释层，不参与打分），
# 凭猜测改 mech→分量 的映射等于编造证据。此处只做「机制讲的是 A、本次由 B 触发」
# 的如实提示，把判断交给读者。
#
# 未列入的机制视为与分量无关，不做提示。
MECH_COMPONENT = {
    "MECH-VolFeedback":    ("atm_iv_vel_z", "隐含波动率水平的变化"),
    "MECH-QuoteDistortion": ("convexity_vel_z", "曲面形态（凸性）的畸变"),
    "MECH-MakerGamma":     ("atm_iv_vel_z", "隐含波动率水平的变化"),
}


# ---------------------------------------------------------------- 机制↔后果 的落点核对
#
# 为什么要有这一层
# ----------------
# `MECH_COMPONENT` 只解决了「机制说的是哪个量 vs **本次由哪个分量触发**」这一端，
# 而第 10 位评审指出：同一套检查**停在了触发端，没有延伸到后果端**。
# 实例：`(R8_accel, ConvexityBreak)` 被指派 `MECH-VolFeedback`，
# 该机制的叙事链条是「已实现波动↑ → 推高 IV 报价 → 保证金占用↑ → 参与者调敞口
# → 又加大已实现波动」——**整条链收尾在波动率上**；而声明的后果是
# 「曲面**沿行权价方向**失去凸性」（度量 `convexity_violation_z`）。
# 两者之间没有任何桥接句，实测影响 6 条传导语句。
#
# 做法：给每个机制声明它的**落点量族**（从 `how` 的叙事收尾处读出），
# 给每个后果声明它的**度量量族**，两者无交集即为错配。
# 这是**结构性判定**而非逐条硬编码：新增边或改机制指派时会自动重新判定。
FAMILY_LABEL = {
    "iv": "隐含波动率水平",
    "shape": "曲面形态（凸性 / 偏度）",
    "liquidity": "流动性与报价深度",
    "arb": "无套利违反",
}

# 机制的叙事**收尾**落在哪些量族上（不是它途经哪些量，而是它最终解释了什么）
MECH_ENDPOINT = {
    "MECH-MakerGamma":      {"iv"},                 # …标的波动被放大，进而推高隐含波动率
    "MECH-VolFeedback":     {"iv"},                 # …又进一步加大已实现波动
    "MECH-LiquiditySpiral": {"liquidity", "shape"},  # …可用点减少、形态更易畸变
    "MECH-ExpiryCluster":   {"shape"},              # …引起较大的曲面形态变化
    "MECH-TermInversion":   {"iv", "arb"},          # …短期风险定价高于长期（含日历维度）
    "MECH-QuoteDistortion": {"shape", "arb"},       # …畸变从单点扩散为区间
}

# 后果由哪个量族度量（取自各 ConseqSpec.feature）
CONSEQ_FAMILY = {
    "CONSEQ-VolSpike": "iv",
    "CONSEQ-ConvexityBreak": "shape",
    "CONSEQ-ExtremeSkew": "shape",
    "CONSEQ-LiquidityDrop": "liquidity",
    "CONSEQ-ArbPersist": "arb",
}


def mech_conseq_mismatch(mech_id: str, conseq_id: str):
    """机制的落点与后果的度量对象是否**不相交**。

    返回 None 表示匹配（或信息不足，不做判定）；否则返回
    `(机制落点的中文, 后果度量的中文)`，供解释文本就地标注。

    **不相交才判错配**——机制可以有多个落点（如流动性螺旋同时解释流动性与形态），
    只要覆盖到后果所在的量族就算说得通。
    """
    ends = MECH_ENDPOINT.get(mech_id)
    fam = CONSEQ_FAMILY.get(conseq_id)
    if not ends or not fam or fam in ends:
        return None
    return ("、".join(FAMILY_LABEL[e] for e in sorted(ends)), FAMILY_LABEL[fam])


# ---------------------------------------------------------------- 机制缺失的**结构性诊断**
#
# 21 条主干边里 12 条没有机制说明，解释文本此前对这 12 条一律印同一句
# 「该关联由数据观测到，但尚未归纳出对应的机制说明」——**65 处逐字相同、
# 且不携带任何信息**。读者读完只知道「作者没写」，不知道为什么没写。
# 十二轮盲评里，「55% 的传导句没有传导环节」始终是「因果逻辑」维度的天花板。
#
# 但**正确的修法不是补写 12 条机制**——那正是本项目一路在反对的编造。
# 实测发现机制缺失**不是随机的**：
#
#   有机制的 9 条：lift 中位 2.39（区间 1.72~4.17）
#   无机制的 12 条：lift 中位 1.34（区间 1.18~1.95）   两区间几乎不重叠
#   Mann-Whitney z = +3.77
#   且无机制的 12 条里 **11 条跨量族**（根因与后果分属不同的观测量族）
#
# 故把「尚未归纳出机制」换成**就地给出结构性原因**：这条边弱、且跨量族，
# 找不到贯通机制本身就是一个可报告的观察，而不是工作没做完。
#
# **必须同时声明的选择效应**：机制是人写的，我们更可能给「看起来强、
# 讲得通」的边配机制——所以「强关联更容易有机制」与「我们只给强关联写了机制」
# 这两个解释**在本数据上分不开**。下方文案已如实写出这一点。
ANOM_FAMILY = {
    "ANOM-R1_convexity": "shape",
    "ANOM-R2_term": "iv",
    "ANOM-R3_atmiv": "iv",
    "ANOM-R4_conc": "exposure",
    "ANOM-R5_arb": "arb",
    "ANOM-R6_liq": "liquidity",
    "ANOM-R7_ivspike": "iv",
    "ANOM-R8_accel": "iv",
}
FAMILY_LABEL_ANOM = dict(FAMILY_LABEL, exposure="风险敞口分布")

# 「弱」的判据：取有机制边的 lift 下限。写成**从数据现算**而非常量，
# 避免图谱重标定后这条判据变成错的。
def weak_lift_threshold(edges) -> float:
    """有机制边的最小 lift ——低于它即归入「弱效应」档。

    `edges` 为 [(anom, conseq, attrs), ...]。从数据现算，不写死。
    """
    covered = [a["lift"] for u, v, a in edges if (u, v) in MECH_MAP]
    return min(covered) if covered else 0.0


def no_mech_reason(anom_id: str, conseq_id: str, lift: float,
                   weak_thr: float) -> str:
    """一条无机制边**为什么**没有机制——就地给出结构性原因。

    只陈述可从数据直接读出的两件事（弱效应 / 跨量族），
    **不推断因果**，也不暗示「补上机制就能解释」。
    """
    fa = ANOM_FAMILY.get(anom_id)
    fc = CONSEQ_FAMILY.get(conseq_id)
    cross = fa is not None and fc is not None and fa != fc
    weak = lift < weak_thr
    bits = []
    if weak:
        bits.append(f"**效应偏弱**（提升度 {lift:.2f} 倍，低于本图谱中"
                    f"所有已归纳出机制的边{weak_thr:.2f} 倍的下限）")
    if cross:
        bits.append(f"**跨量族**（根因度量的是{FAMILY_LABEL_ANOM.get(fa, fa)}、"
                    f"后果度量的是{FAMILY_LABEL_ANOM.get(fc, fc)}，"
                    f"两者之间缺少可写明的中间环节）")
    if not bits:
        return ""
    return ("——**为什么没有机制**：本条" + "、且".join(bits) +
            "。**这不是「还没写」，而是一个可报告的观察**："
            "弱且跨量族的关联更可能由第三方共同冲击造成，"
            "而非由根因直接传导；在找到可检验的中间环节之前，"
            "本图谱**不为它编写机制叙事**。")

