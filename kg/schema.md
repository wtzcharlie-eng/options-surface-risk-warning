# 期权风险知识图谱 Schema

> 本文件由 `scripts/report_kg.py` 从 `kg/schema.py` 自动生成，请勿手工编辑。

## 分层

| layer | 含义 | 权重来源 | 参与推理打分 |
|-------|------|---------|------------|
| `measured` | 实测层 | 87,517 个**训练期**真实截面标定的提升度 lift | 是 |
| `interpretive` | 诠释层 | 业务先验，无权重 | 否 |
| `meta` | 品种/板块等元数据 | — | 否 |

## 节点类型

| type | layer | 数量 | 说明 |
|------|-------|-----|------|
| `ANOM` | `measured` | 8 | 曲面异常模式，与问题1 的 8 条预警规则一一对应 |
| `CONSEQ` | `measured` | 5 | 可测后果，由未来 8 个截面的特征客观判定 |
| `MECH` | `interpretive` | 6 | 传导机制说明，业务先验 |
| `CONSTRAINT` | `interpretive` | 3 | 无套利约束，说明异常为何属定价结构问题 |
| `SYM` | `meta` | 7 | 品种 |
| `SECTOR` | `meta` | 5 | 板块 |
| `XSYM` | `measured` | 48 | 品种级异常节点，承载跨品种共现边 |

## 边类型

| kind | 起点 → 终点 | layer | 权重含义 |
|------|------------|-------|---------|
| `AMPLIFIES` | ANOM → CONSEQ | measured | lift 的 95% 保守下界（推理主干）|
| `PROPAGATES_TO` | XSYM → XSYM | measured | 小时匹配后 lift 的保守下界 |
| `EXPLAINS` | ANOM → MECH → CONSEQ | interpretive | 恒 0，仅供可视化 |
| `VIOLATES` | ANOM → CONSTRAINT | interpretive | 恒 0 |
| `MEMBER_OF` | SYM → SECTOR | meta | 恒 1 |

## 异常节点（ANOM）

| id | 名称 | 对应规则 | 主特征 | 含义 |
|----|------|---------|--------|------|
| `ANOM-R1_convexity` | 曲面凸性违反 | `R1_convexity` | `convexity_violation` | 同一到期上，总方差沿行权价方向不再是凸函数，蝶式组合出现负价值区间 |
| `ANOM-R2_term` | 期限结构斜率急变 | `R2_term_slope_roc` | `term_slope_roc` | 近月与远月的隐含波动率之差在短时间内大幅变化，期限结构形态被快速重塑 |
| `ANOM-R3_atmiv` | ATM IV 异常偏离 | `R3_atm_iv_z` | `atm_iv` | 平值隐含波动率相对自身滚动分布出现显著偏离 |
| `ANOM-R4_conc` | Gamma/Vega 截面集中 | `R4_concentration` | `gamma_concentration` | Gamma 或 Vega 敞口高度集中在少数行权价上，敞口分布的集中度指标抬升 |
| `ANOM-R5_arb` | 无套利条件违反 | `R5_arb` | `arb_score` | 日历、蝶式、买卖权平价三类无套利校验的违反程度综合抬升 |
| `ANOM-R6_liq` | 流动性退化 | `R6_liquidity` | `liquidity_ratio` | 有成交的活跃合约占比下降，曲面报价的支撑变薄 |
| `ANOM-R7_ivspike` | 短期 IV 急升 | `R7_iv_spike` | `iv_spike` | 隐含波动率在数个截面内快速抬升 |
| `ANOM-R8_accel` | IV/凸性加速度异常 | `R8_acceleration` | `atm_iv_vel_z` | 隐含波动率或凸性的变化速度本身出现异动，属导数层面的领先信号 |

## 后果节点（CONSEQ）——判定口径

| id | 名称 | 判定式 | 业务含义 |
|----|------|--------|---------|
| `CONSEQ-VolSpike` | 波动率水平跳升 | 未来 8 个截面内 `atm_iv_z` 的最大值 > 2.0 | 平值隐含波动率显著抬升，期权整体估值与保证金占用随之上行 |
| `CONSEQ-ConvexityBreak` | 曲面凸性破坏 | 未来 8 个截面内 `convexity_violation_z` 的最大值 > 2.0 | 曲面沿行权价方向失去凸性，报价面出现结构性畸变 |
| `CONSEQ-ExtremeSkew` | 偏度走向极端 | 未来 8 个截面内 `skew_percentile` 对中性值 0.5 的偏离幅度[†] 的最大值 > 0.45 | 看涨或看跌一侧的隐含波动率溢价被推到历史分布的尾部 |
| `CONSEQ-LiquidityDrop` | 流动性进一步塌陷 | 未来 8 个截面内 `liquidity_z` 的最小值 < -1.5 | 活跃合约占比继续下滑，做市报价深度变薄 |
| `CONSEQ-ArbPersist` | 无套利违反持续 | 未来 8 个截面内 `arb_score_z` 的均值 > 1.0 | 无套利违反在整个窗口内维持高位，而非单点噪声 |

## 机制节点（MECH，诠释层）

| id | 名称 | 机制说明 |
|----|------|---------|
| `MECH-MakerGamma` | 做市商 Gamma 对冲压力 | 空头 Gamma 敞口集中时，做市商需顺势调整标的对冲头寸，标的价格波动被反馈放大，进而推高隐含波动率 |
| `MECH-VolFeedback` | 波动率反馈环 | 已实现波动上行推高隐含波动率报价，估值与保证金占用同步抬升，促使部分参与者调整敞口，又进一步加大已实现波动 |
| `MECH-LiquiditySpiral` | 流动性螺旋 | 报价深度变薄使成交冲击成本上升，做市商进一步收窄报价或撤单，曲面拟合可用点减少、形态更易畸变 |
| `MECH-ExpiryCluster` | 临到期敞口集中 | 临近到期时 Gamma 与到期损益高度非线性，敞口向平值附近集中，小幅标的变动即可引起较大的曲面形态变化 |
| `MECH-TermInversion` | 期限结构倒挂 | 近月隐含波动率超过远月，反映短期风险定价高于长期，常伴随事件驱动的短期不确定性上升 |
| `MECH-QuoteDistortion` | 报价面畸变传播 | 局部行权价的异常报价通过做市商的相对定价关系扩散到相邻行权价，使畸变从单点扩散为区间 |

> 机制说明是**业务先验**，不参与权重计算，也未经独立验证。它只回答「为什么会这样传导」，其正确性不由本项目的数据背书。

## 边属性

`AMPLIFIES` 边携带以下可审计字段：

| 字段 | 含义 |
|------|------|
| `lift` | 提升度 P(C|A)/P(C) |
| `lift_ci_low` | lift 的 95% Wilson 保守下界，即边权 |
| `p_cond` | 条件概率 P(C|A) |
| `p_conseq` | 无条件基础率 P(C) |
| `n_anom` | 支撑度（异常出现次数） |
| `p_value` | 分块置换检验 p 值 |
| `self_feature` | 根因与后果是否共用同一底层特征 |
| `replicated` | 测试期是否可复现 |
| `mech / how` | 机制注解（诠释层） |
