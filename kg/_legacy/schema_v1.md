# 期权风险知识图谱 Schema 文档

> 对应代码：`kg/schema.py`（节点/边 spec）、`kg/build_graph.py`(NetworkX 实例化)、`kg/reasoner.py`(推理/解释)。
> 与赛题问题2对齐：可解释风险传导 → Schema 交付之一。

---

## 1. 设计目标

用一张**有向、带权、带证据**的图来建模：
- 波动率曲面特征（IV/凸性/偏度/期限结构/集中度）之间的**统计与套利约束**；
- 跨品种、跨板块的**风险传导关系**；
- 异常模式 → 传导机制 → 可能后果 的**可解释路径**。

评分对齐：
- 语义清晰（节点/边有中文标题；`how` 属性说明传导机制）；
- 因果逻辑合理（AMPLIFIES 边有 `how` 文本，约束边有合规性引用）；
- 无具体交易建议（解释只提示风险和分析，不下单）；
- 证据链可回溯（每条解释附 `evidence` 坐标：时间戳+特征+z分）。

---

## 2. 节点类型

| 前缀     | 含义 | 数量 | 主要锚点 |
|---------|------|-----|---------|
| RULE-*  | 预警规则节点（R1-R8） | 8 | 用于验证-解释分离 |
| ANOM-*  | 曲面异常模式 | 8 （+品种×4 扩展共 40) | R1 凸性、R2 期限斜率、R3 ATM IV、R4 集中度、R5 无套利、R6 流动性、R7 急升、R8 加速度 |
| GREEK-* | Greeks 膜态/集中度 | 8 | `gamma_concentration`, `vega_concentration`, `atm_iv`, `atm_iv_vel`, `convexity_vel`, `skew_val`, `convexity_violation`, `term_slope` |
| MECH-*  | 传导机制概念 | 6 | 做市商对冲压力（MakerGamma)、波动率反馈环（VolFeedback)、流动性螺旋（Liquidity)、临到期集中（ExpiryCluster)、期限倒挂（TermInversion)、凸性传播（ConvexitySpread) |
| SYM-*   | 标的品种 | 4 | ag/au/sc/si，挂交易所与板块 |
| ORDER-* | 板块 | 5 | 贵金属/能源原油/新能源/有色金属/大宗全板块 |
| CONSEQ-*| 可能后果 | 5 | 波动放大/偏度极端/流动性塌/无套利持续/保证金压力 |
| CATALOG-*| 类险组合 | 3 | 贵金属冲高/原油冲击/大宗反转 |
| CONSTRAINS-*| 无套利/期限结构约束 | 4 | 日历无套利/蝶式无套利/parity/期限结构单调性 |

每个节点有 `type` 和 `title` 属性（可选中文），用于可视化。

---

## 3. 边类型

| kind | 含义 | 典型用法 |
|------|------|---------|
| TRIGGERS | RULE-Rk 在特征越界时触发 ANOM | 规则锚定到解释链 |
| AMPLIFIES | 异常/机制加强下游 | 推理主线（ANOM→MECH→CONSEQ=>CATALOG) |
| PROPAGATES_TO | 跨品种传导 | 动态加入（anchor 历史) |
| CONSTRAINS | 无套利/几何约束 | 证据引用（非因果，合规性） |
| MEMBER_OF | 品种/品种-异常到板块归属 | 类险组合拼接 |
| CORRELATES | 跨品种同 rule 共现统计 | 权重=anchor 历史里 lookback 内的同现率 |

> 说明：`ANOM---SYM` 的扩展节点 (`ANOM-R1_convexity-ag` 等）用于**品种级**传导。
> CORRELATES 边的 source/target 均为品种级 ANOM 节点，权重对应"传导概率"。

---

## 4. 机制图（AMPLIFIES 主线）

| 下游 MECH | 上游 ANOM | 上游驱动下,下游会连到 |
|----------|-----------|-------------------|
| MECH-MakerGamma | ANOM-R4_conc | CONSEQ-VolSpike |
| MECH-VolFeedback | ANOM-R4_conc / ANOM-R8_accel | CONSEQ-VolSpike / CONSEQ-ExtremeSkew |
| MECH-Liquidity | ANOM-R6_liq | CONSEQ-LiquidityDrop |
| MECH-ExpiryCluster | ANOM-R7_ivspike | CONSEQ-MarginStress |
| MECH-TermInversion | ANOM-R2_term | CONSEQ-ArbPersist |
| MECH-ConvexitySpread | ANOM-R1_convexity | CONSEQ-ArbPersist |

AMPLIFIES 边有 `how` 属性描述传导逻辑（如"近月Gamma急升→做市商反向对冲→放大标的波动"）。
权重 `weight` 是传递力度的先验（当前用启发式 0.6-0.8，后续可用 anchor 数据估计）。

---

## 5. 板块归属与跨品种传导

- 品种 → 板块 (`MEMBER_OF`)：ag(SF)/au(SF)→ORDER-Precious、sc(INE)→ORDER-Energy、si(GF)→ORDER-NewEner。cu 等后续可加 ORDER-Cu.
- 板块 → 类险组合 (`AMPLIFIES`):ORDER-Precious → CATALOG-MetalsSurge、ORDER-Energy → CATALOG-OilShock，全板块事件 → CATALOG-BulkReversal。
- 品种级 ANOM 互传 (`CORRELATES`)：从 anchor 数据实测 rule 预警在同板块/跨板块的共现率。

---

## 6. 证据锚点（evidence）

- 每个 ANOM 节点挂 `feature`（特征名） + `z_feature`(z-score 键，若有）。
- 每条触发解释包含：
  - `ts`: 触发时间戳；
  - `rule`: 触发的规则 id;
  - `feature_value`: 实际曲面读数；
  - `z_value`: 对应 z 分（若为规则自适应阈值）;
  - `path`: 激活节点链（`ANOM→MECH→CONSEQ` 或 `ANOM→SYM→ORDER→CATALOG`);
  - `how`: 传导逻辑的中文描述(AMPLIFIES.how)。

解释格式为：`根因 → 链路(节点标题集) → 可能后果`，每句只描述风险与分析，不含交易指令。

---

## 7. 使用示例（代码接口）

```python
from kg.build_graph import load_graph
from kg.reasoner import explain_alert

g = load_graph("data_out/kg_graph.pkl")
# 一次预警时喂入: ts, symbol, 触发的 rule 列表, 当前特征字典
info = explain_alert(g, ts="20260114103000", symbol="ag",
                     triggered=["R4_concentration", "R7_iv_spike"],
                     feats={"gamma_concentration": 0.12, "iv_spike": 0.09})
print(info["summary"])  # 根因 → 链条 → 后果
```

后续在 dashboard 中把该解释挂在时间轴的 tooltip 上，并展示为一个简版 Sankey 图。
