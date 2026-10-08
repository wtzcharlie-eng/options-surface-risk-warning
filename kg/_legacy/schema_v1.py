"""kg.schema — 期权风险知识图谱 Schema 定义。

设计目标（赛题问题2）：
  把曲面特征之间、特征与标的资产、特征与风险因子之间的统计与套利约束关系
  建模为知识图谱，曲面异常时自动推理潜在影响范围与传导路径，生成自然语言解释。

节点类型（类型 → id 前缀 → 归属说明）
------
- ANOM-xxx    曲面异常模式。锚点=规则 R1-R8，每个节点挂证据特征名与阈值。
- GREEK-xxx   希腊字母因子或截面膜态（GAM_conc/VEGA_conc/ATM_IV/TD_skew）
- MECH-xxx    传导机制概念节点（不可直接观测，但其激活由上游节点驱动，下文详述）
- SYM-xxx     标的品种（ag/au/sc/si），带交易所与板块族
- ORDER-xxx   板块节点（precious金属、ferrous、energy、building、chemicals……）
- CONSEQ-xxx  可能后果节点（波动放大、凹性极端、流动性枯竭、IV期限倒挂）
- CATALOG-xxx 类险组合节点（全板块风险组合，如"贵金属冲高"路径）

边类型（语义）
------
- TRIGGERS      规则 R* 在对应锚点特征越过阈值时触发（rule→ANOM）
- AMPLIFIES     上游节点通过机制加强下游（机制图）
- PROPAGATES_TO 上下跨标的传导（板块内或事件共现）
- CONSTRAINS    无套利/几何约束（calibration, butterfly, parity 约束）
- MEMBER_OF     节点归入概念族（ANOM_MEM_FAMILY, ORDER_MEMBER_OF）
- CORRELATES    伴生统计相似度，weight=同板块/同时间段共现率（动态）

证据锚点（evidence）写在节点/边属性里，给解释器与评分器一条可回溯的证据链。

使用：构建工具在 build_graph.py 把这一套 schema 实例化为 NetworkX 图。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    ANOM = "ANOM"      # 曲面异常模式
    GREEK = "GREEK"
    MECH = "MECH"
    SYM = "SYM"
    ORDER = "ORDER"
    CONSEQ = "CONSEQ"
    CATALOG = "CATALOG"


# ------------------------------- 节点 spec -------------------------------

@dataclass(frozen=True)
class AnomSpec:
    """异常模式节点参数。"""
    id: str               # ANOM-R1_convexity 等
    rule: str             # 对应 alert_rules.py 触发 rule 名
    name: str
    feature: str          # 证据特征（默认features字典键）
    z_feature: str | None # 此项特征的z分名（可选，用于证据链评分）


ANOMS: tuple[AnomSpec, ...] = (
    AnomSpec(id="ANOM-R1_convexity", rule="R1_convexity",
             name="曲面凸性违反", feature="convexity_violation",
             z_feature="convexity_violation_z"),
    AnomSpec(id="ANOM-R2_term", rule="R2_term_slope_roc",
             name="期限结构斜率急变", feature="term_slope_roc",
             z_feature=None),
    AnomSpec(id="ANOM-R3_atmiv", rule="R3_atm_iv_z",
             name="ATM IV 异常偏离", feature="atm_iv",
             z_feature="atm_iv_z"),
    AnomSpec(id="ANOM-R4_conc", rule="R4_concentration",
             name="Gamma/Vega截面集中", feature=None,
             z_feature=None),  # 复合，见 MECH-GAM/VEGA
    AnomSpec(id="ANOM-R5_arb", rule="R5_arb",
             name="无套利违反", feature="arb_score",
             z_feature="arb_score_z"),
    AnomSpec(id="ANOM-R6_liq", rule="R6_liquidity",
             name="流动性枯竭", feature="liquidity_ratio",
             z_feature="liquidity_z"),
    AnomSpec(id="ANOM-R7_ivspike", rule="R7_iv_spike",
             name="短期 IV 急升", feature="iv_spike",
             z_feature=None),
    AnomSpec(id="ANOM-R8_accel", rule="R8_acceleration",
             name="IV/凸性加速度", feature=None,
             z_feature=None),
)


# GREEK 锚点（曲面膜态的子节点，R4 由 G-G GAM/VEGA 两个细分对应）
GREEKS: tuple[str, ...] = (
    "GREEK-gamma_concentration",
    "GREEK-vega_concentration",
    "GREEK-atm_iv",
    "GREEK-atm_iv_vel",
    "GREEK-convexity_vel",
    "GREEK-skew_val",          # 25-delta skew 值（微笑陡度）
    "GREEK-convexity_violation",
    "GREEK-term_slope",
)


# 传导机制（Mechanism）：不可直接观测但其"活性"由上游决定，驱动可解释链路
MECHS: tuple[str, ...] = (
    "MECH-MakerGamma",      # 做市商对冲压力：近月Gamma急升→做市商反向对冲→放大标的波动
    "MECH-VolFeedback",     # 波动率反馈环:Vega暴露+/OI上升→轮次增持→波动放大
    "MECH-Liquidity",       # 流动性螺旋:流动性退化→报价走宽→价差波及情绪
    "MECH-ExpiryCluster",   # 临到期集中:近月期权集中到期→持仓移仓→期限结构扭曲
    "MECH-TermInversion",   # 期限倒挂传递:近月IV上凸→资金移仓→远月传导
    "MECH-ConvexitySpread", # 凸性传播:行权价非凸→蝶市违例→蝶式敞口啃跌
)


# 后果节点（可能后果，给解释器收口）
CONSEQ: tuple[str, ...] = (
    "CONSEQ-VolSpike",      # 月末波动剧烈上扬
    "CONSEQ-ExtremeSkew",   # 偏度极端（单向看多/看空）
    "CONSEQ-LiquidityDrop", # 买卖盘减低，深度骤减
    "CONSEQ-ArbPersist",    # 无套利违例持续,对做市定价造成显著偏移
    "CONSEQ-MarginStress",  # 保证金压力+临近交割压仓
)


# 板块与品种
SYMS: dict[str, dict[str, str]] = {
    "ag": {"exchange": "SF", "order": "ORDER-Precious", "title": "白银"},
    "au": {"exchange": "SF", "order": "ORDER-Precious", "title": "黄金"},
    "sc": {"exchange": "INE", "order": "ORDER-Energy", "title": "原油"},
    "si": {"exchange": "GF", "order": "ORDER-NewEner", "title": "工业硅(新能源)"},
}

ORDERS: dict[str, str] = {
    "ORDER-Precious": "贵金属",
    "ORDER-Energy": "能源(原油)",
    "ORDER-NewEner": "新能源(工业硅等)",
    "ORDER-Cu": "有色(铜)",
    "ORDER-Bulk": "大宗全板块",     # 2024-9.24/2025-4关税 类泛宏观
}

# 可解释的类险组合节点（对应第一阶段可测的风险簇, 上溯自板块）
CATALOGS: dict[str, str] = {
    "CATALOG-MetalsSurge": "贵金属冲高风险组合",
    "CATALOG-OilShock": "原油/能源冲击风险组合",
    "CATALOG-BulkReversal": "大宗行情逆转(全板块反转)组合",
}


# ------------------------------- 边 spec -------------------------------

@dataclass(frozen=True)
class Edge:
    u: str
    v: str
    kind: str               # TRIGGERS / AMPLIFIES / PROPAGATES_TO / CONSTRAINS / MEMBER_OF / CORRELATES
    weight: float = 1.0
    attrs: dict[str, Any] = field(default_factory=dict)


def static_edges(syms: dict[str, dict[str, str]] | None = None) -> tuple[Edge, ...]:
    """构出静态(schema)层边。

    dynamic CORRELATES 权值化由 build_graph.py 用 anchor 数据补齐；
    此处的静态边包含机制图、板块族归属、约束链。
    """
    edges: list[Edge] = []

    # rule→ANOM（规则触发）
    for a in ANOMS:
        edges.append(Edge(f"RULE-{a.rule}", a.id, "TRIGGERS", 1.0,
                          {"threshold": "DEFAULT_PARAMS"}))

    # 异常→GREEK（归属到希腊字母概率）: 只允许连接到已声明的 ANOM
    anom_by_rule = {a.id: a for a in ANOMS}
    anom_by_feature = {}
    for a in ANOMS:
        if a.feature:
            anom_by_feature[a.feature] = a.id
        if a.z_feature:
            anom_by_feature[a.z_feature] = a.id

    for g in GREEKS:
        feat = g.replace("GREEK-", "")
        target = anom_by_feature.get(feat) or anom_by_feature.get(feat + "_z") or \
            anom_by_feature.get(feat.replace("_vel", "")) or \
            anom_by_feature.get(feat.replace("atm_iv_vel", "atm_iv_vel_z"))
        if target is None:
            # 回退: 跟同区段的 ANOM 挂钩
            if "convexity" in feat:
                target = "ANOM-R1_convexity"
            elif "atm_iv" in feat:
                target = "ANOM-R3_atmiv"
            elif "iv" in feat:
                target = "ANOM-R7_ivspike"
        if target:
            edges.append(Edge(g, target, "AMPLIFIES", 0.6,
                              {"how": "同特征作为上游证据"}))

    # 异常模式到机制（上游驱动）
    mech_pairs = (
        ("ANOM-R4_conc",            "MECH-MakerGamma"),
        ("ANOM-R4_conc",            "MECH-VolFeedback"),
        ("ANOM-R7_ivspike",         "MECH-ExpiryCluster"),
        ("ANOM-R6_liq",             "MECH-Liquidity"),
        ("ANOM-R1_convexity",       "MECH-ConvexitySpread"),
        ("ANOM-R2_term",            "MECH-TermInversion"),
        ("ANOM-R8_accel",           "MECH-VolFeedback"),
    )
    for u, v in mech_pairs:
        edges.append(Edge(u, v, "AMPLIFIES", 0.8, {"how": "激活传导上游驱动"}))

    # 机制→后果
    for u, v, how in (
        ("MECH-MakerGamma",       "CONSEQ-VolSpike",     "近月对冲压力放大标的波动"),
        ("MECH-VolFeedback",      "CONSEQ-VolSpike",     "Vega暴露增强放大波动"),
        ("MECH-VolFeedback",      "CONSEQ-ExtremeSkew",  "到期向拉陡偏度"),
        ("MECH-Liquidity",        "CONSEQ-LiquidityDrop","流动性枯竭恶化买卖盘"),
        ("MECH-TermInversion",    "CONSEQ-ArbPersist",   "期限结构违反持续"),
        ("MECH-ConvexitySpread",  "CONSEQ-ArbPersist",   "凸性违例持续蝶式敞口"),
        ("MECH-ExpiryCluster",    "CONSEQ-MarginStress", "临到期集中中段保证金压力"),
    ):
        edges.append(Edge(u, v, "AMPLIFIES", 0.7, {"how": how}))

    # 板块族归属（SYM→ORDER；品种级ANOM→SYM；再给品种级ANOM挂上机制边）
    for sym, meta in (syms or SYMS).items():
        order = meta["order"]
        edges.append(Edge(f"SYM-{sym}", order, "MEMBER_OF", 1.0, {}))
        for a in ANOMS:
            # 品种-异常联合节点 → 品种（用于跨品种传导路径的锚点）
            edges.append(Edge(f"{a.id}-{sym}", f"SYM-{sym}", "MEMBER_OF", 1.0, {}))
            # 品种级异常也要挂上机制边，否则推理会停在 SYM
            # 用品种级ANOM替代通用ANOM接入AMPLIFIES主线
            for u, v in mech_pairs:
                if u == a.id:
                    edges.append(Edge(f"{a.id}-{sym}", v, "AMPLIFIES", 0.8, {"how": "激活传导上游驱动"}))

    # 板块到类险组合（上溯）
    cat_map = (
        ("ORDER-Precious", "CATALOG-MetalsSurge"),
        ("ORDER-Energy",   "CATALOG-OilShock"),
        ("ORDER-Precious", "CATALOG-MetalsSurge"),
        ("ORDER-NewEner",  "CATALOG-BulkReversal"),
        ("ORDER-Precious", "CATALOG-BulkReversal"),
        ("ORDER-Energy",   "CATALOG-BulkReversal"),
        ("ORDER-Cu",       "CATALOG-MetalsSurge"),
    )
    for u, v in cat_map:
        edges.append(Edge(u, v, "AMPLIFIES", 0.6, {"how": "板块证据归并到类险组合"}))

    # 约束（无套利）→异常详述层
    edges.append(Edge("CONSTRAINS-Calendar",    "ANOM-R5_arb",       "CONSTRAINS", 1.0, {"pr": "日历无套利"}))
    edges.append(Edge("CONSTRAINS-Butterfly",   "ANOM-R1_convexity", "CONSTRAINS", 1.0, {"pr": "蝶式无套利"}))
    edges.append(Edge("CONSTRAINS-Parity",      "ANOM-R5_arb",       "CONSTRAINS", 1.0, {"pr": "买-call-卖put 平价"}))
    edges.append(Edge("CONSTRAINS-TermStruct",  "ANOM-R2_term",      "CONSTRAINS", 1.0, {"pr": "期限结构单调性"}))

    return tuple(edges)


# ------------------------------- schema doc -------------------------------

def schema_summary() -> str:
    """把 schema 排序可读的文档头字符串,给 kg/schema.md 生成。"""
    lines = ["# 期权风险知识图谱 Schema",
             "",
             "## 节点类型",
             "",
             "| 类型前缀 | 说明 | 主要锚点 |",
             "|---------|------|----------|",
             "| RULE-*  | 预警规则节点(R1-R8) | 用于验证-解释分离 |",
             "| ANOM-*  | 曲面异常模式 | alert_rules R1-R8(凸性/期限斜率/ATM IV/集中度/无套利/流动性/急升/加速度) |",
             "| GREEK-* | Greeks 与膜态(Gamma/Vega HHI, ATM IV, 加速度,sky_val) | 作为曲面膜态子节点占入 |",
             "| MECH-*  | 传导机制(做市商对冲/反馈环/流动性螺旋/临到期集中/期限倒挂/凸性传播) | 由 ANOM 驱动,激活度由上游决定 |",
             "| SYM-*   | 标的品种(ag/au/sc/si), 挂交易所与板块 | 品种归属 |",
             "| ORDER-* | 板块节点(贵金属/能源原油/新能源工业硅/有色金属) | 品种族 |",
             "| CONSEQ-*| 可能后果(波动放大/偏度极端/流动性塌/无套利持续/保证金压力) | 解释收口 |",
             "| CATALOG-*| 类险组合(贵金属冲高/原油冲击/大宗反转) | 板块上溯 |",
             "| CONSTRAINS-*| 无套利约束 | 解释引证的合规性字典 |",
             "",
             "## 边类型",
             "",
             "| kind         | 语义 | 主要用途 |",
             "|-------------|------|---------|",
             "| TRIGGERS    | 规则 Rk 在锚点特征超阈时触发 ANOM | 把规则锚定到解释链 |",
             "| AMPLIFIES   | 异常/机制 在传导链上加强下游 | 推理主线 |",
             "| PROPAGATES_TO | 跨品种传导(同板块>跨板块) | 板块影响面 |",
             "| CONSTRAINS  | 无套利/几何约束 | 证据口引用 |",
             "| MEMBER_OF   | 品种/异常到板块归属 | 类险组合拼接 |",
             "| CORRELATES  | 伴生统计相关 | weight=同板块共现率,跨板块事件自动体现 |",
             "",
             "## 证据锚点",
             "",
             "- 每条 ANOM 节点挂 `feature/z_feature`(对应 vol_surface.features.*)",
             "- 每条 AMPLIFIES 边挂 `how` 字段(量化解释: 说明传导机制)",
             "- CORRELATES 边权 = `data_out/anchor` 缓存上同板块公开协灾事件的共现率",
             "- 每个预警的解释是 [根因(拉顶机制)→路径(节点集)→后果(可能的曲线形变)] 三段,",
             "  另附 `evidence` 记录该预警时点的 OG 特征 z 值,便于评审回溯",
             ]
    return "\n".join(lines)
