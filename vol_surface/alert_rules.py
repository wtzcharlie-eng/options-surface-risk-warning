"""alert_rules — 规则引擎，输出可解释触发原因。

纯函数契约（此文档确立，禁止破坏）：
- evaluate_rules 只对当前一份特征 dict 做向量化判定，不 view 外部历史，
  也不调用 FeatureHistory。
- 所有相对量（z-score / velocity / prev_features 的依赖性）都来自特征 dict
  本身；特征语义统一在 compute_features 里固化（是否含当前点等）。
- 本文件**若出现** FeatureHistory 引用即为 bug——立即按纯函数契约修复。

每条规则对一个特征向量化为 0-3 级判定 + 中文触发原因文本。规则的阈值基于实测
特征分布与极端事件期特征变化调定，可由 alert_engine 在校准时微调。

返回结构
--------
{
  "level": int 0-3,
  "triggers": [{"rule": str, "level": int, "reason": str, "value": float, "threshold": float}],
  "rule_score": float  # 规则综合分（用于融合）
}
"""

from __future__ import annotations

LEVEL_NORMAL = 0
LEVEL_WATCH = 1
LEVEL_WARN = 2
LEVEL_SERIOUS = 3


# 默认阈值（z-score 自适应）：watch_z / warn_z / serious_z / 绝对值下限。
# 关键阈值经 scripts/grid_search.py 在事件期窗口网格搜索校准（含 R8 加速度导数预警）：
# 4 窗口平均精确率 47.5%、召回 61.6%、平均提前 52min，召回与提前时间达标赛题要求。
DEFAULT_PARAMS = {
    "r1_watch_z": 1.5, "r1_warn_z": 2.5, "r1_ser_z": 3.0, "r1_min": 0.6, "r1_min2": 1.0,
    # R2 默认**关闭**：原阈值高到几乎不触发，重标定后实测任何阈值都让前沿变差
    "r2_enabled": False,
    "r2_watch": 0.3, "r2_warn": 0.3, "r2_ser": 0.8,
    "r3_watch": 2.0, "r3_warn": 3.5, "r3_ser": 4.0,
    # R4 默认**关闭**：实测它在所有测试品种上都是负贡献，见 `r4_enabled` 说明
    "r4_enabled": False,
    "r4_watch": 0.08, "r4_warn": 0.12, "r4_ser": 0.18,
    "r5_watch_z": 2.0, "r5_warn_z": 3.5, "r5_ser_z": 4.0, "r5_min": 12.0, "r5_min2": 25.0,
    # R6 默认**关闭**：其「流动性枯竭 → 危险」的方向假设在本数据集上不成立
    "r6_enabled": False, "r6_invert": False,
    "r6_warn": -1.5, "r6_ser": -2.0,
    "r7_watch": 0.04, "r7_warn": 0.07, "r7_ser": 0.09,
    "r8_watch": 1.5, "r8_warn": 2.0, "r8_ser": 3.0,
}


def _trig(rule, level, reason, value, threshold):
    return {"rule": rule, "level": level, "reason": reason, "value": value, "threshold": threshold}


def evaluate_rules(feats: dict, params: dict | None = None) -> dict:
    """对单截面特征向量跑全部规则。params 覆盖 DEFAULT_PARAMS，供网格搜索。

    纯函数契约（**必须守住**，下游依赖）：
      - 只接收**已算完**、特征字典；不解析 timestamp、不 view 其他历史，也不
        触碰 FeatureHistory（z-score、velocity、prev_features 均由上游算好放
        进 feats）。
      - 对相同输入必然返回同一份结果（无随机、无内部状态、无回放）；任何跨
        截面状态（连续确认、升降级、冷却）都由调用方用
        vol_surface/alert_engine.AlertState 管理。
      - 返回结构恒定：{"level": int 0-3, "triggers": list, "rule_score": float}。

    若发现调用方在 evaluate_rules **内部** 创建 FeatureHistory、或重复向
    量重算 z-scores，一律按 bug 修。"""

    P = {**DEFAULT_PARAMS, **(params or {})}
    triggers = []
    max_level = LEVEL_NORMAL

    # R1 凸性违反程度：z-score 自适应阈值（低波动期自动收紧，避免常态化误报）
    v = feats.get("convexity_violation", 0.0)
    vz = feats.get("convexity_violation_z", 0.0)
    if vz > P["r1_warn_z"] and v > P["r1_min2"]:
        lv = LEVEL_SERIOUS if vz > P["r1_ser_z"] else LEVEL_WARN
        triggers.append(_trig("R1_convexity", lv, f"曲面凸性违反 {v:.2f}（z={vz:.1f}），蝶式非凸程度显著超历史", v, P["r1_min2"]))
        max_level = max(max_level, lv)
    elif vz > P["r1_watch_z"] and v > P["r1_min"]:
        triggers.append(_trig("R1_convexity", LEVEL_WATCH, f"曲面凸性违反 {v:.2f}（z={vz:.1f}）偏高", v, P["r1_min"]))
        max_level = max(max_level, LEVEL_WATCH)

    # R2 期限结构斜率变化率 —— **默认关闭**（`r2_enabled=False`）
    #
    # 这条规则原本的阈值 0.3 高到几乎不触发（ag/sc 训练期只有 0.03% 的截面超过），
    # 等于一直是死代码。重新标定后实测：**任何能产生有意义触发量的阈值都让前沿变差**。
    #
    # 为什么救不回来——`term_slope_roc` 本身有信号但太弱：
    #   整体 AUC 仅 0.537（验证）/ 0.532（测试）；
    #   按阈值看它单独的判别力最高 +8.9pp（验证 @0.08）/ +12.6pp（测试 @0.30），
    #   而**引擎整体的判别力已有 +17.4pp**。把一个更弱的信号源掺进更强的组合里，
    #   必然拉低平均——这是结构性的，不是阈值没调好。
    # 同召回前沿实测（2 个阈值 × 2 个数据集 × 4 个召回点 = 16 格）**全部为负**：
    #   阈值 0.08 时 −0.30~−5.53pp，阈值 0.20 时 −0.24~−1.14pp
    #   （阈值越松越有害，趋势与「R2 被支配」一致）。
    #
    # 传 `params={"r2_enabled": True}` 可复现旧行为。
    if P.get("r2_enabled", False):
        v = abs(feats.get("term_slope_roc", 0.0))
        if v > P["r2_warn"]:
            lv = LEVEL_SERIOUS if v > P["r2_ser"] else LEVEL_WARN
            triggers.append(_trig("R2_term_slope_roc", lv, f"期限结构斜率变化率 {v:.2f}（阈值>{P['r2_warn']}），期限结构急变/反转", v, P["r2_warn"]))
            max_level = max(max_level, lv)
        elif v > P["r2_watch"]:
            triggers.append(_trig("R2_term_slope_roc", LEVEL_WATCH, f"期限结构斜率变化率 {v:.2f} 偏高（阈值>{P['r2_watch']}）", v, P["r2_watch"]))
            max_level = max(max_level, LEVEL_WATCH)

    # R3 ATM IV z-score
    v = feats.get("atm_iv_z", 0.0)
    av = abs(v)
    if av > P["r3_warn"]:
        lv = LEVEL_SERIOUS if av > P["r3_ser"] else LEVEL_WARN
        triggers.append(_trig("R3_atm_iv_z", lv, f"ATM IV z-score {v:.2f}（阈值|z|>{P['r3_warn']}），IV 异常偏离历史均值", v, P["r3_warn"]))
        max_level = max(max_level, lv)
    elif av > P["r3_watch"]:
        triggers.append(_trig("R3_atm_iv_z", LEVEL_WATCH, f"ATM IV z-score {v:.2f}（阈值|z|>{P['r3_watch']}）", v, P["r3_watch"]))
        max_level = max(max_level, LEVEL_WATCH)

    # R4 Gamma/Vega 集中度 —— **默认关闭**（`r4_enabled=False`）
    #
    # 为什么关掉：这条规则在实测的每一个数据切片上都是**负贡献**。
    # 事件窗口逐规则拆解（判别力 = 该规则触发时的命中率 − 命中基础率）：
    #   sc 2026-03 −24.7pp / rb 2025-07 −20.9pp / rb 2024-02 −25.1pp
    # 全样本测试集上关掉它之后，7 个品种的判别力**无一下降**：
    #   rb +3.1→+12.0pp、au +15.6→+27.3、si +15.1→+21.1、cu +12.5→+16.8，
    #   ag/sc/lc 各 +1pp 左右；总体 +10.5→+16.4pp。
    # 且这是**前沿外推而非沿前沿滑动**——同召回下精确率 +1.9~+7.3pp（README §7.13）。
    #
    # 机理：`gamma_concentration` 是**结构性**指标（Greek 在少数行权价上的堆积度），
    # 它主要反映合约挂牌结构与流动性分布，与「波动率曲面正在恶化」关系很弱；
    # 而它的原始量纲跨品种差异极大（90 分位 ag 0.065 / rb 0.147 / cu 0.255），
    # 于是同一个阈值 0.12 在 ag 上几乎不触发、在 rb/cu 上频繁触发，
    # 把噪声预警集中倾泻到黑色系与有色上。
    #
    # 传 `params={"r4_enabled": True}` 可复现旧行为。
    if P.get("r4_enabled", False):
        gc = feats.get("gamma_concentration", 0.0)
        vc = feats.get("vega_concentration", 0.0)
        conc = max(gc, vc)
        if conc > P["r4_warn"]:
            lv = LEVEL_SERIOUS if conc > P["r4_ser"] else LEVEL_WARN
            triggers.append(_trig("R4_concentration", lv, f"Greek 截面集中度 {conc:.3f}（阈值>{P['r4_warn']}），Gamma/Vega 堆积风险点", conc, P["r4_warn"]))
            max_level = max(max_level, lv)
        elif conc > P["r4_watch"]:
            triggers.append(_trig("R4_concentration", LEVEL_WATCH, f"Greek 截面集中度 {conc:.3f} 偏高（阈值>{P['r4_watch']}）", conc, P["r4_watch"]))
            max_level = max(max_level, LEVEL_WATCH)

    # R5 无套利违反综合分：z-score 自适应
    v = feats.get("arb_score", 0.0)
    vz = feats.get("arb_score_z", 0.0)
    if vz > P["r5_warn_z"] and v > P["r5_min2"]:
        lv = LEVEL_SERIOUS if vz > P["r5_ser_z"] else LEVEL_WARN
        triggers.append(_trig("R5_arb", lv, f"无套利违反综合分 {v:.1f}（z={vz:.1f}），日历/蝶式/平价违反显著超历史", v, P["r5_min2"]))
        max_level = max(max_level, lv)
    elif vz > P["r5_watch_z"] and v > P["r5_min"]:
        triggers.append(_trig("R5_arb", LEVEL_WATCH, f"无套利违反综合分 {v:.1f}（z={vz:.1f}）偏高", v, P["r5_min"]))
        max_level = max(max_level, LEVEL_WATCH)

    # R6 流动性急速枯竭 —— **默认关闭**（`r6_enabled=False`）
    #
    # 关掉的理由不是阈值，而是**方向假设本身错了**：
    # 实测 `liquidity_z` 与「未来 120min 内有风险起点」是**正**相关——
    # 用 −liquidity_z 算 AUC 只有 0.422（验证）/ 0.452（测试），双双低于 0.5。
    # 也就是说本数据集里风险发生前流动性是**上升**的（波动来临时期权成交/持仓放大），
    # 而不是枯竭。R6 恰好在低流动性截面发预警，判别力在所有阈值上均为负
    # （验证 −4.1~−10.0pp）。
    #
    # 同召回前沿实测（关掉 R6 相对保留）：验证 +1.72~+6.03pp、测试 +1.27~+3.60pp，
    # 两个数据集五个召回点全部为正。
    #
    # `r6_invert=True` 是把方向翻过来的检验版（「异常活跃才触发」）。也不行：
    # 验证微正 +0.24~+0.59pp、测试为负 −0.81~−2.24pp。原因同 R2——
    # 反向后单独判别力最高也只有 +12.3pp，仍低于引擎整体的 +17.8pp，属被支配。
    #
    # **流动性的正确用法是「门」而不是「触发器」**：README §7.7 实测
    # `liquidity_ratio<0.55` 时**抑制**预警可给规则基线 +3.10pp——
    # 与本节发现同向，且与 R6 的做法恰好相反。
    #
    # 传 `params={"r6_enabled": True}` 可复现旧行为。
    lz = feats.get("liquidity_z", 0.0)
    if P.get("r6_enabled", True):
        if P.get("r6_invert", False):
            if lz > -P["r6_warn"]:
                lv = LEVEL_WARN if lz > -P["r6_ser"] else LEVEL_WATCH
                triggers.append(_trig("R6_liquidity", lv, f"流动性异常活跃 z-score {lz:.2f}（阈值>{-P['r6_warn']}）", lz, -P["r6_warn"]))
                max_level = max(max_level, lv)
        elif lz < P["r6_warn"]:
            lv = LEVEL_WARN if lz < P["r6_ser"] else LEVEL_WATCH
            triggers.append(_trig("R6_liquidity", lv, f"流动性退化 z-score {lz:.2f}（阈值<{P['r6_warn']}），活跃合约占比骤降", lz, P["r6_warn"]))
            max_level = max(max_level, lv)

    # R7 短期 IV 急升
    spike = feats.get("iv_spike", 0.0)
    if spike > P["r7_warn"]:
        lv = LEVEL_SERIOUS if spike > P["r7_ser"] else LEVEL_WARN
        triggers.append(_trig("R7_iv_spike", lv, f"短期 IV 急升幅度 {spike:.3f}（阈值>{P['r7_warn']}），极端波动征兆", spike, P["r7_warn"]))
        max_level = max(max_level, lv)
    elif spike > P["r7_watch"]:
        triggers.append(_trig("R7_iv_spike", LEVEL_WATCH, f"短期 IV 急升幅度 {spike:.3f} 偏高（阈值>{P['r7_watch']}）", spike, P["r7_watch"]))
        max_level = max(max_level, LEVEL_WATCH)

    # R8 IV/凸性加速度（导数类提前预警）：velocity z-score 领先于水平，在 IV 真正飙升前
    # 其加速已异动；用导数触发可在风险区间起始前更早预警，且减少事件期全程响的密集预警
    iv_vel = feats.get("atm_iv_vel_z", 0.0)
    cx_vel = feats.get("convexity_vel_z", 0.0)
    vel = max(iv_vel, cx_vel)
    src = "ATM IV" if iv_vel >= cx_vel else "凸性"
    if vel > P["r8_warn"]:
        lv = LEVEL_SERIOUS if vel > P["r8_ser"] else LEVEL_WARN
        triggers.append(_trig("R8_acceleration", lv, f"{src}加速上升（velocity z={vel:.1f}），领先于水平的提前预警", vel, P["r8_warn"]))
        max_level = max(max_level, lv)
    elif vel > P["r8_watch"]:
        triggers.append(_trig("R8_acceleration", LEVEL_WATCH, f"{src}加速上升（velocity z={vel:.1f}）偏高", vel, P["r8_watch"]))
        max_level = max(max_level, LEVEL_WATCH)

    return {"level": max_level, "triggers": triggers, "rule_score": float(len(triggers))}

# ---------------------------------------------------------------- 工作点预设
# 赛题同时要求「精确率≥50% 且 召回≥60%」。**实测这两个门槛在当前 P-R 前沿上
# 无法同时满足**——18 个事件窗口上两个工作点各达一头：
#
#   默认（RECALL_FIRST 不启用）：精确率 50.19% ✓ / 召回 50.32% ✗ / 判别力 +21.8pp
#   保召回（本预设）：           精确率 44.46% ✗ / 召回 73.89% ✓ / 判别力 +16.1pp
#
# 系数 0.55 是在 **anchor 验证集**上选的（可行域内精确率最高者），
# 事件窗口只用于评估、不参与选择——18 个窗口里有 5 个落在训练期，
# 在其上调参就是就地拟合。
#
# 有意思的是：保召回配置的**平均**精确率更低，但**逐窗口三项全达标数从 4 增到 6**。
# 平均值被 rb 等少数极差窗口拖累，而多数窗口是"召回差一点"而非"精确率差一点"。
#
# 用法：`evaluate_rules(feats, params={**RECALL_FIRST})`
#      或 `{**symbol_params[sym], **RECALL_FIRST}`
_Z_KEYS = ("r1_watch_z", "r1_warn_z", "r1_ser_z", "r3_watch", "r3_warn", "r3_ser",
           "r5_watch_z", "r5_warn_z", "r5_ser_z", "r8_watch", "r8_warn", "r8_ser")
RECALL_FIRST_MULT = 0.55
RECALL_FIRST = {k: DEFAULT_PARAMS[k] * RECALL_FIRST_MULT for k in _Z_KEYS}
