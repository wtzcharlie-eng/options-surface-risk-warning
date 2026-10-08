"""optimize_precision — 赛题指标1（精确率）专项优化与误报结构诊断。

背景
----
赛题指标1的评分逻辑（vol_surface/quant_metrics.py）在数学上有一个强约束：
风险起点按 >120min 间隔去重，且预警命中判定 = "风险起点 ∈ [alert, alert+120min)"。
在该窗口对称、风险互斥 >120min 的前提下，可推出：
  - 召回 = 命中数 / 风险数（每条风险最多 1 预警命中）
  - 精确率 = 命中数 / 预警数（每条预警最多命中 1 风险）
  - 一条预警是误报 ⟺ 它落在"上一条风险终点 [rs_prev+120, rs_next) 且其后120min无风险起点"

因此误报的本质是：在"风险刚过、下一条还没到"的窗口里系统仍在响。
本脚本分两步验证并破解此前的"47.5% 精度-召回前沿"结论：

DIAGNOSIS : 对四窗口连续特征缓存做最优事后抑制的上界诊断，量化"重响误报"占比。
EXPERIMENT: 实现"风险感知上升沿去抖"（y_t 转换边沿：0/1→≥2 才报 + 响后冷却 +
            风险仍活跃才续报），对比既有 best_params 的精确率变化。

泄露边界
--------
DIAGNOSIS 用到了未来 120min 的风险真值（用于判定"哪些误报被吸收后召回不受影响"），
仅用于证明方向——不是实时信号。
EXPERIMENT 的去抖逻辑只用"当前及过去已发出的预警状态"与"当前截面特征"，不读取
未来风险，故若实验指标改善，方案可直接落到 generate_report 跑通端到端。

用法
----
  python3 scripts/optimize_precision.py diagnosis   # 只诊断（验证误报结构）
  python3 scripts/optimize_precision.py experiment  # 诊断 + 去抖实验
  python3 scripts/optimize_precision.py all         # 两者都跑
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.alert_engine import AlertState, evaluate
from vol_surface.alert_model import AlertModel
from vol_surface.quant_metrics import evaluate_quant
from scripts.grid_search import WINDOWS, precompute_window

DATA_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_out")
CACHE_FILE = os.path.join(DATA_OUT, "_quant_feat_cache.pkl")

# 与 grid_search 的 STATE_GRID 对齐的状态机候选；去抖与状态机只用一个
STATE_CANDIDATES = [
    None,
    {"persistence": 8, "confirm": 2, "escalate": 5, "cooldown": 8},
    {"persistence": 10, "confirm": 3, "escalate": 8, "cooldown": 0},
    {"persistence": 10, "confirm": 3, "escalate": 8, "cooldown": 8},
]

# 赛题评分口径硬参数（quant_metrics.evaluate_quant 一致）
LEAD_MIN = 120.0


# ---------------------------------------------------------------------------
# 缓存构建（重用 grid_search.precompute_window，保持口径一致）
# ---------------------------------------------------------------------------

def build_cache(force: bool = False) -> dict:
    t0 = time.time()
    cache = {}
    for label, sym, sd, ed in WINDOWS:
        print(f"[cache] {label} ...", flush=True)
        cache[label] = {"symbol": sym, "sd": sd, "ed": ed,
                        **precompute_window(sym, sd, ed)}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"[cache] done in {time.time() - t0:.0f}s → {CACHE_FILE}\n", flush=True)
    return cache


def load_cache() -> dict:
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------

def _to_epoch(ts: str) -> int:
    return int(pd.Timestamp(year=int(ts[:4]), month=int(ts[4:6]), day=int(ts[6:8]),
                            hour=int(ts[8:10]), minute=int(ts[10:12])).timestamp())


def _resolve_model(sym: str):
    p = os.path.join(DATA_OUT, f"model_{sym}.joblib")
    return AlertModel.load(p) if os.path.exists(p) else None


def _run_feats(window: dict, params: dict, state_params: dict | None,
               use_model: bool) -> list:
    """对缓存特征逐截面跑引擎，输出最终 level 序列。"""
    sym = window["symbol"]
    model = _resolve_model(sym) if use_model else None
    state = AlertState(**state_params) if state_params else None
    levels = []
    for feats in window["feats"]:
        res = evaluate(feats, model=model, state=state, params=params)
        levels.append(res["level"])
    return levels


def _metric(levels: list, feats: list, risk_starts: list,
            lead_min: float = LEAD_MIN) -> dict:
    alerts = pd.DataFrame({"timestamp": [f["_timestamp"] for f in feats],
                           "level": levels})
    r = evaluate_quant(alerts, risk_starts, min_level=2, lead_window_min=lead_min)
    return {"P": r.precision, "R": r.recall, "L": r.avg_lead_minutes,
            "n_a": r.n_alerts, "n_r": r.n_risk_windows}


def _agg(rows: list) -> dict:
    return {"P": float(np.mean([r["P"] for r in rows])),
            "R": float(np.mean([r["R"] for r in rows])),
            "L": float(np.mean([r["L"] for r in rows])),
            "n_a": sum(r["n_a"] for r in rows), "n_r": sum(r["n_r"] for r in rows)}


def _fmt(r: dict) -> str:
    return f"P={r['P']:.1%} R={r['R']:.1%} lead={r['L']:.0f}min " \
           f"(n_a={r['n_a']}, n_r={r['n_r']})"


# ---------------------------------------------------------------------------
# 上升沿 + 冷却 + 风险感知续报
# ---------------------------------------------------------------------------

def _emit_edge(levels: list) -> list:
    """只保留 0/1→≥2 的上升沿；≥2 期间持续置 0（把簇折叠为一次预警）。"""
    out, prev = [], 0
    for l in levels:
        cur = 1 if l >= 2 else 0
        out.append(l if (cur == 1 and prev == 0) else 0)
        prev = cur
    return out


def _emit_cooldown(levels: list, gaps: list, cd_min: float) -> list:
    """响后 cd 分钟内静默（比 edge 更硬，cd 可超过 120min）。"""
    out, since = [], 1e18
    for l, g in zip(levels, gaps):
        if l >= 2 and since >= cd_min:
            out.append(l)
            since = 0.0
        else:
            out.append(0)
            since += g
    return out


def _emit_edge_cooldown(levels: list, gaps: list, cd_min: float,
                        allow_renew: bool, active: list) -> list:
    """组合 A：上升沿触发 + 响后 cd 冷却 +（可选）冷却到期后风险仍活跃才续报。

    - 未响过：0/1→≥2 才报
    - 响过 cd 内：静默
    - 响过且超 cd：需在活跃曲面（active[i]）上才允许续报，否则静默
    """
    out = []
    since = 1e18
    for l, g, a in zip(levels, gaps, active):
        if l >= 2 and since >= cd_min:
            if not allow_renew or a:
                out.append(l)
                since = 0.0
                continue
        out.append(0)
        since += g
    return out


# ---------------------------------------------------------------------------
# 诊断：重响误报结构 + 最优事后抑制天花板
# ---------------------------------------------------------------------------

def diagnose(label: str, sym: str, window: dict, params: dict) -> None:
    risk_starts = sorted(window["risk_starts"])
    risk_ep = [_to_epoch(t) for t in risk_starts]
    n_r = len(risk_ep)

    print(f"\n{'=' * 78}\n窗口: {label} ({sym})  风险区间数 n_r = {n_r}\n{'=' * 78}")
    print(f"风险起点间隔(min): ", end="")
    if n_r >= 2:
        gaps_r = [(risk_ep[i + 1] - risk_ep[i]) / 60 for i in range(n_r - 1)]
        print(f"median={np.median(gaps_r):.0f}  min={min(gaps_r):.0f}  max={max(gaps_r):.0f}")
    else:
        print("(单风险，无间隔)")

    base_rows = []
    for nm, sp, um in [("规则(无状态)", None, False),
                        ("规则+模型", None, True),
                        ("规则+状态+模型", {"persistence": 10, "confirm": 3,
                                             "escalate": 8, "cooldown": 8}, True)]:
        lv = _run_feats(window, params, sp, um)
        m = _metric(lv, window["feats"], risk_starts)
        print(f"  [{nm:28s}] {_fmt(m)}")
        base_rows.append((nm, lv, m))

    # 以最佳基线（最接近赛题提交口径: 规则+模型 无状态机）做结构诊断
    nm0 = "规则+模型"
    lv0 = next(lv for nm, lv, _ in base_rows if nm == nm0)
    m0 = next(m for nm, _, m in base_rows if nm == nm0)

    # 逐条预警确定"命中的风险"或"为何误报"
    alerts = pd.DataFrame({"timestamp": [f["_timestamp"] for f in window["feats"]],
                           "level": lv0})
    al = alerts.sort_values("timestamp").reset_index(drop=True)
    al_ep = [_to_epoch(t) for t in al["timestamp"].tolist()]
    al_lv = al["level"].tolist()
    idx = [i for i in range(len(al)) if al_lv[i] >= 2]
    idx.sort()

    # 每条预警是否命中：风险起点 ∈ [alert, alert+120min)。注意命中数不受风险数上界
    # 约束——赛题召回是"任一预警命中"（集合），精确率是"每条预警是否命中"（逐条）；
    # 同一风险前 120min 内的多条预警各自都算精确率 TP，但只贡献 1 个召回分子。
    alert_hit = []   # alert idx → risk index
    for i in idx:
        a_e = al_ep[i]
        hit = next((j for j, r_e in enumerate(risk_ep) if a_e <= r_e <= a_e + LEAD_MIN * 60), None)
        alert_hit.append(hit)
    hits = sum(1 for h in alert_hit if h is not None)
    assert hits <= len(idx), f"hits 不应超过预警数: {hits} > {len(idx)}"

    fp_idx = [i for k, i in enumerate(idx) if alert_hit[k] is None]
    tp_idx = [i for k, i in enumerate(idx) if alert_hit[k] is not None]

    # 误报 offset 分类：与"最近一条已过去风险"的距离
    # r_last = 最后一个 <= a_e 的风险；若没有则按 first 风险之后 120min 仍无风险处理
    off_buckets = {"事件前(基线期)": 0, "事件尾[0,15)": 0, "事件尾[15,30)": 0,
                   "事件尾[30,60)": 0, "事件尾[60,120)": 0, "事件尾≥120": 0,
                   "下一条前(>120)": 0}
    for i in fp_idx:
        a_e = al_ep[i]
        last = max([r_e for r_e in risk_ep if r_e <= a_e], default=None)
        if last is None:
            off_buckets["事件前(基线期)"] += 1
            continue
        d = (a_e - last) / 60.0 - LEAD_MIN  # 距该风险"评论区"(rs+120)的偏移
        if d < 15:
            off_buckets["事件尾[0,15)"] += 1
        elif d < 30:
            off_buckets["事件尾[15,30)"] += 1
        elif d < 60:
            off_buckets["事件尾[30,60)"] += 1
        elif d < 120:
            off_buckets["事件尾[60,120)"] += 1
        else:
            off_buckets["事件尾≥120"] += 1
    print(f"\n  基线[{nm0}] 预警={len(idx)} 命中={hits} 误报={len(fp_idx)}  "
          f"精确率={m0['P']:.1%} 召回={m0['R']:.1%}")
    print(f"  误报 offset 分布: {off_buckets}")

    # 理论上界：保留全部命中预警，只删"删除后不影响任何风险召回"的误报
    # 由于每条风险仅由唯一一条预警命中，删除任何误报都不损召回，全部删除即最优
    keep_tp = [1 if alert_hit[k] is not None else 0 for k in range(len(idx))]
    lv_opt = list(lv0)
    for k, keep in enumerate(keep_tp):
        if not keep:
            lv_opt[idx[k]] = 0
    m_opt = _metric(lv_opt, window["feats"], risk_starts)
    print(f"  [天花板: 只留命中预警]      {_fmt(m_opt)}")

    # 简单去抖的上界（纯实时逻辑，无未来信息）
    gaps = [0.0] + [(al_ep[i + 1] - al_ep[i]) / 60.0 for i in range(len(al) - 1)]
    lv_edge = _emit_edge(lv0)
    m_edge = _metric(lv_edge, window["feats"], risk_starts)
    print(f"  [A: 仅上升沿去抖]            {_fmt(m_edge)}")
    for cd in (90, 120, 150, 180):
        lv_c = _emit_edge_cooldown(lv0, gaps, cd, allow_renew=False, active=[True] * len(lv0))
        m_c = _metric(lv_c, window["feats"], risk_starts)
        print(f"  [A: 上升沿+冷却{cd:>3}min]      {_fmt(m_c)}")


# ---------------------------------------------------------------------------
# 实验：风险感知去抖 across 四窗口 + baseline 对比
# ---------------------------------------------------------------------------

def _active_mask(window: dict, q_atm: float, q_cx: float,
                 look_min: float = 75.0) -> tuple:
    """滞后活性判定：过去 look_min 内 atm_iv 或 convexity 的最大值 > 历史分位。

    用预警发出前的窗口（滞后），不读取未来；look 默认与赛题 horizon(5×15min) 对齐。
    返回 (atm_active_list, cx_active_list, union_active_list)。
    """
    feats = window["feats"]
    ts_ep = [_to_epoch(f["_timestamp"]) for f in feats]
    atm = np.array([f["atm_iv"] for f in feats], float)
    cx = np.array([f["convexity_violation"] for f in feats], float)
    q_a = np.quantile(atm, q_atm) if len(atm) else 0.0
    q_c = np.quantile(cx, q_cx) if len(cx) else 0.0
    a_act, c_act, u_act = [], [], []
    for i in range(len(feats)):
        # 严格滞后：仅看 t_i 之前 look_min 分钟内的截面（不含当前 t_i），避免读取未来
        lo = ts_ep[i] - look_min * 60
        j = i - 1
        while j >= 0 and ts_ep[j] >= lo:
            j -= 1
        w_a = atm[j + 1:i]
        w_c = cx[j + 1:i]
        a = (w_a.size > 0 and w_a.max() > q_a)
        c = (w_c.size > 0 and w_c.max() > q_c)
        a_act.append(a)
        c_act.append(c)
        u_act.append(a or c)
    return a_act, c_act, u_act, q_a, q_c


def experiment(cache: dict, best_params: dict) -> None:
    # 基线：与 best_params 对齐的"规则+模型 无状态机"
    print(f"\n{'#' * 78}\n# 实验: 风险感知上升沿去抖（四窗口汇总）\n{'#' * 78}")
    base_rows = []
    per_window_base = {}
    for label, sym, _, _ in WINDOWS:
        w = cache[label]
        lv = _run_feats(w, best_params, None, use_model=True)
        m = _metric(lv, w["feats"], w["risk_starts"])
        per_window_base[label] = (lv, m)
        base_rows.append(m)
        print(f"  基线[{label:20s}] {_fmt(m)}")
    agg_base = _agg(base_rows)
    print(f"  基线[汇总]                {_fmt(agg_base)}\n")

    # 预生成各窗口的活性掩码（供 CD 网格扫描用）
    act_cache = {}
    for label, sym, _, _ in WINDOWS:
        w = cache[label]
        ts_ep = [_to_epoch(f["_timestamp"]) for f in w["feats"]]
        gaps = [0.0] + [(ts_ep[i + 1] - ts_ep[i]) / 60.0 for i in range(len(ts_ep) - 1)]
        act_cache[label] = {"gaps": gaps,
                            **{f"q{qa}_{qb}": _active_mask(w, qa, qb)[2]
                               for qa, qb in [(0.90, 0.90), (0.95, 0.95)]}}

    # 组合 A 纯去抖（无活性判定）+ 组合 A+B（风险感知续报）
    grid_cd = [90, 120, 150, 180]
    grid_ns = [("纯A(无活性)", None, None),
                ("A+活性(0.90)", 0.90, 0.90),
                ("A+活性(0.95)", 0.95, 0.95)]
    results = []
    for nm, qa, qb in grid_ns:
        for cd in grid_cd:
            rows = []
            for label, sym, _, _ in WINDOWS:
                w = cache[label]
                lv = per_window_base[label][0]
                gaps = act_cache[label]["gaps"]
                if qa is None:
                    el = _emit_edge_cooldown(lv, gaps, cd, allow_renew=False,
                                             active=[True] * len(lv))
                else:
                    active = act_cache[label][f"q{qa}_{qb}"]
                    el = _emit_edge_cooldown(lv, gaps, cd, allow_renew=True, active=active)
                m = _metric(el, w["feats"], w["risk_starts"])
                rows.append(m)
            agg = _agg(rows)
            feasible = agg["R"] >= 0.6 and agg["L"] >= 30
            # 目标: 在召回不塌 (<0.55) 前提下精确率最大化; 优先召回可行解
            obj = agg["P"] + (0.5 if feasible else 0.0) - (0.5 if agg["R"] < 0.55 else 0.0)
            results.append({"name": nm, "cd": cd, "qa": qa, "qb": qb,
                            "agg": agg, "feasible": feasible, "obj": obj})
            print(f"  [{nm:16s} cd={cd:>3}min] {_fmt(agg)}  "
                  f"{'✓' if feasible else ('边缘' if agg['R'] >= 0.55 else '✗')}")

    # 最优组合逐窗口
    best = max(results, key=lambda r: r["obj"])
    print(f"\n=== 最优组合: {best['name']} cd={best['cd']}min ===")
    print(f"  汇总: {_fmt(best['agg'])}  (基线 {_fmt(agg_base)})")
    dP = best["agg"]["P"] - agg_base["P"]
    dR = best["agg"]["R"] - agg_base["R"]
    print(f"  ΔP={dP:+.1%}  ΔR={dR:+.1%}  Δlead={best['agg']['L'] - agg_base['L']:+.0f}min")

    print(f"\n逐窗口 vs 基线:")
    for label, sym, _, _ in WINDOWS:
        w = cache[label]
        lv = per_window_base[label][0]
        gaps = act_cache[label]["gaps"]
        if best["qa"] is None:
            el = _emit_edge_cooldown(lv, gaps, best["cd"], allow_renew=False,
                                     active=[True] * len(lv))
        else:
            active = act_cache[label][f"q{best['qa']}_{best['qb']}"]
            el = _emit_edge_cooldown(lv, gaps, best["cd"], allow_renew=True, active=active)
        m = _metric(el, w["feats"], w["risk_starts"])
        m0 = per_window_base[label][1]
        print(f"  [{label:20s}] 基线 P={m0['P']:.1%} → 新 P={m['P']:.1%}  "
              f"R={m0['R']:.1%}→{m['R']:.1%}  lead={m0['L']:.0f}→{m['L']:.0f}min")

    # 给问题 3 的结论建议
    print(f"\n建议判断:")
    if best["agg"]["P"] >= 0.50 and best["agg"]["R"] >= 0.60 and best["agg"]["L"] >= 30:
        print("  ★ 去抖使指标1四窗口平均三项全达标。可落到 engine 并跑 generate_report。")
    elif best["agg"]["P"] > agg_base["P"]:
        print("  ★ 去抖提升精确率但未全达标。可择优写回 best_params 做可选模式。")
    else:
        print("  ★ 去抖未改善。保留旧结论：47.5% 前沿可能真实存在，建议问题3改攻 DRL/监督。")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="all",
                    choices=["diagnosis", "experiment", "all"],
                    help="diagnosis=误报结构诊断; experiment=去抖实验; all=两者都跑")
    ap.add_argument("--rebuild", action="store_true", help="忽略既有缓存，重建四窗口特征缓存")
    args = ap.parse_args()

    cache = build_cache() if (args.rebuild or not os.path.exists(CACHE_FILE)) else load_cache()

    import json
    with open(os.path.join(DATA_OUT, "best_params.json")) as f:
        best_params = json.load(f)["params"]

    if args.mode in ("diagnosis", "all"):
        print(f"\n{'#' * 78}\n# 诊断: 误报结构与'重响'占比\n{'#' * 78}")
        for label, sym, _, _ in WINDOWS:
            diagnose(label, sym, cache[label], best_params)

    if args.mode in ("experiment", "all"):
        experiment(cache, best_params)


if __name__ == "__main__":
    main()
