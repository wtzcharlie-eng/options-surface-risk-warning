"""eval_q1_windows — 问题1 的事件窗口评测，落盘 `data_out/q1_7sym.json`。

窗口数由 `EVENTS` 决定（现为 20 个），**不要在文案里写死**——
此前写死为 18，扩到 20 窗后这个数一路过期到落盘 JSON 里，
而 JSON 又原样嵌进 platform.html 的 payload。

为什么要有这个脚本
------------------
`q1_7sym.json` 承载问题1 的交付数字（默认工作点 / 保召回工作点 / 逐窗口达标数），
但它此前是用**临时代码**跑出来的，仓库里没有任何脚本能重新生成——全链路复现时
才暴露：该文件只被 `scripts/build_platform.py` 读取，无人写入。本脚本补回这条链路。

关于窗口边界：**这是一次重建，不是原样恢复**
---------------------------------------------
原始 18 个窗口的起止日期随临时代码一起丢失了。仓库里 committed 的窗口表
（`grid_search.py::WINDOWS`、`generate_report.py::QUANT_WINDOWS`）只有最早的 4 个，
且这 4 个的补边宽度并不统一（前补 10~13 日、后补 −1~8 日），是手工挑的，无从外推。

反推也失败了：按事件表补边、调 `warmup_risk`、加宽区间等多组假设下，
`n_slice` 能对上而 `n_risk`/`n_alert` 差约一倍，说明当时还用了别的参数。

故本脚本改为**写死一套统一、可复算的边界规则**（见 `PAD_BEFORE` / `PAD_AFTER`），
对全部窗口一视同仁。代价是数字与归档版不完全相同；收益是从此可复现。
原归档值保留在 `data_out/archive/q1_7sym_legacy.json` 供对照。

三个配置
--------
- `new`  交付配置：R2/R4/R6 关闭（README §7.13–§7.15）
- `old`  改造前：R2/R4/R6 全开，用于量化「关闭三条规则」的净效应
- `rf`   保召回：在 `new` 上叠 `RECALL_FIRST`（z 阈 ×0.55）

**不叠加 `symbol_params.json`**：现有 20 个窗口里有 10 个落在 anchor 训练期内，
叠加逐品种标定阈值会构成就地拟合。这一条与归档版的口径一致。

汇总口径
--------
`summary_*` 是**窗口等权**平均（每个事件窗口算一票），不是按截面数加权。
两者都成立，等权更保守（实测 50.19% vs 加权 50.88%），且不会被少数长窗口主导。

用法（单次约 15 分钟；支持缓存与分批）::

    python scripts/eval_q1_windows.py                 # 跑全部
    python scripts/eval_q1_windows.py --budget 300    # 跑 300 秒后存盘退出，可反复调用
    python scripts/eval_q1_windows.py --only new      # 只跑某个配置
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vol_surface.alert_rules import RECALL_FIRST
from vol_surface.quant_metrics import run_quant_evaluation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 统一补边规则：事件起始前 PAD_BEFORE 个日历日、结束后 PAD_AFTER 个日历日。
# 前补是为了让滚动分位数与 z-score 有足够历史（另有 prefill_history_days=25 兜底）；
# 后补是为了覆盖事件的余波。数值取自 committed 的 4 个窗口的中位补边宽度。
PAD_BEFORE, PAD_AFTER = 10, 5

# (品种, 标签, 事件起, 事件止)——事件日期取自 extreme_event.csv，品种按 sectors_affected 归属。
# 写死在代码里而不是运行时解析 CSV：CSV 是 GBK 且板块字段是自由文本，
# 解析规则一旦变动会**静默改变评测集**，那正是本项目要避免的失配。
EVENTS = [
    ("ag", "2024-04 贵金属", "20240412", "20240415"),
    ("ag", "2025-04 关税战", "20250403", "20250410"),
    ("ag", "2026-01 贵金属", "20260123", "20260206"),
    ("au", "2024-04 贵金属", "20240412", "20240415"),
    ("au", "2026-01 贵金属", "20260123", "20260206"),
    ("cu", "2024-05 铜逼仓", "20240517", "20240523"),
    ("cu", "2025-09 铜矿事故", "20250924", "20251015"),
    ("cu", "2026-01 有色", "20260123", "20260206"),
    ("lc", "2023-12 碳酸锂", "20231211", "20231213"),
    ("lc", "2025-08 碳酸锂", "20250806", "20250822"),
    ("lc", "2025-12 碳酸锂", "20251229", "20251229"),
    ("rb", "2024-01 螺纹单边", "20240104", "20240315"),
    ("rb", "2024-02 铁矿分化", "20240219", "20240223"),
    ("rb", "2025-07 焦煤黑色", "20250721", "20250728"),
    ("sc", "2023-06 能化", "20230530", "20230602"),
    ("sc", "2024-09 全板块", "20240924", "20241010"),
    ("sc", "2026-03 原油", "20260302", "20260331"),
    ("si", "2025-04 关税战", "20250403", "20250410"),
    # ---- 补覆盖最后两个极端事件（此前 15/17，补后 17/17）
    #
    # 「有色金属；氧化铝」事件在 2024-01，而 **ao（氧化铝）期权 2024-08 才上市**，
    # 数据上不可能覆盖；该事件同时点名「有色金属」，故用 **al（铝）** 覆盖。
    # 这一点必须写明：不是我们挑了个更容易的品种，是氧化铝期权当时还不存在。
    ("al", "2024-01 有色/氧化铝", "20240102", "20240104"),
    ("br", "2023-09 合成橡胶", "20230901", "20230911"),
]

CONFIGS = {
    "new": {},                                    # 交付配置（R2/R4/R6 已在 DEFAULT_PARAMS 关闭）
    "old": {"r2_enabled": True, "r4_enabled": True, "r6_enabled": True},
    "rf": dict(RECALL_FIRST),
}


def _window(sd: str, ed: str) -> tuple:
    import pandas as pd
    s = (pd.to_datetime(sd) - pd.Timedelta(days=PAD_BEFORE)).strftime("%Y%m%d")
    e = (pd.to_datetime(ed) + pd.Timedelta(days=PAD_AFTER)).strftime("%Y%m%d")
    return s, e


def _base_rate(feats, risk_starts, lead_min: float = 120.0) -> float:
    """命中基础率：随机截面在其后 lead_min 内存在风险起点的比例。

    与 `drl.train.hit_base_rate` 同口径——按**时间**而非截面格数判定。
    早前在 `eval_per_symbol.py` 里按「索引 +8 格」近似过，跨夜盘/日盘休市时
    8 格远超 120 分钟，基础率被高估 6~9pp、判别力随之全错。
    """
    import numpy as np
    import pandas as pd
    if not len(feats):
        return 0.0
    ts = pd.to_datetime(feats["timestamp"].astype(str), format="%Y%m%d%H%M%S")
    if not risk_starts:
        return 0.0
    rs = pd.to_datetime(pd.Series(sorted(risk_starts)), format="%Y%m%d%H%M%S")
    idx = np.searchsorted(rs.values, ts.values, side="left")
    ok = idx < len(rs)
    lead = np.full(len(ts), np.inf)
    lead[ok] = ((rs.values[idx[ok]] - ts.values[ok])
                / np.timedelta64(1, "m")).astype(float)
    return float((lead <= lead_min).mean())


def _eval_one(sym: str, sd: str, ed: str, params: dict) -> dict:
    feats, alerts, risk_starts, res = run_quant_evaluation(
        sym, sd, ed, use_state=False, use_model=False,
        params=params or None)
    base = _base_rate(feats, risk_starts)

    # ---- 新口径（命题方 2026-08 答复）：事件级精确率 + 召回不设时间上限
    # 与旧口径**并列**落盘，不替换——两个口径的数字都要能被引用，且必须标明。
    from drl.metrics_v2 import evaluate_v2, trivial_policies
    ts = [str(t) for t in alerts["timestamp"].tolist()]
    lv = alerts["level"].to_numpy()
    pos = {t: i for i, t in enumerate(ts)}
    ridx = [pos[str(r)] for r in risk_starts if str(r) in pos]
    v2 = evaluate_v2(ts, lv, ridx)

    # ---- 平凡策略对照（**每个窗口都算**）
    #
    # 新口径「召回不设上限」自带退化，故任何达标声明都必须配平凡策略对照。
    # 此前问题1 的判别力只在 3 个窗口上临时算过、未落盘，而
    # 「只在开头报一次」在单窗内只产生 1 个预警事件，精确率非 0 即 100——
    # 用 n=3 的抛硬币结果当基线，再拿它论证「达标不是蒙的」，逻辑上自我抵消。
    # 改为全部窗口都算并落盘，由 check_doc_numbers 守住。
    triv = {}
    for nm, seq in trivial_policies(len(ts)).items():
        m = evaluate_v2(ts, seq, ridx)
        triv[nm] = {"n_event": m.n_alert_event, "P": m.precision, "R": m.recall}

    return {"sym": sym, "n_slice": int(len(feats)),
            "n_risk": int(res.n_risk_windows), "n_alert": int(res.n_alerts),
            "P": float(res.precision), "R": float(res.recall),
            "lead": float(res.avg_lead_minutes), "base": base,
            "DP": float(res.precision) - base,
            "alert_rate": float(res.n_alerts / len(feats)) if len(feats) else 0.0,
            "window": [sd, ed],
            "v2": {"n_event": v2.n_alert_event, "P": v2.precision,
                   "R": v2.recall, "lead": v2.avg_lead_min,
                   "n_risk": v2.n_risk},
            "v2_trivial": triv}


def _summarize(rows: list) -> dict:
    """窗口等权平均 + 三项全达标窗口数。旧口径与新口径并列给出。"""
    n = len(rows)
    if not n:
        return {}
    avg = lambda k: sum(r[k] for r in rows) / n
    out = {"n_win": n, "P": avg("P"), "R": avg("R"), "lead": avg("lead"),
           "DP": avg("DP"), "base": avg("base"),
           "n_pass3": sum(1 for r in rows
                          if r["P"] >= .5 and r["R"] >= .6 and r["lead"] >= 30)}
    if all("v2" in r for r in rows):
        v = lambda k: sum(r["v2"][k] for r in rows) / n
        out["v2"] = {
            "P": v("P"), "R": v("R"), "lead": v("lead"),
            "n_event": sum(r["v2"]["n_event"] for r in rows),
            "n_pass3": sum(1 for r in rows
                           if r["v2"]["P"] >= .5 and r["v2"]["R"] >= .6
                           and r["v2"]["lead"] >= 30),
            "_caliber": "事件级精确率 + 召回不设时间上限（命题方 2026-08 答复）",
        }
    # 平凡策略：**全部窗口等权平均**，并据此给出判别力。
    # 单窗口的事件级精确率方差极大（「只在开头报一次」只产生 1 个事件，非 0 即 100），
    # 故必须跨全部窗口平均才有意义。
    if all("v2_trivial" in r for r in rows):
        names = list(rows[0]["v2_trivial"])
        tv = {nm: {"P": sum(r["v2_trivial"][nm]["P"] for r in rows) / n,
                   "R": sum(r["v2_trivial"][nm]["R"] for r in rows) / n,
                   "n_event": sum(r["v2_trivial"][nm]["n_event"] for r in rows)}
              for nm in names}
        best = max(tv.items(), key=lambda kv: kv[1]["P"])
        out["v2_trivial"] = tv
        out["v2_best_trivial"] = {"name": best[0], "P": best[1]["P"]}
        out["v2_discrimination_pp"] = (out["v2"]["P"] - best[1]["P"]) * 100
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data_out", "q1_7sym.json"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "data_out",
                                                    ".q1_windows_cache.json"))
    ap.add_argument("--budget", type=float, default=0.0,
                    help="跑满该秒数即存盘退出（0=不限）；可反复调用直到跑完")
    ap.add_argument("--only", choices=sorted(CONFIGS), default=None)
    a = ap.parse_args()

    cache = {}
    if os.path.exists(a.cache):
        with open(a.cache, encoding="utf-8") as f:
            cache = json.load(f)

    todo = []
    for cfg in ([a.only] if a.only else list(CONFIGS)):
        for sym, label, es, ee in EVENTS:
            sd, ed = _window(es, ee)
            key = f"{cfg}|{sym}|{label}"
            if key not in cache:
                todo.append((key, cfg, sym, label, sd, ed))

    print(f"待跑 {len(todo)} 格（缓存已有 {len(cache)} 格）"
          + (f"，预算 {a.budget:.0f}s" if a.budget else ""))
    t0 = time.time()
    for key, cfg, sym, label, sd, ed in todo:
        if a.budget and time.time() - t0 > a.budget:
            print(f"预算用尽，已完成 {len(cache)} 格，存盘退出（再次运行可续跑）")
            break
        try:
            r = _eval_one(sym, sd, ed, CONFIGS[cfg])
            r["label"] = label
            cache[key] = r
            print(f"  [{cfg:3s}] {sym} {label:14s} {sd}~{ed} "
                  f"n={r['n_slice']:4d} P={r['P']:.3f} R={r['R']:.3f} "
                  f"lead={r['lead']:.0f}m")
        except Exception as e:
            print(f"  [{cfg:3s}] {sym} {label}: 失败 {type(e).__name__}: {e}")
        with open(a.cache, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, default=float)

    done = {c: [cache[k] for k in cache if k.startswith(c + "|")] for c in CONFIGS}
    missing = {c: len(EVENTS) - len(v) for c, v in done.items() if len(v) < len(EVENTS)}
    if missing:
        print(f"\n尚未跑完：{missing}。再次运行本脚本可续跑；未跑完不落盘 q1_7sym.json。")
        return

    order = {f"{s}|{l}": i for i, (s, l, _, _) in enumerate(EVENTS)}
    srt = lambda rows: sorted(rows, key=lambda r: order[f"{r['sym']}|{r['label']}"])
    out = {
        "windows": srt(done["new"]),
        "windows_old": srt(done["old"]),
        "windows_recall_first": srt(done["rf"]),
        "summary_new": _summarize(done["new"]),
        "summary_old": _summarize(done["old"]),
        "workpoints": {
            "default": dict(_summarize(done["new"]), zmult=1.0),
            "recall_first": dict(_summarize(done["rf"]), zmult=0.55),
            "_note": "系数在 anchor 验证集上选（可行域内精确率最大），"
                     f"{len(EVENTS)} 个事件窗口只用于评估、不参与选择",
        },
        "_meta": {
            "rules": "DEFAULT_PARAMS（R2/R4/R6 关闭）",
            "warmup_risk": True, "use_model": False,
            "pad_before_days": PAD_BEFORE, "pad_after_days": PAD_AFTER,
            "summary_caliber": "窗口等权平均（非按截面数加权）",
            # 两个数都现算：此前「18 窗口」与「5 个」都写死，扩到 20 窗后
            # 双双过期，且这段 note 会原样嵌进 platform.html 的 payload。
            "note": (f"未叠加 symbol_params.json：{len(EVENTS)} 窗口中有 "
                     f"{sum(1 for _, _lb, _, _ in EVENTS if str(_lb)[:7] < '2025-01')}"
                     f" 个落在 anchor 训练期内，叠加会构成就地拟合"),
            "reconstructed": "窗口边界为统一规则重建，非原始临时代码的边界；"
                             "原归档值见 data_out/archive/q1_7sym_legacy.json",
        },
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=float)

    s, o = out["summary_new"], out["summary_old"]
    print(f"\n改造前(R2/R4/R6 全开) 精确率 {o['P']:.2%} 判别力 {o['DP']:+.2%} "
          f"达标 {o['n_pass3']}/{len(EVENTS)}")
    print(f"交付配置              精确率 {s['P']:.2%} 判别力 {s['DP']:+.2%} "
          f"达标 {s['n_pass3']}/{len(EVENTS)}")
    rf = out["workpoints"]["recall_first"]
    print(f"保召回配置            精确率 {rf['P']:.2%} 召回 {rf['R']:.2%} "
          f"达标 {rf['n_pass3']}/{len(EVENTS)}")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
