"""kg.calibrate — 在真实截面上标定图谱边权。

要解决的问题
------------
「异常 A 会传导到后果 C」这句话，如果只靠人写，那它就只是一句话。本模块把每条
候选边变成一个可检验的统计命题，并给出三样东西：

1. **提升度 lift = P(C | A) / P(C)**——在异常发生的条件下，后果的发生概率比基础率
   高多少倍。lift≈1 表示这条边没有信息量。
2. **分块置换检验 p 值**——金融时序有强自相关，朴素的卡方/二项检验会把 p 值压得
   极小、几乎条条边都「显著」。这里改用**整日为块的循环平移置换**：只打乱 A 与 C 的
   时间对齐关系，完整保留 A 自身的自相关结构。p 值 = 置换后 lift ≥ 实测 lift 的比例。
3. **训练集标定 / 测试集复现**——边在训练期（2023-01..2024-12）上标定，在测试期
   （2025-07..2026-04）上独立复算。只在训练集上显著、测试集塌掉的边会被标为
   `replicated=false`，报告里如实列出。

设计上的取舍与已知局限
----------------------
- lift 衡量的是**条件共现**，不是因果。曲面特征之间本身高度耦合（例如 R3 的
  `atm_iv_z` 与 CONSEQ-VolSpike 的判定量就是同一个特征的不同时间窗），这类边的
  高 lift 有相当部分来自特征自相关而非"传导"。因此：
    * 凡是 ANOM 与 CONSEQ 共用同一底层特征的边，一律打上 `self_feature=true` 标记，
      在解释与报告中显式提示"该关联部分源自同一特征的时间自相关"；
    * 置换检验只能排除"纯属偶然"，排除不了"同源"。这一点不做隐瞒。
- 因此图谱的定位是**可解释的风险提示**，不是因果推断结论。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from vol_surface.alert_rules import evaluate_rules

# 分块置换检验的重排次数。p 值按 (ge+1)/(n_perm+1) 计，故**分辨率下限**是
# 1/(N_PERM_DEFAULT+1)。解释文本需要引用这个数来说明「p=0.0050 是地板值而非测得值」，
# 所以写成模块常量而不是内联字面量。
N_PERM_DEFAULT = 200

# ---------------------------------------------------------------- 异常检测口径
# 知识图谱**显式启用全部 8 类异常**，包括在问题1 里已被默认关闭的 R2/R4/R6。
#
# 为什么不跟随规则引擎的开关
# --------------------------
# 两者目标不同：
#   - 规则引擎（问题1）的任务是**预测**未来 120min 内的风险区间。R2/R4/R6 在这个
#     任务上判别力为负或被支配，故默认关闭（README §7.13–§7.15）。
#   - 知识图谱（问题2）的任务是**描述**「当前发生了什么异常、历史上它之后常发生什么」。
#     这是条件概率统计，不要求该异常本身是好的预警触发器。
#
# 这三类异常的边是实测显著且测试期可复现的，删掉会让图谱丢掉 15/40 条边、
# 8 类异常里的 3 类：
#   ANOM-R6_liq → CONSEQ-LiquidityDrop  lift 2.80（p=0.005，测试期复现）
#   ANOM-R4_conc → CONSEQ-ArbPersist    lift 1.95（p=0.005，测试期复现）
#   ANOM-R2_term → CONSEQ-ConvexityBreak lift 1.37（p=0.030，测试期复现）
#
# **写成显式常量而不是依赖默认值**：否则问题1 那边一改开关，图谱会在无人察觉的情况下
# 少掉三类节点，而所有落盘的标定结果仍是旧的——这正是本项目反复踩过的那类静默失配。
KG_RULE_PARAMS = {"r2_enabled": True, "r4_enabled": True, "r6_enabled": True}


from .schema import ANOMS, CONSEQS, HORIZON

# 异常判定：规则触发到 watch(1) 及以上即视为该异常出现
ANOM_MIN_LEVEL = 1


# ---------------------------------------------------------------- 事件序列

def anom_matrix(X: np.ndarray, feature_names: list,
                min_level: int = ANOM_MIN_LEVEL) -> np.ndarray:
    """(n, n_anom) 布尔矩阵：每个截面上各异常是否出现。

    直接调用 vol_surface.alert_rules.evaluate_rules 本体，保证与问题1 的判定一致。
    """
    n = len(X)
    out = np.zeros((n, len(ANOMS)), dtype=bool)
    rule_idx = {a.rule: j for j, a in enumerate(ANOMS)}
    for i in range(n):
        feats = {k: float(X[i, j]) for j, k in enumerate(feature_names)}
        for t in evaluate_rules(feats, params=KG_RULE_PARAMS)["triggers"]:
            if t["level"] >= min_level:
                j = rule_idx.get(t["rule"])
                if j is not None:
                    out[i, j] = True
    return out


def conseq_matrix(X: np.ndarray, feature_names: list,
                  horizon: int = HORIZON) -> np.ndarray:
    """(n, n_conseq) 布尔矩阵：每个截面**未来 horizon 个截面内**各后果是否发生。

    末尾 horizon 行因窗口不完整而为 False，调用方需用 valid_mask 排除。
    """
    idx = {k: j for j, k in enumerate(feature_names)}
    n = len(X)
    out = np.zeros((n, len(CONSEQS)), dtype=bool)
    for j, c in enumerate(CONSEQS):
        v = X[:, idx[c.feature]].astype(float)
        if c.transform == "abs_dev":
            v = np.abs(v - 0.5)
        s = pd.Series(v)
        roll = {"max": s.rolling(horizon).max(),
                "min": s.rolling(horizon).min(),
                "mean": s.rolling(horizon).mean()}[c.agg]
        fwd = roll.shift(-horizon).to_numpy()      # 位置 i 处 = 窗口 [i+1, i+horizon]
        hit = (fwd > c.thr) if c.op == ">" else (fwd < c.thr)
        out[:, j] = np.nan_to_num(hit, nan=False).astype(bool)
    return out


def valid_mask(n: int, horizon: int = HORIZON) -> np.ndarray:
    m = np.ones(n, dtype=bool)
    m[max(0, n - horizon):] = False
    return m


# ---------------------------------------------------------------- 统计量

def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 得分区间——小样本下比正态近似稳健。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class EdgeStat:
    anom: str
    conseq: str
    n: int                # 有效样本数
    n_anom: int           # 异常出现次数
    p_conseq: float       # 后果基础率 P(C)
    p_cond: float         # 条件概率 P(C|A)
    lift: float
    ci_low: float         # P(C|A) 的 95% Wilson 下界
    ci_high: float
    lift_ci_low: float    # lift 的保守下界 = ci_low / P(C)
    p_value: float        # 分块置换检验
    self_feature: bool    # ANOM 与 CONSEQ 是否共用同一底层特征

    def as_dict(self) -> dict:
        return asdict(self)


def _counts(A: np.ndarray, C: np.ndarray, mask: np.ndarray) -> tuple:
    a, c = A[mask], C[mask]
    return int(a.sum()), int((a & c).sum()), int(c.sum()), int(mask.sum())


def calibrate_pair(episodes, ai: int, ci: int, cache: dict,
                   n_perm: int = N_PERM_DEFAULT, seed: int = 0) -> EdgeStat:
    """标定一条 (ANOM, CONSEQ) 边。cache 里存好各 episode 的 A/C/mask 矩阵。"""
    rng = np.random.default_rng(seed)
    n_a = n_ac = n_c = n_tot = 0
    for e in episodes:
        A, C, m = cache[id(e)]
        na, nac, nc, nt = _counts(A[:, ai], C[:, ci], m)
        n_a += na; n_ac += nac; n_c += nc; n_tot += nt

    p_c = n_c / n_tot if n_tot else 0.0
    p_ca = n_ac / n_a if n_a else 0.0
    lift = p_ca / p_c if p_c > 0 else float("nan")
    lo, hi = _wilson(n_ac, n_a)
    lift_lo = lo / p_c if p_c > 0 else float("nan")

    # 分块置换：按「整日」为块对 A 做循环平移，保留 A 的自相关、破坏与 C 的对齐
    ge = 0
    for _ in range(n_perm):
        pn_a = pn_ac = 0
        for e in episodes:
            A, C, m = cache[id(e)]
            a_col = A[:, ai]
            day_len = max(1, int(len(a_col) / max(1, e.n_days)))
            shift = int(rng.integers(1, max(2, e.n_days))) * day_len
            pa = np.roll(a_col, shift)
            pn_a += int(pa[m].sum())
            pn_ac += int((pa[m] & C[m, ci]).sum())
        pl = (pn_ac / pn_a / p_c) if (pn_a and p_c > 0) else 0.0
        ge += int(pl >= lift)
    p_value = (ge + 1) / (n_perm + 1)

    a_spec, c_spec = ANOMS[ai], CONSEQS[ci]
    same = bool(a_spec.feature and (
        a_spec.feature == c_spec.feature or a_spec.z_feature == c_spec.feature))

    return EdgeStat(a_spec.id, c_spec.id, n_tot, n_a, p_c, p_ca, lift,
                    lo, hi, lift_lo, p_value, same)


# ---------------------------------------------------------------- 主流程

def build_cache(episodes, feature_names: list) -> dict:
    cache = {}
    for e in episodes:
        A = anom_matrix(e.X, feature_names)
        C = conseq_matrix(e.X, feature_names)
        m = valid_mask(len(e))
        # 每幕的交易日数，供分块置换用
        e.n_days = int(pd.to_datetime(pd.Series(e.ts), format="%Y%m%d%H%M%S")
                       .dt.date.nunique())
        cache[id(e)] = (A, C, m)
    return cache


def calibrate(train_eps, test_eps, feature_names: list,
              n_perm: int = 200, seed: int = 0,
              min_support: int = 200, max_p: float = 0.05,
              min_lift: float = 1.10, verbose: bool = True) -> dict:
    """标定全部 (ANOM, CONSEQ) 组合，返回 {edges: [...], meta: {...}}。

    保留标准（三条同时满足才算 strong）：
      - 支撑度 n_anom ≥ min_support
      - 分块置换 p ≤ max_p
      - lift 的 95% 保守下界 > min_lift
    """
    tr_cache = build_cache(train_eps, feature_names)
    te_cache = build_cache(test_eps, feature_names) if test_eps else {}

    edges = []
    for ai, a in enumerate(ANOMS):
        for ci, c in enumerate(CONSEQS):
            st = calibrate_pair(train_eps, ai, ci, tr_cache, n_perm, seed)
            rec = st.as_dict()
            rec["strong"] = bool(st.n_anom >= min_support and st.p_value <= max_p
                                 and np.isfinite(st.lift_ci_low)
                                 and st.lift_ci_low > min_lift)
            if te_cache:
                ts = calibrate_pair(test_eps, ai, ci, te_cache, n_perm, seed + 1)
                rec["test"] = {"n_anom": ts.n_anom, "p_cond": ts.p_cond,
                               "p_conseq": ts.p_conseq, "lift": ts.lift,
                               "lift_ci_low": ts.lift_ci_low, "p_value": ts.p_value}
                # 复现判据分两档，**都要落盘**。
                #
                # 早前只有一个 `replicated`，判据是「n≥50 且 lift 下界 > 1.0」——
                # 不含 p 值检验，比训练期的入选门槛（支撑度 + p≤0.05 + 下界>1.1）宽得多，
                # 而字段名却叫 "replicated"，读起来就是「统计上复现了」。
                # 后果：测试期 p=0.1542（不显著）的边被标成「测试期可复现 ✓」，
                # 报告还据此宣称「21 条全部测试期可复现」。这是**对外的假陈述**，
                # 由第 5 位独立评审点名。
                #
                # 现改为：`replicated` = 与训练期**同标准**（严格）；
                #         `replicated_loose` = 原宽松标准，保留以说明差异。
                lo = bool(ts.n_anom >= 50 and np.isfinite(ts.lift_ci_low)
                          and ts.lift_ci_low > 1.0)
                rec["replicated_loose"] = lo
                rec["replicated"] = bool(lo and ts.p_value <= max_p
                                         and ts.lift_ci_low > min_lift)
            edges.append(rec)
            if verbose:
                flag = "strong" if rec["strong"] else "weak  "
                rep = ("" if "replicated" not in rec
                       else ("  复现✓" if rec["replicated"] else "  复现✗"))
                print(f"  [{flag}] {a.id:22s} → {c.id:24s} "
                      f"lift={st.lift:5.2f} (下界{st.lift_ci_low:5.2f}) "
                      f"P(C|A)={st.p_cond:.3f} 基础率={st.p_conseq:.3f} "
                      f"n_A={st.n_anom:6d} p={st.p_value:.4f}"
                      f"{' 同源' if st.self_feature else '    '}{rep}", flush=True)

    n_strong = sum(1 for e in edges if e["strong"])
    # **只对入选的 strong 边计数。** 早前对全部候选计数，于是控制台会打印
    # 「候选 40 条 → 通过 21 条（测试期可复现 24 条）」——21 条里复现 24 条，
    # 字面不可能。两个数来自两个总体（24/40 候选 vs 21/40 入选），并排放就是误导。
    # 图谱与报告只承载 strong 边，故复现计数也应以 strong 为分母。
    n_rep = sum(1 for e in edges if e["strong"] and e.get("replicated"))
    n_rep_loose = sum(1 for e in edges
                      if e["strong"] and e.get("replicated_loose")
                      and not e.get("replicated"))
    return {
        "edges": edges,
        "meta": {
            "horizon": HORIZON, "anom_min_level": ANOM_MIN_LEVEL,
            "n_perm": n_perm, "seed": seed,
            "criteria": {"min_support": min_support, "max_p": max_p,
                         "min_lift_ci_low": min_lift},
            "n_candidate": len(edges), "n_strong": n_strong, "n_replicated": n_rep,
            "n_replicated_loose": n_rep_loose,
            "n_train_episodes": len(train_eps), "n_test_episodes": len(test_eps or []),
            "n_train_slices": int(sum(len(e) for e in train_eps)),
        },
    }


def session_profile(episodes, feature_names: list, rule: str = "R6_liquidity",
                    min_level: int = ANOM_MIN_LEVEL) -> dict:
    """交易时段画像——为「跨品种共现是日内季节性假象」这一判断留下可复算的证据。

    返回每个品种：该异常触发时刻的小时分布、有数据覆盖的小时集合，以及品种两两之间
    交易小时集合的 Jaccard 相似度。报告里关于「夜盘集中度」「si 与其他品种时段不重合」
    的论断全部引用本函数的输出，而不是写死在文案里。
    """
    j = next((k for k, a in enumerate(ANOMS) if a.rule == rule), None)
    if j is None:
        return {}
    hours_fired: dict = {}
    hours_seen: dict = {}
    for e in episodes:
        A = anom_matrix(e.X, feature_names, min_level)
        h = (pd.to_datetime(pd.Series(e.ts), format="%Y%m%d%H%M%S")
             .dt.hour.to_numpy())
        hours_seen.setdefault(e.symbol, set()).update(h.tolist())
        fired = h[A[:, j]]
        if len(fired):
            hours_fired.setdefault(e.symbol, []).extend(fired.tolist())

    out = {"rule": rule, "by_symbol": {}, "jaccard": {}}
    NIGHT = {21, 22, 23, 0, 1, 2}
    for s, hs in sorted(hours_fired.items()):
        arr = np.array(hs)
        vc = pd.Series(arr).value_counts(normalize=True).sort_values(ascending=False)
        out["by_symbol"][s] = {
            "n_fired": int(len(arr)),
            "top_hours": [[int(k), float(v)] for k, v in vc.head(4).items()],
            "night_share": float(np.isin(arr, list(NIGHT)).mean()),
            "traded_hours": sorted(int(x) for x in hours_seen.get(s, [])),
        }
    syms = sorted(hours_seen)
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            u = hours_seen[a] | hours_seen[b]
            out["jaccard"][f"{a}|{b}"] = (len(hours_seen[a] & hours_seen[b]) / len(u)
                                          if u else float("nan"))
    return out


def save(res: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    return path


# ---------------------------------------------------------------- 跨品种传导

def _alert_minutes(episodes, feature_names: list, min_level: int = ANOM_MIN_LEVEL) -> dict:
    """{symbol: {rule: 有序的触发分钟数组}} 以及 {symbol: 有序的全部观测分钟数组}。"""
    fired = {}
    seen = {}
    for e in episodes:
        A = anom_matrix(e.X, feature_names, min_level)
        mins = (pd.to_datetime(pd.Series(e.ts), format="%Y%m%d%H%M%S")
                .astype("int64").to_numpy() // 60_000_000_000)
        seen.setdefault(e.symbol, []).append(mins)
        for j, a in enumerate(ANOMS):
            if A[:, j].any():
                fired.setdefault(e.symbol, {}).setdefault(a.rule, []).append(mins[A[:, j]])
    fired = {s: {r: np.sort(np.concatenate(v)) for r, v in d.items()}
             for s, d in fired.items()}
    seen = {s: np.sort(np.concatenate(v)) for s, v in seen.items()}
    return fired, seen


def _within(src: np.ndarray, tgt: np.ndarray, window: int) -> np.ndarray:
    """对 src 中每个时刻 t，判断 tgt 中是否存在落在 [t, t+window] 的元素。"""
    if len(tgt) == 0 or len(src) == 0:
        return np.zeros(len(src), dtype=bool)
    i = np.searchsorted(tgt, src, side="left")
    ok = i < len(tgt)
    out = np.zeros(len(src), dtype=bool)
    out[ok] = tgt[i[ok]] <= src[ok] + window
    return out


def _hours(mins: np.ndarray) -> np.ndarray:
    """分钟数 → 当地小时（原始时间戳按当地时间解析，故直接取模即可）。"""
    return ((mins // 60) % 24).astype(int)


def _hour_matched_base(cov: np.ndarray, hit: np.ndarray,
                       src: np.ndarray) -> float:
    """按 src 的「小时分布」加权的基线发生率。

    为什么需要它：实测发现 ag/au/sc 的流动性异常有 90% 以上集中在 0–2 点的夜盘，
    而 si（广期所）不交易夜盘、异常集中在 10–14 点。若用全时段平均作基线，
    「白银流动性枯竭 → 黄金流动性枯竭」会得到 lift≈3.2 的漂亮数字——但那几乎完全是
    「两者都在夜盘、夜盘流动性本来就薄」造成的日内季节性，不是风险传导。
    按小时重新加权基线后，剩下的才是超出日内节律的额外共现。
    """
    ch, sh = _hours(cov), _hours(src)
    base_h = {}
    for h in range(24):
        m = ch == h
        if m.any():
            base_h[h] = float(hit[m].mean())
    if not base_h:
        return float("nan")
    tot, acc = 0, 0.0
    for h in range(24):
        w = int((sh == h).sum())
        if w and h in base_h:
            acc += w * base_h[h]
            tot += w
    return acc / tot if tot else float("nan")


def calibrate_cross_symbol(episodes, feature_names: list, window: int = 120,
                           n_perm: int = 200, seed: int = 0,
                           min_support: int = 100, max_p: float = 0.05,
                           min_lift: float = 1.10, verbose: bool = True) -> list:
    """实测「品种 s1 的某异常出现后，s2 在 window 分钟内出现同类异常」的提升度。

    报两个基线，两个都写进产物：
      - `lift`         ：全时段平均基线（朴素口径）
      - `lift_matched` ：**按 s1 触发时刻的小时分布加权**的基线（去日内季节性）

    保留标准以 `lift_matched` 的置信下界为准。原因见 `_hour_matched_base` 的说明：
    朴素口径会把「两个品种都在夜盘」当成风险传导。

    即便如此，lift 高也只说明**超出日内节律的同步共现**，仍不能区分「传导」与
    「共同暴露于同一宏观冲击」。报告与解释文本一律按「同步共现」措辞，不写「A 导致 B」。
    """
    rng = np.random.default_rng(seed)
    fired, seen = _alert_minutes(episodes, feature_names)
    out = []
    day = 24 * 60
    for rule in sorted({a.rule for a in ANOMS}):
        for s1 in sorted(fired):
            for s2 in sorted(fired):
                if s1 == s2:
                    continue
                src = fired.get(s1, {}).get(rule)
                tgt = fired.get(s2, {}).get(rule)
                if src is None or tgt is None or len(src) < min_support:
                    continue
                # 只保留 s2 也有数据覆盖的时刻，避免把「没数据」当成「没传导」
                cov = seen[s2]
                keep = _within(src, cov, window)
                src2 = src[keep]
                if len(src2) < min_support:
                    continue
                hit_src = _within(src2, tgt, window)
                hit_cov = _within(cov, tgt, window)
                p_cond = float(hit_src.mean())
                p_base = float(hit_cov.mean())
                p_base_m = _hour_matched_base(cov, hit_cov, src2)
                lift = p_cond / p_base if p_base > 0 else float("nan")
                lift_m = p_cond / p_base_m if p_base_m and p_base_m > 0 else float("nan")
                lo, _ = _wilson(int(hit_src.sum()), len(src2))
                lift_lo = lo / p_base_m if p_base_m and p_base_m > 0 else float("nan")

                ge = 0
                span = max(1, int(cov[-1] - cov[0]))
                for _ in range(n_perm):
                    sh = int(rng.integers(1, max(2, span // day))) * day
                    ps = np.sort(((src2 - cov[0] + sh) % span) + cov[0])
                    pl = (float(_within(ps, tgt, window).mean()) / p_base_m
                          if p_base_m and p_base_m > 0 else 0.0)
                    ge += int(pl >= lift_m)
                p_value = (ge + 1) / (n_perm + 1)

                rec = {"rule": rule, "src": s1, "dst": s2, "n_src": int(len(src2)),
                       "p_cond": p_cond, "p_base": p_base,
                       "p_base_matched": p_base_m,
                       "lift": lift, "lift_matched": lift_m,
                       "lift_ci_low": lift_lo, "p_value": p_value,
                       "strong": bool(len(src2) >= min_support and p_value <= max_p
                                      and np.isfinite(lift_lo) and lift_lo > min_lift)}
                out.append(rec)
                if verbose:
                    print(f"  [{'strong' if rec['strong'] else 'weak  '}] "
                          f"{rule:20s} {s1}→{s2}  lift朴素={lift:5.2f} "
                          f"→ 小时匹配={lift_m:5.2f} (下界{lift_lo:5.2f}) "
                          f"n={len(src2):5d} p={p_value:.4f}", flush=True)
    return out
