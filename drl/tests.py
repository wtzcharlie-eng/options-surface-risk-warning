"""drl.tests — 验证门。不通过就不许把结果写进报告。

设计意图：DRL 在金融数据上"效果好得可疑"是常态，绝大多数时候是标签泄漏、
指标口径错位或奖励可被套利。本模块把这几类失败模式逐个变成可执行断言。

运行::

    python -m drl.tests                 # 全部
    python -m drl.tests --quick         # 跳过需要训练的门

门列表
------
G0 标签复现      重算风险起点 → 反推标签，须与 anchor 存档 `y` 逐行一致
G1 状态因果性    篡改 t 之后的全部特征，t 时刻状态必须逐位不变
G2 指标一致性    drl.metrics 与 vol_surface.quant_metrics 在同一输入上须给出同一结果
G3 环境不变量    cover==召回数、cover+redundant==命中预警数、cover+miss==风险起点数
G4 奖励不可套利  全部平凡策略（恒0/恒2/恒3/随机）累计奖励须劣于规则基线
G5 归一化无泄漏  归一化统计量只依赖训练集，改动 val/test 不得影响
G6 训练可复现    同 seed 两次训练须得到同一份验证集指标
G7 打乱标签      风险起点随机重排后重训，**判别力**须塌陷（真泄漏门）
G8 优于盲节奏    DRL 须超过最优「盲节奏」策略，且精确率显著高于 hit 基础率
G9 续训等价性    断点续训须与一次跑完逐位等价
G10 新口径非退化 新口径下的达标须建立在对平凡策略的真实优势上（口径本身会退化）
G11 CVaR 非饱和   CVaR 改善率若与「恒报警」无差别，即为仓位饱和，不得当作达标证据
G12 死区过滤安全  休市阈值须 > **实测日内最长休市**（现算，本数据集 135min，非 120）
G13 否决式融合     不启用时逐位不变；启用时须与交付脚本 rule_ml.fuse 一致
G14 异常分健全性   批不变 + 非退化 + 方向单调 + 端到端四级可达（走真模型的 evaluate）

注：G6/G7 为控制耗时用的是小规模子集，其绝对数值与主结果（全测试集、多 seed）
不可直接比较——门消息里已标注各自口径。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.baseline import (ConstantPolicy, PeriodicPolicy, RandomPolicy, RulePolicy,
                          load_best_params)
from drl.dataset import (FEATURES, Normalizer, _lead_minutes,
                         apply_continuous_risk, identify_risk_starts,
                         labels_from_risk_starts, load_episodes, split_episodes)
from drl.env import AlertEnv, RewardSpec
from drl.metrics import aggregate, evaluate_alerts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANCHOR = os.path.join(ROOT, "data_out", "anchor")

_RESULTS = []
_DETAILS = {}   # 各门的结构化数值，供 --emit 落盘给报告引用


def gate(name: str):
    def deco(fn):
        def wrapped(*a, **kw):
            try:
                msg = fn(*a, **kw)
                _RESULTS.append((name, True, msg or ""))
                print(f"  [PASS] {name}  {msg or ''}")
                return True
            except AssertionError as e:
                _RESULTS.append((name, False, str(e)))
                print(f"  [FAIL] {name}  {e}")
                return False
            except Exception as e:                       # noqa: BLE001
                _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
                print(f"  [ERROR] {name}  {type(e).__name__}: {e}")
                return False
        return wrapped
    return deco


# ---------------------------------------------------------------- G0

@gate("G0 标签复现与 anchor 存档一致")
def test_label_reconstruction(eps, monthly: bool = False) -> str:
    """标签可复现性。

    逐月口径：直接对每幕重算风险起点，应与 `e.y` 逐行一致。
    连续口径：`e.y` 已被 `apply_continuous_risk` 按品种全时间轴改写，逐幕重算
    不再等价（这正是修正的目的）。此时改为检验两条更强的性质：
      (a) 载入时 `verify=True` 已逐幕核对过存档标签与逐月口径一致（parquet 没坏）；
      (b) 连续口径下每幕开头的「结构性死区」应当消失——逐月口径下**全样本**
          前 72 根截面的风险起点数恒为 0，连续口径下应显著大于 0。
    """
    if monthly:
        n_check = 0
        for e in eps[:: max(1, len(eps) // 12)]:
            idx = identify_risk_starts(e.ts, e.X[:, FEATURES.index("atm_iv")],
                                       e.X[:, FEATURES.index("convexity_violation")])
            y = labels_from_risk_starts(len(e), idx)
            assert np.array_equal(y, e.y), f"{e.symbol} {e.ym} 标签不一致"
            n_check += 1
        return f"[逐月口径] 抽查 {n_check} 幕，逐行一致"

    n_dead = sum(int((e.risk_idx < 72).sum()) for e in eps)
    n_all = sum(len(e.risk_idx) for e in eps)
    # 载入时 verify=True 已断言存档标签与逐月口径一致；这里查连续口径的自洽性。
    # 注意跨幕效应：y[i]=1 的条件是「(i, i+8] 内有风险起点」，而该起点可能落在
    # **下一幕**（月末的截面本就应当被次月初的风险区间标为正——线上正是如此）。
    # 故只有前 n−8 根能与本幕 risk_idx 严格对齐，末尾 8 根只能要求「不减」。
    H = 8
    for e in eps[:: max(1, len(eps) // 12)]:
        assert len(e.y) == len(e.ts) == len(e.lead_min), f"{e.symbol} {e.ym} 长度不齐"
        local = labels_from_risk_starts(len(e), e.risk_idx.tolist())
        k = max(0, len(e) - H)
        assert np.array_equal(local[:k], e.y[:k]), \
            f"{e.symbol} {e.ym} 幕内标签与 risk_idx 不自洽"
        assert np.all(e.y[k:] >= local[k:]), \
            f"{e.symbol} {e.ym} 幕尾标签少于本幕 risk_idx 所蕴含的"
    n_cross = sum(int((e.y[max(0, len(e) - H):] >
                       labels_from_risk_starts(len(e), e.risk_idx.tolist())
                       [max(0, len(e) - H):]).sum()) for e in eps)
    _DETAILS["G0"] = {"n_risk": n_all, "n_risk_in_first72": n_dead,
                      "n_cross_episode_labels": n_cross}
    assert n_dead > 0, ("连续口径下幕首 72 根仍无任何风险起点——"
                        "说明 apply_continuous_risk 没有真正跨月延续滚动窗口")
    return (f"[连续口径] 幕内自洽；幕首死区已消除（前 72 根内风险起点 "
            f"{n_dead}/{n_all}，逐月口径下恒为 0）；跨幕标签 {n_cross} 处")


# ---------------------------------------------------------------- G1

@gate("G1 状态不含未来信息")
def test_state_causality(eps, norm) -> str:
    e = eps[len(eps) // 2]
    spec = RewardSpec()
    env = AlertEnv(e, norm, spec)
    probe = [1, len(e) // 3, len(e) // 2, len(e) - 5]
    rng = np.random.default_rng(0)

    for t in probe:
        env.reset()
        for i in range(t):
            env.step(0)
        s_ref = env._state(t).copy()

        # 把 t 之后的所有原始特征彻底打乱后重建环境，t 时刻状态必须不变
        import copy
        e2 = copy.deepcopy(e)
        e2.X[t + 1:] = rng.normal(size=e2.X[t + 1:].shape) * 100
        env2 = AlertEnv(e2, norm, spec)
        env2.reset()
        for i in range(t):
            env2.step(0)
        s_new = env2._state(t)
        assert np.allclose(s_ref, s_new), f"t={t} 处状态受未来特征影响"
    return f"探针 {probe} 全部通过"


# ---------------------------------------------------------------- G2

@gate("G2 指标实现与 quant_metrics 一致")
def test_metrics_vs_quant_metrics(eps) -> str:
    try:
        from vol_surface.quant_metrics import evaluate_quant
    except Exception as e:                                # noqa: BLE001
        return f"SKIP（无法导入 vol_surface.quant_metrics: {type(e).__name__}）"

    rng = np.random.default_rng(7)
    n_cmp = 0
    for e in eps[:: max(1, len(eps) // 8)]:
        levels = rng.integers(0, 4, size=len(e))
        mine = evaluate_alerts(e.ts, levels, e.risk_idx)
        alerts = pd.DataFrame({"timestamp": e.ts, "level": levels})
        risk_starts = [e.ts[i] for i in e.risk_idx]
        ref = evaluate_quant(alerts, risk_starts, lead_window_min=120.0, min_level=2)
        assert mine.n_alert == ref.n_alerts, f"{e.ym} 预警数 {mine.n_alert}!={ref.n_alerts}"
        assert mine.n_risk == ref.n_risk_windows, f"{e.ym} 风险数不一致"
        assert abs(mine.precision - ref.precision) < 1e-9, \
            f"{e.ym} 精确率 {mine.precision}!={ref.precision}"
        assert abs(mine.recall - ref.recall) < 1e-9, \
            f"{e.ym} 召回率 {mine.recall}!={ref.recall}"
        assert abs(mine.avg_lead_min - ref.avg_lead_minutes) < 1e-6, \
            f"{e.ym} 提前时间 {mine.avg_lead_min}!={ref.avg_lead_minutes}"
        n_cmp += 1
    return f"随机等级序列对比 {n_cmp} 幕，精确率/召回率/提前时间逐位一致"


# ---------------------------------------------------------------- G3

@gate("G3 环境奖励会计与赛题指标一致")
def test_env_invariants(eps, norm) -> str:
    spec = RewardSpec()
    bp = load_best_params()
    pols = [("rule", lambda e: RulePolicy(e, params=bp)),
            ("const2", lambda e: ConstantPolicy(2)),
            ("random", lambda e: RandomPolicy(seed=1))]
    for nm, mk in pols:
        cover = redundant = miss = 0
        ros = []
        for e in eps:
            env = AlertEnv(e, norm, spec)
            ro = env.rollout(mk(e))
            ros.append(ro)
            for x in ro["events"]:
                for p in x.split("+"):
                    if p == "cover":
                        cover += 1
                    elif p == "redundant":
                        redundant += 1
                    elif p == "miss":
                        miss += 1
        m = aggregate(ros)["micro"]
        n_rec = round(m["n_risk"] * m["recall"])
        n_hit = round(m["n_alert"] * m["precision"])
        assert cover == n_rec, f"{nm}: cover={cover} != 召回数={n_rec}"
        assert abs((cover + redundant) - n_hit) <= 1, \
            f"{nm}: cover+redundant={cover + redundant} != 命中预警数={n_hit}"
        assert cover + miss == m["n_risk"], \
            f"{nm}: cover+miss={cover + miss} != 风险起点数={m['n_risk']}"
    return "rule/const2/random 三策略三项不变量全部成立"


# ---------------------------------------------------------------- G4

@gate("G4 奖励不可被平凡策略套利")
def test_reward_not_gameable(eps, norm) -> str:
    spec = RewardSpec()
    bp = load_best_params()

    def total(mk):
        return sum(AlertEnv(e, norm, spec).rollout(mk(e))["total_reward"] for e in eps)

    rule = total(lambda e: RulePolicy(e, params=bp))
    trivial = {"const0": total(lambda e: ConstantPolicy(0)),
               "const2": total(lambda e: ConstantPolicy(2)),
               "const3": total(lambda e: ConstantPolicy(3)),
               "random": total(lambda e: RandomPolicy(seed=0))}
    bad = {k: v for k, v in trivial.items() if v >= rule}
    assert not bad, (f"平凡策略 {bad} 的累计奖励不低于规则基线 {rule:.0f}"
                     f"——奖励函数可被套利，DRL 的'胜利'将毫无意义")
    return f"规则基线 {rule:.0f} 高于全部平凡策略 " + \
           str({k: round(v) for k, v in trivial.items()})


# ---------------------------------------------------------------- G5

@gate("G5 归一化统计量只来自训练集")
def test_normalizer_train_only(eps) -> str:
    tr, va, te = split_episodes(eps)
    n1 = Normalizer.fit(tr)
    import copy
    te2 = copy.deepcopy(te)
    for e in te2:
        e.X *= 1000.0
    n2 = Normalizer.fit(tr)
    assert np.allclose(n1.mean, n2.mean) and np.allclose(n1.std, n2.std), \
        "训练集统计量受到了测试集改动的影响"
    assert not np.allclose(n1.mean, np.concatenate([e.X for e in te2]).mean(axis=0)), \
        "归一化统计量疑似来自测试集"
    return "改动测试集不影响训练集统计量"


# ---------------------------------------------------------------- G6/G7

@gate("G6 训练可复现（同 seed 同结果）")
def test_reproducible(eps, norm) -> str:
    from drl.train import evaluate_agent, train
    tr, va, _ = split_episodes(eps)
    tr_s, va_s = tr[:12], va[:6]
    spec = RewardSpec()
    a1, _ = train(tr_s, va_s, norm, spec, epochs=1, seed=7, warmup=200, verbose=False)
    a2, _ = train(tr_s, va_s, norm, spec, epochs=1, seed=7, warmup=200, verbose=False)
    m1 = evaluate_agent(va_s, norm, spec, a1)
    m2 = evaluate_agent(va_s, norm, spec, a2)
    assert abs(m1["total_reward"] - m2["total_reward"]) < 1e-6, \
        f"两次训练结果不同: {m1['total_reward']} vs {m2['total_reward']}"
    return (f"两次独立训练验证集累计奖励均为 {m1['total_reward']:.2f}"
            f"（12 幕训练/6 幕验证、1 epoch 的小规模复现跑，量级与主结果不可比）")


@gate("G7 打乱风险起点后判别力塌陷")
def test_shuffled_labels(eps, norm) -> str:
    """真泄漏门：把风险起点随机重排（数量不变），特征与时间戳都不动。

    **判据用「判别力」而非累计奖励**——这一点踩过坑：初版拿"打乱后 DRL 的累计
    奖励是否高于随机策略"作判据，结果 FAIL。排查发现并非泄漏，而是判据本身错了：
    一个完全不看特征、只按固定节奏刷预警的盲策略，累计奖励就能远高于随机策略
    （实测打乱数据上盲策略 +317 vs 随机 −657）。也就是说累计奖励里混了
    "预警节奏优化"这一与信号无关的成分，不能用来判泄漏。

    正确判据：**精确率相对 hit 基础率的提升**。无判别力的策略精确率恒等于基础率；
    若打乱标签后智能体仍有明显判别力，才说明状态里混进了未来信息。
    """
    from drl.train import evaluate_agent, hit_base_rate, train
    import copy
    tr, va, _ = split_episodes(eps)
    spec = RewardSpec()

    def lift(train_g, val_g, seed):
        ag, _ = train(train_g, val_g, norm, spec, epochs=3, seed=seed, verbose=False)
        m = evaluate_agent(val_g, norm, spec, ag)
        return m["precision"] - hit_base_rate(val_g), m

    real_lift, _ = lift(tr[:40], va[:12], 11)

    tr_s, va_s = copy.deepcopy(tr[:40]), copy.deepcopy(va[:12])
    rng = np.random.default_rng(999)
    for e in tr_s + va_s:
        k = len(e.risk_idx)
        if k:
            e.risk_idx = np.sort(rng.choice(np.arange(len(e)), size=k, replace=False))
        e.lead_min = _lead_minutes(pd.Series(pd.to_datetime(e.dt)), e.risk_idx.tolist())
    shuf_lift, _ = lift(tr_s, va_s, 11)

    _DETAILS["G7"] = {"real_lift": float(real_lift), "shuffled_lift": float(shuf_lift),
                      "ratio": float(shuf_lift / real_lift) if real_lift else float("nan")}
    assert shuf_lift < 0.5 * real_lift, (
        f"打乱后判别力 {shuf_lift:+.2%} 未显著低于真实数据的 {real_lift:+.2%}"
        f"（要求 < 一半）——状态中疑似存在未来信息泄漏")
    assert shuf_lift < 0.08, f"打乱后判别力 {shuf_lift:+.2%} 绝对值仍偏高"
    return (f"真实判别力 {real_lift:+.2%} → 打乱后 {shuf_lift:+.2%}"
            f"（塌陷至 {shuf_lift / real_lift:.0%}；口径：40 幕训练/12 幕验证子集、"
            f"单 seed、3 epoch，在**验证集**上评估，故数值与主结果表的测试集判别力不可直接比较）")


@gate("G8 DRL 须优于「盲节奏」策略且具备真实判别力")
def test_beats_blind(eps, norm, drl_dir: str | None = None) -> str:
    """最关键的一条：证明增益来自曲面信号，而不只是预警节奏。

    盲节奏策略不看任何特征，其精确率必然≈hit 基础率。DRL 若只是学会了"多久报
    一次"，精确率也会贴着基础率走。因此要求 DRL 同时满足：
      (a) 累计奖励高于该数据集上**最优**的盲节奏策略（对基线取优，对自己从严）
      (b) 精确率显著高于 hit 基础率
    """
    import glob
    from drl.dqn import DQNAgent
    from drl.train import aggregate as _agg, best_blind, hit_base_rate

    _, _, te = split_episodes(eps)
    spec = RewardSpec()
    files = sorted(f for f in glob.glob(os.path.join(
        drl_dir or os.path.join(ROOT, "data_out", "drl"), "agent_seed*.npz"))
        if "shuf_" not in os.path.basename(f))
    if not files:
        return "SKIP（尚无训练好的 checkpoint，先跑 scripts/train_drl.py）"

    base = hit_base_rate(te)
    period, m_blind = best_blind(te, norm, spec)
    # checkpoint 的输入维度必须与当前载入的特征集匹配。不匹配时 numpy 会在
    # 网络前向里抛 `matmul: ... size 34 is different from 28`——**看不出根因**，
    # 复现者只会以为模型坏了。实际原因永远是同一个：忘了加 `--underlying`
    # （不带该开关只载入 26 维曲面特征，带上才是 32 维含标的侧）。
    # 门可以失败，但**不该以让人看不懂的方式失败**。
    _probe, _ = DQNAgent.load(files[0])
    _need, _got = _probe.state_dim, AlertEnv(te[0], norm, spec).reset().shape[0]
    if _need != _got:
        raise AssertionError(
            f"checkpoint 期望 {_need} 维状态，当前数据集给出 {_got} 维。"
            f"本项目的交付模型用 32 维特征集训练——请改跑 "
            f"`python -m drl.tests --underlying`（见 README「快速开始」）。")
    rewards, lifts = [], []
    for f in files:
        ag, _ = DQNAgent.load(f)
        pol = ag.greedy_policy()
        ros = [AlertEnv(e, norm, spec).rollout(pol) for e in te]
        m = aggregate(ros)["micro"]
        rewards.append(m["total_reward"])
        lifts.append(m["precision"] - base)
    mr, ml = float(np.mean(rewards)), float(np.mean(lifts))

    _DETAILS["G8"] = {"drl_mean_reward": mr, "drl_mean_lift": ml,
                      "blind_period": int(period),
                      "blind_reward": float(m_blind["total_reward"]),
                      "blind_lift": float(m_blind["precision"] - base),
                      "hit_base_rate": float(base)}
    assert mr > m_blind["total_reward"], (
        f"DRL 均值累计奖励 {mr:.0f} 未超过最佳盲节奏(每{period}步) "
        f"{m_blind['total_reward']:.0f}——增益无法归因于曲面信号")
    assert ml > 0.05, f"DRL 精确率仅比基础率高 {ml:+.2%}，判别力不足"
    blind_lift = m_blind["precision"] - base
    return (f"DRL {mr:.0f} > 最佳盲节奏(每{period}步) {m_blind['total_reward']:.0f}；"
            f"判别力 DRL {ml:+.2%} vs 盲节奏 {blind_lift:+.2%}"
            f"（基础率 {base:.1%}；口径：测试集全集、{len(files)} 个 seed 均值）")


@gate("G9 断点续训与一次跑完逐位等价")
def test_resume_equivalence(eps, norm) -> str:
    """断点续训必须与一次跑完**逐位等价**。

    训练支持按 epoch 存盘续跑（单次执行时长受限的环境需要它）。这带来一个隐患：
    若快照漏存了任何一处可变状态——Adam 的一二阶矩、目标网权重、回放池写指针，
    尤其是三个随机数发生器（训练采样序 / ε-贪心 / 回放采样）——续跑轨迹就会
    与直跑发散，「分几次跑」实际上变成了另一个算法，而且**表面上看不出来**。

    故用同一配置跑两遍：一次直跑 3 epoch，一次在第 1 个 epoch 后存盘、
    反序列化、再续跑，要求最终权重与逐次验证集奖励全部逐位相同。
    """
    import pickle as _pkl

    from drl.train import train as _train

    tr, va, _ = split_episodes(eps)
    tr, va = tr[:6], va[:2]          # 小样本即可暴露发散，跑满没必要
    kw = dict(epochs=2, seed=7, verbose=False, warmup=200, buffer=20000,
              evals_per_epoch=2)
    a1, h1 = _train(tr, va, norm, RewardSpec(), **kw)

    class _Stop(Exception):
        pass

    box = {}

    def _hook(i, st):
        box[i] = _pkl.loads(_pkl.dumps(st))   # 真的走一遍序列化
        if i == 0:
            raise _Stop
    try:
        _train(tr, va, norm, RewardSpec(), on_epoch_end=_hook, **kw)
    except _Stop:
        pass
    a2, h2 = _train(tr, va, norm, RewardSpec(), resume=box[0], **kw)

    w1, w2 = a1.q.params(), a2.q.params()
    same_w = all(np.array_equal(w1[k], w2[k]) for k in w1)
    r1 = [r["val_reward"] for r in h1]
    r2 = [r["val_reward"] for r in h2]
    _DETAILS["G9"] = {"weights_identical": bool(same_w),
                      "history_identical": bool(r1 == r2),
                      "val_rewards": r1}
    assert same_w, "续跑后网络权重与直跑不一致——快照漏存了状态"
    assert r1 == r2, f"续跑的验证集奖励序列与直跑不一致：{r1} vs {r2}"
    return f"续跑 = 直跑，权重逐位相同，验证奖励序列一致（{len(r1)} 个 checkpoint）"


@gate("G10 新口径的达标不得来自口径退化")
def test_v2_not_degenerate() -> str:
    """新口径（命题方 2026-08 答复）下，达标必须建立在对平凡策略的真实优势上。

    为什么必须有这道门
    ------------------
    新口径规定「召回匹配不设时间上限」，其字面推论是：幕内任意一次早期预警
    即覆盖其后**全部**风险起点。实测所有平凡策略（含「只在开头报一次」）
    召回均为 98.9%、平均提前虚高至 20651 分钟——**三条门槛实际塌缩成只剩精确率**。

    也就是说「三项全达标」在这个口径下本身不构成成绩。唯一有意义的量是
    **精确率相对最强平凡策略的增量**。本门强制该增量为正且非平凡，
    并强制落盘里带着平凡策略成绩，防止只报达标、不报对照。

    这与本项目此前发现 CVaR「仅首次减仓」导致改善率恒为 +50% 是同一类问题：
    评分规则可能自带退化，必须用平凡策略去探。
    """
    p = os.path.join(ROOT, "data_out", "metrics_v2.json")
    if not os.path.exists(p):
        raise AssertionError(f"缺 {p}，请先跑 scripts/eval_v2.py")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)

    assert d.get("trivial"), "落盘缺平凡策略对照——新口径结论不得单独发布"
    assert d.get("degeneracy_note"), "落盘缺退化说明"

    triv = d["trivial"]
    # 退化事实本身要被断言住：若哪天平凡策略召回不再接近 1，说明口径被改过
    rec = [m["recall"] for m in triv.values()]
    assert max(rec) > 0.9, (
        f"平凡策略最高召回仅 {max(rec):.1%}，与「召回不设上限」的口径不符——"
        f"口径实现可能被改动，请核对 drl/metrics_v2.py")

    best = d["best_trivial"]["precision"]
    drl = d["drl"]["test"]["precision"]
    disc = (drl - best) * 100
    _DETAILS["G10"] = {"drl_precision": drl, "best_trivial": best,
                       "discrimination_pp": disc,
                       "trivial_max_recall": max(rec),
                       "pass3": d["drl"]["pass3"]}
    assert disc > 5.0, (
        f"DRL 精确率 {drl:.1%} 相对最强平凡策略 {best:.1%} 仅 {disc:+.1f}pp，"
        f"不足以说明达标来自真实判别力而非口径退化")
    return (f"DRL 精确率 {drl:.1%} vs 最强平凡策略「{d['best_trivial']['name']}」"
            f"{best:.1%}，判别力 {disc:+.1f}pp；已确认平凡策略召回达"
            f"{max(rec):.1%}（口径退化事实已落盘并披露）")



@gate("G11 CVaR 口径的改善率不得来自仓位饱和")
def test_cvar_not_saturated() -> str:
    """凡是报出 CVaR 改善率，必须同时证明该口径**没有退化成「全程降杠杆」**。

    为什么单设一门
    --------------
    本项目在同一族退化上栽过三次：
      ① `backtest_cvar` 的「首次预警后持有至期末」→ 全程半仓 → 7 品种一律 +50.0%；
      ② 命题方新口径的「召回不设时间上限」→ 所有平凡策略召回均达 98.9%（由 G10 守）；
      ③ `backtest_cvar_path` 在 15 分钟频预警的密度下 → 又回到全程半仓。
    三次的共同形态是**某个参数把策略空间压缩到一个点**，于是指标不再区分策略。
    三次都只有靠「并列平凡策略对照」才发现——没有对照就会被当成好成绩报出去。

    判据（两条都要过）
      (a) 恒报警的 excess 必须 ≈ 0——它是「不看数据也能拿到的白拿部分」的定义；
      (b) 被评估策略的 excess 必须**明显高于**恒报警，否则该口径在当前密度下无判别力，
          其 improve 数值**不得作为达标证据**。
    """
    import json as _json
    p = os.path.join(ROOT, "data_out", "cvar_drl.json")
    if not os.path.exists(p):
        return "SKIP（尚无 cvar_drl.json，先跑 scripts/cvar_drl.py）"
    d = _json.load(open(p, encoding="utf-8"))
    sm = d.get("summary", {})
    assert "恒报警" in sm, "CVaR 结果缺少「恒报警」对照——**没有对照就不许报这个指标**"
    const_ex = sm["恒报警"]["excess_mean"]
    assert abs(const_ex) < 0.005, (
        f"恒报警的 excess = {const_ex:+.2%}，应≈0。不为 0 说明对照本身构造有误")
    drl = sm.get("drl", {})
    degenerate = abs(drl.get("excess_mean", 0.0) - const_ex) < 0.01
    _DETAILS["G11"] = {
        "drl_excess": drl.get("excess_mean"),
        "const_excess": const_ex,
        "drl_improve": drl.get("improve_mean"),
        "const_improve": sm["恒报警"]["improve_mean"],
        "degenerate": bool(degenerate),
        "verdict": d.get("preregistered_verdict", {}),
    }
    # **退化本身不判 FAIL**——退化是数据事实，不是代码缺陷。本门要拦的是
    # 「退化了却把 improve 当成绩报出去」。故要求：一旦退化，落盘里必须已经
    # 记录了未通过的预登记结论，材料才不会误导读者。
    if degenerate:
        assert d.get("preregistered_verdict", {}).get("passed") is False, (
            f"CVaR 口径已退化（DRL excess {drl.get('excess_mean'):+.2%} 与恒报警 "
            f"{const_ex:+.2%} 无区别），却把预登记判据标成通过——这会把"
            f"「降杠杆」误报成「风控有效」")
        return (f"口径在预登记参数下**已退化**：DRL improve {drl['improve_mean']:+.1%} "
                f"与恒报警 {sm['恒报警']['improve_mean']:+.1%} 几乎相同、"
                f"excess 均 ≈0；落盘已如实记为「未通过」，未被当作达标证据")
    return (f"口径非退化：DRL excess {drl['excess_mean']:+.2%} vs "
            f"恒报警 {const_ex:+.2%}，差距明显")


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--quick", action="store_true", help="跳过 G6/G7（需训练）")
    ap.add_argument("--underlying", action="store_true",
                    help="按 32 维特征集校验（含标的侧 6 维）")
    ap.add_argument("--monthly-risk", action="store_true",
                    help="风险起点退回逐月判定（复现旧结果用，见 README §7.10）")
    ap.add_argument("--drl-dir", default=os.path.join(ROOT, "data_out", "drl"),
                    help="G8 读取 checkpoint 的目录")
    ap.add_argument("--emit", nargs="?", const=os.path.join(ROOT, "data_out", "drl",
                                                            "gates.json"),
                    help="把各门的通过状态与关键数值落盘为 JSON，供报告引用")
    args = ap.parse_args()

    print("载入 anchor 数据集 ...")
    eps = load_episodes(args.anchor, verify=True, underlying=args.underlying)
    risk_mode = "逐月(旧)" if args.monthly_risk else "跨月连续"
    if not args.monthly_risk:
        st = apply_continuous_risk(eps)
        print(f"  风险起点口径：跨月连续（{st['risk_before']} → {st['risk_after']}）")
    tr, va, te = split_episodes(eps)
    norm = Normalizer.fit(tr)
    n_feat = eps[0].X.shape[1] if eps else 0
    print(f"  {len(eps)} 幕 / {sum(len(e) for e in eps)} 截面 / {n_feat} 维特征\n")

    print("验证门:")
    test_label_reconstruction(eps, monthly=args.monthly_risk)
    test_state_causality(eps, norm)
    test_metrics_vs_quant_metrics(eps)
    test_env_invariants(te, norm)
    test_reward_not_gameable(te, norm)
    test_normalizer_train_only(eps)
    if not args.quick:
        test_reproducible(eps, norm)
        test_shuffled_labels(eps, norm)
    test_beats_blind(eps, norm, drl_dir=args.drl_dir)
    test_resume_equivalence(eps, norm)
    test_v2_not_degenerate()
    test_cvar_not_saturated()
    test_dead_zone_threshold(eps)
    test_veto_fusion(eps)
    test_ml_score_batch_invariance(eps)

    n_pass = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{n_pass}/{len(_RESULTS)} 通过")

    if args.emit:
        import json
        payload = {
            "n_pass": n_pass, "n_total": len(_RESULTS), "quick": bool(args.quick),
            "risk_mode": risk_mode, "n_features": int(n_feat),
            "drl_dir": os.path.basename(args.drl_dir.rstrip("/")),
            "gates": [{"name": n, "passed": bool(ok), "message": m}
                      for n, ok, m in _RESULTS],
            "details": _DETAILS,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.emit)), exist_ok=True)
        with open(args.emit, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=float)
        print(f"→ {args.emit}")

    sys.exit(0 if n_pass == len(_RESULTS) else 1)


@gate("G12 死区过滤的休市阈值须落在真实间隔分布的空隙里")
def test_dead_zone_threshold(eps) -> str:
    """休市阈值必须**落在观测到的间隔分布的空隙中**，且两侧留有余量。

    为什么单设一门
    --------------
    该过滤要处理的缺陷是：精确率命中窗口是**墙钟分钟制**的 `[a, a+120min]`，
    而风险起点是**根数制**的。收盘前的预警承诺的 120 分钟里几乎没有可交易时间，
    结构上不可能命中（实测 15:00 命中率 0.65%，全样本 24.39%）。

    但**午休前的预警实测仍接近基础率**（11:00/11:15/11:30 为 23.4%/22.8%/22.1%），
    若阈值取得太小，把午休也当成「休市」，这批预警会被误杀，
    而**精确率照样上升**（它们命中率略低于均值）——指标变好、系统变差。

    两次踩坑，两次都是「写死一个凭印象的常量」
    ------------------------------------------
    v1 断言 `> 120`，依据是「午休 120 分钟」——实测**午休其实是 135 分钟**
    （早市末根 11:30 → 下午首根 13:45）。有人把阈值设成 130 时 v1 照样通过。
    v2 改为现算日内最长间隔，但又引入了 `OVERNIGHT=200.0` 这个**新的写死常量**
    来区分「日内」与「隔夜」——结构上仍是同一类隐患，只是位置更安全。

    v3（本版）**不再需要任何分类常量**：只问一件事——
    阈值是否落在「观测不到任何间隔」的空隙里，且离两侧最近的观测值都够远。
    这样无论将来加入什么交易时段的品种，判据都自动跟着数据走。

    判据（四条都要过）
      (a) 阈值两侧最近的观测间隔各留 ≥ `MARGIN` 分钟余量（阈值不压在观测值上）；
      (b) 凡「其后间隔 < 阈值」的根，一律**不**得标记为死区；
      (c) 凡「其后间隔 > 阈值」的根，一律**必须**标记为死区；
      (d) 全量扫描所有幕，不抽样——抽样能覆盖全部品种是巧合，不是保证。
    """
    import numpy as _np
    from drl.dataset import (DEAD_ZONE_BARS, DEAD_ZONE_BREAK_MIN,
                             session_dead_zone)

    MARGIN = 10.0      # 阈值距最近观测间隔的最小余量（分钟）
    BAR = 20.0         # 小于它视为「相邻正常 K 线」，不属于休市

    # ---- (d) 全量扫描
    gaps = []
    for e in eps:
        t = (pd.to_datetime(pd.Series(e.ts), format="%Y%m%d%H%M%S")
             .astype("int64").to_numpy() // 10 ** 9)
        g = (t[1:] - t[:-1]) / 60.0
        gaps.extend(g[g > BAR].tolist())
    assert gaps, "样本里找不到任何休市间隔，无法校准阈值"
    gaps = _np.asarray(gaps)

    # ---- (a) 阈值必须落在空隙里，两侧留余量
    below = gaps[gaps < DEAD_ZONE_BREAK_MIN]
    above = gaps[gaps > DEAD_ZONE_BREAK_MIN]
    assert below.size and above.size, (
        f"阈值 {DEAD_ZONE_BREAK_MIN}min 落在观测间隔分布的一侧之外，无法区分两类休市")
    lo, hi = float(below.max()), float(above.min())
    assert not _np.any(_np.isclose(gaps, DEAD_ZONE_BREAK_MIN)), (
        f"阈值 {DEAD_ZONE_BREAK_MIN}min 恰好压在一个观测间隔上，判定不稳定")
    assert DEAD_ZONE_BREAK_MIN - lo >= MARGIN, (
        f"阈值 {DEAD_ZONE_BREAK_MIN}min 距下方最近观测间隔 {lo:.0f}min 仅 "
        f"{DEAD_ZONE_BREAK_MIN - lo:.0f}min（<{MARGIN:.0f}），"
        f"再低一点就会把 {lo:.0f}min 的休市也当成隔夜、误杀其前的有效预警")
    assert hi - DEAD_ZONE_BREAK_MIN >= MARGIN, (
        f"阈值 {DEAD_ZONE_BREAK_MIN}min 距上方最近观测间隔 {hi:.0f}min 仅 "
        f"{hi - DEAD_ZONE_BREAK_MIN:.0f}min（<{MARGIN:.0f}）")

    # ---- (b)(c) 在真实时间轴上验证标记位置（同样全量）
    n_below = n_above = 0
    for e in eps:
        t = (pd.to_datetime(pd.Series(e.ts), format="%Y%m%d%H%M%S")
             .astype("int64").to_numpy() // 10 ** 9)
        g = (t[1:] - t[:-1]) / 60.0
        dead = session_dead_zone(e.ts, DEAD_ZONE_BARS, DEAD_ZONE_BREAK_MIN)
        for i in _np.flatnonzero((g > BAR) & (g < DEAD_ZONE_BREAK_MIN)):
            assert not dead[i], (
                f"{e.symbol} {e.ym} 第 {i} 根其后间隔 {g[i]:.0f}min "
                f"< 阈值，却被标记为死区")
            n_below += 1
        for i in _np.flatnonzero(g > DEAD_ZONE_BREAK_MIN):
            assert dead[i], (
                f"{e.symbol} {e.ym} 第 {i} 根其后间隔 {g[i]:.0f}min "
                f"> 阈值，却未被标记")
            n_above += 1

    _DETAILS["G12"] = {
        "break_min": DEAD_ZONE_BREAK_MIN, "bars": DEAD_ZONE_BARS,
        "nearest_below_min": lo, "nearest_above_min": hi,
        "margin_below_min": DEAD_ZONE_BREAK_MIN - lo,
        "margin_above_min": hi - DEAD_ZONE_BREAK_MIN,
        "n_episodes_scanned": len(eps),
        "n_checked_below": n_below, "n_checked_above": n_above,
        "distinct_gaps": sorted({round(float(x)) for x in gaps}),
    }
    return (f"阈值 {DEAD_ZONE_BREAK_MIN:.0f}min 落在 [{lo:.0f}, {hi:.0f}] 的空隙里"
            f"（下余量 {DEAD_ZONE_BREAK_MIN - lo:.0f}min / 上余量 "
            f"{hi - DEAD_ZONE_BREAK_MIN:.0f}min）；全量 {len(eps)} 幕，"
            f"阈值下方 {n_below} 处均未标记、上方 {n_above} 处均已标记")


@gate("G13 否决式融合：不启用时逐位不变，启用时与交付脚本一致")
def test_veto_fusion(eps) -> str:
    """新加的 `veto_prob/veto_thr` 必须满足两条，缺一不可。

    **(a) 不启用时逐位不变。**
    问题1 的既有结果全部是 `veto_prob=None` 跑出来的。若新参数哪怕在默认路径上
    改变了一点行为，那些已交付的数字就全部作废且无人察觉——
    加可选参数最常见的事故就是「顺手改了公共路径」。

    **(b) 启用时与交付脚本一致。**
    `scripts/rule_ml.py` 里的融合是脚本自己实现的（`fuse()` 直接改 numpy 数组），
    而 `alert_engine` 是**交付本体**。两者若不一致，报告里那个 53.86%
    就不代表线上系统的行为——**评测与交付必须走同一条代码路径**，
    这是本项目 §7.16 已经立过的规矩。
    """
    import numpy as _np
    from vol_surface.alert_engine import evaluate as _ev
    from drl.dataset import feature_names as _fn

    FN = _fn(eps[0].X.shape[1] == 32)
    e = eps[len(eps) // 2]
    n = min(len(e), 400)

    # ---- (a) 公共路径未被污染：对**金标准夹具**逐例比对
    # ⚠ 初版这里是拿 `_ev(...)` 与 `_ev(..., veto_prob=None)` 互比——
    # **那是自指的**：两者走同一段代码，被污染时会一起变、仍然相等。
    # 实测注入 `if level >= 3: level = 2` 后本门照样通过。
    # 故改为对照**改动之前生成的金标准夹具**，那才是外部参照。
    import json as _json
    _gp = os.path.join(ROOT, "data_out", "alert_engine_golden.json")
    assert os.path.exists(_gp), (
        "缺 data_out/alert_engine_golden.json —— 没有外部参照就无法证明"
        "公共路径未被改动，本门失去意义")
    _g = _json.load(open(_gp, encoding="utf-8"))
    _bad = []
    for _c in _g["cases"]:
        _lv = _ev(_c["feats"], params=None, state=None, model=None)["level"]
        if _lv != _c["level"]:
            _bad.append((_c["level"], _lv))
    assert not _bad, (
        f"alert_engine.evaluate 在 {len(_bad)}/{_g['n']} 个金标准样例上改变了等级"
        f"（如 {_bad[0][0]}→{_bad[0][1]}）——**公共路径被改动**，"
        f"问题1 的既有结果全部失效。若这是有意的行为变更，"
        f"须显式重生成 alert_engine_golden.json 并在 README 说明")

    # 顺带确认：只给一半 veto 参数不得生效
    for i in range(min(n, 100)):
        f = {k: float(e.X[i, j]) for j, k in enumerate(FN)}
        base = _ev(f, params=None, state=None, model=None)["level"]
        # ⚠ 探针值必须选在**会触发 bug 的那一侧**。初版用 veto_prob=0.99（高分），
        # 而典型的错误实现是 `veto_prob < (veto_thr or 0.5)` —— 0.99 本来就不会被否决，
        # 于是注入了缺陷本门照样通过。改用**极低**的 0.01：任何「只看一个参数就否决」
        # 的实现都会在这里把等级压到 0。
        for kw in ({"veto_prob": 0.01}, {"veto_thr": 0.5}):
            assert _ev(f, params=None, state=None, model=None, **kw)["level"] == base, \
                f"只给 {list(kw)[0]} 时 veto 生效了——两个参数必须同时给才算启用"

    # ---- (b) 启用时与交付脚本的实现一致
    rng = _np.random.default_rng(0)
    probs = rng.random(n)
    thr = 0.35
    mism = 0
    for i in range(n):
        f = {k: float(e.X[i, j]) for j, k in enumerate(FN)}
        base = _ev(f, params=None, state=None, model=None)["level"]
        eng = _ev(f, params=None, state=None, model=None,
                  veto_prob=float(probs[i]), veto_thr=thr)["level"]
        ref = 0 if probs[i] < thr else base          # scripts/rule_ml.py::fuse 的语义
        if eng != ref:
            mism += 1
    assert mism == 0, (
        f"引擎的否决式融合与 scripts/rule_ml.py::fuse 在 {mism}/{n} 个截面上不一致——"
        f"评测与交付走的不是同一条路径，报告里的数字不代表线上行为")

    _DETAILS["G13"] = {"n_checked": n, "n_golden": _g["n"],
                       "golden_sha16": _g["levels_sha256_16"],
                       "unchanged_when_off": True,
                       "matches_delivery_script": True, "veto_thr_probe": thr}
    return (f"金标准 {_g['n']} 例逐位一致（指纹 {_g['levels_sha256_16']}）；"
            f"只给一半参数不生效；启用时与 rule_ml.fuse 语义在 {n} 个截面上逐位一致")


@gate("G14 异常分与批大小无关，且交付模型未退化为常数")
def test_ml_score_batch_invariance(eps) -> str:
    """`AlertModel.score_samples` 必须满足两条，缺一不可。

    **(a) 批不变性。** 同一个截面单独打分，与它混在一批里打分，必须得到同一个数。
    这是根因性质——本项目那次事故就是它不成立：原实现 `norm / norm.max()` 是
    **批内相对**的，而所有生产路径（`alert_engine.evaluate` 每次只传一个样本、
    `scan.py`、`build_features.py`）都是 batch=1，于是分子分母同一个数、
    异常分**恒等于 1.0**。落盘的 2,191 行 `ml_score` 全部为 1.0。

    **(b) 交付模型未退化。** (a) 只验代码，验不出「代码修好了但模型没重训」。
    故对**仓库里实际交付的** `model_*.joblib` 逐个打分，要求分数有真实分散度。

    判据为什么要先四舍五入
    ----------------------
    裸 `len(set(scores)) > 1` **会放那个 bug 过去**：`norm/(norm.max()+1e-9)` 里的
    `1e-9` 让 300 个样本在**第 9 位小数**上互不相同，唯一值数是 300 而非 1。
    故判据取 `round(s, 4)` 的唯一值数，并同时看**极差**——
    bug 版极差 2.3e-9，修复版 0.999。
    """
    import numpy as _np
    from drl.dataset import feature_names as _fn
    from vol_surface.alert_model import AlertModel, MODEL_FEATURES

    try:
        AlertModel()
    except ImportError as e:                                   # noqa: BLE001
        return f"跳过（未装 sklearn，ML 层不可用）：{e}"

    FN = _fn(eps[0].X.shape[1] == 32)
    idx = [FN.index(k) for k in MODEL_FEATURES]      # 13 维全部在 DRL 特征里
    e = eps[len(eps) // 2]
    Q = _np.asarray(e.X[:, idx], float)              # 真实特征向量，非合成数据

    # ---- (a) 批不变性：用一个当场训练的模型验代码本身
    tr = _np.concatenate([_np.asarray(ep.X[:, idx], float) for ep in eps[:40]])
    m = AlertModel().fit(tr)
    one = _np.array([m.score_samples(Q[i:i + 1])[0] for i in range(len(Q))])
    bat = m.score_samples(Q)
    d = float(_np.abs(one - bat).max())
    assert d < 1e-12, (
        f"逐样本打分与整批打分不一致（最大差 {d:.2e}）——异常分是**批内相对**的。"
        f"生产路径每次只传一个样本，这会让分数退化成常数")
    # 半批拼接也须一致（批不变性的更强形式：不只是 1 vs N）
    half = _np.concatenate([m.score_samples(Q[:len(Q) // 2]),
                            m.score_samples(Q[len(Q) // 2:])])
    assert _np.allclose(half, bat), "分半批打分与整批不一致——分数仍依赖批的构成"

    # ---- (b) 交付模型未退化：对仓库里实际存在的模型逐个打分
    import glob as _glob
    paths = sorted(_glob.glob(os.path.join(ROOT, "data_out", "model_*.joblib")))
    assert paths, "data_out 下没有 model_*.joblib —— 仪表板与 scan.py 的 ML 层无模型可用"
    worst, per = None, {}
    for p in paths:
        sym = os.path.basename(p)[len("model_"):-len(".joblib")]
        s = AlertModel.load(p).score_samples(Q)
        nuniq = int(len(set(_np.round(s, 4))))       # 先取整再数，见 docstring
        rng_ = float(s.max() - s.min())
        per[sym] = {"n_unique": nuniq, "range": rng_, "mean": float(s.mean())}
        assert nuniq >= 20 and rng_ > 0.1, (
            f"交付模型 model_{sym}.joblib 的异常分退化了："
            f"{len(s)} 个截面只有 {nuniq} 个不同取值、极差 {rng_:.2e}。"
            f"代码可能已修好但模型未重训——请重跑 scripts/build_features.py")
        if worst is None or rng_ < per[worst]["range"]:
            worst = sym

    # ---- (b2) **方向**：原始分越低越异常 ⇒ 映射后的异常分必须单调**递减**于 raw。
    # (a)(b)(c) 都测不出方向：把映射整个翻过来（`return k/len(_ref)`）之后，
    # 分数照样在 [0,1]、照样非退化、四级照样全部可达——**但 ML 层开始给「正常」
    # 截面加级**。实测该注入能同时溜过批不变性、非退化、端到端四级三项。
    for p in paths:
        sym = os.path.basename(p)[len("model_"):-len(".joblib")]
        m2 = AlertModel.load(p)
        raw = m2.iforest.score_samples(m2.scaler.transform(
            _np.nan_to_num(_np.asarray(Q, float))))
        s2 = m2.score_samples(Q)
        o = _np.argsort(raw)                      # raw 升序 = 由异常到正常
        dif = _np.diff(s2[o])
        n_up = int((dif > 1e-12).sum())
        per[sym]["monotone_violations"] = n_up
        assert n_up == 0, (
            f"交付模型 model_{sym}.joblib 的异常分**方向反了**："
            f"按原始分升序（由异常到正常）排列后，映射分出现 {n_up} 处上升。"
            f"原始分越低越异常，故映射分必须单调递减——"
            f"方向反了会让 ML 层给「正常」截面加级，而值域/分散度/四级可达全都看不出来")

    # ---- (c) **端到端**：用真模型走 `alert_engine.evaluate`，四级必须全部可达。
    # (a)(b) 都只测 `AlertModel.score_samples` 这一个函数。而 §7.25 那次事故
    # **暴露在 `evaluate(feats, model=m)` 这条路径上**——四级预警塌成两级。
    # 实测：把 `ml_score = 1.0` 直接写死在 `evaluate` 里（**不碰 alert_model**），
    # ag 等级分布变成 {1:285, 3:91}（0 与 2 双双不可达，事故原样复现），
    # 而 (a)(b) 与 G13 **三道全部 PASS**——因为全仓库没有任何一处用真模型调过 evaluate。
    # 门要盖住事故真正发生的那条路径，不是盖住它的某个零件。
    import pandas as _pd
    from vol_surface.alert_engine import evaluate as _ev
    from vol_surface.features import FEATURE_COLUMNS as _FC
    e2e = {}
    for p in paths:
        sym = os.path.basename(p)[len("model_"):-len(".joblib")]
        ap = os.path.join(ROOT, "data_out", f"alerts_{sym}.parquet")
        if not os.path.exists(ap):
            continue
        df = _pd.read_parquet(ap)
        cols = [c for c in _FC if c in df.columns]
        mm = AlertModel.load(p)
        lv, sc = [], []
        for _, r in df.iterrows():
            f = {c: (0.0 if _pd.isna(r[c]) else float(r[c])) for c in cols}
            res = _ev(f, model=mm)
            lv.append(int(res["level"]))
            sc.append(float(res["ml_score"]))
        sc = _np.asarray(sc)
        seen = sorted(set(lv))
        e2e[sym] = {"levels": seen, "n": len(lv),
                    "score_min": float(sc.min()), "score_max": float(sc.max())}
        assert sc.min() >= 0.0 and sc.max() <= 1.0, (
            f"{sym}: evaluate 返回的 ml_score 越界 [{sc.min():.3f}, {sc.max():.3f}]，"
            f"应在 [0,1]")
        assert len(seen) == 4, (
            f"{sym}: 走 evaluate(feats, model=…) 后只出现了等级 {seen}，"
            f"缺 {sorted(set(range(4)) - set(seen))}——"
            f"四级预警塌成了 {len(seen)} 级，正是 README §7.25 那次事故的形态")

    _DETAILS["G14"] = {"n_slices": int(len(Q)), "max_batch_diff": d,
                       "n_models": len(paths), "per_symbol": per,
                       "worst_symbol": worst, "end_to_end": e2e}
    return (f"{len(Q)} 截面逐样本 vs 整批最大差 {d:.1e}；"
            f"{len(paths)} 个交付模型分散度最低者 {worst}"
            f"；端到端 {len(e2e)} 个品种四级全部可达"
            f"（{per[worst]['n_unique']} 个取值、极差 {per[worst]['range']:.2f}）")


if __name__ == "__main__":
    main()