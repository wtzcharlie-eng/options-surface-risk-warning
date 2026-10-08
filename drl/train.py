"""drl.train — DQN 训练循环。

纪律
----
- **只用训练集拟合**：归一化统计量、网络参数都只见训练集（2023-01..2024-12）
- **只用验证集选模**：早停/挑最优 checkpoint 看验证集（2025-01..2025-06）累计奖励
- **测试集只跑一次**：最终报告的数字来自测试集（2025-07..2026-04），全程不参与选择
- 固定 seed，训练日志逐轮落盘，便于复现与查收敛
"""

from __future__ import annotations

import time

import numpy as np

from .baseline import (ConstantPolicy, PeriodicPolicy, RandomPolicy, RulePolicy,
                       load_best_params)
from .env import AlertEnv, RewardSpec, oracle_reward
from .metrics import aggregate
from .dqn import DQNAgent

BLIND_PERIODS = (2, 3, 4, 5, 6, 8, 10, 14, 20)


def evaluate_policy(episodes, norm, spec, policy_factory) -> dict:
    """对一组 episode 跑策略并汇总赛题指标。"""
    ros = [AlertEnv(e, norm, spec).rollout(policy_factory(e)) for e in episodes]
    return aggregate(ros)["micro"]


def evaluate_agent(episodes, norm, spec, agent) -> dict:
    pol = agent.greedy_policy()
    return evaluate_policy(episodes, norm, spec, lambda e: pol)


def snapshot_state(agent, buf, rng, step: int, history: list, best: dict,
                   next_epoch: int) -> dict:
    """把训练的**全部**可变状态打包成可 pickle 的 dict（供断点续训）。

    回放池只存前 `n` 条有效样本，避免把 200k 容量的空槽也写盘。
    """
    def _mlp(m):
        return {"W": [w.copy() for w in m.W], "b": [b.copy() for b in m.b],
                "mW": [x.copy() for x in m._mW], "vW": [x.copy() for x in m._vW],
                "mb": [x.copy() for x in m._mb], "vb": [x.copy() for x in m._vb],
                "t": m._t}
    n = buf.n
    return {
        "q": _mlp(agent.q), "tgt": _mlp(agent.tgt),
        "agent_rng": agent.rng.bit_generator.state,
        "buf": {"s": buf.s[:n].copy(), "a": buf.a[:n].copy(), "r": buf.r[:n].copy(),
                "s2": buf.s2[:n].copy(), "d": buf.d[:n].copy(),
                "n": n, "ptr": buf.ptr, "rng": buf.rng.bit_generator.state},
        "rng": rng.bit_generator.state,
        "step": step, "history": list(history),
        "best": {"val_reward": best["val_reward"], "tag": best["tag"],
                 "params": (None if best["params"] is None
                            else {k: v.copy() for k, v in best["params"].items()})},
        "next_epoch": next_epoch,
    }


def restore_state(st: dict, agent, buf, rng) -> tuple:
    """`snapshot_state` 的逆操作，就地写回 agent / buf / rng。"""
    def _mlp(m, d):
        m.W = [w.copy() for w in d["W"]]
        m.b = [b.copy() for b in d["b"]]
        m._mW = [x.copy() for x in d["mW"]]
        m._vW = [x.copy() for x in d["vW"]]
        m._mb = [x.copy() for x in d["mb"]]
        m._vb = [x.copy() for x in d["vb"]]
        m._t = d["t"]
    _mlp(agent.q, st["q"])
    _mlp(agent.tgt, st["tgt"])
    agent.rng.bit_generator.state = st["agent_rng"]
    b = st["buf"]
    n = b["n"]
    buf.s[:n], buf.a[:n], buf.r[:n] = b["s"], b["a"], b["r"]
    buf.s2[:n], buf.d[:n] = b["s2"], b["d"]
    buf.n, buf.ptr = n, b["ptr"]
    buf.rng.bit_generator.state = b["rng"]
    rng.bit_generator.state = st["rng"]
    return st["step"], list(st["history"]), dict(st["best"]), st["next_epoch"]


def train(
    train_eps, val_eps, norm, spec: RewardSpec | None = None,
    *, epochs: int = 6, hidden: int = 128, lr: float = 1e-3, gamma: float = 0.95,
    weight_decay: float = 0.0,
    batch_size: int = 128, buffer: int = 200_000, warmup: int = 5_000,
    train_every: int = 2, target_sync: int = 1_000, evals_per_epoch: int = 4,
    eps_start: float = 1.0, eps_end: float = 0.05, eps_frac: float = 0.5,
    seed: int = 42, log_every: int = 1, verbose: bool = True,
    resume: dict | None = None, on_epoch_end=None,
) -> tuple:
    """训练并返回 (best_agent, history)。best 按**验证集**累计奖励挑选。

    `evals_per_epoch`：每个 epoch 内均匀评估若干次。实测本任务在约 1 个 epoch 后
    就开始过拟合（训练奖励继续升、验证奖励掉头向下），只在 epoch 末评估会错过峰值，
    故按小批次粒度评估并保留最优 checkpoint。

    断点续训（`resume` / `on_epoch_end`）
    ------------------------------------
    完整一次训练约 110 秒，超出本项目沙箱单次调用的时长上限，故支持按 epoch 续跑。
    `on_epoch_end(ep_i, state)` 在每个 epoch 结束时被调用，`state` 由
    `snapshot_state` 产出，可直接 pickle；下次把它传给 `resume` 即从该处继续。

    **续跑必须逐位等价于一次跑完**，否则「分几次跑」就成了另一个算法。为此
    快照里除网络权重外还包含：Adam 一二阶矩与步数 `_t`、目标网权重、回放池的
    全部有效样本与写指针、以及**三个随机数发生器各自的状态**（训练采样序、
    ε-贪心、回放采样）。少存任何一个都会让续跑轨迹发散——
    `drl/tests.py::G9` 用一次跑完 vs 分两次跑的逐位比对守住这一点。
    """
    spec = spec or RewardSpec()
    rng = np.random.default_rng(seed)
    sdim = AlertEnv(train_eps[0], norm, spec).state_dim
    agent = DQNAgent(sdim, hidden=hidden, lr=lr, gamma=gamma, seed=seed,
                     weight_decay=weight_decay)

    from .dqn import Replay
    buf = Replay(buffer, sdim, seed=seed)

    total_steps = epochs * sum(len(e) for e in train_eps)
    decay_steps = max(1, int(total_steps * eps_frac))
    step = 0
    history, best = [], {"val_reward": -np.inf, "params": None, "tag": "-"}
    start_epoch = 0
    if resume is not None:
        step, history, best, start_epoch = restore_state(resume, agent, buf, rng)
    t0 = time.time()
    eps_now = eps_start

    def _checkpoint(tag: str, ep_reward: float, losses: list) -> None:
        nonlocal best
        val = evaluate_agent(val_eps, norm, spec, agent)
        rec = {"tag": tag, "steps": step, "eps": round(eps_now, 4),
               "train_reward": round(ep_reward, 1),
               "loss": round(float(np.mean(losses)) if losses else np.nan, 5),
               "val_reward": round(val["total_reward"], 1),
               "val_precision": round(val["precision"], 4),
               "val_recall": round(val["recall"], 4),
               "val_lead": round(val["avg_lead_min"], 1),
               "val_alert_rate": round(val["alert_rate"], 4),
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        if val["total_reward"] > best["val_reward"]:
            best = {"val_reward": val["total_reward"],
                    "params": {k: v.copy() for k, v in agent.q.params().items()},
                    "tag": tag}
        if verbose:
            print(f"  {tag:>8s} | step {step:7d} | eps {eps_now:.3f} | "
                  f"loss {rec['loss']:.4f} | train R {rec['train_reward']:8.0f} | "
                  f"VAL R {rec['val_reward']:8.0f} "
                  f"P {val['precision']:.1%} R {val['recall']:.1%} "
                  f"lead {val['avg_lead_min']:.0f}m rate {val['alert_rate']:.1%} | "
                  f"{rec['sec']:.0f}s", flush=True)

    for ep_i in range(start_epoch, epochs):
        order = rng.permutation(len(train_eps))
        ep_reward, losses = 0.0, []
        marks = {int(len(order) * (j + 1) / evals_per_epoch) - 1
                 for j in range(evals_per_epoch)}
        for pos, k in enumerate(order):
            env = AlertEnv(train_eps[k], norm, spec)
            s = env.reset()
            done = False
            while not done:
                eps_now = max(eps_end, eps_start - (eps_start - eps_end) * step / decay_steps)
                a = agent.act(s, eps=eps_now) if step >= warmup else int(rng.integers(0, 4))
                s2, r, done, _ = env.step(a)
                buf.push(s, a, r, s2, done)
                s = s2
                ep_reward += r
                step += 1
                if step >= warmup and step % train_every == 0:
                    loss, _ = agent.update(buf.sample(batch_size))
                    losses.append(loss)
                if step % target_sync == 0:
                    agent.sync_target()
            if pos in marks:
                _checkpoint(f"e{ep_i}.{sorted(marks).index(pos)}", ep_reward, losses)
        if on_epoch_end is not None:
            on_epoch_end(ep_i, snapshot_state(agent, buf, rng, step, history,
                                              best, ep_i + 1))

    if best["params"] is not None:
        agent.q.load_params(best["params"])
        agent.sync_target()
    if verbose:
        print(f"  选中 checkpoint {best['tag']}（验证集累计奖励 {best['val_reward']:.0f}）")
    return agent, history


def hit_base_rate(episodes) -> float:
    """hit 的基础率 = 无判别力策略的精确率天花板。

    任何策略的精确率若不显著高于它，就说明该策略并没有真正在"识别"风险，
    只是在调节预警节奏。这是区分"学到信号"与"刷召回"的唯一硬指标。
    """
    import numpy as _np
    from .env import LEAD_WINDOW_MIN
    return float(_np.concatenate(
        [(e.lead_min <= LEAD_WINDOW_MIN).astype(float) for e in episodes]).mean())


def best_blind(episodes, norm, spec, periods=BLIND_PERIODS) -> tuple:
    """盲节奏策略族里在**该数据集上**最优的一条（对基线取优，对自己的结论从严）。"""
    best = (None, None)
    for p in periods:
        m = evaluate_policy(episodes, norm, spec, lambda e, p=p: PeriodicPolicy(p))
        if best[1] is None or m["total_reward"] > best[1]["total_reward"]:
            best = (p, m)
    return best


def baseline_table(episodes, norm, spec, bp: dict | None = None) -> dict:
    """跑齐全部对照策略，返回 {名称: 指标}。"""
    bp = load_best_params() if bp is None else bp
    out = {}
    out["规则基线(校准阈值)"] = evaluate_policy(episodes, norm, spec,
                                                lambda e: RulePolicy(e, params=bp))
    period, m_blind = best_blind(episodes, norm, spec)
    out[f"最佳盲节奏(每{period}步,不看特征)"] = m_blind
    out["恒不预警(level0)"] = evaluate_policy(episodes, norm, spec, lambda e: ConstantPolicy(0))
    out["恒警告(level2)"] = evaluate_policy(episodes, norm, spec, lambda e: ConstantPolicy(2))
    out["恒危险(level3)"] = evaluate_policy(episodes, norm, spec, lambda e: ConstantPolicy(3))
    out["随机策略"] = evaluate_policy(episodes, norm, spec, lambda e: RandomPolicy(seed=0))
    out["预言机(上界)"] = {"total_reward": sum(oracle_reward(e, norm, spec) for e in episodes),
                            "precision": float("nan"), "recall": float("nan"),
                            "avg_lead_min": float("nan"), "n_alert": -1, "alert_rate": float("nan")}
    return out
