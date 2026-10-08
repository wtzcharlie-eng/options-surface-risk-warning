"""drl.dqn — 纯 NumPy 实现的 Double DQN。

为什么不用 PyTorch
------------------
本项目 requirements.txt 原本没有任何深度学习框架，而这里要学的是一个
28→128→128→4 的小 MLP。为一个几万参数的网络引入 torch 依赖不划算，且：
  - 纯 NumPy 可完全确定性复现（固定 seed 即逐位可复现，无 cuDNN 非确定性算子）
  - 训练全程 CPU，几分钟即可跑完，评审复现零环境成本
  - 反向传播代码显式可读，便于核对梯度而非当黑箱
若后续要换更大的网络或连续动作空间，再引入 torch 不迟。

算法要点
--------
- **Double DQN**：用在线网络选动作、目标网络估值，抑制 Q 值高估
  （金融数据信噪比低，高估会直接表现为过度预警）
- **目标网络**：每 `target_sync` 步硬同步一次
- **经验回放**：打散时序相关性
- **Huber 损失**：对离群 TD 误差稳健——事件级奖励里 cover(+9~13) 与 miss(−10)
  都是大数，平方损失会被它们主导
- **ε-greedy**：线性退火，训练末期保留小探索率
"""

from __future__ import annotations

import json
import os

import numpy as np


# ---------------------------------------------------------------- 网络

class MLP:
    """两层隐藏层的 MLP + Adam。手写前反向，便于核验。"""

    def __init__(self, dims: list, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.dims = dims
        self.W, self.b = [], []
        for i in range(len(dims) - 1):
            # He 初始化（ReLU）
            scale = np.sqrt(2.0 / dims[i])
            self.W.append(rng.normal(0, scale, (dims[i], dims[i + 1])))
            self.b.append(np.zeros(dims[i + 1]))
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    def forward(self, X: np.ndarray, cache: bool = False):
        a = X
        acts = [a]
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = np.maximum(z, 0) if i < len(self.W) - 1 else z   # 输出层线性
            acts.append(a)
        return (a, acts) if cache else a

    def backward(self, acts: list, dout: np.ndarray, lr: float,
                 b1=0.9, b2=0.999, eps=1e-8, clip: float = 10.0, wd: float = 0.0):
        """dout: dL/d(输出)，形状 (B, out)。就地做一次 Adam 更新。

        wd: L2 权重衰减（只作用于权重矩阵，不衰减偏置）——缓解小样本上的过拟合。
        """
        self._t += 1
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            grads_W[i] = acts[i].T @ d
            grads_b[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        if wd:
            for i in range(len(self.W)):
                grads_W[i] = grads_W[i] + wd * self.W[i]
        gn = np.sqrt(sum((g ** 2).sum() for g in grads_W + grads_b))
        scale = min(1.0, clip / (gn + 1e-12))
        for i in range(len(self.W)):
            for g, m, v, p in ((grads_W[i] * scale, self._mW, self._vW, self.W),
                               (grads_b[i] * scale, self._mb, self._vb, self.b)):
                m[i] = b1 * m[i] + (1 - b1) * g
                v[i] = b2 * v[i] + (1 - b2) * g * g
                mh = m[i] / (1 - b1 ** self._t)
                vh = v[i] / (1 - b2 ** self._t)
                p[i] -= lr * mh / (np.sqrt(vh) + eps)
        return float(gn)

    def copy_from(self, other: "MLP") -> None:
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]

    def params(self) -> dict:
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        return d

    def load_params(self, d) -> None:
        self.W = [d[f"W{i}"] for i in range(len(self.W))]
        self.b = [d[f"b{i}"] for i in range(len(self.b))]


# ---------------------------------------------------------------- 回放池

class Replay:
    def __init__(self, cap: int, sdim: int, seed: int = 0):
        self.cap = cap
        self.s = np.zeros((cap, sdim), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.s2 = np.zeros((cap, sdim), dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.n, self.ptr = 0, 0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, float(done)
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, bs: int):
        idx = self.rng.integers(0, self.n, size=bs)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


# ---------------------------------------------------------------- 智能体

class DQNAgent:
    """Double DQN 智能体。状态→4 个动作的 Q 值。"""

    def __init__(self, state_dim: int, n_actions: int = 4, hidden: int = 128,
                 lr: float = 1e-3, gamma: float = 0.95, seed: int = 0,
                 weight_decay: float = 0.0):
        self.state_dim, self.n_actions = state_dim, n_actions
        self.gamma, self.lr, self.weight_decay = gamma, lr, weight_decay
        self.q = MLP([state_dim, hidden, hidden, n_actions], seed=seed)
        self.tgt = MLP([state_dim, hidden, hidden, n_actions], seed=seed)
        self.tgt.copy_from(self.q)
        self.rng = np.random.default_rng(seed)
        self.cfg = {"state_dim": state_dim, "n_actions": n_actions, "hidden": hidden,
                    "lr": lr, "gamma": gamma, "seed": seed,
                    "weight_decay": weight_decay}

    # ---- 推理

    def q_values(self, s: np.ndarray) -> np.ndarray:
        return self.q.forward(np.atleast_2d(s).astype(np.float64))

    def act(self, s: np.ndarray, eps: float = 0.0) -> int:
        if eps > 0 and self.rng.random() < eps:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_values(s)[0]))

    def greedy_policy(self):
        """返回可直接喂给 AlertEnv.rollout 的确定性策略。"""
        return lambda s, i: self.act(s, eps=0.0)

    # ---- 训练

    def update(self, batch) -> tuple:
        s, a, r, s2, d = batch
        s, s2 = s.astype(np.float64), s2.astype(np.float64)
        # Double DQN：在线网选 argmax，目标网给估值
        a_star = np.argmax(self.q.forward(s2), axis=1)
        q_next = self.tgt.forward(s2)[np.arange(len(a_star)), a_star]
        target = r + self.gamma * (1.0 - d) * q_next

        pred_all, acts = self.q.forward(s, cache=True)
        idx = np.arange(len(a))
        pred = pred_all[idx, a]
        delta = pred - target
        # Huber：|delta|>1 时梯度饱和为 ±1
        gerr = np.clip(delta, -1.0, 1.0)
        dout = np.zeros_like(pred_all)
        dout[idx, a] = gerr / len(a)
        gn = self.q.backward(acts, dout, self.lr, wd=self.weight_decay)
        loss = float(np.mean(np.where(np.abs(delta) <= 1, 0.5 * delta ** 2,
                                      np.abs(delta) - 0.5)))
        return loss, gn

    def sync_target(self) -> None:
        self.tgt.copy_from(self.q)

    # ---- 存取

    def save(self, path: str, extra: dict | None = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, **self.q.params(),
                 _cfg=json.dumps(self.cfg), _extra=json.dumps(extra or {}))

    @classmethod
    def load(cls, path: str) -> tuple:
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["_cfg"]))
        extra = json.loads(str(z["_extra"]))
        ag = cls(**cfg)
        ag.q.load_params({k: z[k] for k in z.files if k[0] in "Wb" and k != "_cfg"})
        ag.tgt.copy_from(ag.q)
        return ag, extra
