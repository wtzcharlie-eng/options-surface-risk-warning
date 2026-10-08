"""alert_model — 轻量级 IsolationForest 异常检测。

用多变量特征向量训练 IsolationForest，输出异常分（0-1）。规则引擎擅长可解释的
单指标触发，ML 层擅长捕捉多变量复合畸变（例如期限结构 + 集中度 + 套利同时轻微
异常，单看都不超阈，合在一起已异常）。两层在 alert_engine 融合。

模型可持久化（joblib），由 build_features 离线训练、scan/dashboard 在线推理。

**sklearn 是可选依赖**：本模块把 sklearn 的导入推迟到真正要训练/加载模型时。
这样纯规则路径（`evaluate(..., model=None)`）在没装 sklearn 的环境里也能跑——
问题1 的 7 品种对照评测正需要这一点：新扩的 lc/cu/rb 没有训练好的 IF 模型，
若让 4 个老品种走「规则+ML」而新品种只走规则，跨品种比较就不成立了。
"""

from __future__ import annotations

import numpy as np

MODEL_FEATURES = [
    "convexity_violation", "skew_val", "term_slope", "term_slope_roc",
    "gamma_concentration", "vega_concentration", "atm_iv", "iv_change_rate",
    "fit_degradation", "arb_score", "liquidity_ratio", "iv_near", "iv_far",
]


def _sklearn():
    """按需导入 sklearn，给出可操作的报错而不是裸 ImportError。"""
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:                                   # noqa: BLE001
        raise ImportError(
            "AlertModel 需要 scikit-learn（仅 ML 层需要，纯规则路径不需要）。"
            "请 `pip install scikit-learn`，或改用 use_model=False 只跑规则层。"
        ) from e
    return IsolationForest, StandardScaler


class AlertModel:
    """IsolationForest + StandardScaler 封装。"""

    def __init__(self, n_estimators: int = 200, contamination: float = 0.05, random_state: int = 42):
        IsolationForest, StandardScaler = _sklearn()
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=random_state
        )
        self.fitted = False
        self._ref: np.ndarray | None = None   # 训练集 raw 分数的分位网格，见 score_samples

    # 参照分布的采样点数（存进 joblib，故不宜过大；1001 点分位精度已远超阈值需求）
    _REF_N = 1001

    def fit(self, X: np.ndarray) -> "AlertModel":
        X = np.asarray(X, float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.scaler.fit(X)
        self.iforest.fit(self.scaler.transform(X))
        # **标定参照分布**：记下训练集 raw 分数的分位网格，供打分时算分位。
        # 这一步是 score_samples 与批大小无关的前提，见该函数的说明。
        raw = self.iforest.score_samples(self.scaler.transform(X))
        self._ref = np.quantile(raw, np.linspace(0.0, 1.0, self._REF_N))
        self.fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """返回异常分 0-1（越大越异常）——**与批大小无关**。

        分数的定义是「训练集里有多大比例的样本比它更正常」，即对训练集 raw 分数
        分布取分位。同一个截面单独打分与混在一批里打分得到同一个数。

        为什么不能用批内归一化（一处已修复的缺陷，见 README §7.25）
        ------------------------------------------------------
        原实现是 `norm / norm.max()`——**批内相对**。而生产路径
        （`alert_engine.evaluate` → `model.featurize(feats).reshape(1, -1)`、
        `scan.py`、`build_features.py`）**每次只传一个样本**，于是分子分母是同一个数，
        异常分**恒等于 1.0**。实测落盘的 2,191 行 `ml_score` 全部为 1.0。

        后果是 `_ml_level` 恒走「ml_score > ML_HIGH」分支，于是
        `level = max(rule_level, ml_level)` 把 rule_level 0→1、2→3 全线抬升：
        **仪表板从不显示「正常」，且「预警 WARN」这一级结构上不可达**，
        四级预警退化成两级——而且是朝「系统看起来一直在报警」的方向退化。

        模型本身没问题：同一模型整批打分时，真异常点均分 0.98、正常点 0.46。
        """
        X = np.asarray(X, float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.fitted:
            return np.zeros(X.shape[0])
        if getattr(self, "_ref", None) is None:
            raise RuntimeError(
                "该 AlertModel 缺少标定参照分布，说明它由修复前的版本训练："
                "那一版的异常分是**批内相对**的，单样本调用恒返回 1.0（见 README §7.25）。"
                "请重跑 `python scripts/build_features.py --symbols <品种>` 重新训练。"
            )
        raw = self.iforest.score_samples(self.scaler.transform(X))  # 越小越异常
        # raw 越小越异常 ⇒ 异常分 = 训练集中 raw 不小于它的比例
        k = np.searchsorted(self._ref, raw, side="left")
        return 1.0 - k / len(self._ref)

    def featurize(self, feats: dict) -> np.ndarray:
        return np.array([feats.get(k, 0.0) for k in MODEL_FEATURES], float)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump({"scaler": self.scaler, "iforest": self.iforest,
                     "fitted": self.fitted, "ref": self._ref}, path)

    @classmethod
    def load(cls, path: str) -> "AlertModel":
        import joblib
        obj = joblib.load(path)
        m = cls()
        m.scaler, m.iforest, m.fitted = obj["scaler"], obj["iforest"], obj["fitted"]
        # `ref` 缺失 ⇒ 修复前训练的模型。**不静默降级**：那一版单样本恒返回 1.0，
        # 悄悄用下去只会让「四级退化成两级」再次发生且无人察觉。
        # 这里只标记，由 score_samples 在真正要用时抛出可操作的报错。
        m._ref = obj.get("ref")
        return m
