"""Numeric-only frozen histogram boosting inference; no pickle or sklearn load."""
from pathlib import Path
import json
import numpy as np


class PortableRuleML:
    def __init__(self, path):
        self.artifact = json.loads(Path(path).read_text())
        if self.artifact["format"] != "numeric-hist-gradient-boosting-v1":
            raise ValueError("unsupported model format")
        self.features = self.artifact["feature_names"]

    def predict_proba(self, X):
        x = np.asarray(X, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.features):
            raise ValueError("ordered feature matrix does not match fingerprint")
        scores = np.full(len(x), self.artifact["baseline"], dtype=float)
        for nodes in self.artifact["trees"]:
            at = np.zeros(len(x), dtype=int)
            pending = np.ones(len(x), dtype=bool)
            while pending.any():
                for node_id in np.unique(at[pending]):
                    selected = pending & (at == node_id)
                    value, feature, threshold, missing_left, left, right, leaf = nodes[node_id]
                    if leaf:
                        scores[selected] += value
                        pending[selected] = False
                    else:
                        v = x[selected, feature]
                        go_left = np.where(np.isnan(v), missing_left, v <= threshold)
                        at[selected] = np.where(go_left, left, right)
        positive = 1. / (1. + np.exp(-np.clip(scores, -700, 700)))
        return np.column_stack([1. - positive, positive])
