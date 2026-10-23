"""One-time conversion of the repository's trusted frozen sklearn 1.7.2 artifact.

Run in scikit-learn==1.7.2 with pandas/numpy/pyarrow/joblib. Does not train.
The private Cython module alias repairs the source pickle's module path only in
this offline converter. Runtime uses validated numeric JSON and needs no alias.
"""
from pathlib import Path
import hashlib
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import sklearn
import sklearn._loss._loss as loss_module
from vol_surface.portable_rule_ml import PortableRuleML


def main():
    if sklearn.__version__ != "1.7.2":
        raise SystemExit("export requires the original scikit-learn==1.7.2")
    path = ROOT / "data_out/rule_ml_model.joblib"
    fp = json.loads((ROOT / "data_out/rule_ml_model.fingerprint.json").read_text())
    # The supplied pickle names this private Cython extension `_loss`.
    sys.modules.setdefault("_loss", loss_module)
    model = joblib.load(path)
    trees = []
    for stage in model._predictors:
        if len(stage) != 1:
            raise ValueError("only binary numerical models are supported")
        nodes = stage[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical trees are not supported")
        trees.append([[float(n["value"]), int(n["feature_idx"]), float(n["num_threshold"]),
                       int(n["missing_go_to_left"]), int(n["left"]), int(n["right"]),
                       int(n["is_leaf"])] for n in nodes])
    artifact = dict(format="numeric-hist-gradient-boosting-v1", feature_names=fp["feature_names"],
                    baseline=float(model._baseline_prediction[0, 0]), trees=trees,
                    source_joblib_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_sklearn=sklearn.__version__, training_performed=False)
    target = ROOT / "data_out/rule_ml_portable.json"
    target.write_text(json.dumps(artifact, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n")
    portable = PortableRuleML(target)
    max_error, n_rows = 0., 0
    for file in sorted(ROOT.glob("data_out/anchor/*/*/features.parquet")):
        f = pd.read_parquet(file).sort_values("_timestamp").reset_index(drop=True)
        u = pd.read_parquet(file.with_name("underlying.parquet")).sort_values("_timestamp").reset_index(drop=True)
        if f._timestamp.astype(str).tolist() != u._timestamp.astype(str).tolist():
            raise ValueError("unaligned source features")
        x = pd.concat([f, u[[name for name in portable.features if name not in f]]], axis=1)
        x = np.nan_to_num(x[portable.features].to_numpy(float), nan=0., posinf=0., neginf=0.)
        expected, actual = model.predict_proba(x), portable.predict_proba(x)
        max_error = max(max_error, float(np.max(np.abs(actual-expected))))
        n_rows += len(x)
    if max_error > 1e-12:
        target.unlink()
        raise ValueError(f"migration not equivalent: {max_error}")
    proof = dict(source_joblib_sha256=artifact["source_joblib_sha256"], source_sklearn=sklearn.__version__,
                 portable_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 n_compared_rows=n_rows, max_probability_error=max_error,
                 training_performed=False, passed=True)
    (ROOT / "data_out/rule_ml_portable.validation.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
