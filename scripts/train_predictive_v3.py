"""Train the fixed V3 development walk-forward experiment and final vintage."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vol_surface.predictive import load_anchor_data
from vol_surface.predictive_v3_validation import readable_report, train_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data_out/anchor"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_out/predictive_v3"))
    parser.add_argument("--v2-predictions", type=Path, default=Path("data_out/predictive_v2/test_predictions.parquet"))
    parser.add_argument("--symbols", help="Optional comma-separated research subset")
    args = parser.parse_args()
    history = load_anchor_data(args.data_dir, args.symbols.split(",") if args.symbols else None)
    print(f"Loaded {len(history):,} raw snapshots; starting fixed development walk-forward", flush=True)
    artifact, report, predictions = train_models(history, progress=lambda message: print(message, flush=True), v2_predictions_path=args.v2_predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_text = json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False)
    (args.output_dir / "model.json").write_text(model_text, encoding="utf-8")
    report["model_sha256"] = hashlib.sha256(model_text.encode()).hexdigest()
    report["training_command"] = " ".join(sys.argv)
    report["artifact_format"] = "JSON numeric arrays only; no executable deserialization"
    root = Path(__file__).resolve().parents[1]
    report["algorithm_source_sha256"] = {name: hashlib.sha256((root / "vol_surface" / name).read_bytes()).hexdigest()
                                         for name in ("predictive.py", "predictive_v2.py", "predictive_validation.py", "predictive_v3.py", "predictive_v3_validation.py")}
    report["training_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    import numpy, pandas, scipy
    report["runtime_versions"] = {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__, "scipy": scipy.__version__}
    report["design_sha256"] = hashlib.sha256((root / "docs/V3实验设计.md").read_bytes()).hexdigest()
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.output_dir / "report.md").write_text(readable_report(report), encoding="utf-8")
    predictions.to_parquet(args.output_dir / "test_predictions.parquet", index=False)
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(f"Saved fixed final vintage and all three fold predictions to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
