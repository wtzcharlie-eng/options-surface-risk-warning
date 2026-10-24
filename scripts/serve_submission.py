"""One Linux entry point. No fitting, browser launch or model download at startup."""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deployment.runtime import load_settings, DeploymentConfigError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--require-model", action="store_true")
    args = ap.parse_args()
    try:
        settings = load_settings(ROOT)
        required = ["data_out/predictive_v3/model.json", "data_out/submission/audit.json",
                    "data_out/drl/agent_seed42.npz", "data_out/kg/kg_graph.json"]
        if any(not (ROOT / p).is_file() for p in required):
            raise DeploymentConfigError("交付模型或审计文件缺失，请重新解压完整项目包；启动时不会自动重训。")
        if not list((ROOT / "data_out/anchor").glob("*/*/features.parquet")):
            raise DeploymentConfigError("缺少随包提供的历史特征数据。")
    except (ValueError, OSError):
        print("部署配置或文件检查失败，请核对 .env、模型与数据文件。", file=sys.stderr)
        raise SystemExit(1)
    if args.check:
        print(json.dumps(settings.public_status(), ensure_ascii=False, indent=2))
        if args.require_model and not settings.model_ready:
            raise SystemExit(2)
        return
    if args.require_model and not settings.model_ready:
        print("平台模型配置未齐，尚不能验证真实模型问答。", file=sys.stderr)
        raise SystemExit(2)
    settings.apply()
    os.chdir(ROOT)
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard/submission_app.py"),
           "--server.address", settings.host, "--server.port", str(settings.port),
           "--server.baseUrlPath", settings.base_path, "--server.headless", "true",
           "--browser.gatherUsageStats", "false"]
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
