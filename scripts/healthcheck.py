"""Minimal container liveness check; never implies LLM or business readiness."""
from pathlib import Path
import sys
from urllib.request import urlopen
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deployment.runtime import load_settings

if __name__ == "__main__":
    s = load_settings(ROOT)
    prefix = "/" + s.base_path if s.base_path else ""
    with urlopen(f"http://127.0.0.1:{s.port}{prefix}/_stcore/health", timeout=4) as r:
        assert r.status == 200 and r.read().strip() == b"ok"
