#!/bin/zsh
set -e
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements-predictive.lock.txt
  else
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements-predictive.lock.txt
  fi
fi
if [[ ! -f data_out/predictive_v2/model.json ]]; then
  .venv/bin/python scripts/train_predictive_v2.py
fi
open http://127.0.0.1:8502
exec .venv/bin/python -m streamlit run dashboard/predictive_app.py --server.address 127.0.0.1 --server.port 8502
