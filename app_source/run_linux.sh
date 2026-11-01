#!/bin/bash
# Kronos Forecast on Linux (runs from source):  bash run_linux.sh
# First time only: sets up a private Python environment with the CPU build of PyTorch.
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is needed (e.g. sudo apt install python3 python3-venv), then run this again."; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up (this takes a few minutes, once)..."
  python3 -m venv .venv && ./.venv/bin/pip install --upgrade pip \
    && ./.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && ./.venv/bin/pip install -r requirements.txt || exit 1
fi
./.venv/bin/python kronos_app.py
