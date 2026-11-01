#!/bin/bash
# Kronos Forecast on macOS (runs from source). Double-click this file, or run it in Terminal.
# First time only: sets up a private Python environment with PyTorch (a few minutes, ~300 MB).
cd "$(dirname "$0")" || exit 1
if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is needed. Install it from https://www.python.org/downloads/ and run this again."; exit 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up (this takes a few minutes, once)..."
  python3 -m venv .venv && ./.venv/bin/pip install --upgrade pip && ./.venv/bin/pip install -r requirements.txt || exit 1
fi
./.venv/bin/python kronos_app.py
