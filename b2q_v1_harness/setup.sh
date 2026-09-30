#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv-lab
if [ "$(uname -s)" = Darwin ]; then
  .venv-lab/bin/python -m pip install torch==2.10.0 numpy==2.3.5
else
  .venv-lab/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
  .venv-lab/bin/python -m pip install numpy==2.3.5
fi
.venv-lab/bin/python lab.py doctor
.venv-lab/bin/python lab.py selftest
