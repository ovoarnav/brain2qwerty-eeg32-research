#!/usr/bin/env bash
# Reproduce the original pinned V1 environment without changing the small harness.
set -euo pipefail
task_repo=$(cd "$(dirname "$0")/../.." && pwd)
export UV_CACHE_DIR=/workspace/.cache/uv
export CC=gcc CXX=g++
task_python=/workspace/venvs/brain2qwerty/bin/python
mkdir -p /workspace/venvs /home/agent/.cache/neuralset
if [ ! -x "$task_python" ]; then
  uv venv /workspace/venvs/brain2qwerty --python python3.12
fi
cd "$task_repo"
uv pip install --python "$task_python" --build-constraint b2q_v1_harness/tools/native-build-constraints.txt -r requirements.lock pytest==8.4.2
uv pip install --python "$task_python" -e . --no-deps
uv pip check --python "$task_python"
