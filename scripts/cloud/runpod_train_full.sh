#!/usr/bin/env bash
set -euo pipefail
ROOT="${B2Q_ROOT:-/workspace/brain2qwerty-eeg32-research}"
VENV="${B2Q_VENV:-/workspace/.venv-b2q}"
RESULTS="${B2Q_CLOUD_RESULTS:-/workspace/cloud_results}"
WORKERS="${B2Q_WORKERS:-8}"
source "$(dirname "$0")/runpod_env.sh"
source "$VENV/bin/activate"
cd "$ROOT"
python -m brain2qwerty_v1.main train --modality eeg --condition full \
  --num-workers "$WORKERS" --output-dir /workspace/results/full \
  2>&1 | tee "$RESULTS/full_train.log"
