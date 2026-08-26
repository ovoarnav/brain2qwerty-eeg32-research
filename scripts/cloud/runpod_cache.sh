#!/usr/bin/env bash
set -euo pipefail
ROOT="${B2Q_ROOT:-/workspace/brain2qwerty-eeg32-research}"
VENV="${B2Q_VENV:-/workspace/.venv-b2q}"
WORKERS="${B2Q_WORKERS:-8}"
source "$(dirname "$0")/runpod_env.sh"
source "$VENV/bin/activate"
cd "$ROOT"
for condition in full eeg32; do
  python -m brain2qwerty_v1.main cache --debug --modality eeg --condition "$condition" --num-workers "$WORKERS"
done
for condition in full eeg32; do
  python -m brain2qwerty_v1.main cache --modality eeg --condition "$condition" --num-workers "$WORKERS"
done
