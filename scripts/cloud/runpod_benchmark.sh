#!/usr/bin/env bash
set -euo pipefail
ROOT="${B2Q_ROOT:-/workspace/brain2qwerty-eeg32-research}"
VENV="${B2Q_VENV:-/workspace/.venv-b2q}"
RESULTS="${B2Q_CLOUD_RESULTS:-/workspace/cloud_results}"
WORKERS="${B2Q_WORKERS:-8}"
RATE="${RUNPOD_HOURLY_USD:?Set RUNPOD_HOURLY_USD to the active Pod's hourly rate}"
mkdir -p "$RESULTS"
source "$VENV/bin/activate"
cd "$ROOT"

nvidia-smi dmon -s pucvmet -d 1 > "$RESULTS/nvidia_benchmark.csv" &
NVIDIA_MON_PID=$!
trap 'kill "$NVIDIA_MON_PID" 2>/dev/null || true' EXIT

for condition in full eeg32; do
  /usr/bin/time -v python -m brain2qwerty_v1.main train \
    --modality eeg --condition "$condition" --max-epochs 1 \
    --limit-train-batches 200 --limit-val-batches 50 --limit-test-batches 50 \
    --num-workers "$WORKERS" --output-dir "/workspace/results/benchmark_$condition" \
    2>&1 | tee "$RESULTS/benchmark_$condition.log"
done

python scripts/cloud/runpod_collect_results.py benchmark --hourly-usd "$RATE" --results-dir "$RESULTS"
