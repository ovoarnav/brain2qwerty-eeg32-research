#!/usr/bin/env bash
set -euo pipefail
ROOT="${B2Q_ROOT:-/workspace/brain2qwerty-eeg32-research}"
VENV="${B2Q_VENV:-/workspace/.venv-b2q}"
RESULTS="${B2Q_CLOUD_RESULTS:-/workspace/cloud_results}"
WORKERS="${B2Q_WORKERS:-8}"
RATE="${RUNPOD_HOURLY_USD:?Set RUNPOD_HOURLY_USD to the active Pod hourly rate}"
source "$(dirname "$0")/runpod_env.sh"
mkdir -p "$RESULTS"
source "$VENV/bin/activate"
cd "$ROOT"

nvidia-smi dmon -s pucvmet -d 1 > "$RESULTS/nvidia_benchmark.csv" &
NVIDIA_MON_PID=$!
trap "kill $NVIDIA_MON_PID 2>/dev/null || true" EXIT

for condition in full eeg32; do
  log="$RESULTS/benchmark_$condition.log"
  start_seconds="$(python -c 'import time; print(time.perf_counter())')"
  set +e
  python -m brain2qwerty_v1.main train \
    --modality eeg --condition "$condition" --max-epochs 1 \
    --limit-train-batches 200 --limit-val-batches 50 --limit-test-batches 50 \
    --num-workers "$WORKERS" --output-dir "/workspace/results/benchmark_$condition" \
    2>&1 | tee "$log"
  train_status="${PIPESTATUS[0]}"
  set -e
  end_seconds="$(python -c 'import time; print(time.perf_counter())')"
  python - "$start_seconds" "$end_seconds" <<'PY' | tee -a "$log"
import sys
print(f"B2Q_ELAPSED_SECONDS: {float(sys.argv[2]) - float(sys.argv[1]):.6f}")
PY
  if [[ "$train_status" -ne 0 ]]; then
    exit "$train_status"
  fi
done

python scripts/cloud/runpod_collect_results.py benchmark --hourly-usd "$RATE" --results-dir "$RESULTS"
