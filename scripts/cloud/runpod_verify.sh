#!/usr/bin/env bash
set -euo pipefail

ROOT="${B2Q_ROOT:-/workspace/brain2qwerty-eeg32-research}"
VENV="${B2Q_VENV:-/workspace/.venv-b2q}"
RESULTS="${B2Q_CLOUD_RESULTS:-/workspace/cloud_results}"
mkdir -p "$RESULTS"
source "$VENV/bin/activate"
cd "$ROOT"

{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  nvidia-smi
  nproc
  free -h
  df -h /workspace
  python --version
  python - <<'PY'
import torch
print('torch_version:', torch.__version__)
print('cuda_runtime:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
assert torch.__version__.startswith('2.6.0'), torch.__version__
assert torch.cuda.is_available(), 'CUDA is not available'
assert torch.cuda.device_count() >= 1
print('gpu:', torch.cuda.get_device_name(0))
print('vram_gb:', torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY
} | tee "$RESULTS/environment.txt"

pytest tests -q | tee "$RESULTS/pytest.log"
