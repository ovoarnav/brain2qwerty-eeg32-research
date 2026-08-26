#!/usr/bin/env bash
set -euo pipefail

# Cloud-only setup. It deliberately leaves the scientific configuration intact.
ROOT="${B2Q_ROOT:-/workspace/brain2qwerty-eeg32-research}"
VENV="${B2Q_VENV:-/workspace/.venv-b2q}"
COMMIT="8a43f03c82bf889e989e37f48f38c9ed2c630674"

mkdir -p /workspace/{data,results,b2q_cache,hf_cache,cloud_results}
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-/workspace/data/SpanishBCBL}"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-/workspace/b2q_cache}"
export BRAIN2QWERTY_RESULTS="${BRAIN2QWERTY_RESULTS:-/workspace/results}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

if [[ ! -d "$ROOT/.git" ]]; then
  echo "Clone the private repository into $ROOT first, then rerun this script." >&2
  exit 2
fi

cd "$ROOT"
git fetch --all --tags
git checkout experiment/b2q-v1-eeg32-runpod
git pull --ff-only origin experiment/b2q-v1-eeg32-runpod
git merge-base --is-ancestor "$COMMIT" HEAD

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.12
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$VENV/bin/python" -e '.[dev]'
