#!/usr/bin/env bash
# Shared cloud paths.  These locations are intentionally outside the git worktree.
export BRAIN2QWERTY_STUDIES="${BRAIN2QWERTY_STUDIES:-/workspace/data/SpanishBCBL}"
export BRAIN2QWERTY_CACHE="${BRAIN2QWERTY_CACHE:-/workspace/b2q_cache}"
export BRAIN2QWERTY_RESULTS="${BRAIN2QWERTY_RESULTS:-/workspace/results}"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

mkdir -p "$BRAIN2QWERTY_STUDIES" "$BRAIN2QWERTY_CACHE" "$BRAIN2QWERTY_RESULTS" "$HF_HOME" \
  "${B2Q_CLOUD_RESULTS:-/workspace/cloud_results}"
