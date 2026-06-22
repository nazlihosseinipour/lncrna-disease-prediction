#!/usr/bin/env bash
# Binary (pairwise) model comparison alongside the structured-output runs.
# Each lncRNA-disease pair is one example; lncRNA sequence features are concatenated
# with an ontology-based (leakage-free) disease-similarity vector.
#
# Binary mode expands rows to n_lncRNA x n_disease, so this is heavier than it looks
# on V2. Run under nohup. Use MAX_FOLDS for a quick check first.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
MAX_FOLDS="${MAX_FOLDS:-}"            # e.g. MAX_FOLDS=1 for a quick validation
EXTRA=""
[ -n "$MAX_FOLDS" ] && EXTRA="--max-folds $MAX_FOLDS"

echo "Binary comparison (rflda ipcarf rf), V1 + V2 ..."
"$PYTHON" scripts/run_binary_comparison.py \
  --models rflda ipcarf rf \
  --n-splits 10 \
  --threshold-mode youden \
  $EXTRA \
  --outdir results/binary_comparison

echo "DONE: binary comparison -> results/binary_comparison/"
