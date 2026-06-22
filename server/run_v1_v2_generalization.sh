#!/usr/bin/env bash
# Cross-dataset generalization (disease-space transfer) over the shared, normalized
# disease label space. Trains on all source-version lncRNAs, tests on all
# target-version lncRNAs (disjoint samples -> clean inductive test).
#   V1 -> V2  (highest priority)
#   V2 -> V1
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

echo "[1/2] V1 -> V2 transfer ..."
"$PYTHON" scripts/run_inductive_transfer_experiments.py \
  --source-version v1 --target-version v2 \
  --models rflda ipcarf rf \
  --label-match normalized --label-space both --min-positives 1 --keep-rule gt \
  --n-splits 10 --random-state 0 \
  --skip-target-cv \
  --outdir results/transfer

echo "[2/2] V2 -> V1 transfer (+ V1 common-space CV baseline) ..."
"$PYTHON" scripts/run_inductive_transfer_experiments.py \
  --source-version v2 --target-version v1 \
  --models rflda ipcarf rf \
  --label-match normalized --label-space both --min-positives 1 --keep-rule gt \
  --n-splits 10 --random-state 0 \
  --outdir results/transfer_v2_to_v1

echo "DONE: transfer -> results/transfer/ and results/transfer_v2_to_v1/"
