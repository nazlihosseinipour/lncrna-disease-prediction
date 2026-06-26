#!/usr/bin/env bash
# Feature comparison: within-version 10-fold CV for RFLDA / IPCARF / RF over the
# leakage-free (sequence-only) feature representations, for V1 and V2.
#
# This is the EXPENSIVE step (V2 RFLDA does a nested feature search ~hours/feature).
# Run it in the background / under nohup on the server.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

echo "[1/2] Preparing leakage-free manifest (ALL sequence features, no SVD/GIP/LFS) ..."
"$PYTHON" scripts/prepare_inductive_feature_representations.py \
  --versions v1 v2 --all-compatible \
  --feature-set "kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix" \
  --min-positives 5 --keep-rule gt --n-splits 10 --random-state 0
# --all-compatible sweeps every leakage-free sequence feature; the explicit --feature-set
# adds the concatenated representation. For a QUICK run, replace this with the explicit
# 3-4 --feature-set lines (kmer_matrix_k4 / rc_kmer_matrix_k4 / psednc_matrix / combined).

echo "[2/2] Within-version CV (rflda ipcarf rf) ..."
"$PYTHON" scripts/run_inductive_within_cv.py \
  --models rflda ipcarf rf \
  --n-splits 10 \
  --threshold-mode youden \
  --outdir results/within_version_cv
# Add a leakage-free thresholded pass with: --threshold-mode fixed --outdir results/within_version_cv_fixed

echo "DONE: feature comparison -> results/within_version_cv/"
