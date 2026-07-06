#!/usr/bin/env bash
# Model comparison across train/test dataset combinations, using IDENTICAL feature
# representations:  V1->V1,  V2->V2,  V1->V2  (and V2->V1).
#
# This is CHEAP: it only AGGREGATES results already produced by
#   server/run_feature_comparison.sh    (within-version V1->V1, V2->V2)
#   server/run_v1_v2_generalization.sh  (cross-version V1->V2, V2->V1)
# into the model-comparison table and the generalization matrix (with V1->V2 as an
# additional comparison column). Run it AFTER those two have finished.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"

echo "[1/2] Generalization matrix: V1->V1 / V2->V2 / V1->V2 / V2->V1 (per feature x model) ..."
"$PYTHON" scripts/build_generalization_matrix.py
# -> results/generalization_matrix.csv  (columns: feature_set, model, metric, V1->V1, V2->V2, V1->V2, V2->V1)
#    results/generalization_matrix_label_counts.csv

echo "[2/2] Model-comparison table (RFLDA vs IPCARF vs RF, per train-dataset) ..."
"$PYTHON" scripts/build_final_deliverables.py
# -> results/model_comparison.csv, results/feature_comparison.csv, results/final_comparison.csv

echo "DONE: model comparison -> results/generalization_matrix.csv + results/model_comparison.csv"
