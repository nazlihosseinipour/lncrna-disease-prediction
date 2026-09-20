#!/bin/bash
# Authors' parse_results/main.py (commit 73b21ee text) + authors' evaluate.py (commit 73b21ee: per-label Youden fit on the test fold).
# usage: run_eval_authors.sh [pred_root] [out_dir]   (defaults: this folder's predictions_*, eval/)
PY="/Users/nazlihosseinipour/Nazli Second Brain/03 Projects/Internship/lncrna-gip-project/.venv/bin/python"
R="/Users/nazlihosseinipour/Nazli Second Brain/03 Projects/Internship/lncrna-gip-project/results/paper_reproduction_reference_2026-09-20/ipcarf"
AC="/private/tmp/claude-501/-Users-nazlihosseinipour-git-logicalcollections/d46db87b-e760-4f53-8270-db0680928abb/scratchpad/ipcarf_fresh/authors_copy"
PR="${1:-$R}"; OUT="${2:-$R/eval}"; mkdir -p "$OUT"
export PYTHONPATH="$AC/parse_results:$AC"
cd "$OUT" && "$PY" "$AC/parse_results/main.py" \
  "$AC/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv" \
  "$PR/predictions_mlc/labels_flat_preprocessed_IPCARF_inductive_multilabel_predictions_test_fold1.csv" \
  false \
  "$AC/create_dataset/datasets/splits/inductive/inductive_splits.csv" \
  "$AC/create_dataset/datasets/hierarchical_labels/mesh_labels_final.csv" \
  "$PR/predictions_hmc/mesh_labels_final_IPCARF_inductive_multilabel_predictions_test_fold1.csv"
