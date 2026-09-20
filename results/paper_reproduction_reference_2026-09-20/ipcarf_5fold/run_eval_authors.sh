#!/bin/bash
# Authors' parse_results/main.py (73b21ee) + evaluate.py (73b21ee: per-label Youden fit on the test fold), 5 outer folds.
# usage: run_eval_authors.sh [pred_root] [out_dir]   (defaults: this folder's predictions_*, eval/)
PY="/Users/nazlihosseinipour/Nazli Second Brain/03 Projects/Internship/lncrna-gip-project/.venv/bin/python"
R="/Users/nazlihosseinipour/Nazli Second Brain/03 Projects/Internship/lncrna-gip-project/results/paper_reproduction_reference_2026-09-20/ipcarf_5fold"
AC="/private/tmp/claude-501/-Users-nazlihosseinipour-git-logicalcollections/d46db87b-e760-4f53-8270-db0680928abb/scratchpad/ipcarf_5fold/authors_copy"
PR="${1:-$R}"; OUT="${2:-$R/eval}"; mkdir -p "$OUT"
cd "$OUT" && "$PY" "$R/run_eval_authors.py" \
  "$AC/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv" \
  "$PR/predictions_mlc/labels_flat_preprocessed_IPCARF_inductive_multilabel_predictions_test_fold1.csv" \
  false \
  "$R/inductive_splits_5fold.csv" \
  "$AC/create_dataset/datasets/hierarchical_labels/mesh_labels_final.csv" \
  "$PR/predictions_hmc/mesh_labels_final_IPCARF_inductive_multilabel_predictions_test_fold1.csv"
