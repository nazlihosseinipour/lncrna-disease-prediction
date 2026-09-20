#!/bin/bash
# 5 OUTER folds. MLC (flat 44 labels) folds 0-4, then HMC (369 MeSH-expanded labels) folds 0-4, sequentially (n_jobs=-1 inside each fold).
PY="/Users/nazlihosseinipour/Nazli Second Brain/03 Projects/Internship/lncrna-gip-project/.venv/bin/python"
R="/Users/nazlihosseinipour/Nazli Second Brain/03 Projects/Internship/lncrna-gip-project/results/paper_reproduction_reference_2026-09-20/ipcarf_5fold"
AC="/private/tmp/claude-501/-Users-nazlihosseinipour-git-logicalcollections/d46db87b-e760-4f53-8270-db0680928abb/scratchpad/ipcarf_5fold/authors_copy"
export PYTHONHASHSEED=0
date +"START %F %T" > "$R/logs/status.log"
cd "$R/predictions_mlc" && "$PY" -u "$R/run_ipcarf_authors.py" "$AC/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv" 0,1,2,3,4 2>&1 | while IFS= read -r l; do echo "$(date +%T) $l"; done > "$R/logs/mlc_all_folds.log"
date +"MLC_DONE %F %T" >> "$R/logs/status.log"
cd "$R/predictions_hmc" && "$PY" -u "$R/run_ipcarf_authors.py" "$AC/create_dataset/datasets/hierarchical_labels/mesh_labels_final.csv" 0,1,2,3,4 2>&1 | while IFS= read -r l; do echo "$(date +%T) $l"; done > "$R/logs/hmc_all_folds.log"
date +"HMC_DONE %F %T" >> "$R/logs/status.log"
