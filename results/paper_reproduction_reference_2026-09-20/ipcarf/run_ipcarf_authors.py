"""Run the authors' IPCARF/main.py (verbatim source text) for a chosen subset of folds.

The only textual change to the authors' script: `for fold in range(n_folds):` becomes
`for fold in FOLDS:` so a crashed run can be resumed. Everything else (IPCARF class,
iterator_cross_validation, predict_proba, CSV writer) is the authors' own code.
Output CSVs land in the current directory (cd into predictions_mlc/ or predictions_hmc/ first).

usage: python run_ipcarf_authors.py <labels.csv> <comma-separated 0-based folds>
LEAKAGE-INFLATED REFERENCE: features_preprocessed.csv contains a 285-column association-matrix block.
"""
import os, sys
AC = "/private/tmp/claude-501/-Users-nazlihosseinipour-git-logicalcollections/d46db87b-e760-4f53-8270-db0680928abb/scratchpad/ipcarf_fresh/authors_copy"
D = AC + "/create_dataset/datasets/"
labels, folds = sys.argv[1], [int(f) for f in sys.argv[2].split(",")]
src = open(AC + "/IPCARF/main.py").read()
assert src.count("for fold in range(n_folds):") == 1
src = src.replace("for fold in range(n_folds):", "for fold in FOLDS:")
sys.path[:0] = [AC + "/IPCARF", AC]
sys.argv = ["main.py", D + "features/features_preprocessed.csv", labels,
            D + "splits/inductive/inductive_splits.csv", "false"]   # binary_mode=false -> multi-label
exec(compile(src, "IPCARF/main.py", "exec"), {"__name__": "__main__", "FOLDS": folds})
