"""Run the authors' IPCARF/main.py (verbatim source text) for a chosen subset of OUTER folds, 5-FOLD VARIANT.

Differences from the 10-fold driver (../ipcarf/run_ipcarf_authors.py), and nothing else:
  * splits file = ./inductive_splits_5fold.csv (5 outer folds) instead of the authors' 10-fold file;
  * utils.utils.iterator_cross_validation default n_folds 10 -> 5 (it reads train_fold1..N columns);
  * `for fold in range(n_folds):` becomes `for fold in FOLDS` (as in the 10-fold driver).
Everything else (IPCARF class with 5-fold inner CV, 150 trees, grid 2..128) is the authors' own code, untouched.
Output CSVs land in the current directory (cd into predictions_mlc/ or predictions_hmc/ first).

usage: python run_ipcarf_authors.py <labels.csv> <comma-separated 0-based folds>
LEAKAGE-INFLATED REFERENCE: features_preprocessed.csv contains a 285-column association-matrix block.
"""
import os, sys
NOUTER = 5
HERE = os.path.dirname(os.path.abspath(__file__))
AC = "/private/tmp/claude-501/-Users-nazlihosseinipour-git-logicalcollections/d46db87b-e760-4f53-8270-db0680928abb/scratchpad/ipcarf_5fold/authors_copy"
D = AC + "/create_dataset/datasets/"
SPLITS = HERE + "/inductive_splits_5fold.csv"
labels, folds = sys.argv[1], [int(f) for f in sys.argv[2].split(",")]
assert all(0 <= f < NOUTER for f in folds)
src = open(AC + "/IPCARF/main.py").read()
assert src.count("for fold in range(n_folds):") == 1
src = src.replace("for fold in range(n_folds):", "for fold in FOLDS:")
sys.path[:0] = [AC + "/IPCARF", AC]
import utils.utils as U
assert U.iterator_cross_validation.__defaults__ == (10,)
U.iterator_cross_validation.__defaults__ = (NOUTER,)
sys.argv = ["main.py", D + "features/features_preprocessed.csv", labels, SPLITS, "false"]   # binary_mode=false -> multi-label
os.chdir(os.getcwd())
exec(compile(src, "IPCARF/main.py", "exec"), {"__name__": "__main__", "FOLDS": folds})
