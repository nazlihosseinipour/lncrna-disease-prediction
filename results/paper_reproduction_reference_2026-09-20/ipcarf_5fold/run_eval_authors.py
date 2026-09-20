"""Authors' parse_results/main.py (commit 73b21ee text, verbatim) for 5 OUTER folds.
Patches, and nothing else: `n_folds = 10` -> 5; iterator_cross_validation default 10 -> 5;
Evaluator.columns_to_add_df = fold1..fold5 + mean(std). evaluate.py itself (73b21ee, per-label Youden fit on the test fold) is untouched.
usage (cwd = output dir): python run_eval_authors.py <flat_labels> <flat_pred_fold1> false <splits> <hier_labels> <hier_pred_fold1>"""
import os, sys
NOUTER = 5
AC = "/private/tmp/claude-501/-Users-nazlihosseinipour-git-logicalcollections/d46db87b-e760-4f53-8270-db0680928abb/scratchpad/ipcarf_5fold/authors_copy"
sys.path[:0] = [AC + "/parse_results", AC]
import utils.utils as U
from evaluate import Evaluator
assert U.iterator_cross_validation.__defaults__ == (10,)
U.iterator_cross_validation.__defaults__ = (NOUTER,)
Evaluator.columns_to_add_df = ["fold%d" % i for i in range(1, NOUTER + 1)] + ["mean(std)"]
src = open(AC + "/parse_results/main.py").read()
assert src.count("n_folds = 10") == 1
src = src.replace("n_folds = 10", "n_folds = %d" % NOUTER)
sys.argv = ["main.py"] + sys.argv[1:]
exec(compile(src, "parse_results/main.py", "exec"), {"__name__": "__main__"})
