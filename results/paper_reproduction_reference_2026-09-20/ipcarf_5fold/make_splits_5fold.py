"""Authors' split_inductive.py logic (MultilabelStratifiedKFold, shuffle, seed 0, hierarchical Y) with n_splits as a parameter.
usage: python make_splits_5fold.py <n_splits> <out.csv>
n_splits==10 also reports whether the output equals the authors' shipped file (it does not, see FIDELITY.md)."""
import sys, pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
n = int(sys.argv[1]); out = sys.argv[2]
D = "/private/tmp/claude-501/-Users-nazlihosseinipour-git-logicalcollections/d46db87b-e760-4f53-8270-db0680928abb/scratchpad/ipcarf_5fold/authors_copy/create_dataset/datasets/"
x = pd.read_csv(D + "features/features_preprocessed.csv", index_col=0)
y = pd.read_csv(D + "hierarchical_labels/mesh_labels_final.csv", index_col=0)
mskf = MultilabelStratifiedKFold(n_splits=n, shuffle=True, random_state=0)
columns = [a for i in range(1, n + 1) for a in ("train_fold%d" % i, "test_fold%d" % i)]
splits = pd.DataFrame([index.astype(int) for indexes in mskf.split(x, y) for index in indexes]).T
splits.columns = columns
splits.to_csv(out, index=False)
if n == 10:
    ref = pd.read_csv(D + "splits/inductive/inductive_splits.csv")
    same = ref.equals(pd.read_csv(out))   # NOT fatal: see FIDELITY.md (does not match with the shipped label matrix)
    print("10-fold regeneration %s authors' shipped inductive_splits.csv" % ("matches" if same else "DIFFERS from"))
