"""Tiny synthetic probe for RF / IPCARF / RFLDA wrapper classes.

Read-only investigation aid for the felipe audit. Not part of the repo;
written to a scratch/output directory only.

40 samples x 20 features x 3 labels, with one label held constant (all zeros)
in the training fold to check one-class handling and predict_proba shape.
"""
import sys
import time

sys.path.insert(0, "/Users/nazlihosseinipour/03 Projects/Internship/lncrna-gip-project/lncRNA_CIBCB2025-main")

import numpy as np
import pandas as pd

rng = np.random.RandomState(0)
n, p, k = 40, 20, 3
X = pd.DataFrame(rng.rand(n, p))
Y = pd.DataFrame(rng.randint(0, 2, size=(n, k)))
Y.iloc[:, 2] = 0  # constant (one-class) label

X_train, X_test = X.iloc[:30], X.iloc[30:]
Y_train, Y_test = Y.iloc[:30], Y.iloc[30:]

results = {}

from RF.rf import RF
t0 = time.time()
rf = RF(binary_mode=False)
rf.fit(X_train, Y_train)
preds = rf.predict_proba(X_test)
results["RF"] = (np.asarray(preds).shape, time.time() - t0)

from IPCARF.ipcarf import IPCARF
t0 = time.time()
ip = IPCARF(binary_mode=False, n_components=2)
ip.fit(X_train, Y_train)
preds = ip.predict_proba(X_test)
results["IPCARF"] = (np.asarray(preds).shape, time.time() - t0)

t0 = time.time()
ip_paper = IPCARF(binary_mode=False, n_components=2, standardize=False, allow_pca_fallback=False)
ip_paper.fit(X_train, Y_train)
preds = ip_paper.predict_proba(X_test)
results["IPCARF_paper"] = (np.asarray(preds).shape, time.time() - t0)

from RFLDA.rflda import RFLDA
t0 = time.time()
rl = RFLDA(binary_mode=False)
rl.fit(X_train, Y_train)
preds = rl.predict_proba(X_test)
results["RFLDA"] = (np.asarray(preds).shape, time.time() - t0)

t0 = time.time()
rl_paper = RFLDA(binary_mode=False, n_folds=5, selection_metric="subset_accuracy")
rl_paper.fit(X_train, Y_train)
preds = rl_paper.predict_proba(X_test)
results["RFLDA_paper"] = (np.asarray(preds).shape, time.time() - t0)

for name, (shape, dt) in results.items():
    print(f"{name}: predict_proba shape={shape} expected=({X_test.shape[0]}, {k})  time={dt:.2f}s")
print("OK - no exceptions on constant-label training fold")
