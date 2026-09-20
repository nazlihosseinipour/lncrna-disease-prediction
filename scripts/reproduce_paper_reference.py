"""Reproduce the CIBCB2025 paper's RF numbers from the authors' own released files.

LEAKAGE-INFLATED REFERENCE, NOT A VALID INDUCTIVE RESULT.
Columns 462-746 of the supplied feature file are the raw lncRNA-disease
association matrix (each row's own label vector for rows 0-230, shifted by 1-2
after that). F1/P/R additionally use a per-label Youden threshold fitted on each
test fold's own labels, as in the authors' original evaluate.py (commit 73b21ee).
Never mix these numbers into the clean-pipeline tables without that label.

Nothing here touches the clean pipeline; it only reads the authors' files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets"
OUT = ROOT / "results/paper_reproduction_reference_2026-09-20"
BLOCK = slice(462, 747)  # 285 columns == raw association matrix
# Table III ranges for RFLDA/IPCARF (paper p.7)
PAPER = {"AUROC": (0.76, 0.83), "AUPRC": (0.31, 0.44), "F1": (0.38, 0.48)}

X = pd.read_csv(DS / "features/features_preprocessed.csv", index_col=0)
Y = pd.read_csv(DS / "flat_labels/labels_flat_preprocessed.csv", index_col=0)
S = pd.read_csv(DS / "splits/inductive/inductive_splits.csv")
assert (X.index == Y.index).all() and X.shape == (355, 827) and Y.shape == (355, 44)
Xv, Yv = X.values, Y.values.astype(int)
no_block = np.delete(Xv, np.arange(462, 747), axis=1)


def youden(yt, ys):
    """Per-label Youden cutoff fitted on the evaluated fold's own labels (paper protocol).

    Mirrors the authors' sensivity_specifity_cutoff (evaluate.py @73b21ee): when
    sum(tpr - fpr) is not positive the cutoff falls back to 0.5 (single-class labels
    give NaN, which also falls back to 0.5).
    """
    th = []
    for c in range(yt.shape[1]):
        if len(np.unique(yt[:, c])) < 2:
            th.append(0.5)
            continue
        fpr, tpr, t = roc_curve(yt[:, c], ys[:, c])
        s = tpr - fpr
        th.append(t[np.nanargmax(s)] if s.sum() > 0 else 0.5)
    return np.array(th)


def evaluate(predict):
    rows = []
    for f in range(1, 11):
        tr = S[f"train_fold{f}"].dropna().astype(int).values
        te = S[f"test_fold{f}"].dropna().astype(int).values
        yt, ys = Yv[te], predict(tr, te)
        _, _, f_fixed, _ = precision_recall_fscore_support(
            yt, (ys >= 0.5).astype(int), average="micro", zero_division=0)
        p, r, f1, _ = precision_recall_fscore_support(
            yt, (ys >= youden(yt, ys)).astype(int), average="micro", zero_division=0)
        rows.append([roc_auc_score(yt, ys, average="micro"),
                     average_precision_score(yt, ys, average="micro"), f_fixed, p, r, f1])
    return np.mean(rows, axis=0)


def rf(X_):
    def g(tr, te):
        m = RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=-1)
        m.fit(X_[tr], Yv[tr])
        return m.predict(X_[te])
    return g


def prior(tr, te):
    return np.tile(Yv[tr].mean(axis=0), (len(te), 1))


configs = {
    "prior_only_no_features": prior,
    "paper_542_block_removed": rf(no_block),
    "paper_827_full__PAPER_REPRODUCTION": rf(Xv),
    "block_only_285": rf(Xv[:, BLOCK]),
}
cols = ["AUROC", "AUPRC", "F1_fixed0.5", "P_youden", "R_youden", "F1_youden"]
res = pd.DataFrame({k: evaluate(v) for k, v in configs.items()}, index=cols).T.round(3)

OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / "paper_reproduction_reference.csv")
(OUT / "README.md").write_text(
    "# LEAKAGE-INFLATED reference reproduction of the CIBCB2025 RF numbers\n\n"
    "Not a valid inductive result. See scripts/reproduce_paper_reference.py docstring.\n"
    "* 285 of 827 features (cols 462-746) are the raw association matrix (own labels for rows 0-230).\n"
    "* F1/P/R use a per-label Youden threshold fitted on each test fold's own labels.\n\n"
    + "```\n" + res.to_string() + "\n```\n")
print(res.to_string())

full = res.loc["paper_827_full__PAPER_REPRODUCTION"]
ok = (PAPER["AUROC"][0] <= full["AUROC"] <= PAPER["AUROC"][1]
      and PAPER["AUPRC"][0] <= full["AUPRC"] <= PAPER["AUPRC"][1]
      and PAPER["F1"][0] <= full["F1_youden"] <= PAPER["F1"][1])
print("\nwithin paper Table III ranges:", ok)
sys.exit(0 if ok else 1)
