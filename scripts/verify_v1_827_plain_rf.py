"""Independent check of the V1 827 plain-RF run from its saved probabilities.

Own sklearn/numpy implementation (does not import the paper Evaluator): verifies
output hashes against run_contract.json, test-row identities against the released
folds and labels (joined by RNA ID), recomputes the per-label Youden thresholds and
all eight measures per fold, and compares them with the saved values and with the
pre-existing RF summary (results/paper_reproduction_reference_2026-09-20).
Writes <outdir>/verification.json; refuses to overwrite it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, hamming_loss,
                             label_ranking_loss, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets"
EXISTING = ROOT / "results/paper_reproduction_reference_2026-09-20/paper_reproduction_reference.csv"
MEASURES = ["hamming", "label_ranking", "micro_roc", "micro_auprc", "precision", "recall", "fscore", "accuracy"]
EXISTING_KEYS = {"micro_roc": "AUROC", "micro_auprc": "AUPRC", "precision": "P_youden",
                 "recall": "R_youden", "fscore": "F1_youden"}


def youden(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    out = []
    for j in range(y.shape[1]):
        fpr, tpr, t = roc_curve(y[:, j], s[:, j])
        d = tpr - fpr
        out.append(t[np.nanargmax(d)] if d.sum() > 0 else 0.5)
    return np.array(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "results/felipe_paper_exact_2026-09-29T0014/v1_rf_827")
    args = parser.parse_args(argv)
    run = args.run_dir
    target = run.parent / "v1_rf_827_verification.json"
    if target.exists():
        raise FileExistsError(target)
    warnings.filterwarnings("ignore")
    contract = json.loads((run / "run_contract.json").read_text())
    hash_bad = [k for k, v in contract["outputs"].items()
                if hashlib.sha256((run / k).read_bytes()).hexdigest() != v]
    y_all = pd.read_csv(DS / "flat_labels/labels_flat_preprocessed.csv", index_col=0)
    splits = pd.read_csv(DS / "splits/inductive/inductive_splits.csv")
    thresholds = pd.read_csv(run / "rf_mlc_thresholds.csv")
    saved = pd.read_csv(run / "rf_mlc_per_fold.csv")
    rows, th_diff, ids = [], 0.0, []
    for fold in range(1, 11):
        pred = pd.read_csv(run / "predictions" / f"rf_mlc_predictions_test_fold{fold}.csv",
                           index_col=0, float_precision="round_trip")
        positions = pred.pop("test_position").tolist()
        assert positions == splits[f"test_fold{fold}"].dropna().astype(int).tolist()
        assert pred.index.tolist() == y_all.index[positions].tolist()
        assert pred.columns.tolist() == y_all.columns.tolist()
        ids += pred.index.tolist()
        y = y_all.loc[pred.index].to_numpy()  # explicit join by RNA ID
        s = pred.to_numpy()
        th = youden(y, s)
        th_diff = max(th_diff, float(np.abs(th - thresholds[thresholds.fold == fold].threshold.to_numpy()).max()))
        b = (s >= th).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, b, average="micro", zero_division=0)
        rows.append([hamming_loss(y, b), label_ranking_loss(y, b), roc_auc_score(y, s, average="micro"),
                     average_precision_score(y, s, average="micro"), p, r, f, accuracy_score(y, b)])
    recomputed = np.array(rows)
    existing = pd.read_csv(EXISTING, index_col=0).loc["paper_827_full__PAPER_REPRODUCTION"]
    result = {
        "output_hash_mismatches": hash_bad,
        "rows_cover_all_once": len(ids) == len(set(ids)) == len(y_all),
        "threshold_max_abs_diff": th_diff,
        "per_fold_max_abs_diff": float(np.abs(recomputed - saved[MEASURES].to_numpy()).max()),
        "recomputed": {m: {"mean": float(recomputed[:, i].mean()), "std_population": float(recomputed[:, i].std())}
                       for i, m in enumerate(MEASURES)},
        "existing_summary": {m: float(existing[k]) for m, k in EXISTING_KEYS.items()},
    }
    result["matches_existing_summary_3dp"] = all(
        round(result["recomputed"][m]["mean"], 3) == round(v, 3) for m, v in result["existing_summary"].items())
    result["passed"] = (not hash_bad and result["rows_cover_all_once"] and th_diff < 1e-12
                        and result["per_fold_max_abs_diff"] < 1e-12 and result["matches_existing_summary_3dp"])
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
