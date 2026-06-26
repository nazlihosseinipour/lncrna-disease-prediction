"""Re-score already-saved within-version fold predictions under a chosen threshold mode.

No retraining: reads results/within_version_cv/predictions/<cell>/<model>/test_fold*.csv,
recovers the matching y_test from the manifest's Y + splits, and re-evaluates with the
inductive Evaluator. Used to produce a leakage-free (fixed-0.5) thresholded variant of the
metrics alongside the legacy Youden numbers. Threshold-free metrics (AUROC/AUPRC) are
identical across modes by construction; only the thresholded metrics change.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS = ["hamming", "label_ranking", "micro_roc", "micro_auprc",
           "precision", "recall", "fscore", "accuracy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",
                   default="inductive_inputs/feature_representations/feature_representation_manifest.csv")
    p.add_argument("--project-dir", default="lncRNA_CIBCB2025-main")
    p.add_argument("--pred-root", default="results/within_version_cv/predictions")
    p.add_argument("--models", nargs="+", default=["rflda", "ipcarf", "rf"])
    p.add_argument("--threshold-mode", choices=["youden", "fixed"], default="fixed")
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--outdir", default="results/within_version_cv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = PROJECT_ROOT / args.project_dir
    sys.path.insert(0, (project_dir / "parse_results").as_posix())
    sys.path.insert(0, project_dir.as_posix())
    from parse_results.evaluate import Evaluator
    from utils.utils import iterator_cross_validation

    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
    pred_root = PROJECT_ROOT / args.pred_root
    rows = []

    for _, r in manifest.iterrows():
        version, feature_key = r["version"], r["feature_key"]
        cell = f"{version}_{feature_key}"
        y = pd.read_csv(Path(r["y_path"]), index_col=0)
        splits = pd.read_csv(Path(r["split_path"]))
        dummy = pd.DataFrame(index=y.index)
        _, test = iterator_cross_validation(splits, dummy, y, n_folds=args.n_splits)

        for model in args.models:
            cell_dir = pred_root / cell / model
            if not cell_dir.exists():
                continue
            ev = Evaluator(threshold_mode=args.threshold_mode)
            perf = []
            for fold in range(1, args.n_splits + 1):
                fp = cell_dir / f"test_fold{fold}.csv"
                if not fp.exists():
                    break
                pred = pd.read_csv(fp, index_col=0)
                y_test = test[fold - 1][1]
                pred.columns = y_test.columns
                pred.index = y_test.index
                perf.append(ev.evaluate(y_test, pred))
            if not perf:
                continue
            mean = pd.DataFrame(perf, columns=METRICS).mean()
            row = {"experiment": f"within_{version}_cv", "version": version,
                   "feature_set": r["feature_set"], "feature_key": feature_key,
                   "model": model, "threshold_mode": args.threshold_mode}
            for m in METRICS:
                row[f"{m}_mean"] = round(float(mean[m]), 4)
            rows.append(row)

    out = pd.DataFrame(rows).sort_values(["version", "feature_set", "model"], kind="stable")
    out_path = PROJECT_ROOT / args.outdir / f"within_version_cv_summary_threshold_{args.threshold_mode}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} rows, threshold_mode={args.threshold_mode})")
    print(out[["version", "feature_set", "model", "micro_auprc_mean", "fscore_mean",
               "precision_mean", "recall_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
