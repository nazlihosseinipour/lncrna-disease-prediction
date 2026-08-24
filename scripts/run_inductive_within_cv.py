"""Run within-version 10-fold inductive CV for RFLDA / IPCARF / RF.

Self-contained: for every feature representation in the prepared manifest it runs
each model fold by fold, saves the per-fold predictions into a clean results tree,
evaluates them through the shared inductive Evaluator (fixed, legacy Youden, or
training-only OOB-calibrated threshold; micro metrics), and writes per-experiment
performance CSVs plus an aggregated summary.

Outputs (under --outdir, default results/within_version_cv):
    predictions/<version>_<feature_key>/<model>/test_fold{1..10}.csv
    performance/<version>_<feature_key>_<model>_performance.csv
    within_version_cv_summary.csv
    run_config.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results")

METRICS = [
    "hamming",
    "label_ranking",
    "micro_roc",
    "micro_auprc",
    "precision",
    "recall",
    "fscore",
    "accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Explicit path to a prepared, contamination-audited feature manifest.",
    )
    parser.add_argument("--project-dir", default="lncRNA_CIBCB2025-main")
    parser.add_argument(
        "--models", nargs="+", choices=["rflda", "ipcarf", "rf"],
        default=["rflda", "ipcarf", "rf"],
    )
    parser.add_argument("--versions", nargs="+", choices=["v1", "v2"], default=None)
    parser.add_argument(
        "--contains", default=None,
        help="Optional substring filter over feature_set.",
    )
    parser.add_argument(
        "--feature-keys", nargs="+", default=None,
        help="Optional exact filter over feature_key (sanitized feature set).",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N outer folds while retaining the original split definition. "
             "Intended for controlled probes; omit for the final full run.",
    )
    parser.add_argument(
        "--fold-workers",
        type=int,
        default=1,
        help="Number of CV folds to fit concurrently.",
    )
    parser.add_argument(
        "--model-n-jobs",
        type=int,
        default=None,
        help="CPU workers per model. Defaults to all cores for one fold, or "
             "CPU count divided by --fold-workers for parallel folds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid existing fold-prediction CSVs in the output directory.",
    )
    parser.add_argument(
        "--threshold-mode", choices=["youden", "fixed", "train_oob_f1"], default="fixed",
        help="Thresholding for the thresholded metrics. 'youden' is the legacy "
             "(test-fit) cut; 'fixed' uses 0.5; 'train_oob_f1' learns one global "
             "micro-F1 threshold from Random Forest out-of-bag training scores.",
    )
    parser.add_argument("--outdir", default=f"{RESULTS_ROOT}/within_version_cv")
    return parser.parse_args()


def parse_mean_std(value: str) -> tuple[float, float]:
    left, right = value.rsplit("(", 1)
    return float(left), float(right.rstrip(")"))


def select_global_f1_threshold(
    y_true: pd.DataFrame | np.ndarray,
    y_score: pd.DataFrame | np.ndarray,
    *,
    default: float = 0.5,
) -> float:
    """Choose one micro-F1 threshold using training-only scores.

    A single global threshold is intentionally used instead of one threshold per
    label: many disease labels are sparse, so per-label calibration would be
    unstable and would add a large number of fitted hyperparameters.
    """
    truth = np.asarray(y_true).reshape(-1)
    score = np.asarray(y_score).reshape(-1)
    if truth.shape != score.shape:
        raise ValueError("Threshold calibration requires equal truth/score shapes.")
    finite = np.isfinite(score)
    truth = truth[finite]
    score = score[finite]
    if truth.size == 0 or np.unique(truth).size < 2:
        return float(default)
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Threshold calibration requires binary targets.")

    precision, recall, thresholds = precision_recall_curve(truth, score)
    if thresholds.size == 0:
        return float(default)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    if not np.isfinite(f1).any():
        return float(default)
    best = np.flatnonzero(f1 == np.nanmax(f1))
    # Prefer the largest equally optimal threshold to avoid gratuitous positives.
    return float(np.clip(thresholds[int(best[-1])], 0.0, 1.0))


def build_model_factory(model_key: str, *, oob_score: bool = False):
    if model_key == "rflda":
        from RFLDA.rflda import RFLDA
        return lambda: RFLDA(binary_mode=False)
    if model_key == "ipcarf":
        from IPCARF.ipcarf import IPCARF
        return lambda: IPCARF(binary_mode=False)
    from RF.rf import RF
    return lambda: RF(binary_mode=False, oob_score=oob_score)


def main() -> None:
    args = parse_args()
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    project_dir = PROJECT_ROOT / args.project_dir
    outdir = PROJECT_ROOT / args.outdir
    (outdir / "predictions").mkdir(parents=True, exist_ok=True)
    (outdir / "performance").mkdir(parents=True, exist_ok=True)

    # Put professor-code imports before PYTHONPATH=mainfolder so its top-level
    # `utils` package is not shadowed by mainfolder/utils.
    sys.path.insert(0, (project_dir / "parse_results").as_posix())
    sys.path.insert(0, project_dir.as_posix())

    from parse_results.evaluate import Evaluator
    from utils.utils import iterator_cross_validation

    manifest = pd.read_csv(manifest_path)
    if args.versions:
        manifest = manifest[manifest["version"].isin(args.versions)]
    if args.contains:
        manifest = manifest[
            manifest["feature_set"].str.contains(args.contains, case=False, na=False)
        ]
    if args.feature_keys:
        manifest = manifest[manifest["feature_key"].isin(args.feature_keys)]

    summary_rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        version = row["version"]
        feature_key = row["feature_key"]
        x_path = Path(row["x_path"])
        y_path = Path(row["y_path"])
        split_path = Path(row["split_path"])
        if not split_path.exists():
            print(f"Skipping {version}/{feature_key}: split file missing.")
            continue

        x = pd.read_csv(x_path, index_col=0)
        y = pd.read_csv(y_path, index_col=0)
        splits = pd.read_csv(split_path)
        train, test = iterator_cross_validation(splits, x, y, n_folds=args.n_splits)
        selected_folds = list(range(args.n_splits))
        if args.max_folds is not None:
            if not 1 <= args.max_folds <= args.n_splits:
                raise ValueError("--max-folds must be between 1 and --n-splits.")
            selected_folds = selected_folds[: args.max_folds]

        for model_key in args.models:
            if args.threshold_mode == "train_oob_f1" and model_key != "rf":
                raise ValueError(
                    "--threshold-mode train_oob_f1 currently supports only --models rf."
                )
            uses_oob_threshold = args.threshold_mode == "train_oob_f1"
            factory = build_model_factory(model_key, oob_score=uses_oob_threshold)
            evaluator = Evaluator(
                threshold_mode="fixed" if uses_oob_threshold else args.threshold_mode
            )
            pred_dir = outdir / "predictions" / f"{version}_{feature_key}" / model_key
            pred_dir.mkdir(parents=True, exist_ok=True)

            fold_workers = max(1, args.fold_workers)
            if args.model_n_jobs is not None:
                model_n_jobs = args.model_n_jobs
            elif fold_workers == 1:
                model_n_jobs = -1
            else:
                model_n_jobs = max(1, (os.cpu_count() or 1) // fold_workers)

            def load_existing_prediction(fold: int, y_test: pd.DataFrame):
                pred_path = pred_dir / f"test_fold{fold + 1}.csv"
                # OOB calibration is fold-specific. A prediction alone is not
                # enough to reproduce the fitted threshold, so refit this mode.
                if uses_oob_threshold or not args.resume or not pred_path.exists():
                    return None
                pred = pd.read_csv(pred_path, index_col=0)
                pred.index = pred.index.astype(str)
                y_index = y_test.index.astype(str)
                if list(pred.columns) != list(y_test.columns) or set(pred.index) != set(y_index):
                    return None
                return pred.reindex(y_index)

            def fit_fold(fold: int):
                x_train, y_train = train[fold]
                x_test, y_test = test[fold]
                existing = load_existing_prediction(fold, y_test)
                if existing is not None:
                    return fold, evaluator.evaluate(y_test, existing), None, True, np.nan
                model = factory()
                if hasattr(model, "n_jobs"):
                    model.n_jobs = model_n_jobs
                model.fit(x_train, y_train)
                pred = pd.DataFrame(
                    model.predict_proba(x_test, y_test),
                    columns=y_test.columns,
                    index=y_test.index,
                )
                pred.to_csv(pred_dir / f"test_fold{fold + 1}.csv")
                if uses_oob_threshold:
                    threshold = select_global_f1_threshold(
                        y_train, model.get_oob_predictions()
                    )
                    fold_evaluator = Evaluator(
                        threshold=threshold,
                        threshold_mode="fixed",
                    )
                    fixed_baseline_perf = Evaluator(
                        threshold=0.5,
                        threshold_mode="fixed",
                    ).evaluate(y_test, pred)
                else:
                    threshold = 0.5 if args.threshold_mode == "fixed" else np.nan
                    fold_evaluator = evaluator
                    fixed_baseline_perf = None
                return (
                    fold,
                    fold_evaluator.evaluate(y_test, pred),
                    fixed_baseline_perf,
                    False,
                    threshold,
                )

            perf_by_fold: dict[int, tuple[float, ...]] = {}
            fixed_perf_by_fold: dict[int, tuple[float, ...]] = {}
            threshold_by_fold: dict[int, float] = {}
            if fold_workers == 1:
                fold_results = (fit_fold(fold) for fold in selected_folds)
            else:
                executor = ThreadPoolExecutor(max_workers=fold_workers)
                futures = {
                    executor.submit(fit_fold, fold): fold
                    for fold in selected_folds
                }
                fold_results = (future.result() for future in as_completed(futures))

            try:
                for fold, fold_perf, fixed_fold_perf, reused, threshold in fold_results:
                    perf_by_fold[fold] = fold_perf
                    if fixed_fold_perf is not None:
                        fixed_perf_by_fold[fold] = fixed_fold_perf
                    threshold_by_fold[fold] = threshold
                    status = "reused" if reused else "fitted"
                    print(
                        f"  {version}/{feature_key} | {model_key} | "
                        f"fold {fold + 1:02d} ({status})",
                        flush=True,
                    )
            finally:
                if fold_workers > 1:
                    executor.shutdown(wait=True, cancel_futures=True)

            perf = [perf_by_fold[fold] for fold in selected_folds]

            perf_df = evaluator.create_df_inductive(
                perf,
                fold_labels=[f"fold{fold + 1}" for fold in selected_folds],
            )
            perf_path = outdir / "performance" / f"{version}_{feature_key}_{model_key}_performance.csv"
            perf_df.to_csv(perf_path, index=False)

            fixed_baseline_mean_row = None
            if uses_oob_threshold:
                calibration_path = (
                    outdir / "performance"
                    / f"{version}_{feature_key}_{model_key}_oob_thresholds.csv"
                )
                pd.DataFrame(
                    {
                        "fold": [fold + 1 for fold in selected_folds],
                        "threshold": [
                            threshold_by_fold[fold] for fold in selected_folds
                        ],
                        "calibration_source": "training_fold_rf_oob_predictions",
                    }
                ).to_csv(calibration_path, index=False)
                fixed_perf = [fixed_perf_by_fold[fold] for fold in selected_folds]
                fixed_baseline_df = evaluator.create_df_inductive(
                    fixed_perf,
                    fold_labels=[f"fold{fold + 1}" for fold in selected_folds],
                )
                fixed_baseline_path = (
                    outdir / "performance"
                    / f"{version}_{feature_key}_{model_key}_fixed_baseline_performance.csv"
                )
                fixed_baseline_df.to_csv(fixed_baseline_path, index=False)
                fixed_baseline_mean_row = fixed_baseline_df[
                    fixed_baseline_df["folds"] == "mean(std)"
                ].iloc[0].to_dict()

            mean_row = perf_df[perf_df["folds"] == "mean(std)"].iloc[0].to_dict()
            summary_row: dict[str, object] = {
                "experiment": f"within_{version}_cv",
                "version": version,
                "source_version": version,
                "target_version": version,
                "feature_set": row["feature_set"],
                "feature_key": feature_key,
                "model": model_key,
                "threshold_mode": args.threshold_mode,
                "n_samples": int(row["n_samples"]),
                "n_features": int(row["n_features"]),
                "n_labels": int(row["n_labels"]),
                "completed_folds": len(selected_folds),
                "performance_csv": perf_path.as_posix(),
                "prediction_dir": pred_dir.as_posix(),
            }
            for metric in METRICS:
                summary_row[f"{metric}_meanstd"] = mean_row[metric]
                mean_value, std_value = parse_mean_std(mean_row[metric])
                summary_row[f"{metric}_mean"] = mean_value
                summary_row[f"{metric}_std"] = std_value
                if fixed_baseline_mean_row is not None:
                    fixed_mean, fixed_std = parse_mean_std(
                        fixed_baseline_mean_row[metric]
                    )
                    summary_row[f"fixed_baseline_{metric}_mean"] = fixed_mean
                    summary_row[f"fixed_baseline_{metric}_std"] = fixed_std
            summary_rows.append(summary_row)
            print(f"Saved {perf_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "within_version_cv_summary.csv"
    # Merge cumulatively: a per-version invocation must not clobber rows from a
    # previous run on the other version (or other feature sets in Stage B).
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        if not summary_df.empty:
            key = ["version", "feature_key", "model"]
            done = set(map(tuple, summary_df[key].itertuples(index=False, name=None)))
            existing = existing[
                ~existing[key].apply(tuple, axis=1).isin(done)
            ]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["version", "feature_set", "model"], kind="stable"
        )
    summary_df.to_csv(summary_path, index=False)

    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "models": args.models,
        "versions": args.versions,
        "contains": args.contains,
        "n_splits": args.n_splits,
        "max_folds": args.max_folds,
        "fold_workers": max(1, args.fold_workers),
        "model_n_jobs": args.model_n_jobs,
        "resume": args.resume,
        "random_state": 0,
        "model_n_estimators": 150,
        "threshold_mode": args.threshold_mode,
        "outdir": outdir.as_posix(),
    }
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved run config: {outdir / 'run_config.json'}")


if __name__ == "__main__":
    main()
