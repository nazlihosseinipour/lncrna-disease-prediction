"""Run within-version 10-fold inductive CV for RFLDA / IPCARF / RF.

Self-contained: for every feature representation in the prepared manifest it runs
each model fold by fold, saves the per-fold predictions into a clean results tree,
evaluates them through the shared inductive Evaluator (Youden threshold, micro
metrics), and writes per-experiment performance CSVs plus an aggregated summary.

Outputs (under --outdir, default results/within_version_cv):
    predictions/<version>_<feature_key>/<model>/test_fold{1..10}.csv
    performance/<version>_<feature_key>_<model>_performance.csv
    within_version_cv_summary.csv
    run_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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
        default="inductive_inputs/feature_representations/feature_representation_manifest.csv",
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
        "--threshold-mode", choices=["youden", "fixed"], default="youden",
        help="Thresholding for the thresholded metrics. 'youden' is the legacy "
             "(test-fit) cut; 'fixed' uses 0.5 and avoids the optimistic bias.",
    )
    parser.add_argument("--outdir", default=f"{RESULTS_ROOT}/within_version_cv")
    return parser.parse_args()


def parse_mean_std(value: str) -> tuple[float, float]:
    left, right = value.rsplit("(", 1)
    return float(left), float(right.rstrip(")"))


def build_model_factory(model_key: str):
    if model_key == "rflda":
        from RFLDA.rflda import RFLDA
        return lambda: RFLDA(binary_mode=False)
    if model_key == "ipcarf":
        from IPCARF.ipcarf import IPCARF
        return lambda: IPCARF(binary_mode=False)
    from RF.rf import RF
    return lambda: RF(binary_mode=False)


def main() -> None:
    args = parse_args()
    manifest_path = PROJECT_ROOT / args.manifest
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

        for model_key in args.models:
            factory = build_model_factory(model_key)
            evaluator = Evaluator(threshold_mode=args.threshold_mode)
            pred_dir = outdir / "predictions" / f"{version}_{feature_key}" / model_key
            pred_dir.mkdir(parents=True, exist_ok=True)

            perf: list[tuple[float, ...]] = []
            for fold in range(args.n_splits):
                x_train, y_train = train[fold]
                x_test, y_test = test[fold]
                model = factory()
                model.fit(x_train, y_train)
                pred = pd.DataFrame(
                    model.predict_proba(x_test, y_test),
                    columns=y_test.columns,
                    index=y_test.index,
                )
                pred.to_csv(pred_dir / f"test_fold{fold + 1}.csv")
                perf.append(evaluator.evaluate(y_test, pred))
                print(f"  {version}/{feature_key} | {model_key} | fold {fold + 1:02d}")

            perf_df = evaluator.create_df_inductive(perf)
            perf_path = outdir / "performance" / f"{version}_{feature_key}_{model_key}_performance.csv"
            perf_df.to_csv(perf_path, index=False)

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
                "performance_csv": perf_path.as_posix(),
                "prediction_dir": pred_dir.as_posix(),
            }
            for metric in METRICS:
                summary_row[f"{metric}_meanstd"] = mean_row[metric]
                mean_value, std_value = parse_mean_std(mean_row[metric])
                summary_row[f"{metric}_mean"] = mean_value
                summary_row[f"{metric}_std"] = std_value
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
        "models": args.models,
        "versions": args.versions,
        "contains": args.contains,
        "n_splits": args.n_splits,
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
