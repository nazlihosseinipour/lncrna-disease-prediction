"""Run isolated 5/10-fold RF audits on one explicitly selected V1 feature file.

This runner never modifies prepared inputs or production results.  The supplied
paper artifact is accepted only with an explicit audit flag because it contains
an unresolved target-like binary block.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
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


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-source", choices=["paper", "ours"], required=True)
    parser.add_argument(
        "--paper-x",
        default=(
            "lncRNA_CIBCB2025-main/create_dataset/datasets/features/"
            "features_preprocessed.csv"
        ),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[10, 5])
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--fold-workers", type=int, default=1)
    parser.add_argument("--outdir", required=True)
    parser.add_argument(
        "--allow-target-like-paper-artifact",
        action="store_true",
        help="Required for paper reproduction/audit runs; never denotes a clean model.",
    )
    return parser.parse_args()


def load_x(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_column = "sample_id" if "sample_id" in frame.columns else "ID"
    if id_column not in frame.columns:
        raise ValueError(f"{path} has neither sample_id nor ID.")
    frame[id_column] = frame[id_column].astype(str)
    if not frame[id_column].is_unique:
        raise ValueError(f"{path} contains duplicate IDs.")
    return frame.set_index(id_column).apply(pd.to_numeric, errors="raise")


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest).resolve()
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 1:
        raise ValueError("Controlled audit requires a one-row V1 manifest.")
    row = manifest.iloc[0]
    y_path = resolve_path(str(row["y_path"])).resolve()
    x_path = (
        resolve_path(args.paper_x).resolve()
        if args.feature_source == "paper"
        else resolve_path(str(row["x_path"])).resolve()
    )
    if args.feature_source == "paper" and not args.allow_target_like_paper_artifact:
        raise ValueError(
            "Paper features are quarantined. Pass --allow-target-like-paper-artifact "
            "only for a clearly labelled reproduction/audit run."
        )

    y = pd.read_csv(y_path, index_col="ID")
    y.index = y.index.astype(str)
    y = y.apply(pd.to_numeric, errors="raise").astype(int)
    x_raw = load_x(x_path)
    missing = y.index.difference(x_raw.index)
    if len(missing):
        raise ValueError(f"Feature file is missing {len(missing)} label IDs.")
    x = x_raw.loc[y.index]
    if not x.index.equals(y.index):
        raise AssertionError("Aligned X/Y indexes differ.")

    sys.path.insert(0, (ROOT / "lncRNA_CIBCB2025-main/parse_results").as_posix())
    sys.path.insert(0, (ROOT / "lncRNA_CIBCB2025-main").as_posix())
    from parse_results.evaluate import Evaluator
    from RF.rf import RF

    evaluator = Evaluator(threshold=0.5, threshold_mode="fixed")
    outdir = resolve_path(args.outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"Refusing to reuse audit output directory: {outdir}")
    outdir.mkdir(parents=True)
    x.rename_axis("ID").to_csv(outdir / "aligned_X.csv")
    y.rename_axis("ID").to_csv(outdir / "aligned_Y.csv")

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for n_splits in args.folds:
        splitter = MultilabelStratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=args.random_state,
        )
        prediction_dir = outdir / f"{n_splits}fold_predictions"
        prediction_dir.mkdir()
        fold_rows: list[dict[str, object]] = []
        splits = list(splitter.split(x, y))

        def fit_fold(fold: int, train_idx, test_idx):
            model = RF(binary_mode=False)
            model.n_estimators = args.n_estimators
            model.n_jobs = args.n_jobs
            model.random_state = args.random_state
            model.fit(x.iloc[train_idx], y.iloc[train_idx])
            y_test = y.iloc[test_idx]
            prediction = pd.DataFrame(
                model.predict_proba(x.iloc[test_idx], y_test),
                index=y_test.index,
                columns=y_test.columns,
            )
            prediction.to_csv(prediction_dir / f"test_fold{fold}.csv")
            values = evaluator.evaluate(y_test, prediction)
            record = {"n_splits": n_splits, "fold": fold}
            record.update(dict(zip(METRICS, values)))
            return record

        if args.fold_workers == 1:
            completed = (
                fit_fold(fold, train_idx, test_idx)
                for fold, (train_idx, test_idx) in enumerate(splits, start=1)
            )
        else:
            executor = ThreadPoolExecutor(max_workers=args.fold_workers)
            futures = {
                executor.submit(fit_fold, fold, train_idx, test_idx): fold
                for fold, (train_idx, test_idx) in enumerate(splits, start=1)
            }
            completed = (future.result() for future in as_completed(futures))
        try:
            for record in completed:
                fold = int(record["fold"])
                fold_rows.append(record)
                all_rows.append(record)
                print(
                    f"{args.feature_source} audit | {n_splits}-fold | fold {fold:02d}",
                    flush=True,
                )
        finally:
            if args.fold_workers > 1:
                executor.shutdown(wait=True, cancel_futures=True)

        fold_rows.sort(key=lambda item: int(item["fold"]))
        fold_frame = pd.DataFrame(fold_rows)
        summary = {
            "audit_label": f"{args.feature_source}_feature_reproduction_audit",
            "feature_source": args.feature_source,
            "n_splits": n_splits,
            "n_samples": len(x),
            "n_features": x.shape[1],
            "n_labels": y.shape[1],
            "positive_interactions": int(y.to_numpy().sum()),
            "threshold_mode": "fixed",
            "threshold": 0.5,
        }
        for metric in METRICS:
            summary[f"{metric}_mean"] = float(fold_frame[metric].mean())
            summary[f"{metric}_std"] = float(fold_frame[metric].std(ddof=0))
        summaries.append(summary)

    pd.DataFrame(all_rows).to_csv(outdir / "per_fold_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(outdir / "summary.csv", index=False)
    config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "audit_only": args.feature_source == "paper",
        "feature_source": args.feature_source,
        "x_path": x_path.as_posix(),
        "x_sha256": sha256(x_path),
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": sha256(manifest_path),
        "y_path": y_path.as_posix(),
        "y_sha256": sha256(y_path),
        "n_samples": len(x),
        "n_features": x.shape[1],
        "n_labels": y.shape[1],
        "positive_interactions": int(y.to_numpy().sum()),
        "folds": args.folds,
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "fold_workers": args.fold_workers,
        "threshold_mode": "fixed",
        "threshold": 0.5,
        "paper_artifact_warning": (
            "The supplied feature file contains an unresolved 285-column "
            "target-like binary block; these are reproduction/audit results."
            if args.feature_source == "paper"
            else None
        ),
    }
    (outdir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved isolated audit: {outdir}", flush=True)


if __name__ == "__main__":
    main()
