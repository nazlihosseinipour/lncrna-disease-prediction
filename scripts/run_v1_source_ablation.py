"""Isolate feature-space and label-space causes of the corrected V1 RF gap."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = [
    "current2023_y44",
    "composition583_y44",
]
CONTAMINATED_AUDIT_CONFIGS = {
    "paper827_y44",
    "paper827_y24",
    "paper_binary285_y44",
    "paper_continuous542_y44",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Explicit path to the audited V1 feature manifest.",
    )
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N folds for a controlled probe.",
    )
    parser.add_argument("--outdir", default="results/v1_source_ablation_2026-08-17")
    parser.add_argument(
        "--allow-contaminated-paper-artifact",
        action="store_true",
        help=(
            "Audit-only escape hatch for the supplied 827-column paper artifact. "
            "Never use these configurations as clean benchmark results."
        ),
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_inputs(manifest_path: Path):
    paper_x = pd.read_csv(
        PROJECT_ROOT
        / "lncRNA_CIBCB2025-main/create_dataset/datasets/features/features_preprocessed.csv",
        index_col="ID",
    )
    paper_y = pd.read_csv(
        PROJECT_ROOT
        / "lncRNA_CIBCB2025-main/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv",
        index_col="ID",
    )
    manifest = pd.read_csv(manifest_path).iloc[0]
    current_x = pd.read_csv(manifest["x_path"], index_col="sample_id")

    for frame in (paper_x, paper_y, current_x):
        frame.index = frame.index.astype(str)
    common_ids = current_x.index.intersection(paper_x.index).intersection(paper_y.index)
    if len(common_ids) != len(current_x):
        raise ValueError(
            f"Expected every current sample in paper matrices; got {len(common_ids)}/{len(current_x)}"
        )
    ids = current_x.index
    paper_x = paper_x.loc[ids].apply(pd.to_numeric, errors="raise")
    paper_y = paper_y.loc[ids].apply(pd.to_numeric, errors="raise").astype(int)
    current_x = current_x.loc[ids].apply(pd.to_numeric, errors="raise")
    composition_cols = [
        column
        for column in current_x.columns
        if not column.startswith("mp_sequence__")
        and not column.startswith("mp_sequence_chunked__")
    ]
    composition_x = current_x[composition_cols]
    counts = paper_y.sum(axis=0)
    y24 = paper_y.loc[:, counts >= 10]
    if y24.shape[1] != 24:
        raise ValueError(f"Expected 24 paper labels with >=10 positives, got {y24.shape[1]}")
    return paper_x, paper_y, y24, current_x, composition_x


def build_configs(manifest_path: Path):
    paper_x, y44, y24, current_x, composition_x = load_inputs(manifest_path)
    paper_binary_cols = [
        column
        for column in paper_x.columns
        if set(paper_x[column].dropna().unique()).issubset({0, 1})
    ]
    if len(paper_binary_cols) != 285:
        raise ValueError(
            f"Expected 285 binary association-derived paper columns, got {len(paper_binary_cols)}"
        )
    paper_binary_x = paper_x[paper_binary_cols]
    paper_continuous_x = paper_x.drop(columns=paper_binary_cols)
    return {
        "paper827_y44": (paper_x, y44, "Paper features; all 44 paper labels"),
        "paper827_y24": (
            paper_x,
            y24,
            "Paper features; 24 labels with >=10 positives after 353-sample alignment",
        ),
        "current2023_y44": (current_x, y44, "Current features; all 44 paper labels"),
        "current2023_y24": (
            current_x,
            y24,
            "Current features; 24 labels with >=10 positives after 353-sample alignment",
        ),
        "composition583_y44": (
            composition_x,
            y44,
            "Current composition features only; all 44 paper labels",
        ),
        "composition583_y24": (
            composition_x,
            y24,
            "Current composition features only; 24 labels with >=10 positives after 353-sample alignment",
        ),
        "paper_binary285_y44": (
            paper_binary_x,
            y44,
            "Paper binary association-derived block only; all 44 paper labels",
        ),
        "paper_continuous542_y44": (
            paper_continuous_x,
            y44,
            "Paper continuous block after removing 285 binary columns; all 44 paper labels",
        ),
    }


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest).resolve()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    if args.max_folds is not None and not 1 <= args.max_folds <= args.n_splits:
        raise ValueError("--max-folds must be between 1 and --n-splits.")
    run_config_path = outdir / "run_config.json"
    if run_config_path.exists():
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        if previous.get("manifest_sha256") != manifest_sha256:
            raise ValueError(
                f"Refusing to mix manifests in {outdir}; choose a new --outdir."
            )
    fold_path = outdir / "per_fold_metrics.csv"
    rows = (
        pd.read_csv(fold_path).to_dict("records")
        if fold_path.exists()
        else []
    )
    completed = {(str(row["config"]), int(row["fold"])) for row in rows}
    configs = build_configs(manifest_path)
    unknown = sorted(set(args.configs) - set(configs))
    if unknown:
        raise ValueError(f"Unknown configs: {unknown}. Available: {sorted(configs)}")
    contaminated = sorted(set(args.configs) & CONTAMINATED_AUDIT_CONFIGS)
    if contaminated and not args.allow_contaminated_paper_artifact:
        raise ValueError(
            "Refusing contaminated paper feature configurations: "
            f"{contaminated}. Use the clean current2023/composition configurations. "
            "The --allow-contaminated-paper-artifact flag is for leakage auditing only."
        )

    # One shared split assignment, derived from the full 44-label paper matrix,
    # ensures that feature and label ablations see the same held-out samples.
    reference_y = configs["paper827_y44"][1]
    splitter = MultilabelStratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=0
    )
    splits = list(splitter.split(np.zeros((len(reference_y), 1)), reference_y))
    if args.max_folds is not None:
        splits = splits[: args.max_folds]

    model_dir = PROJECT_ROOT / "lncRNA_CIBCB2025-main"
    sys.path.insert(0, (model_dir / "parse_results").as_posix())
    sys.path.insert(0, model_dir.as_posix())
    from parse_results.evaluate import Evaluator
    from RF.rf import RF

    evaluator = Evaluator(threshold_mode="fixed")

    for config_name in args.configs:
        x, y, description = configs[config_name]
        for fold, (train_index, test_index) in enumerate(splits, start=1):
            if (config_name, fold) in completed:
                print(f"{config_name} fold {fold:02d} reused", flush=True)
                continue
            model = RF(binary_mode=False)
            model.n_estimators = args.n_estimators
            model.n_jobs = -1
            model.fit(x.iloc[train_index], y.iloc[train_index])
            y_test = y.iloc[test_index]
            scores = pd.DataFrame(
                model.predict_proba(x.iloc[test_index], y_test),
                columns=y_test.columns,
                index=y_test.index,
            )
            metrics = evaluator.evaluate(y_test, scores)
            row = {
                "config": config_name,
                "description": description,
                "fold": fold,
                "n_samples": len(x),
                "n_features": x.shape[1],
                "n_labels": y.shape[1],
                "positive_interactions": int(y.to_numpy().sum()),
                "density": float(y.to_numpy().mean()),
                "micro_auroc": metrics[2],
                "micro_auprc": metrics[3],
                "micro_f1_fixed_0.5": metrics[6],
            }
            rows.append(row)
            pd.DataFrame(rows).to_csv(fold_path, index=False)
            print(
                f"{config_name} fold {fold:02d}: "
                f"AUROC={row['micro_auroc']:.3f} "
                f"AUPRC={row['micro_auprc']:.3f} "
                f"F1={row['micro_f1_fixed_0.5']:.3f}",
                flush=True,
            )

    folds = pd.DataFrame(rows)
    folds = folds[folds["config"].isin(args.configs)]
    summary = (
        folds.groupby(
            [
                "config",
                "description",
                "n_samples",
                "n_features",
                "n_labels",
                "positive_interactions",
                "density",
            ],
            as_index=False,
        )
        .agg(
            completed_folds=("fold", "nunique"),
            micro_auroc_mean=("micro_auroc", "mean"),
            micro_auroc_std=("micro_auroc", "std"),
            micro_auprc_mean=("micro_auprc", "mean"),
            micro_auprc_std=("micro_auprc", "std"),
            micro_f1_mean=("micro_f1_fixed_0.5", "mean"),
            micro_f1_std=("micro_f1_fixed_0.5", "std"),
        )
        .sort_values("config")
    )
    summary.to_csv(outdir / "ablation_summary.csv", index=False)
    run_config_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "manifest": manifest_path.as_posix(),
                "manifest_sha256": manifest_sha256,
                "configs": args.configs,
                "n_estimators": args.n_estimators,
                "n_splits": args.n_splits,
                "max_folds": args.max_folds,
                "threshold_mode": "fixed",
                "random_state": 0,
                "contaminated_paper_artifact_allowed": (
                    args.allow_contaminated_paper_artifact
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {outdir / 'ablation_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
