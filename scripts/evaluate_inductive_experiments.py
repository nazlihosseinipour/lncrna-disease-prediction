from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved inductive RFLDA/IPCARF fold predictions and build a summary table."
    )
    parser.add_argument(
        "--manifest",
        default="inductive_inputs/feature_representations/feature_representation_manifest.csv",
        help="Manifest created by prepare_inductive_feature_representations.py",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rflda", "ipcarf"],
        default=["rflda", "ipcarf"],
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--project-dir",
        default="lncRNA_CIBCB2025-main",
        help="Path to the professor code folder containing RFLDA/IPCARF/parse_results.",
    )
    parser.add_argument(
        "--outdir",
        default="lncRNA_CIBCB2025-main/parse_results/feature_representations",
        help="Directory for per-run performance CSVs and the aggregated summary.",
    )
    return parser.parse_args()


def parse_mean_std(value: str) -> tuple[float, float]:
    left, right = value.rsplit("(", 1)
    return float(left), float(right.rstrip(")"))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    project_dir = Path(args.project_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Put professor-code imports before PYTHONPATH=mainfolder so its top-level
    # `utils` package is not shadowed by mainfolder/utils.
    sys.path.insert(0, (project_dir / "parse_results").as_posix())
    sys.path.insert(0, project_dir.as_posix())

    from parse_results.evaluate import Evaluator
    from utils.utils import iterator_cross_validation

    try:
        manifest = pd.read_csv(manifest_path)
    except EmptyDataError:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    model_dir_map = {"rflda": "RFLDA", "ipcarf": "IPCARF"}
    summary_rows: list[dict[str, object]] = []
    summary_columns = [
        "version",
        "feature_set",
        "feature_key",
        "model",
        "n_samples",
        "n_features",
        "n_labels",
        "performance_csv",
        "hamming_meanstd",
        "hamming_mean",
        "hamming_std",
        "label_ranking_meanstd",
        "label_ranking_mean",
        "label_ranking_std",
        "micro_roc_meanstd",
        "micro_roc_mean",
        "micro_roc_std",
        "micro_auprc_meanstd",
        "micro_auprc_mean",
        "micro_auprc_std",
        "precision_meanstd",
        "precision_mean",
        "precision_std",
        "recall_meanstd",
        "recall_mean",
        "recall_std",
        "fscore_meanstd",
        "fscore_mean",
        "fscore_std",
        "accuracy_meanstd",
        "accuracy_mean",
        "accuracy_std",
    ]

    for _, row in manifest.iterrows():
        y_path = Path(row["y_path"])
        split_path = Path(row["split_path"])
        if not split_path.exists():
            print(f"Skipping {y_path.stem}: split file missing at {split_path}")
            continue

        y = pd.read_csv(y_path, index_col=0)
        splits = pd.read_csv(split_path)
        dummy = pd.DataFrame(index=y.index)
        _, test = iterator_cross_validation(splits, dummy, y)

        for model_key in args.models:
            model_dir = project_dir / model_dir_map[model_key]
            model_token = model_dir_map[model_key]
            pred_pattern = f"{y_path.stem}_{model_token}_inductive_multilabel_predictions_test_fold{{}}.csv"
            missing = [
                str(model_dir / pred_pattern.format(fold))
                for fold in range(1, 11)
                if not (model_dir / pred_pattern.format(fold)).exists()
            ]
            if missing:
                print(f"Skipping {y_path.stem} {model_key}: missing prediction files.")
                continue

            evaluator = Evaluator()
            perf = []
            for fold in range(1, 11):
                pred = pd.read_csv(model_dir / pred_pattern.format(fold))
                test_y = test[fold - 1][1]
                pred.columns = test_y.columns
                perf.append(evaluator.evaluate(test_y, pred))

            perf_df = evaluator.create_df_inductive(perf)
            perf_path = outdir / f"{y_path.stem}_{model_key}_performance.csv"
            perf_df.to_csv(perf_path, index=False)

            mean_row = perf_df[perf_df["folds"] == "mean(std)"].iloc[0].to_dict()
            summary_row: dict[str, object] = {
                "version": row["version"],
                "feature_set": row["feature_set"],
                "feature_key": row["feature_key"],
                "model": model_key,
                "n_samples": int(row["n_samples"]),
                "n_features": int(row["n_features"]),
                "n_labels": int(row["n_labels"]),
                "performance_csv": perf_path.as_posix(),
            }
            for metric in [
                "hamming",
                "label_ranking",
                "micro_roc",
                "micro_auprc",
                "precision",
                "recall",
                "fscore",
                "accuracy",
            ]:
                summary_row[f"{metric}_meanstd"] = mean_row[metric]
                mean_value, std_value = parse_mean_std(mean_row[metric])
                summary_row[f"{metric}_mean"] = mean_value
                summary_row[f"{metric}_std"] = std_value

            summary_rows.append(summary_row)
            print(f"Saved {perf_path}")

    summary_df = pd.DataFrame(summary_rows, columns=summary_columns)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["version", "feature_set", "model"], kind="stable"
        )
    summary_path = outdir / "inductive_feature_representation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
