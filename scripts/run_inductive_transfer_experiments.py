from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except Exception:  # pragma: no cover - handled at runtime
    MultilabelStratifiedKFold = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VERSION_TO_LABEL_CSV = {
    "v1": PROJECT_ROOT / "Data/output_data/sequences_for_oop.csv",
    "v2": PROJECT_ROOT / "Data/output_data/website_full_matrix.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-version inductive transfer experiments over the shared disease space "
            "for feature sets prepared by prepare_inductive_feature_representations.py."
        )
    )
    parser.add_argument(
        "--manifest",
        default="inductive_inputs/feature_representations/feature_representation_manifest.csv",
        help="Manifest created by prepare_inductive_feature_representations.py",
    )
    parser.add_argument(
        "--project-dir",
        default="lncRNA_CIBCB2025-main",
        help="Path to the professor code folder containing RFLDA/IPCARF/parse_results.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rflda", "ipcarf", "rf"],
        default=["rflda", "ipcarf", "rf"],
        help="Models to evaluate.",
    )
    parser.add_argument(
        "--label-match",
        choices=["exact", "normalized"],
        default="normalized",
        help=(
            "How to match disease columns across versions. 'exact' uses raw column "
            "names (only ~20 common); 'normalized' lowercases and collapses "
            "non-alphanumeric characters so e.g. V1 'acute myeloid leukemia' matches "
            "V2 'Acute Myeloid Leukemia' (~73 common)."
        ),
    )
    parser.add_argument(
        "--source-version",
        choices=["v1", "v2"],
        default="v1",
        help="Training dataset version for the transfer experiment.",
    )
    parser.add_argument(
        "--target-version",
        choices=["v1", "v2"],
        default="v2",
        help="Evaluation dataset version for the transfer experiment.",
    )
    parser.add_argument(
        "--label-space",
        choices=["train", "both", "all_common"],
        default="train",
        help=(
            "How to keep common disease labels. "
            "'train' keeps labels that pass the threshold in the source version only, "
            "'both' requires both source and target to pass, "
            "'all_common' keeps every shared disease label."
        ),
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=5,
        help="Positive-count threshold used when selecting common labels.",
    )
    parser.add_argument(
        "--keep-rule",
        choices=["gt", "ge"],
        default="gt",
        help="Threshold rule: 'gt' means > min-positives, 'ge' means >= min-positives.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of target-side CV folds used for transfer testing and common-space target CV.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for the target-side multilabel splits.",
    )
    parser.add_argument(
        "--skip-target-cv",
        action="store_true",
        help=(
            "Only run the cross-version transfer evaluation, skipping the "
            "common-label target-side CV. Use this to avoid re-running the "
            "expensive within-target CV (e.g. RFLDA on V2) when only the "
            "transfer result is needed."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="lncRNA_CIBCB2025-main/parse_results/transfer_feature_representations",
        help="Directory for per-run performance CSVs and the aggregated summary.",
    )
    return parser.parse_args()


def parse_mean_std(value: str) -> tuple[float, float]:
    left, right = value.rsplit("(", 1)
    return float(left), float(right.rstrip(")"))


def load_label_csv(version: str) -> pd.DataFrame:
    path = VERSION_TO_LABEL_CSV[version]
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "seqs"], errors="ignore")
    if "ID" not in df.columns:
        raise ValueError(f"{path} is missing required ID column.")
    df["ID"] = df["ID"].astype(str)
    return df


def align_y_to_x(x_path: Path, version: str) -> pd.DataFrame:
    x = pd.read_csv(x_path)
    if "sample_id" not in x.columns:
        raise ValueError(f"{x_path} is missing required sample_id column.")
    x["sample_id"] = x["sample_id"].astype(str)

    y = load_label_csv(version)
    aligned = x[["sample_id"]].merge(y, left_on="sample_id", right_on="ID", how="inner")
    aligned = aligned.drop(columns=["sample_id"]).rename(columns={"ID": "ID"})
    return aligned


def resolve_manifest_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_disease_name(name: str) -> str:
    """Canonical disease key: lowercase, non-alphanumerics collapsed to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _passes(count: float, min_positives: int, keep_rule: str) -> bool:
    return count > min_positives if keep_rule == "gt" else count >= min_positives


def select_common_label_pairs(
    y_source: pd.DataFrame,
    y_target: pd.DataFrame,
    *,
    label_space: str,
    min_positives: int,
    keep_rule: str,
    label_match: str,
) -> list[tuple[str, str, str]]:
    """Return (source_col, target_col, canonical_name) triples for shared diseases.

    With label_match='normalized' the canonical key harmonizes naming differences
    across versions (e.g. V1 'acute myeloid leukemia' / V2 'Acute Myeloid Leukemia').
    """
    source_cols = [c for c in y_source.columns if c != "ID"]
    target_cols = [c for c in y_target.columns if c != "ID"]

    def key(col: str) -> str:
        return col if label_match == "exact" else normalize_disease_name(col)

    source_by_key: dict[str, str] = {}
    for col in source_cols:
        source_by_key.setdefault(key(col), col)
    target_by_key: dict[str, str] = {}
    for col in target_cols:
        target_by_key.setdefault(key(col), col)

    common_keys = sorted(set(source_by_key) & set(target_by_key))
    pairs = [(source_by_key[k], target_by_key[k], k) for k in common_keys]
    if label_space == "all_common":
        return pairs

    kept: list[tuple[str, str, str]] = []
    for source_col, target_col, canonical in pairs:
        keep_source = _passes(
            y_source[source_col].sum(), min_positives, keep_rule
        )
        if label_space == "train":
            if keep_source:
                kept.append((source_col, target_col, canonical))
        else:  # "both"
            keep_target = _passes(
                y_target[target_col].sum(), min_positives, keep_rule
            )
            if keep_source and keep_target:
                kept.append((source_col, target_col, canonical))
    return kept


def make_target_splits(
    x: pd.DataFrame, y: pd.DataFrame, n_splits: int, random_state: int
) -> list[tuple[pd.Index, pd.Index]]:
    if MultilabelStratifiedKFold is None:
        raise ImportError(
            "iterative-stratification is required for transfer split generation. "
            "Install it with: pip install iterative-stratification"
    )
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    x_model = x.set_index("sample_id")
    y_model = y.set_index("ID")
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(x_model, y_model)]


def load_splits_from_csv(path: Path, n_splits: int) -> list[tuple[pd.Index, pd.Index]]:
    splits_df = pd.read_csv(path)
    splits: list[tuple[pd.Index, pd.Index]] = []
    for fold in range(1, n_splits + 1):
        train_col = f"train_fold{fold}"
        test_col = f"test_fold{fold}"
        train_idx = splits_df[train_col].dropna().astype(int).to_numpy()
        test_idx = splits_df[test_col].dropna().astype(int).to_numpy()
        splits.append((train_idx, test_idx))
    return splits


def build_model_factory(model_key: str):
    if model_key == "rflda":
        from RFLDA.rflda import RFLDA

        return lambda: RFLDA(binary_mode=False)

    if model_key == "rf":
        from RF.rf import RF

        return lambda: RF(binary_mode=False)

    from IPCARF.ipcarf import IPCARF

    return lambda: IPCARF(binary_mode=False)


def evaluate_runs(
    *,
    evaluator,
    model_factory,
    x_train_full: pd.DataFrame,
    y_train_full: pd.DataFrame,
    x_target_full: pd.DataFrame,
    y_target_full: pd.DataFrame,
    splits: list[tuple[pd.Index, pd.Index]],
    mode: str,
) -> list[tuple[float, ...]]:
    performance: list[tuple[float, ...]] = []

    # In transfer mode the training set is the full source version and is identical
    # across folds, so fit a single model once and reuse it (important for the
    # V2-source case where fitting on 4k+ samples per fold would dominate runtime).
    shared_model = None
    if mode == "transfer":
        shared_model = model_factory()
        shared_model.fit(x_train_full, y_train_full)

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        x_test = x_target_full.iloc[test_idx]
        y_test = y_target_full.iloc[test_idx]

        if mode == "transfer":
            model = shared_model
        else:
            model = model_factory()
            model.fit(x_target_full.iloc[train_idx], y_target_full.iloc[train_idx])

        pred = pd.DataFrame(model.predict_proba(x_test, y_test), columns=y_test.columns, index=y_test.index)
        performance.append(evaluator.evaluate(y_test, pred))
        print(f"  fold {fold_idx:02d} | mode={mode}")

    return performance


def main() -> None:
    args = parse_args()
    manifest_path = PROJECT_ROOT / args.manifest
    outdir = PROJECT_ROOT / args.outdir
    project_dir = PROJECT_ROOT / args.project_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # Put professor-code imports before PYTHONPATH=mainfolder so its top-level
    # `utils` package is not shadowed by mainfolder/utils.
    sys.path.insert(0, (project_dir / "parse_results").as_posix())
    sys.path.insert(0, project_dir.as_posix())

    from parse_results.evaluate import Evaluator

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["version"].isin([args.source_version, args.target_version])]

    summary_rows: list[dict[str, object]] = []

    source_rows = manifest[manifest["version"] == args.source_version].set_index("feature_key")
    target_rows = manifest[manifest["version"] == args.target_version].set_index("feature_key")
    common_feature_keys = sorted(set(source_rows.index) & set(target_rows.index))

    for feature_key in common_feature_keys:
        source_row = source_rows.loc[feature_key]
        target_row = target_rows.loc[feature_key]

        if isinstance(source_row, pd.DataFrame) or isinstance(target_row, pd.DataFrame):
            raise ValueError(f"Duplicate manifest rows found for feature_key={feature_key}.")

        x_source_path = resolve_manifest_path(str(source_row["x_path"]))
        x_target_path = resolve_manifest_path(str(target_row["x_path"]))

        x_source = pd.read_csv(x_source_path)
        x_target = pd.read_csv(x_target_path)
        x_source["sample_id"] = x_source["sample_id"].astype(str)
        x_target["sample_id"] = x_target["sample_id"].astype(str)

        y_source_raw = align_y_to_x(x_source_path, args.source_version)
        y_target_raw = align_y_to_x(x_target_path, args.target_version)
        y_source_raw["ID"] = y_source_raw["ID"].astype(str)
        y_target_raw["ID"] = y_target_raw["ID"].astype(str)

        pairs = select_common_label_pairs(
            y_source_raw,
            y_target_raw,
            label_space=args.label_space,
            min_positives=args.min_positives,
            keep_rule=args.keep_rule,
            label_match=args.label_match,
        )
        if not pairs:
            print(f"Skipping {feature_key}: no labels survived common-space selection.")
            continue

        labels = [canonical for _, _, canonical in pairs]
        y_source = y_source_raw[["ID"] + [s for s, _, _ in pairs]]
        y_source.columns = ["ID"] + labels
        y_target = y_target_raw[["ID"] + [t for _, t, _ in pairs]]
        y_target.columns = ["ID"] + labels
        split_path_value = str(target_row.get("split_path", "")).strip()
        split_path = Path(split_path_value) if split_path_value else None
        if split_path is not None and not split_path.is_absolute():
            split_path = PROJECT_ROOT / split_path
        if split_path is not None and split_path.exists():
            splits = load_splits_from_csv(split_path, args.n_splits)
        else:
            splits = make_target_splits(
                x_target, y_target, args.n_splits, args.random_state
            )

        x_source_model = x_source.set_index("sample_id")
        x_target_model = x_target.set_index("sample_id")
        y_source_model = y_source.set_index("ID")
        y_target_model = y_target.set_index("ID")

        print(
            f"\nFeature {feature_key} | labels={len(labels)} "
            f"| source_samples={len(x_source)} | target_samples={len(x_target)}"
        )

        for model_key in args.models:
            model_factory = build_model_factory(model_key)
            evaluator = Evaluator()

            transfer_perf = evaluate_runs(
                evaluator=evaluator,
                model_factory=model_factory,
                x_train_full=x_source_model,
                y_train_full=y_source_model,
                x_target_full=x_target_model,
                y_target_full=y_target_model,
                splits=splits,
                mode="transfer",
            )
            transfer_df = evaluator.create_df_inductive(transfer_perf)
            transfer_path = outdir / (
                f"{feature_key}_{model_key}_{args.source_version}_to_{args.target_version}_transfer_performance.csv"
            )
            transfer_df.to_csv(transfer_path, index=False)

            experiments = [
                (f"{args.source_version}_to_{args.target_version}_transfer", transfer_df, transfer_path),
            ]

            if not args.skip_target_cv:
                target_cv_perf = evaluate_runs(
                    evaluator=evaluator,
                    model_factory=model_factory,
                    x_train_full=x_source_model,
                    y_train_full=y_source_model,
                    x_target_full=x_target_model,
                    y_target_full=y_target_model,
                    splits=splits,
                    mode="target_cv",
                )
                target_cv_df = evaluator.create_df_inductive(target_cv_perf)
                target_cv_path = outdir / (
                    f"{feature_key}_{model_key}_{args.target_version}_common_cv_performance.csv"
                )
                target_cv_df.to_csv(target_cv_path, index=False)
                experiments.append(
                    (f"{args.target_version}_common_cv", target_cv_df, target_cv_path)
                )

            for experiment_name, perf_df, perf_path in [
                *experiments,
            ]:
                mean_row = perf_df[perf_df["folds"] == "mean(std)"].iloc[0].to_dict()
                summary_row: dict[str, object] = {
                    "experiment": experiment_name,
                    "source_version": args.source_version,
                    "target_version": args.target_version,
                    "feature_key": feature_key,
                    "feature_set": str(source_row["feature_set"]),
                    "domains": str(source_row.get("domains", "")),
                    "model": model_key,
                    "label_space": args.label_space,
                    "label_match": args.label_match,
                    "min_positives": args.min_positives,
                    "keep_rule": args.keep_rule,
                    "n_source_samples": int(len(x_source)),
                    "n_target_samples": int(len(x_target)),
                    "n_labels": int(len(labels)),
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

            print(f"Saved {transfer_path}")
            if not args.skip_target_cv:
                print(f"Saved {target_cv_path}")

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["experiment", "feature_set", "model"], kind="stable"
        )
    summary_path = outdir / "transfer_feature_representation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
