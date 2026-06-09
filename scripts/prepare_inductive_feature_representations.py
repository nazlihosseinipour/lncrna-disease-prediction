from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except Exception:  # pragma: no cover - handled at runtime
    MultilabelStratifiedKFold = None


VERSION_CONFIG = {
    "v1": {
        "feature_prefix": "sequences_for_oop",
        "label_csv": Path("Data/output_data/sequences_for_oop.csv"),
        "feature_dir": Path("final_output/v1/rna"),
    },
    "v2": {
        "feature_prefix": "website_full_matrix",
        "label_csv": Path("Data/output_data/website_full_matrix.csv"),
        "feature_dir": Path("final_output/v2/rna"),
    },
}

DEFAULT_FEATURE_SETS = [
    "kmer_matrix_k4",
    "rc_kmer_matrix_k4",
    "psednc_matrix",
]


def sanitize_feature_key(feature_set: str) -> str:
    return feature_set.replace("+", "__")


def resolve_feature_paths(version: str, feature_set: str) -> list[Path]:
    cfg = VERSION_CONFIG[version]
    prefix = cfg["feature_prefix"]
    feature_dir = cfg["feature_dir"]
    paths = []
    for feature_name in feature_set.split("+"):
        path = feature_dir / f"{prefix}_{feature_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        paths.append(path)
    return paths


def load_feature_representation(paths: list[Path], feature_set: str) -> pd.DataFrame:
    merged = None
    parts = feature_set.split("+")
    for feature_name, path in zip(parts, paths):
        df = pd.read_csv(path)
        if "sample_id" not in df.columns:
            raise ValueError(f"{path} is missing required column 'sample_id'.")
        if merged is None:
            merged = df.copy()
            if len(parts) > 1:
                merged = merged.rename(
                    columns={
                        c: f"{feature_name}__{c}"
                        for c in merged.columns
                        if c != "sample_id"
                    }
                )
            continue

        renamed = df.rename(
            columns={c: f"{feature_name}__{c}" for c in df.columns if c != "sample_id"}
        )
        merged = merged.merge(renamed, on="sample_id", how="inner")

    if merged is None:
        raise ValueError(f"No feature files resolved for feature set: {feature_set}")
    return merged


def filter_labels(
    label_csv: Path, min_positives: int, keep_rule: str
) -> tuple[pd.DataFrame, pd.Series]:
    y = pd.read_csv(label_csv)
    y = y.drop(columns=["Unnamed: 0", "seqs"], errors="ignore")
    label_cols = [c for c in y.columns if c != "ID"]
    counts = y[label_cols].sum(axis=0)
    if keep_rule == "gt":
        keep = counts[counts > min_positives].index.tolist()
    else:
        keep = counts[counts >= min_positives].index.tolist()
    if not keep:
        raise ValueError(
            f"No labels survived the filter ({keep_rule} {min_positives}) for {label_csv}."
        )
    return y[["ID"] + keep], counts[keep]


def make_splits(
    x: pd.DataFrame, y: pd.DataFrame, n_splits: int, random_state: int
) -> pd.DataFrame:
    if MultilabelStratifiedKFold is None:
        raise ImportError(
            "iterative-stratification is required for split generation. "
            "Install it with: pip install iterative-stratification"
        )
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    x_model = x.set_index("sample_id")
    y_model = y.set_index("ID")
    columns = [item for i in range(1, n_splits + 1) for item in (f"train_fold{i}", f"test_fold{i}")]
    rows = [index.astype(int) for indexes in splitter.split(x_model, y_model) for index in indexes]
    return pd.DataFrame(rows).T.set_axis(columns, axis=1)


def build_run_commands(x_path: Path, y_path: Path, split_path: Path) -> tuple[str, str]:
    rflda = (
        "python main.py "
        f"../../{x_path.as_posix()} ../../{y_path.as_posix()} ../../{split_path.as_posix()} "
        "false 2>&1 | tee "
        f"{y_path.stem.replace('_Y_', '_')}_rflda.log"
    )
    ipcarf = (
        "python main.py "
        f"../../{x_path.as_posix()} ../../{y_path.as_posix()} ../../{split_path.as_posix()} "
        "false 2>&1 | tee "
        f"{y_path.stem.replace('_Y_', '_')}_ipcarf.log"
    )
    return rflda, ipcarf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare aligned inductive X/Y/splits for multiple RNA feature representations."
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=sorted(VERSION_CONFIG),
        default=["v1", "v2"],
        help="Dataset versions to prepare.",
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        dest="feature_sets",
        default=None,
        help=(
            "Feature representation to prepare. Can be repeated. "
            "Use '+' to concatenate multiple feature files, e.g. "
            "'kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix'."
        ),
    )
    parser.add_argument(
        "--outdir",
        default="inductive_inputs/feature_representations",
        help="Directory for prepared X/Y/splits and manifest.",
    )
    parser.add_argument(
        "--min-positives",
        type=int,
        default=5,
        help="Minimum positive-count threshold applied to disease labels.",
    )
    parser.add_argument(
        "--keep-rule",
        choices=["gt", "ge"],
        default="gt",
        help="Label filter rule: 'gt' means > min-positives, 'ge' means >= min-positives.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=10,
        help="Number of multilabel stratified CV folds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for split generation.",
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Prepare X/Y only and skip split generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_sets = args.feature_sets or list(DEFAULT_FEATURE_SETS)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for version in args.versions:
        cfg = VERSION_CONFIG[version]
        filtered_y, kept_counts = filter_labels(
            cfg["label_csv"], args.min_positives, args.keep_rule
        )

        for feature_set in feature_sets:
            feature_paths = resolve_feature_paths(version, feature_set)
            x = load_feature_representation(feature_paths, feature_set)
            merged = x.merge(filtered_y, left_on="sample_id", right_on="ID", how="inner")

            if merged.empty:
                raise ValueError(
                    f"No overlapping sample IDs between {feature_set} and {cfg['label_csv']} for {version}."
                )

            x_out = merged[x.columns]
            y_out = merged[["sample_id"] + list(kept_counts.index)].rename(
                columns={"sample_id": "ID"}
            )

            feature_key = sanitize_feature_key(feature_set)
            suffix = f"{args.keep_rule}{args.min_positives}"
            x_path = outdir / f"{version}_{feature_key}_X.csv"
            y_path = outdir / f"{version}_{feature_key}_Y_{suffix}.csv"
            x_out.to_csv(x_path, index=False)
            y_out.to_csv(y_path, index=False)

            split_path = None
            if not args.no_splits:
                split_path = outdir / f"{version}_{feature_key}_X_splits.csv"
                splits = make_splits(x_out, y_out, args.n_splits, args.random_state)
                splits.to_csv(split_path, index=False)

            rflda_cmd, ipcarf_cmd = build_run_commands(
                x_path, y_path, split_path or Path("<generate_splits_first>")
            )

            manifest_rows.append(
                {
                    "version": version,
                    "feature_set": feature_set,
                    "feature_key": feature_key,
                    "x_path": x_path.as_posix(),
                    "y_path": y_path.as_posix(),
                    "split_path": "" if split_path is None else split_path.as_posix(),
                    "n_samples": int(x_out.shape[0]),
                    "n_features": int(x_out.shape[1] - 1),
                    "n_labels": int(y_out.shape[1] - 1),
                    "min_label_positives": int(kept_counts.min()),
                    "median_label_positives": float(kept_counts.median()),
                    "max_label_positives": int(kept_counts.max()),
                    "labels_lt_n_splits": int((kept_counts < args.n_splits).sum()),
                    "rflda_command": rflda_cmd,
                    "ipcarf_command": ipcarf_cmd,
                }
            )

            print(
                f"{version} | {feature_set} -> samples={x_out.shape[0]} "
                f"features={x_out.shape[1] - 1} labels={y_out.shape[1] - 1}"
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = outdir / "feature_representation_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nSaved manifest: {manifest_path}")
    if not manifest.empty:
        warned = manifest["labels_lt_n_splits"].sum()
        if warned:
            print(
                "Warning: some prepared label spaces still contain labels with "
                "fewer positives than the number of folds; fold-level ROC/Youden "
                "warnings may still appear during evaluation."
            )


if __name__ == "__main__":
    main()
