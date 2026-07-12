from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except Exception:  # pragma: no cover - handled at runtime
    MultilabelStratifiedKFold = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VERSION_CONFIG = {
    "v1": {
        "label_csv": Path("Data/output_data/sequences_for_oop.csv"),
        "feature_prefix": "sequences_for_oop",
        "version_dirs": ("v1", "V1"),
    },
    "v2": {
        "label_csv": Path("Data/output_data/website_full_matrix.csv"),
        "feature_prefix": "website_full_matrix",
        "version_dirs": ("v2", "V2"),
    },
}

DOMAIN_CONFIG = {
    "rna": {
        "dir_names": ("rna", "Rna"),
        "summary_file": "rna_feature_shapes.csv",
    },
    "disease": {
        "dir_names": ("disease", "Dis"),
        "summary_file": "disease_feature_shapes.csv",
    },
    "cross": {
        "dir_names": ("cross", "Cross"),
        "summary_file": "cross_feature_shapes.csv",
    },
    "nn": {
        "dir_names": ("nn", "NN"),
        "summary_file": "nn_feature_shapes.csv",
    },
}

DEFAULT_FEATURE_SETS = [
    "kmer_matrix_k4",
    "rc_kmer_matrix_k4",
    "psednc_matrix",
]

# Features derived from the lncRNA-disease association matrix Y. Using them as inputs
# leaks the target, so they are removed from the catalogue. See the leakage audit.
LEAKY_FEATURE_NAMES = frozenset(
    {"svd_lncRNA", "svd_disease", "gip_lncRNA", "gip_disease", "lfs_from_Y"}
)


@dataclass(frozen=True)
class FeatureSpec:
    version: str
    domain: str
    feature_name: str
    method_id: int | None
    source_name: str
    output_path: Path
    loader_kind: str
    reason: str


def sanitize_feature_key(feature_set: str) -> str:
    key = feature_set.replace("+", "__").replace("::", "__")
    # A physical concatenation of every safe representation exceeds common
    # filesystem component limits if all component names are embedded in filenames.
    # The manifest retains the complete ordered component list.
    if len(key.encode("utf-8")) > 180:
        return "all_safe_concatenated"
    return key


def resolve_version_dir(version: str) -> Path:
    for candidate in VERSION_CONFIG[version]["version_dirs"]:
        path = PROJECT_ROOT / "final_output" / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find final_output directory for version {version}.")


def resolve_domain_dir(version: str, domain: str) -> Path:
    version_dir = resolve_version_dir(version)
    for candidate in DOMAIN_CONFIG[domain]["dir_names"]:
        path = version_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {domain} directory under {version_dir}. Tried: {DOMAIN_CONFIG[domain]['dir_names']}"
    )


def resolve_output_path(version: str, domain: str, output_path: str) -> Path:
    candidate = Path(output_path)
    absolute = PROJECT_ROOT / candidate
    if absolute.exists():
        return absolute

    domain_dir = resolve_domain_dir(version, domain)
    basename = candidate.name
    basename_candidate = domain_dir / basename
    if basename_candidate.exists():
        return basename_candidate

    raise FileNotFoundError(
        f"Could not resolve output_path={output_path!r} for version={version}, domain={domain}."
    )


def load_label_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0", "seqs"], errors="ignore")
    if "ID" not in df.columns:
        raise ValueError(f"{path} is missing required ID column.")
    df["ID"] = df["ID"].astype(str)
    return df


def filter_labels(
    label_csv: Path, min_positives: int, keep_rule: str
) -> tuple[pd.DataFrame, pd.Series]:
    y = load_label_csv(label_csv)
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
    columns = [
        item
        for i in range(1, n_splits + 1)
        for item in (f"train_fold{i}", f"test_fold{i}")
    ]
    rows = [
        index.astype(int)
        for indexes in splitter.split(x_model, y_model)
        for index in indexes
    ]
    return pd.DataFrame(rows).T.set_axis(columns, axis=1)


def format_main_path_arg(path: Path) -> str:
    if path.is_absolute():
        return path.as_posix()
    return f"../../{path.as_posix()}"


def build_run_commands(x_path: Path, y_path: Path, split_path: Path) -> tuple[str, str]:
    rflda = (
        "python main.py "
        f"{format_main_path_arg(x_path)} {format_main_path_arg(y_path)} {format_main_path_arg(split_path)} "
        "false 2>&1 | tee "
        f"{y_path.stem.replace('_Y_', '_')}_rflda.log"
    )
    ipcarf = (
        "python main.py "
        f"{format_main_path_arg(x_path)} {format_main_path_arg(y_path)} {format_main_path_arg(split_path)} "
        "false 2>&1 | tee "
        f"{y_path.stem.replace('_Y_', '_')}_ipcarf.log"
    )
    return rflda, ipcarf


def classify_feature(domain: str, feature_name: str, method_id: int | None) -> tuple[str | None, str]:
    if domain == "rna":
        return "sample_id_csv", "sample-level RNA representation"

    if domain == "nn":
        if feature_name == "mp_tokens" or method_id == 101:
            return None, "token-level NN output; not a one-row-per-sample representation"
        return "sample_id_csv", "sample-level NN representation"

    # Association-derived features (SVD/GIP/LFS) are computed from the label matrix Y
    # and therefore leak the target if used as inputs. They are excluded from the
    # catalogue so they can never enter a manifest. See results/audit/leakage_audit_report.md.
    if feature_name in LEAKY_FEATURE_NAMES:
        return None, "association-derived feature (built from Y); excluded to prevent target leakage"

    if domain == "disease":
        return None, "disease-level matrix; not a valid X matrix for the current multilabel setup"

    if domain == "cross":
        return None, "disease-level cross feature; not a valid X matrix for the current multilabel setup"

    return None, "unsupported domain"


def discover_feature_specs(
    version: str, domains: list[str]
) -> tuple[dict[str, FeatureSpec], list[dict[str, object]]]:
    specs: dict[str, FeatureSpec] = {}
    inventory_rows: list[dict[str, object]] = []

    for domain in domains:
        try:
            domain_dir = resolve_domain_dir(version, domain)
        except FileNotFoundError:
            continue
        summary_path = domain_dir / DOMAIN_CONFIG[domain]["summary_file"]
        if not summary_path.exists():
            continue

        summary_df = pd.read_csv(summary_path).fillna("")
        for record in summary_df.to_dict("records"):
            feature_name = str(record.get("feature_name", "")).strip()
            method_raw = str(record.get("method_id", "")).strip()
            method_id = int(method_raw) if method_raw else None
            loader_kind, reason = classify_feature(domain, feature_name, method_id)
            resolved_path = resolve_output_path(
                version, domain, str(record.get("output_path", ""))
            )
            spec = FeatureSpec(
                version=version,
                domain=domain,
                feature_name=feature_name,
                method_id=method_id,
                source_name=str(record.get("source_name", "")),
                output_path=resolved_path,
                loader_kind=loader_kind or "",
                reason=reason,
            )
            inventory_rows.append(
                {
                    "version": version,
                    "domain": domain,
                    "feature_name": feature_name,
                    "method_id": method_id,
                    "source_name": spec.source_name,
                    "output_path": spec.output_path.as_posix(),
                    "supported": bool(loader_kind),
                    "loader_kind": loader_kind or "",
                    "reason": reason,
                }
            )
            if not loader_kind:
                continue

            specs[feature_name] = spec
            specs[f"{domain}::{feature_name}"] = spec

    return specs, inventory_rows


def select_feature_sets(
    *,
    feature_specs: dict[str, FeatureSpec],
    explicit_feature_sets: list[str] | None,
    all_compatible: bool,
) -> list[str]:
    # --all-compatible and explicit --feature-set are unioned (dedup, order-preserving)
    # so a single invocation can sweep every leakage-free single feature AND include
    # an explicit concatenation like "kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix".
    sets: list[str] = []
    if all_compatible:
        canonical = {
            spec.feature_name
            for key, spec in feature_specs.items()
            if "::" not in key
        }
        sets.extend(sorted(canonical))
    if explicit_feature_sets:
        for feature_set in explicit_feature_sets:
            if feature_set not in sets:
                sets.append(feature_set)
    if sets:
        return sets
    return list(DEFAULT_FEATURE_SETS)


def load_feature_frame(spec: FeatureSpec, label_ids_in_order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(spec.output_path)

    if spec.loader_kind == "sample_id_csv":
        if "sample_id" not in df.columns:
            raise ValueError(f"{spec.output_path} is missing required column 'sample_id'.")
        df["sample_id"] = df["sample_id"].astype(str)
        return df

    if spec.loader_kind == "id_first_col":
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "sample_id"})
        df["sample_id"] = df["sample_id"].astype(str)
        return df

    if spec.loader_kind == "sample_square_matrix":
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "sample_id"})
        df["sample_id"] = df["sample_id"].astype(str)
        label_id_set = set(label_ids_in_order)
        overlap = int(df["sample_id"].isin(label_id_set).sum())
        if overlap == 0:
            if len(df) != len(label_ids_in_order):
                raise ValueError(
                    f"{spec.output_path} does not expose sample IDs and row count {len(df)} "
                    f"does not match label ID count {len(label_ids_in_order)}."
                )
            df["sample_id"] = label_ids_in_order
        return df

    if spec.loader_kind == "row_order_from_labels":
        if "sample_id" in df.columns:
            df["sample_id"] = df["sample_id"].astype(str)
            return df
        if len(df) != len(label_ids_in_order):
            raise ValueError(
                f"{spec.output_path} row count {len(df)} does not match label ID count {len(label_ids_in_order)}."
            )
        df = df.copy()
        df.insert(0, "sample_id", label_ids_in_order)
        df["sample_id"] = df["sample_id"].astype(str)
        return df

    raise ValueError(f"Unsupported loader kind {spec.loader_kind!r} for {spec.output_path}.")


def load_feature_representation(
    specs: list[FeatureSpec],
    feature_set: str,
    label_ids_in_order: list[str],
) -> pd.DataFrame:
    merged = None
    parts = feature_set.split("+")
    for feature_name, spec in zip(parts, specs):
        df = load_feature_frame(spec, label_ids_in_order)
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
        raise ValueError(f"No feature files resolved for feature set {feature_set}.")
    return merged


def resolve_feature_specs(
    *,
    version: str,
    feature_set: str,
    feature_specs: dict[str, FeatureSpec],
) -> list[FeatureSpec]:
    resolved: list[FeatureSpec] = []
    for part in feature_set.split("+"):
        key = part.strip()
        component = key.split("::")[-1]
        if component in LEAKY_FEATURE_NAMES:
            raise ValueError(
                f"{version}: feature component {component!r} is association-derived "
                f"(built from the label matrix Y) and is excluded to prevent target "
                f"leakage. See results/audit/leakage_audit_report.md."
            )
        spec = feature_specs.get(key)
        if spec is None:
            available = sorted(k for k in feature_specs if "::" not in k)
            raise KeyError(
                f"{version}: unknown feature component {key!r}. "
                f"Available supported features: {', '.join(available)}"
            )
        resolved.append(spec)
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare aligned inductive X/Y/splits for sample-level feature representations "
            "across RNA, NN, disease, and cross outputs."
        )
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=sorted(VERSION_CONFIG),
        default=["v1", "v2"],
        help="Dataset versions to prepare.",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(DOMAIN_CONFIG),
        default=["rna", "disease", "cross", "nn"],
        help="Feature domains to scan for compatible sample-level outputs.",
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        dest="feature_sets",
        default=None,
        help=(
            "Feature representation to prepare. Can be repeated. Use '+' to concatenate "
            "multiple feature files, e.g. 'kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix'. "
            "You can also disambiguate with domain::feature_name."
        ),
    )
    parser.add_argument(
        "--all-compatible",
        action="store_true",
        help="Prepare every compatible sample-level feature discovered in the selected domains.",
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
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []

    for version in args.versions:
        cfg = VERSION_CONFIG[version]
        label_csv = PROJECT_ROOT / cfg["label_csv"]
        filtered_y, kept_counts = filter_labels(
            label_csv, args.min_positives, args.keep_rule
        )
        label_ids_in_order = load_label_csv(label_csv)["ID"].astype(str).tolist()

        feature_specs, discovered_inventory = discover_feature_specs(version, args.domains)
        inventory_rows.extend(discovered_inventory)
        feature_sets = select_feature_sets(
            feature_specs=feature_specs,
            explicit_feature_sets=args.feature_sets,
            all_compatible=args.all_compatible,
        )

        for feature_set in feature_sets:
            specs = resolve_feature_specs(
                version=version,
                feature_set=feature_set,
                feature_specs=feature_specs,
            )
            x = load_feature_representation(specs, feature_set, label_ids_in_order)
            merged = x.merge(filtered_y, left_on="sample_id", right_on="ID", how="inner")

            if merged.empty:
                raise ValueError(
                    f"No overlapping sample IDs between {feature_set} and {label_csv} for {version}."
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
                    "domains": "+".join(spec.domain for spec in specs),
                    "components": "+".join(spec.feature_name for spec in specs),
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

    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["version", "domain", "feature_name"], kind="stable"
    )
    inventory_path = outdir / "feature_inventory.csv"
    inventory.to_csv(inventory_path, index=False)

    print(f"\nSaved manifest: {manifest_path}")
    print(f"Saved feature inventory: {inventory_path}")

    if not manifest.empty:
        warned = int(manifest["labels_lt_n_splits"].sum())
        if warned:
            print(
                "Warning: some prepared label spaces still contain labels with "
                "fewer positives than the number of folds; fold-level ROC/Youden "
                "warnings may still appear during evaluation."
            )


if __name__ == "__main__":
    main()
