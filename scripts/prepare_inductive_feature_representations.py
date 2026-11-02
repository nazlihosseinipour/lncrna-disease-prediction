from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
        candidate_path = Path(candidate)
        paths = (
            [candidate_path]
            if candidate_path.is_absolute()
            else [PROJECT_ROOT / candidate_path, PROJECT_ROOT / "final_output" / candidate_path]
        )
        for path in paths:
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


def load_label_source(path: Path) -> pd.DataFrame:
    """Load IDs, sequences (when available), and every candidate label.

    Label filtering must happen *after* feature alignment and sequence
    deduplication.  Filtering the source table first made the manifest counts
    stale whenever feature extraction dropped samples.
    """
    df = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
    if "ID" not in df.columns:
        raise ValueError(f"{path} is missing required ID column.")
    df["ID"] = df["ID"].astype(str).str.strip()
    if df["ID"].duplicated().any():
        examples = df.loc[df["ID"].duplicated(False), "ID"].head(8).tolist()
        raise ValueError(f"{path} contains duplicate IDs: {examples}")
    return df


def normalize_sequence_key(value: object) -> str:
    """Return a stable RNA sequence identity key without exposing sequence text."""
    raw = "" if pd.isna(value) else str(value)
    sequence = "".join(raw.split()).upper().replace("T", "U")
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else ""


def collapse_identical_sequences(
    x: pd.DataFrame,
    y: pd.DataFrame,
    sequences: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Collapse identical biological inputs and OR their association labels.

    Identical sequences cannot be independent train/test samples.  Keeping them
    as separate rows caused both leakage and contradictory targets in V2.  The
    first ID is retained deterministically and all positive associations from
    its aliases are preserved with a logical OR.
    """
    if len(x) != len(y):
        raise ValueError("X/Y row counts differ before sequence deduplication.")
    if x["sample_id"].astype(str).tolist() != y["ID"].astype(str).tolist():
        raise ValueError("X/Y IDs are not aligned before sequence deduplication.")

    label_cols = [c for c in y.columns if c != "ID"]
    feature_cols = [c for c in x.columns if c != "sample_id"]
    if sequences is None:
        keys = [f"id:{sample_id}" for sample_id in x["sample_id"].astype(str)]
    else:
        if len(sequences) != len(x):
            raise ValueError("Sequence/X row counts differ before deduplication.")
        keys = [
            normalize_sequence_key(seq) or f"id:{sample_id}"
            for seq, sample_id in zip(sequences, x["sample_id"].astype(str))
        ]

    groups: dict[str, list[int]] = {}
    for position, key in enumerate(keys):
        groups.setdefault(key, []).append(position)

    x_rows: list[pd.Series] = []
    y_rows: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []
    for key, positions in groups.items():
        representative = positions[0]
        representative_features = x.iloc[representative][feature_cols].to_numpy(dtype=float)
        for position in positions[1:]:
            candidate = x.iloc[position][feature_cols].to_numpy(dtype=float)
            if not np.allclose(
                representative_features,
                candidate,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                ids = x.iloc[positions]["sample_id"].astype(str).tolist()
                raise ValueError(
                    "Identical source sequences produced different feature rows: "
                    f"{ids}. Regenerate the feature representation before continuing."
                )

        x_row = x.iloc[representative].copy()
        y_row = y.iloc[representative].copy()
        if len(positions) > 1:
            group_y = y.iloc[positions][label_cols].apply(pd.to_numeric, errors="raise")
            y_row.loc[label_cols] = group_y.max(axis=0).astype(int)
            varying = int((group_y.max(axis=0) != group_y.min(axis=0)).sum())
            member_ids = x.iloc[positions]["sample_id"].astype(str).tolist()
            audit_rows.append(
                {
                    "representative_id": member_ids[0],
                    "member_ids": "|".join(member_ids),
                    "group_size": len(member_ids),
                    "conflicting_label_count": varying,
                    "sequence_sha256": key if not key.startswith("id:") else "",
                }
            )
        x_rows.append(x_row)
        y_rows.append(y_row)

    x_out = pd.DataFrame(x_rows, columns=x.columns).reset_index(drop=True)
    y_out = pd.DataFrame(y_rows, columns=y.columns).reset_index(drop=True)
    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "representative_id",
            "member_ids",
            "group_size",
            "conflicting_label_count",
            "sequence_sha256",
        ],
    )
    return x_out, y_out, audit


def drop_duplicate_feature_columns(
    x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove exact duplicate columns so concatenation cannot overweight a block."""
    feature_cols = [c for c in x.columns if c != "sample_id"]
    transposed = x[feature_cols].T
    duplicate_mask = transposed.duplicated(keep="first")
    dropped = transposed.index[duplicate_mask].tolist()
    audit_rows: list[dict[str, str]] = []
    retained = transposed.index[~duplicate_mask].tolist()
    for duplicate in dropped:
        duplicate_values = transposed.loc[duplicate]
        original = next(
            name for name in retained if transposed.loc[name].equals(duplicate_values)
        )
        audit_rows.append(
            {"retained_feature": original, "dropped_duplicate_feature": duplicate}
        )
    return (
        x.drop(columns=dropped),
        pd.DataFrame(
            audit_rows,
            columns=["retained_feature", "dropped_duplicate_feature"],
        ),
    )


def audit_target_contamination(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    min_block_positive_f1: float = 0.5,
    min_block_exact_row_fraction: float = 0.5,
) -> dict[str, object]:
    """Detect label-matrix data accidentally embedded in an input feature file.

    The published 827-column V1 artifact contains a contiguous 285-column
    binary block that tracks the raw 285-label association matrix. Two invalid
    sequences shifted part of that block relative to the IDs, so checking only
    for perfectly identical feature/target columns is insufficient. This audit
    checks both informative exact column copies and label-width contiguous
    binary blocks with substantial positive overlap and row-wise identity.

    Sequence descriptors can legitimately be binary, so a block is rejected
    only when both its positive-class F1 and exact-row fraction are high.
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("X/Y row counts differ during target-contamination audit.")
    if "sample_id" not in x.columns or "ID" not in y.columns:
        raise ValueError("Target-contamination audit requires sample_id in X and ID in Y.")
    if x["sample_id"].astype(str).tolist() != y["ID"].astype(str).tolist():
        raise ValueError("X/Y IDs are not aligned during target-contamination audit.")

    feature_cols = [column for column in x.columns if column != "sample_id"]
    label_cols = [column for column in y.columns if column != "ID"]
    feature_values = x[feature_cols].apply(pd.to_numeric, errors="raise")
    label_values = y[label_cols].apply(pd.to_numeric, errors="raise")
    if not label_values.isin([0, 1]).all(axis=None):
        raise ValueError("Target-contamination audit requires binary labels.")

    binary_mask = feature_values.isin([0, 1]).all(axis=0).to_numpy()
    binary_feature_cols = [
        column for column, is_binary in zip(feature_cols, binary_mask) if is_binary
    ]

    # Constant-zero columns are deliberately ignored: an unused label and a
    # legitimate all-zero descriptor would otherwise look like an exact copy.
    informative_labels = [
        column for column in label_cols if 0 < int(label_values[column].sum()) < len(y)
    ]
    exact_copy_pairs: list[tuple[str, str]] = []
    for feature_column in binary_feature_cols:
        feature = feature_values[feature_column].to_numpy(dtype=np.uint8)
        if feature.min() == feature.max():
            continue
        for label_column in informative_labels:
            if np.array_equal(
                feature,
                label_values[label_column].to_numpy(dtype=np.uint8),
            ):
                exact_copy_pairs.append((feature_column, label_column))

    best_block_positive_f1 = 0.0
    best_block_exact_row_fraction = 0.0
    best_block_start = ""
    label_width = len(label_cols)
    labels_array = label_values.to_numpy(dtype=np.uint8)

    # Find contiguous binary runs, then test every label-width window. In clean
    # sequence-only inputs this loop normally has no eligible window.
    run_start = 0
    while run_start < len(feature_cols):
        if not binary_mask[run_start]:
            run_start += 1
            continue
        run_end = run_start
        while run_end + 1 < len(feature_cols) and binary_mask[run_end + 1]:
            run_end += 1
        run_length = run_end - run_start + 1
        if label_width and run_length >= label_width:
            for window_start in range(run_start, run_end - label_width + 2):
                block = feature_values.iloc[
                    :, window_start : window_start + label_width
                ].to_numpy(dtype=np.uint8)
                true_positive = int(np.logical_and(block == 1, labels_array == 1).sum())
                false_positive = int(np.logical_and(block == 1, labels_array == 0).sum())
                false_negative = int(np.logical_and(block == 0, labels_array == 1).sum())
                denominator = 2 * true_positive + false_positive + false_negative
                positive_f1 = 2 * true_positive / denominator if denominator else 0.0
                exact_row_fraction = float(np.all(block == labels_array, axis=1).mean())
                if (
                    positive_f1 > best_block_positive_f1
                    or (
                        positive_f1 == best_block_positive_f1
                        and exact_row_fraction > best_block_exact_row_fraction
                    )
                ):
                    best_block_positive_f1 = positive_f1
                    best_block_exact_row_fraction = exact_row_fraction
                    best_block_start = feature_cols[window_start]
        run_start = run_end + 1

    suspicious_block = (
        best_block_positive_f1 >= min_block_positive_f1
        and best_block_exact_row_fraction >= min_block_exact_row_fraction
    )
    return {
        "flagged": bool(exact_copy_pairs or suspicious_block),
        "binary_feature_columns": len(binary_feature_cols),
        "exact_label_copy_count": len(exact_copy_pairs),
        "exact_label_copy_examples": "|".join(
            f"{feature}->{label}" for feature, label in exact_copy_pairs[:5]
        ),
        "best_label_width_block_start": best_block_start,
        "best_label_width_block_positive_f1": best_block_positive_f1,
        "best_label_width_block_exact_row_fraction": best_block_exact_row_fraction,
    }


def assert_no_target_contamination(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    context: str,
) -> dict[str, object]:
    """Run the target audit and fail before writing a contaminated manifest."""
    audit = audit_target_contamination(x, y)
    if audit["flagged"]:
        raise ValueError(
            f"{context}: feature input appears to contain target-derived data "
            f"(binary_features={audit['binary_feature_columns']}, "
            f"exact_label_copies={audit['exact_label_copy_count']}, "
            f"best_label_width_block_positive_f1="
            f"{audit['best_label_width_block_positive_f1']:.3f}, "
            f"best_label_width_block_exact_row_fraction="
            f"{audit['best_label_width_block_exact_row_fraction']:.3f}). "
            "Regenerate features from sequences with explicit sample IDs; do not "
            "remove the suspicious columns and reuse potentially shifted rows."
        )
    return audit


def filter_aligned_labels(
    y: pd.DataFrame, min_positives: int, keep_rule: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Filter labels using counts from the final aligned, deduplicated samples."""
    label_cols = [c for c in y.columns if c != "ID"]
    numeric = y[label_cols].apply(pd.to_numeric, errors="raise")
    counts = numeric.sum(axis=0)
    keep_mask = counts > min_positives if keep_rule == "gt" else counts >= min_positives
    keep = counts.index[keep_mask].tolist()
    if not keep:
        raise ValueError(
            f"No aligned labels survived the filter ({keep_rule} {min_positives})."
        )
    dropped = counts.loc[~keep_mask].rename("aligned_positive_count").reset_index()
    dropped = dropped.rename(columns={"index": "label"})
    return y[["ID"] + keep].copy(), counts.loc[keep], dropped


def preserve_aligned_inputs(
    x_aligned: pd.DataFrame,
    y_aligned: pd.DataFrame,
    *,
    source_ids: list[str],
    label_cols: list[str],
    context: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return aligned inputs unchanged, failing if alignment removed or reordered rows."""
    aligned_ids = x_aligned["sample_id"].astype(str).tolist()
    if aligned_ids != source_ids:
        aligned_set = set(aligned_ids)
        source_set = set(source_ids)
        missing = [sample_id for sample_id in source_ids if sample_id not in aligned_set]
        extra = [sample_id for sample_id in aligned_ids if sample_id not in source_set]
        raise ValueError(
            f"{context}: preservation mode would change source rows. Expected "
            f"{len(source_ids)} IDs in source order, got {len(aligned_ids)}; "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    sequence_audit = pd.DataFrame(
        columns=[
            "representative_id",
            "member_ids",
            "group_size",
            "conflicting_label_count",
            "sequence_sha256",
        ]
    )
    feature_audit = pd.DataFrame(
        columns=["retained_feature", "dropped_duplicate_feature"]
    )
    kept_counts = y_aligned[label_cols].apply(pd.to_numeric, errors="raise").sum(axis=0)
    dropped_labels = pd.DataFrame(columns=["label", "aligned_positive_count"])
    return (
        x_aligned.copy(),
        y_aligned.copy(),
        sequence_audit,
        feature_audit,
        kept_counts,
        dropped_labels,
    )


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
    for version in sorted(VERSION_CONFIG):
        parser.add_argument(
            f"--{version}-label-csv",
            help=f"Override the {version.upper()} label/sequence source CSV.",
        )
        parser.add_argument(
            f"--{version}-feature-dir",
            help=f"Override the {version.upper()} feature-version directory.",
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
        "--preserve-rows-and-columns",
        action="store_true",
        help="Preserve every source row, feature column, and label column. Fail if "
             "feature availability or ID alignment would remove a row. This disables "
             "sequence collapsing, duplicate-feature removal, and label-frequency filtering.",
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
    for version in sorted(VERSION_CONFIG):
        label_override = getattr(args, f"{version}_label_csv")
        feature_override = getattr(args, f"{version}_feature_dir")
        if label_override:
            VERSION_CONFIG[version]["label_csv"] = Path(label_override)
        if feature_override:
            VERSION_CONFIG[version]["version_dirs"] = (feature_override,)
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []

    for version in args.versions:
        cfg = VERSION_CONFIG[version]
        label_csv = PROJECT_ROOT / cfg["label_csv"]
        label_source = load_label_source(label_csv)
        label_ids_in_order = label_source["ID"].astype(str).tolist()
        sequence_col = "seqs" if "seqs" in label_source.columns else None
        label_cols = [
            c for c in label_source.columns if c not in {"ID", "seqs", "seq"}
        ]

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
            merge_cols = ["ID", *label_cols]
            if sequence_col:
                merge_cols.append(sequence_col)
            merged = x.merge(
                label_source[merge_cols],
                left_on="sample_id",
                right_on="ID",
                how="inner",
                validate="one_to_one",
            )

            if merged.empty:
                raise ValueError(
                    f"No overlapping sample IDs between {feature_set} and {label_csv} for {version}."
                )

            x_aligned = merged[x.columns].copy()
            y_aligned = merged[["sample_id", *label_cols]].rename(
                columns={"sample_id": "ID"}
            )
            sequences = merged[sequence_col] if sequence_col else None

            target_audit = assert_no_target_contamination(
                x_aligned,
                y_aligned,
                context=f"{version}/{feature_set}",
            )

            if args.preserve_rows_and_columns:
                (
                    x_out,
                    y_out,
                    sequence_audit,
                    feature_audit,
                    kept_counts,
                    dropped_labels,
                ) = preserve_aligned_inputs(
                    x_aligned,
                    y_aligned,
                    source_ids=label_ids_in_order,
                    label_cols=label_cols,
                    context=f"{version}/{feature_set}",
                )
            else:
                x_out, y_out, sequence_audit = collapse_identical_sequences(
                    x_aligned, y_aligned, sequences
                )
                x_out, feature_audit = drop_duplicate_feature_columns(x_out)
                y_out, kept_counts, dropped_labels = filter_aligned_labels(
                    y_out, args.min_positives, args.keep_rule
                )

            if x_out["sample_id"].astype(str).tolist() != y_out["ID"].astype(str).tolist():
                raise ValueError(f"{version}/{feature_set}: final X/Y IDs are not aligned.")

            feature_key = sanitize_feature_key(feature_set)
            suffix = (
                "unfiltered"
                if args.preserve_rows_and_columns
                else f"{args.keep_rule}{args.min_positives}"
            )
            x_path = outdir / f"{version}_{feature_key}_X.csv"
            y_path = outdir / f"{version}_{feature_key}_Y_{suffix}.csv"
            x_out.to_csv(x_path, index=False)
            y_out.to_csv(y_path, index=False)

            sequence_audit_path = outdir / f"{version}_{feature_key}_duplicate_sequences.csv"
            feature_audit_path = outdir / f"{version}_{feature_key}_duplicate_features.csv"
            dropped_labels_path = outdir / f"{version}_{feature_key}_dropped_labels.csv"
            sequence_audit.to_csv(sequence_audit_path, index=False)
            feature_audit.to_csv(feature_audit_path, index=False)
            dropped_labels.to_csv(dropped_labels_path, index=False)

            split_path = None
            if not args.no_splits:
                split_path = outdir / f"{version}_{feature_key}_X_splits.csv"
                splits = make_splits(x_out, y_out, args.n_splits, args.random_state)
                splits.to_csv(split_path, index=False)

            def project_path(path: Path) -> str:
                try:
                    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
                except ValueError:
                    return path.resolve().as_posix()

            command_x_path = Path(project_path(x_path))
            command_y_path = Path(project_path(y_path))
            command_split_path = (
                Path(project_path(split_path))
                if split_path is not None
                else Path("<generate_splits_first>")
            )
            rflda_cmd, ipcarf_cmd = build_run_commands(
                command_x_path, command_y_path, command_split_path
            )

            manifest_rows.append(
                {
                    "version": version,
                    "feature_set": feature_set,
                    "feature_key": feature_key,
                    "domains": "+".join(spec.domain for spec in specs),
                    "components": "+".join(spec.feature_name for spec in specs),
                    "x_path": project_path(x_path),
                    "y_path": project_path(y_path),
                    "split_path": "" if split_path is None else project_path(split_path),
                    "preprocessing_mode": (
                        "preserve_rows_and_columns"
                        if args.preserve_rows_and_columns
                        else "aligned_filtered"
                    ),
                    "n_samples": int(x_out.shape[0]),
                    "n_features": int(x_out.shape[1] - 1),
                    "n_labels": int(y_out.shape[1] - 1),
                    "min_label_positives": int(kept_counts.min()),
                    "median_label_positives": float(kept_counts.median()),
                    "max_label_positives": int(kept_counts.max()),
                    "labels_lt_n_splits": int((kept_counts < args.n_splits).sum()),
                    "source_samples": int(label_source.shape[0]),
                    "aligned_samples_before_sequence_dedup": int(x_aligned.shape[0]),
                    "duplicate_sequence_groups": int(sequence_audit.shape[0]),
                    "collapsed_duplicate_sequence_rows": int(
                        sequence_audit["group_size"].sub(1).sum()
                        if not sequence_audit.empty
                        else 0
                    ),
                    "duplicate_feature_columns_dropped": int(feature_audit.shape[0]),
                    "binary_feature_columns_audited": int(
                        target_audit["binary_feature_columns"]
                    ),
                    "exact_label_copy_pairs": int(
                        target_audit["exact_label_copy_count"]
                    ),
                    "best_label_width_block_positive_f1": float(
                        target_audit["best_label_width_block_positive_f1"]
                    ),
                    "best_label_width_block_exact_row_fraction": float(
                        target_audit["best_label_width_block_exact_row_fraction"]
                    ),
                    "duplicate_sequence_audit": project_path(sequence_audit_path),
                    "duplicate_feature_audit": project_path(feature_audit_path),
                    "dropped_label_audit": project_path(dropped_labels_path),
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
