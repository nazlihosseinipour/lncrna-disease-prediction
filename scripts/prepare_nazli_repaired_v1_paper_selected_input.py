#!/usr/bin/env python3
"""Build the complete repaired V1 sequence-only input from Nazli's data.

This preparer is intentionally write-once: the output directory must not
already exist.  Nazli's processed sequence table is the sequence source.  The
released paper labels supply the exact RNA identifiers and their order; the
single known trailing-space discrepancy is repaired explicitly before any
exact-ID alignment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

try:
    from scripts.reconstruct_paper_sequence_features import normalize_sequence, reconstruct
except ModuleNotFoundError:  # Direct execution: python scripts/prepare_....py
    from reconstruct_paper_sequence_features import normalize_sequence, reconstruct


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "nazli_repaired_v1_felipe_2026-10-05" / "inputs"
SEQUENCES = ROOT / "Data" / "output_data" / "sequences_for_oop.csv"
PROPERTIES = ROOT / "config" / "pse_in_one_rna_dcc_properties.csv"
PAPER_Y = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv"
PAPER_SPLITS = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/splits/inductive/inductive_splits.csv"
ACCEPTED_353_X = (
    ROOT
    / "results/nazli_selected_v1_felipe_2026-10-05/inputs"
    / "v1_nazli_selected_paper_sequence_542_X.csv"
)
INDEPENDENT_RECOVERED_355_X = (
    ROOT / "results/felipe_basics_2026-09-27/inputs/v1_paper_sequence_542_X.csv"
)

EXPECTED_ROWS = 355
EXPECTED_FEATURES = 542
EXPECTED_LABELS = 44
KNOWN_ID_REPAIRS = {"BX374997.2": "BX374997.2 "}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def repaired_sequences_and_labels() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    """Return Nazli sequences keyed to the paper's exact 355 identifiers."""
    source = pd.read_csv(SEQUENCES, dtype={"ID": str}, keep_default_na=False)
    if len(source) != EXPECTED_ROWS or source["ID"].nunique() != EXPECTED_ROWS:
        raise ValueError("Nazli's processed V1 source must contain 355 unique RNA IDs")
    if not {"ID", "seqs"}.issubset(source.columns):
        raise ValueError("Nazli's processed V1 source must contain ID and seqs columns")

    sequences = source[["ID", "seqs"]].copy()
    repairs: list[dict[str, str]] = []
    for source_id, paper_id in KNOWN_ID_REPAIRS.items():
        count = int((sequences["ID"] == source_id).sum())
        if count != 1 or (sequences["ID"] == paper_id).any():
            raise ValueError(
                f"Expected exactly one {source_id!r} and no pre-existing {paper_id!r}"
            )
        sequences.loc[sequences["ID"] == source_id, "ID"] = paper_id
        repairs.append({"source_id": source_id, "paper_exact_id": paper_id})

    paper_y = pd.read_csv(PAPER_Y, index_col=0)
    paper_y.index = paper_y.index.astype(str)
    if paper_y.shape != (EXPECTED_ROWS, EXPECTED_LABELS):
        raise ValueError(
            f"Expected paper Y shape {(EXPECTED_ROWS, EXPECTED_LABELS)}, got {paper_y.shape}"
        )
    if paper_y.index.duplicated().any() or not np.isin(paper_y.to_numpy(), [0, 1]).all():
        raise ValueError("Paper Y must have unique exact IDs and binary labels")

    sequence_ids = sequences["ID"].astype(str).tolist()
    paper_ids = paper_y.index.tolist()
    missing_from_sequences = sorted(set(paper_ids) - set(sequence_ids))
    extra_sequence_ids = sorted(set(sequence_ids) - set(paper_ids))
    if missing_from_sequences or extra_sequence_ids:
        raise ValueError(
            "Exact-ID alignment failed after the declared repair: "
            f"missing={missing_from_sequences}, extra={extra_sequence_ids}"
        )
    sequences = sequences.set_index("ID", drop=False).loc[paper_ids].reset_index(drop=True)
    if sequences["ID"].tolist() != paper_ids:
        raise ValueError("Exact-ID alignment did not preserve the paper label order")
    return sequences, paper_y, repairs


def removed_symbol_counts(sequences: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for sample_id, raw_sequence in sequences[["ID", "seqs"]].itertuples(index=False):
        _cleaned, removed = normalize_sequence(raw_sequence)
        if removed:
            result[str(sample_id)] = dict(sorted((symbol, int(count)) for symbol, count in removed.items()))
    return result


def validate_released_folds(splits: pd.DataFrame) -> dict[str, object]:
    expected_columns = [
        name
        for fold in range(1, 11)
        for name in (f"train_fold{fold}", f"test_fold{fold}")
    ]
    if splits.columns.tolist() != expected_columns:
        raise ValueError("Released split columns/order do not match the ten-fold contract")

    expected = set(range(EXPECTED_ROWS))
    test_occurrences: Counter[int] = Counter()
    fold_sizes: dict[str, dict[str, int]] = {}
    for fold in range(1, 11):
        train_values = splits[f"train_fold{fold}"].dropna().astype(int).tolist()
        test_values = splits[f"test_fold{fold}"].dropna().astype(int).tolist()
        train = set(train_values)
        test = set(test_values)
        if len(train) != len(train_values) or len(test) != len(test_values):
            raise ValueError(f"Released fold {fold} contains duplicate indices")
        if train & test or train | test != expected:
            raise ValueError(f"Released fold {fold} is not a full 355-row partition")
        test_occurrences.update(test_values)
        fold_sizes[str(fold)] = {"train": len(train), "test": len(test)}

    if set(test_occurrences) != expected or set(test_occurrences.values()) != {1}:
        raise ValueError("Every V1 row must occur in exactly one released test fold")
    return {
        "fold_sizes": fold_sizes,
        "test_rows": len(test_occurrences),
        "test_occurrence_min": min(test_occurrences.values()),
        "test_occurrence_max": max(test_occurrences.values()),
        "each_row_in_exactly_one_test_fold": True,
    }


def compare_independent_reconstruction(generated: pd.DataFrame, generated_path: Path) -> dict[str, object]:
    independent = pd.read_csv(INDEPENDENT_RECOVERED_355_X, index_col=0)
    independent.index = independent.index.astype(str)
    same_shape = generated.shape == independent.shape
    same_ids = generated.index.astype(str).tolist() == independent.index.tolist()
    same_columns = generated.columns.tolist() == independent.columns.tolist()
    same_values = bool(
        same_shape
        and same_ids
        and same_columns
        and np.array_equal(generated.to_numpy(), independent.to_numpy(), equal_nan=True)
    )
    return {
        "path": repository_relative(INDEPENDENT_RECOVERED_355_X),
        "generated_sha256": sha256(generated_path),
        "independent_sha256": sha256(INDEPENDENT_RECOVERED_355_X),
        "hashes_match": sha256(generated_path) == sha256(INDEPENDENT_RECOVERED_355_X),
        "shape_match": same_shape,
        "id_order_match": same_ids,
        "column_order_match": same_columns,
        "values_match_exactly": same_values,
    }


def run(outdir: Path) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=False)

    sequences, paper_y, id_repairs = repaired_sequences_and_labels()
    sequence_path = outdir / "v1_processed_sequences_355.csv"
    sequences.to_csv(sequence_path, index=False)

    x, reconstruction = reconstruct(
        sequence_path,
        PROPERTIES,
        id_column="ID",
        sequence_column="seqs",
    )
    if x.shape != (EXPECTED_ROWS, EXPECTED_FEATURES):
        raise ValueError(f"Expected repaired X shape (355, 542), got {x.shape}")
    if x.index.astype(str).tolist() != paper_y.index.tolist():
        raise ValueError("Repaired X and paper Y IDs/order must match exactly")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Repaired X contains non-finite values")

    accepted_schema = pd.read_csv(ACCEPTED_353_X, nrows=0).columns.tolist()[1:]
    if x.columns.tolist() != accepted_schema:
        raise ValueError("Repaired X schema/order differs from the accepted 353-row V1 X")

    symbol_counts = removed_symbol_counts(sequences)
    expected_symbol_counts = {
        "AK153778.1": {"N": 1},
        "BX374997.2 ": {"D": 1, "K": 3, "Y": 3},
    }
    if symbol_counts != expected_symbol_counts:
        raise ValueError(f"Unexpected invalid-symbol repair set: {symbol_counts}")
    if reconstruction["output_rows"] != EXPECTED_ROWS or reconstruction["missing_sequence_count"] != 0:
        raise ValueError("Reconstruction must retain all 355 rows")
    if reconstruction["cleaned_sequence_ids"] != list(expected_symbol_counts):
        raise ValueError("Reconstruction did not record the two cleaned IDs exactly")

    y = paper_y.loc[x.index].copy()
    x_path = outdir / "v1_nazli_repaired_paper_sequence_542_X.csv"
    y_path = outdir / "v1_paper_44_Y.csv"
    split_path = outdir / "v1_released_folds_355.csv"
    x.to_csv(x_path)
    y.to_csv(y_path)
    shutil.copyfile(PAPER_SPLITS, split_path)

    folds = validate_released_folds(pd.read_csv(split_path))
    if sha256(split_path) != sha256(PAPER_SPLITS):
        raise ValueError("The released split file was not preserved byte-for-byte")
    independent_comparison = compare_independent_reconstruction(x, x_path)
    if not all(
        independent_comparison[key]
        for key in (
            "hashes_match",
            "shape_match",
            "id_order_match",
            "column_order_match",
            "values_match_exactly",
        )
    ):
        raise ValueError("Repaired X differs from the independent recovered 355-row matrix")

    manifest_path = outdir / "manifest.csv"
    manifest = pd.DataFrame(
        [
            {
                "version": "v1",
                "label_space": "all44",
                "population": "v1_nazli_processed_sequence_355",
                "feature_set": "paper_artifact_verified_sequence_families_542",
                "feature_key": "nazli_repaired_paper_sequence_542",
                "domains": "rna",
                "components": "Pse-in-One DCC all 22 properties lag1 + normalized kmer2 + normalized kmer3",
                "x_path": repository_relative(x_path),
                "y_path": repository_relative(y_path),
                "split_path": repository_relative(split_path),
                "n_samples": EXPECTED_ROWS,
                "n_features": EXPECTED_FEATURES,
                "n_labels": EXPECTED_LABELS,
                "x_sha256": sha256(x_path),
                "y_sha256": sha256(y_path),
                "split_sha256": sha256(split_path),
                "preprocessing_mode": "reconstructed_paper_sequence_features_from_nazli_processed_sequences",
                "split_method": "released_paper_ten_folds_unchanged",
            }
        ]
    )
    manifest.to_csv(manifest_path, index=False)

    record: dict[str, object] = {
        "purpose": "Complete repaired Nazli V1 sequence-only input for Felipe's MLC pipeline",
        "immutability": "fresh output directory required; existing directories are rejected",
        "selected_features": {
            "dcc": 462,
            "normalized_kmer2": 16,
            "normalized_kmer3": 64,
            "total": EXPECTED_FEATURES,
        },
        "source_hashes": {
            "processed_sequences": sha256(SEQUENCES),
            "properties": sha256(PROPERTIES),
            "paper_labels": sha256(PAPER_Y),
            "paper_splits": sha256(PAPER_SPLITS),
            "accepted_353_schema_source": sha256(ACCEPTED_353_X),
            "independent_recovered_355_x": sha256(INDEPENDENT_RECOVERED_355_X),
        },
        "id_repairs_before_exact_alignment": id_repairs,
        "label_alignment": {
            "method": "exact ID after the one declared source-ID repair",
            "rows": EXPECTED_ROWS,
            "x_y_id_order_equal": True,
        },
        "invalid_symbol_removals_by_exact_id": symbol_counts,
        "dropped_rows": 0,
        "reconstruction": reconstruction,
        "accepted_353_x_schema": {
            "path": repository_relative(ACCEPTED_353_X),
            "feature_columns": len(accepted_schema),
            "column_names_and_order_match": True,
        },
        "released_folds": {
            "source_path": repository_relative(PAPER_SPLITS),
            "copied_byte_for_byte": True,
            **folds,
        },
        "independent_recovered_355_comparison": independent_comparison,
        "outputs": {
            "sequences": repository_relative(sequence_path),
            "x": repository_relative(x_path),
            "y": repository_relative(y_path),
            "splits": repository_relative(split_path),
            "manifest": repository_relative(manifest_path),
        },
        "output_hashes": {
            "sequences": sha256(sequence_path),
            "x": sha256(x_path),
            "y": sha256(y_path),
            "splits": sha256(split_path),
        },
    }
    (outdir / "preparation_record.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    print(json.dumps(run(outdir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
