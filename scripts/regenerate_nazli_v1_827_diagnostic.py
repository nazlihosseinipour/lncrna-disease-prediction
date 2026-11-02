#!/usr/bin/env python3
"""Regenerate the *diagnostic* V1 827-column construction from Nazli's data.

The output deliberately includes 285 raw RNA--disease association outcomes
between the 462 DCC and 80 k-mer columns.  It is useful for explaining the
released Felipe artifact, but it is target-derived and therefore is not a
valid sequence-only predictive input.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

try:
    from scripts.reconstruct_paper_sequence_features import (
        DEFAULT_PROPERTIES,
        reconstruct,
        sha256_file,
    )
except ModuleNotFoundError:  # direct execution from scripts/
    from reconstruct_paper_sequence_features import (  # type: ignore
        DEFAULT_PROPERTIES,
        reconstruct,
        sha256_file,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "results/nazli_827_regeneration_audit_2026-10-08/v1"
NAZLI_PROCESSED = ROOT / "Data/output_data/sequences_for_oop.csv"
FELIPE_RELEASED = (
    ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/features/features_preprocessed.csv"
)
EXPECTED_RELEASED_SHA256 = "b264dd75545ca424f1db96622555066791c52ba6105001770ae869e211428887"
ID_REPAIRS = {"BX374997.2": "BX374997.2 "}
EXPECTED_MAPPING_COUNTS = {"identity": 231, "shifted": 121, "unmatched": 3}
ATOL = 1e-6

FEATURE_FILE = "v1_nazli_diagnostic_827.csv"
SEQUENCE_FILE = "v1_nazli_processed_sequences_355.csv"
ROW_MAPPING_FILE = "row_mapping.csv"
BLOCK_COMPARISON_FILE = "block_comparison.csv"
AUDIT_FILE = "audit.json"
REQUIRED_FILES = (
    FEATURE_FILE,
    SEQUENCE_FILE,
    ROW_MAPPING_FILE,
    BLOCK_COMPARISON_FILE,
    AUDIT_FILE,
)


class ValidationError(RuntimeError):
    """Raised when a generated diagnostic artifact violates its contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def require_frame_close(stored: pd.DataFrame, expected: pd.DataFrame, message: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            stored,
            expected,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValidationError(message) from error


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_nazli_components() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load processed sequences and the 285 outcome columns in released ID order."""
    source = pd.read_csv(NAZLI_PROCESSED, dtype={"ID": str}, keep_default_na=False)
    require(source.shape == (355, 288), f"unexpected Nazli source shape: {source.shape}")
    require(source.columns[:3].tolist() == ["Unnamed: 0", "ID", "seqs"], "unexpected source prefix")
    association_names = source.columns[3:].astype(str).tolist()
    require(len(association_names) == 285, "Nazli source must have 285 association columns")
    association_values = source.iloc[:, 3:].apply(pd.to_numeric, errors="raise")
    require(
        bool(np.isin(association_values.to_numpy(), [0, 1]).all()),
        "association columns must be binary",
    )
    require(int(association_values.to_numpy().sum()) == 1132, "unexpected association positive count")

    source = source.copy()
    for old, new in ID_REPAIRS.items():
        require(int((source["ID"] == old).sum()) == 1, f"missing unique repair source ID {old!r}")
        require(not bool((source["ID"] == new).any()), f"repair target already exists: {new!r}")
        source.loc[source["ID"] == old, "ID"] = new

    released = pd.read_csv(FELIPE_RELEASED, index_col=0)
    released.index = released.index.astype(str)
    require(released.shape == (355, 827), f"unexpected Felipe shape: {released.shape}")
    require(not source["ID"].duplicated().any(), "Nazli source IDs must be unique")
    require(set(source["ID"]) == set(released.index), "Nazli and Felipe ID sets differ")
    aligned = source.set_index("ID", drop=False).loc[released.index].copy()
    require(aligned["ID"].tolist() == released.index.tolist(), "ID order alignment failed")

    sequences = aligned[["ID", "seqs"]].reset_index(drop=True)
    associations = aligned.iloc[:, 3:].copy()
    associations.index = released.index
    associations.columns = [f"association_outcome::{name}" for name in association_names]
    return sequences, associations, association_names


def assemble_matrix(sequence_features: pd.DataFrame, associations: pd.DataFrame) -> pd.DataFrame:
    """Place the target-derived block at released positions 462..746."""
    require(sequence_features.shape[1] == 542, "sequence matrix must contain 542 columns")
    require(associations.shape[1] == 285, "association matrix must contain 285 columns")
    require(sequence_features.index.tolist() == associations.index.tolist(), "component IDs/order differ")
    dcc = sequence_features.iloc[:, :462]
    kmers = sequence_features.iloc[:, 462:]
    combined = pd.concat([dcc, associations, kmers], axis=1)
    require(combined.shape == (len(sequence_features), 827), "assembled matrix has wrong shape")
    require(not combined.columns.duplicated().any(), "assembled feature names are not unique")
    return combined


def sequence_row_mapping(released: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    """Independently recover the released-to-generated row mapping from sequence blocks."""
    rel = np.hstack(
        [released.iloc[:, :462].to_numpy(float), released.iloc[:, 747:].to_numpy(float)]
    )
    gen = np.hstack(
        [generated.iloc[:, :462].to_numpy(float), generated.iloc[:, 747:].to_numpy(float)]
    )
    rows: list[dict[str, Any]] = []
    for released_row, vector in enumerate(rel):
        distances = np.max(np.abs(gen - vector), axis=1)
        order = np.argsort(distances, kind="stable")
        best = int(order[0])
        best_abs = float(distances[best])
        if best_abs > ATOL:
            mapped = -1
            mapping_class = "unmatched"
        else:
            mapped = best
            mapping_class = "identity" if best == released_row else "shifted"
        rows.append(
            {
                "released_row": released_row,
                "released_id": str(released.index[released_row]),
                "generated_same_position_id": str(generated.index[released_row]),
                "mapped_generated_row": mapped,
                "mapped_generated_id": "" if mapped < 0 else str(generated.index[mapped]),
                "mapping_class": mapping_class,
                "mapping_offset": np.nan if mapped < 0 else mapped - released_row,
                "best_max_abs": best_abs,
                "second_best_max_abs": float(distances[int(order[1])]),
                "n_generated_rows_within_1e6": int((distances <= ATOL).sum()),
            }
        )
    return pd.DataFrame(rows)


def comparison_stats(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    require(left.shape == right.shape, "comparison arrays differ in shape")
    exact = left == right
    within = np.isclose(left, right, atol=ATOL, rtol=0)
    return {
        "rows_compared": int(left.shape[0]),
        "columns_compared": int(left.shape[1]),
        "cells_compared": int(left.size),
        "exact_cells": int(exact.sum()),
        "exact_cell_fraction": float(exact.mean()),
        "within_1e6_cells": int(within.sum()),
        "within_1e6_cell_fraction": float(within.mean()),
        "exact_rows": int(exact.all(axis=1).sum()),
        "exact_columns": int(exact.all(axis=0).sum()),
        "max_abs_difference": float(np.abs(left - right).max()),
    }


def build_block_comparison(
    released: pd.DataFrame, generated: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    blocks = {
        "dcc_462": (0, 462),
        "association_outcomes_285": (462, 747),
        "normalized_k2_k3_80": (747, 827),
        "all_827": (0, 827),
    }
    rel = released.to_numpy(float)
    gen = generated.to_numpy(float)
    mapped_rows = mapping.loc[mapping["mapped_generated_row"] >= 0]
    rel_indices = mapped_rows["released_row"].to_numpy(int)
    gen_indices = mapped_rows["mapped_generated_row"].to_numpy(int)
    records: list[dict[str, Any]] = []
    for block_name, (start, stop) in blocks.items():
        for alignment, left, right in (
            ("same_id_and_position", rel[:, start:stop], gen[:, start:stop]),
            (
                "recovered_sequence_row_mapping",
                rel[rel_indices, start:stop],
                gen[gen_indices, start:stop],
            ),
        ):
            records.append(
                {
                    "block": block_name,
                    "start_zero_based": start,
                    "stop_exclusive": stop,
                    "alignment": alignment,
                    **comparison_stats(left, right),
                }
            )
    return pd.DataFrame(records)


def compute(outdir: Path, *, write: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sequences, associations, association_names = load_nazli_components()
    if write:
        outdir.mkdir(parents=True, exist_ok=False)
        sequence_path = outdir / SEQUENCE_FILE
        sequences.to_csv(sequence_path, index=False)
    else:
        sequence_path = outdir / SEQUENCE_FILE
        require(sequence_path.is_file(), f"missing {sequence_path}")
        stored_sequences = pd.read_csv(sequence_path, dtype={"ID": str}, keep_default_na=False)
        require(stored_sequences.equals(sequences), "stored processed sequences differ from source alignment")

    sequence_features, reconstruction = reconstruct(
        sequence_path,
        DEFAULT_PROPERTIES,
        id_column="ID",
        sequence_column="seqs",
    )
    require(sequence_features.shape == (355, 542), "reconstructed sequence shape is not 355 x 542")
    require(sequence_features.index.tolist() == associations.index.tolist(), "sequence/association IDs differ")
    generated = assemble_matrix(sequence_features, associations)
    released = pd.read_csv(FELIPE_RELEASED, index_col=0)
    released.index = released.index.astype(str)
    require(generated.index.tolist() == released.index.tolist(), "generated/released IDs or order differ")
    require(np.isfinite(generated.to_numpy(float)).all(), "generated features contain non-finite values")

    mapping = sequence_row_mapping(released, generated)
    counts = {str(k): int(v) for k, v in mapping["mapping_class"].value_counts().items()}
    require(counts == EXPECTED_MAPPING_COUNTS, f"unexpected row mapping counts: {counts}")
    require(
        int((mapping["n_generated_rows_within_1e6"] > 1).sum()) == 0,
        "matched sequence rows must have unique candidates",
    )
    comparison = build_block_comparison(released, generated, mapping)
    mapped_all = comparison[
        (comparison["block"] == "all_827")
        & (comparison["alignment"] == "recovered_sequence_row_mapping")
    ].iloc[0]
    # Pse-in-One's serialized k-mer values differ by 1e-8 in 15 cells; all
    # 352 mapped rows still agree within the frozen audit tolerance.
    require(
        int(mapped_all["within_1e6_cells"]) == int(mapped_all["cells_compared"]),
        "all mapped 827-column rows should agree within 1e-6",
    )

    audit: dict[str, Any] = {
        "purpose": "diagnostic reconstruction of the released V1 827-column artifact",
        "valid_sequence_only_predictive_input": False,
        "association_columns_are_outcome_derived": True,
        "leakage_warning": (
            "Columns 462-746 are raw RNA-disease outcomes. Using them as predictors for the same "
            "associations leaks target information; this artifact is diagnostic only."
        ),
        "v2_guidance": (
            "Do not copy or resize this V1 outcome block for V2. Use the clean 542 sequence features, "
            "or construct association-derived transforms strictly inside each training fold."
        ),
        "shape": [int(generated.shape[0]), int(generated.shape[1])],
        "blocks": [
            {"name": "dcc_462", "start": 0, "stop_exclusive": 462, "source": "processed sequences"},
            {
                "name": "association_outcomes_285",
                "start": 462,
                "stop_exclusive": 747,
                "source": "Nazli processed RNA-disease matrix",
                "outcome_derived": True,
            },
            {
                "name": "normalized_k2_k3_80",
                "start": 747,
                "stop_exclusive": 827,
                "source": "processed sequences",
            },
        ],
        "association_columns": association_names,
        "association_positive_count": int(associations.to_numpy().sum()),
        "id_repairs": ID_REPAIRS,
        "row_mapping_counts": counts,
        "row_mapping_verified_fresh": True,
        "reconstruction": reconstruction,
        "inputs": {
            "nazli_processed": relative(NAZLI_PROCESSED),
            "nazli_processed_sha256": sha256_file(NAZLI_PROCESSED),
            "felipe_released": relative(FELIPE_RELEASED),
            "felipe_released_sha256": sha256_file(FELIPE_RELEASED),
            "felipe_released_expected_sha256": EXPECTED_RELEASED_SHA256,
            "properties": relative(Path(DEFAULT_PROPERTIES)),
            "properties_sha256": sha256_file(Path(DEFAULT_PROPERTIES)),
        },
        "comparison": comparison.to_dict(orient="records"),
    }
    require(
        audit["inputs"]["felipe_released_sha256"] == EXPECTED_RELEASED_SHA256,
        "released Felipe feature file hash changed",
    )
    return generated, mapping, comparison, audit


def generate(outdir: Path) -> None:
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {outdir}")
    generated, mapping, comparison, audit = compute(outdir, write=True)
    feature_path = outdir / FEATURE_FILE
    mapping_path = outdir / ROW_MAPPING_FILE
    comparison_path = outdir / BLOCK_COMPARISON_FILE
    generated.to_csv(feature_path)
    mapping.to_csv(mapping_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    audit["outputs"] = {
        FEATURE_FILE: sha256_file(feature_path),
        SEQUENCE_FILE: sha256_file(outdir / SEQUENCE_FILE),
        ROW_MAPPING_FILE: sha256_file(mapping_path),
        BLOCK_COMPARISON_FILE: sha256_file(comparison_path),
    }
    (outdir / AUDIT_FILE).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"wrote diagnostic V1 matrix: {generated.shape[0]} rows x {generated.shape[1]} columns")
    print(audit["leakage_warning"])


def check(outdir: Path) -> None:
    for name in REQUIRED_FILES:
        require((outdir / name).is_file(), f"missing artifact: {outdir / name}")
    expected, expected_mapping, expected_comparison, expected_audit = compute(outdir, write=False)
    stored = pd.read_csv(outdir / FEATURE_FILE, index_col=0)
    stored.index = stored.index.astype(str)
    require(stored.shape == (355, 827), "stored feature matrix shape changed")
    require(stored.index.tolist() == expected.index.tolist(), "stored feature IDs/order changed")
    require(stored.columns.tolist() == expected.columns.tolist(), "stored feature schema/order changed")
    require(
        np.array_equal(stored.to_numpy(float), expected.to_numpy(float), equal_nan=True),
        "stored feature values do not equal a fresh reconstruction",
    )
    stored_mapping = pd.read_csv(outdir / ROW_MAPPING_FILE, keep_default_na=False)
    # The empty unmatched-ID cells are intentionally strings; restore the
    # numeric missing-value representation used by the in-memory mapping.
    stored_mapping["mapping_offset"] = pd.to_numeric(
        stored_mapping["mapping_offset"], errors="coerce"
    )
    require_frame_close(stored_mapping, expected_mapping, "row mapping changed")
    stored_comparison = pd.read_csv(outdir / BLOCK_COMPARISON_FILE)
    require_frame_close(stored_comparison, expected_comparison, "block comparison changed")
    audit = json.loads((outdir / AUDIT_FILE).read_text(encoding="utf-8"))
    require(audit["shape"] == [355, 827], "audit shape changed")
    require(audit["row_mapping_counts"] == EXPECTED_MAPPING_COUNTS, "audit mapping counts changed")
    require(audit["association_columns_are_outcome_derived"] is True, "outcome warning missing")
    require(audit["valid_sequence_only_predictive_input"] is False, "invalid predictive claim")
    require(audit["comparison"] == expected_audit["comparison"], "audit comparison changed")
    for name, expected_hash in audit["outputs"].items():
        require(sha256_file(outdir / name) == expected_hash, f"output hash changed: {name}")
    print("V1 diagnostic 827 regeneration passed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--check", action="store_true", help="validate existing artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    try:
        if args.check:
            check(outdir)
        else:
            generate(outdir)
    except (ValidationError, FileExistsError, FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
