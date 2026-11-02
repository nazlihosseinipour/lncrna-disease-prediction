#!/usr/bin/env python3
"""Build immutable V2 inputs for Nazli's paper-selected sequence feature lane.

The feature contract is identical to the accepted V1 lane: Pse-in-One RNA DCC
with all 22 properties at lag 1, normalized 2-mers, and normalized 3-mers. The
436 labels are aligned by RNA ID, and identical normalized sequences are kept
in the same one of ten multilabel-stratified folds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from make_v2_group_splits import build as build_group_splits
from reconstruct_paper_sequence_features import reconstruct


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/nazli_selected_v2_felipe_2026-10-05/inputs"
SOURCE = ROOT / "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv"
PROPERTIES = ROOT / "config/pse_in_one_rna_dcc_properties.csv"
ACCEPTED_V1_X = (
    ROOT
    / "results/nazli_selected_v1_felipe_2026-10-05/inputs/"
    "v1_nazli_selected_paper_sequence_542_X.csv"
)
INDEPENDENT_V2_X = ROOT / "results/felipe_basics_2026-09-27/inputs/v2_paper_sequence_542_X.csv"
INDEPENDENT_V2_Y = ROOT / "results/felipe_basics_2026-09-27/inputs/v2_all_436_Y.csv"
GROUP_SPLIT_SCRIPT = ROOT / "scripts/make_v2_group_splits.py"
RECONSTRUCTION_SCRIPT = ROOT / "scripts/reconstruct_paper_sequence_features.py"

EXPECTED_SOURCE_ROWS = 5_338
EXPECTED_ROWS = 4_130
EXPECTED_FEATURES = 542
EXPECTED_LABELS = 436


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    """Return a repository-relative path and refuse paths outside the repository."""
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def align_labels(source: pd.DataFrame, ids: list[str]) -> pd.DataFrame:
    """Align every non-sequence source column to ``ids`` by the unique ID column."""
    required = {"ID", "seqs"}
    missing_columns = sorted(required - set(source.columns))
    if missing_columns:
        raise ValueError(f"V2 source is missing required columns: {missing_columns}")
    source = source.copy()
    source["ID"] = source["ID"].astype(str)
    if source["ID"].duplicated().any():
        raise ValueError("V2 source contains duplicate ID values")
    missing_ids = sorted(set(ids) - set(source["ID"]))
    if missing_ids:
        raise ValueError(f"Feature IDs are absent from V2 label source: {missing_ids[:5]}")
    labels = source.drop(columns=["seqs"]).set_index("ID").loc[ids].copy()
    labels.index.name = "ID"
    if labels.shape[1] != EXPECTED_LABELS:
        raise ValueError(f"Expected {EXPECTED_LABELS} V2 labels, got {labels.shape[1]}")
    numeric = labels.apply(pd.to_numeric, errors="raise")
    if not np.isin(numeric.to_numpy(), [0, 1]).all():
        raise ValueError("V2 labels must all be binary 0/1 values")
    return numeric.astype(int)


def require_accepted_v1_schema(x: pd.DataFrame, reference_path: Path = ACCEPTED_V1_X) -> str:
    """Fail unless X has the exact accepted V1 feature names in the same order."""
    reference_columns = pd.read_csv(reference_path, index_col=0, nrows=0).columns.tolist()
    actual_columns = x.columns.astype(str).tolist()
    if len(reference_columns) != EXPECTED_FEATURES:
        raise ValueError(
            f"Accepted V1 schema must contain {EXPECTED_FEATURES} columns, "
            f"got {len(reference_columns)}"
        )
    if actual_columns != reference_columns:
        first_difference = next(
            (
                position
                for position, (actual, expected) in enumerate(
                    zip(actual_columns, reference_columns, strict=False)
                )
                if actual != expected
            ),
            min(len(actual_columns), len(reference_columns)),
        )
        raise ValueError(
            "V2 feature schema differs from the accepted V1 X columns "
            f"at position {first_difference}; no manifest will be written"
        )
    return sha256(reference_path)


def write_csv_exclusive(frame: pd.DataFrame, path: Path, *, index: bool) -> None:
    """Write a CSV exactly once."""
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        frame.to_csv(handle, index=index)


def write_json_exclusive(value: Any, path: Path) -> None:
    with Path(path).open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def independent_comparison(generated: Path, recovered: Path) -> dict[str, object]:
    """Record a byte-level comparison without using the recovered file as generation input."""
    generated_hash = sha256(generated)
    if not recovered.is_file():
        return {
            "generated_path": relative(generated),
            "generated_sha256": generated_hash,
            "recovered_path": relative(recovered),
            "recovered_available": False,
            "byte_identical": None,
        }
    recovered_hash = sha256(recovered)
    return {
        "generated_path": relative(generated),
        "generated_sha256": generated_hash,
        "recovered_path": relative(recovered),
        "recovered_available": True,
        "recovered_sha256": recovered_hash,
        "byte_identical": generated_hash == recovered_hash,
    }


def run(outdir: Path) -> dict[str, object]:
    source = pd.read_csv(SOURCE)
    if source.shape != (EXPECTED_SOURCE_ROWS, EXPECTED_LABELS + 2):
        raise ValueError(
            f"Expected V2 source shape ({EXPECTED_SOURCE_ROWS}, {EXPECTED_LABELS + 2}), "
            f"got {source.shape}"
        )
    source["ID"] = source["ID"].astype(str)
    if source["ID"].duplicated().any():
        raise ValueError("V2 source contains duplicate ID values")

    # Reconstruction happens directly from the designated source. The independently
    # recovered matrices below are used only for a post-generation hash comparison.
    x, reconstruction = reconstruct(
        SOURCE,
        PROPERTIES,
        id_column="ID",
        sequence_column="seqs",
    )
    x.index = x.index.astype(str)
    x.index.name = "ID"
    if x.shape != (EXPECTED_ROWS, EXPECTED_FEATURES):
        raise ValueError(
            f"Expected reconstructed X shape ({EXPECTED_ROWS}, {EXPECTED_FEATURES}), got {x.shape}"
        )
    if x.index.has_duplicates or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Reconstructed V2 X must have unique IDs and only finite values")
    accepted_v1_x_hash = require_accepted_v1_schema(x)

    ids = x.index.tolist()
    y = align_labels(source, ids)
    if y.shape != (EXPECTED_ROWS, EXPECTED_LABELS) or y.index.tolist() != ids:
        raise ValueError(f"Expected aligned Y shape/order ({EXPECTED_ROWS}, {EXPECTED_LABELS})")

    by_id = source.set_index("ID", drop=False)
    processed_sequences = by_id.loc[ids, ["ID", "seqs"]].copy()
    if processed_sequences.shape != (EXPECTED_ROWS, 2):
        raise ValueError("Processed V2 sequence selection did not retain exactly 4,130 rows")

    # The schema gate above intentionally runs before the immutable output directory
    # is created, so an incompatible X cannot leave a plausible manifest behind.
    outdir.mkdir(parents=True, exist_ok=False)
    x_path = outdir / "v2_nazli_selected_paper_sequence_542_X.csv"
    y_path = outdir / "v2_all_436_Y.csv"
    split_path = outdir / "v2_group_splits.csv"
    split_metadata_path = outdir / "v2_group_splits.csv.metadata.json"
    sequences_path = outdir / "v2_processed_sequences_4130.csv"
    manifest_path = outdir / "manifest.csv"
    record_path = outdir / "preparation_record.json"

    write_csv_exclusive(processed_sequences, sequences_path, index=False)
    write_csv_exclusive(x, x_path, index=True)
    write_csv_exclusive(y, y_path, index=True)

    splits, split_metadata = build_group_splits(x_path, y_path, sequences_path)
    split_metadata["population"] = "v2_nazli_processed_sequence_4130"
    write_csv_exclusive(splits, split_path, index=False)
    split_metadata["output"] = {
        "path": relative(split_path),
        "sha256": sha256(split_path),
    }
    write_json_exclusive(split_metadata, split_metadata_path)

    x_comparison = independent_comparison(x_path, INDEPENDENT_V2_X)
    y_comparison = independent_comparison(y_path, INDEPENDENT_V2_Y)
    for name, comparison in (("X", x_comparison), ("Y", y_comparison)):
        if comparison["recovered_available"] and not comparison["byte_identical"]:
            raise ValueError(
                f"Generated {name} differs from the independent recovered input; "
                "no manifest will be written"
            )

    split_method = (
        "group-level MultilabelStratifiedKFold on OR of member labels; "
        "groups = SHA-256 of normalized sequence; identical X rows verified isolated"
    )
    manifest = pd.DataFrame(
        [
            {
                "version": "v2",
                "label_space": "all436",
                "population": "v2_nazli_processed_sequence_4130",
                "feature_set": "paper_artifact_verified_sequence_families_542",
                "feature_key": "nazli_selected_paper_sequence_542",
                "domains": "rna",
                "components": (
                    "Pse-in-One DCC all 22 properties lag1 + normalized kmer2 + "
                    "normalized kmer3"
                ),
                "x_path": relative(x_path),
                "y_path": relative(y_path),
                "split_path": relative(split_path),
                "n_samples": EXPECTED_ROWS,
                "n_features": EXPECTED_FEATURES,
                "n_labels": EXPECTED_LABELS,
                "x_sha256": sha256(x_path),
                "y_sha256": sha256(y_path),
                "split_sha256": sha256(split_path),
                "preprocessing_mode": (
                    "verified_paper_sequence_family_selection_from_nazli_processed_sequences"
                ),
                "split_method": split_method,
            }
        ]
    )
    write_csv_exclusive(manifest, manifest_path, index=False)

    record: dict[str, object] = {
        "purpose": (
            "Nazli processed V2 sequences, accepted V1-identical paper-verified feature "
            "selection, Felipe MLC pipeline"
        ),
        "selected_features": {
            "dcc": 462,
            "normalized_kmer2": 16,
            "normalized_kmer3": 64,
            "total": EXPECTED_FEATURES,
        },
        "source_contract": {
            "source_rows": EXPECTED_SOURCE_ROWS,
            "usable_sequence_rows": EXPECTED_ROWS,
            "labels": EXPECTED_LABELS,
        },
        "schema_verification": {
            "accepted_v1_x_path": relative(ACCEPTED_V1_X),
            "accepted_v1_x_sha256": accepted_v1_x_hash,
            "exact_column_names_and_order": True,
        },
        "group_split_verification": {
            "method": split_method,
            "n_splits": split_metadata["n_splits"],
            "n_groups": split_metadata["n_groups"],
            "identical_sequence_train_test_overlap": split_metadata[
                "identical_sequence_train_test_overlap"
            ],
            "identical_x_row_train_test_overlap": split_metadata[
                "identical_x_row_train_test_overlap"
            ],
        },
        "source_hashes": {
            "website_full_matrix": sha256(SOURCE),
            "properties": sha256(PROPERTIES),
            "reconstruction_script": sha256(RECONSTRUCTION_SCRIPT),
            "group_split_script": sha256(GROUP_SPLIT_SCRIPT),
        },
        "reconstruction": reconstruction,
        "independent_recovered_input_comparison": {
            "x": x_comparison,
            "y": y_comparison,
            "generation_used_recovered_matrices": False,
        },
        "outputs": {
            "processed_sequences": {
                "path": relative(sequences_path),
                "sha256": sha256(sequences_path),
            },
            "x": {"path": relative(x_path), "sha256": sha256(x_path)},
            "y": {"path": relative(y_path), "sha256": sha256(y_path)},
            "splits": {"path": relative(split_path), "sha256": sha256(split_path)},
            "split_metadata": {
                "path": relative(split_metadata_path),
                "sha256": sha256(split_metadata_path),
            },
            "manifest": {"path": relative(manifest_path), "sha256": sha256(manifest_path)},
        },
    }
    write_json_exclusive(record, record_path)
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
