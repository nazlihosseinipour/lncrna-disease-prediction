#!/usr/bin/env python3
"""Audit whether Felipe's apparent V1 827-column recipe can be applied to V2.

This script is deliberately an audit, not a feature generator or model runner.  It
measures the corrected V2 population, the valid sequence-only matrix, and the V2
association outcomes that would have to be appended to imitate the apparent V1
construction.  Appending outcomes to predictors is reported as label leakage and
is never written as a model input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V2_SOURCE = (
    ROOT / "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv"
)
DEFAULT_V1_SOURCE = ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv"
DEFAULT_V2_SEQUENCE_X = (
    ROOT / "results/felipe_basics_2026-09-27/inputs/v2_paper_sequence_542_X.csv"
)
DEFAULT_OUTDIR = ROOT / "results/nazli_827_regeneration_audit_2026-10-08/v2"
DEFAULT_REPORT = ROOT / "Reports/nazli_v2_827_feasibility_2026-10-08.md"

ID_COLUMN = "ID"
SEQUENCE_COLUMN = "seqs"
EXPECTED_V2_ROWS = 5_338
EXPECTED_USABLE_ROWS = 4_130
EXPECTED_SEQUENCE_FEATURES = 542
EXPECTED_V1_ASSOCIATIONS = 285
EXPECTED_V2_ASSOCIATIONS = 436


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo_relative(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def normalized_label(value: str) -> str:
    """Normalize a disease label exactly as the established transfer audit does."""
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _association_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {ID_COLUMN, SEQUENCE_COLUMN, "Unnamed: 0"}
    return [str(column) for column in frame.columns if str(column) not in excluded]


def _require_unique_ids(frame: pd.DataFrame, name: str) -> pd.Series:
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"{name} is missing the {ID_COLUMN!r} column")
    ids = frame[ID_COLUMN].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{name} contains duplicate RNA IDs")
    return ids


def _require_binary(frame: pd.DataFrame, columns: list[str], name: str) -> pd.DataFrame:
    values = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isin(values.to_numpy(), [0, 1]).all():
        raise ValueError(f"{name} association outcomes must all be binary 0/1 values")
    return values.astype(np.int8)


def _normalized_groups(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for column in columns:
        groups.setdefault(normalized_label(column), []).append(column)
    return groups


def analyze_frames(
    v2_source: pd.DataFrame,
    v1_source: pd.DataFrame,
    sequence_x: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Return deterministic audit facts and two evidence tables."""
    v2 = v2_source.copy()
    v1 = v1_source.copy()
    v2_ids = _require_unique_ids(v2, "V2 source")
    _require_unique_ids(v1, "V1 source")
    if SEQUENCE_COLUMN not in v2.columns:
        raise ValueError(f"V2 source is missing the {SEQUENCE_COLUMN!r} column")

    usable_mask = v2[SEQUENCE_COLUMN].notna() & v2[SEQUENCE_COLUMN].astype(str).str.strip().ne("")
    usable_ids = v2_ids.loc[usable_mask].tolist()
    missing_ids = v2_ids.loc[~usable_mask].tolist()

    x = sequence_x.copy()
    x.index = x.index.astype(str)
    x.index.name = ID_COLUMN
    if x.index.has_duplicates:
        raise ValueError("V2 sequence feature matrix contains duplicate RNA IDs")
    if x.index.tolist() != usable_ids:
        raise ValueError(
            "V2 sequence feature rows do not exactly match usable corrected-source IDs/order"
        )
    numeric_x = x.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric_x.to_numpy(dtype=float)).all():
        raise ValueError("V2 sequence feature matrix contains non-finite values")

    v1_columns = _association_columns(v1)
    v2_columns = _association_columns(v2)
    v2_y = _require_binary(v2.loc[usable_mask], v2_columns, "V2")

    exact = set(v1_columns) & set(v2_columns)
    v1_groups = _normalized_groups(v1_columns)
    v2_groups = _normalized_groups(v2_columns)
    normalized_overlap = sorted(set(v1_groups) & set(v2_groups))

    overlap_rows = []
    for v1_column in v1_columns:
        norm = normalized_label(v1_column)
        matches = v2_groups.get(norm, [])
        overlap_rows.append(
            {
                "v1_disease": v1_column,
                "normalized_name": norm,
                "exact_v2_match": v1_column if v1_column in exact else "",
                "normalized_v2_matches": " | ".join(matches),
                "normalized_match_count": len(matches),
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    missing = pd.DataFrame({ID_COLUMN: missing_ids})

    sequence_width = int(x.shape[1])
    v2_association_width = len(v2_columns)
    apparent_v1_association_width = len(v1_columns)
    all_v2_outcomes_width = sequence_width + v2_association_width
    forced_827_association_count = 827 - sequence_width

    summary: dict[str, Any] = {
        "purpose": "V2 feasibility audit only; no model training and no leaky X matrix generated",
        "v2_population": {
            "full_rows": int(len(v2)),
            "unique_rna_ids": int(v2_ids.nunique()),
            "usable_sequence_rows": int(len(usable_ids)),
            "missing_sequence_rows": int(len(missing_ids)),
            "sequence_feature_rows": int(len(x)),
        },
        "feature_contract": {
            "valid_sequence_features": sequence_width,
            "dcc": 462 if sequence_width == 542 else None,
            "normalized_kmer2": 16 if sequence_width == 542 else None,
            "normalized_kmer3": 64 if sequence_width == 542 else None,
        },
        "association_outcomes": {
            "v1_raw_disease_columns": apparent_v1_association_width,
            "v2_raw_disease_columns": v2_association_width,
            "v2_normalized_unique_diseases": len(v2_groups),
            "v2_normalized_duplicate_groups": sum(len(group) > 1 for group in v2_groups.values()),
            "v2_usable_population_positive_cells": int(v2_y.to_numpy().sum()),
            "these_columns_are_prediction_outcomes": True,
            "using_them_as_inputs_is_label_leakage": True,
        },
        "width_analysis": {
            "sequence_only_width": sequence_width,
            "sequence_plus_all_v2_outcomes_width": all_v2_outcomes_width,
            "association_columns_needed_to_force_827": forced_827_association_count,
            "v2_outcomes_omitted_if_arbitrarily_forced_to_827": (
                v2_association_width - forced_827_association_count
            ),
            "sequence_plus_exact_v1_name_overlap_width": sequence_width + len(exact),
            "sequence_plus_normalized_v1_name_overlap_width": sequence_width
            + len(normalized_overlap),
        },
        "v1_v2_disease_overlap": {
            "exact_name_matches": len(exact),
            "normalized_name_matches": len(normalized_overlap),
            "normalization_rule": "lowercase; replace non-alphanumeric runs with spaces; trim",
        },
        "decision": {
            "valid_v2_input": "4130 rows x 542 sequence-only features",
            "append_v2_association_outcomes": False,
            "arbitrary_827_column_selection": False,
            "paper_equivalent_827_reproduction_available": False,
            "reason": (
                "V2 has 436 association outcomes, so appending all outcomes yields 978 columns; "
                "using any of those outcomes as predictors leaks labels, and choosing 285 only to "
                "reach 827 has no methodological basis."
            ),
            "v2_model_training_performed": False,
        },
    }
    return summary, overlap, missing


def load_and_analyze(
    v2_source_path: Path = DEFAULT_V2_SOURCE,
    v1_source_path: Path = DEFAULT_V1_SOURCE,
    sequence_x_path: Path = DEFAULT_V2_SEQUENCE_X,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    v2_source_path = Path(v2_source_path)
    v1_source_path = Path(v1_source_path)
    sequence_x_path = Path(sequence_x_path)
    v2 = pd.read_csv(v2_source_path)
    v1 = pd.read_csv(v1_source_path)
    x = pd.read_csv(sequence_x_path, index_col=0)
    summary, overlap, missing = analyze_frames(v2, v1, x)
    summary["sources"] = {
        "corrected_v2_full_matrix": {
            "path": repo_relative(v2_source_path),
            "sha256": sha256(v2_source_path),
        },
        "v1_raw_association_reference": {
            "path": repo_relative(v1_source_path),
            "sha256": sha256(v1_source_path),
        },
        "v2_sequence_542_matrix": {
            "path": repo_relative(sequence_x_path),
            "sha256": sha256(sequence_x_path),
        },
    }
    return summary, overlap, missing


def render_report(summary: dict[str, Any]) -> str:
    pop = summary["v2_population"]
    outcomes = summary["association_outcomes"]
    widths = summary["width_analysis"]
    overlap = summary["v1_v2_disease_overlap"]
    return f"""# Nazli V2 827-feature feasibility audit — 2026-10-08

## Scope and measured population

This is a data and feature-schema audit. **V2 model training remains deferred**. The canonical corrected V2 table contains **{pop['full_rows']:,} RNA rows** and **{outcomes['v2_raw_disease_columns']} disease outcomes**. Of those rows, **{pop['usable_sequence_rows']:,} have usable sequences** and **{pop['missing_sequence_rows']:,} do not**.

The defensible V2 predictor matrix has {pop['usable_sequence_rows']:,} rows and **542 sequence features**: 462 DCC descriptors, 16 normalized 2-mers, and 64 normalized 3-mers. Its RNA IDs and order exactly match the sequence-available population.

## What happens if the V1 apparent construction is copied

The V1 raw source has 285 disease columns, while V2 has {outcomes['v2_raw_disease_columns']}. Appending all V2 disease outcomes to the 542 sequence descriptors would produce **{widths['sequence_plus_all_v2_outcomes_width']} columns**, not 827. Selecting only {widths['association_columns_needed_to_force_827']} of the {outcomes['v2_raw_disease_columns']} outcomes would discard {widths['v2_outcomes_omitted_if_arbitrarily_forced_to_827']} solely to reach a target width.

These are **association-derived** inputs: in a within-V2 experiment they are the same RNA-disease outcomes the model is asked to predict. Adding them to `X` creates direct **label leakage**. The fact that 542 + 285 equals 827 does not provide a valid feature method.

V1 and V2 share only **{overlap['exact_name_matches']} exact disease names**. Name normalization gives **{overlap['normalized_name_matches']} shared disease concepts**. These yield widths of {widths['sequence_plus_exact_v1_name_overlap_width']} or {widths['sequence_plus_normalized_v1_name_overlap_width']}, respectively, and neither defines an 827-column V2 method.

## Decision

Use the 542 sequence-only matrix for a future controlled V1/V2 comparison after that experiment is approved. Appending V2 association outcomes is invalid, and selecting arbitrary columns merely to total 827 is **not an 827-feature paper reproduction**. Felipe's released 827 artifact remains a V1 positive control; it does not supply a reproducible, leakage-free V2 feature recipe.

No V2 model was trained by this audit and no source artifact was modified.
"""


def write_artifacts(
    summary: dict[str, Any],
    overlap: pd.DataFrame,
    missing: pd.DataFrame,
    outdir: Path = DEFAULT_OUTDIR,
    report_path: Path = DEFAULT_REPORT,
) -> None:
    outdir = Path(outdir)
    report_path = Path(report_path)
    if outdir.exists():
        raise FileExistsError(f"Refusing to overwrite audit directory: {outdir}")
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    outdir.mkdir(parents=True, exist_ok=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    overlap.to_csv(outdir / "v1_v2_disease_name_overlap.csv", index=False)
    missing.to_csv(outdir / "v2_missing_sequence_ids.csv", index=False)
    report_path.write_text(render_report(summary), encoding="utf-8")


def validate_artifacts(
    expected: dict[str, Any],
    expected_overlap: pd.DataFrame,
    expected_missing: pd.DataFrame,
    outdir: Path = DEFAULT_OUTDIR,
    report_path: Path = DEFAULT_REPORT,
) -> None:
    outdir = Path(outdir)
    actual = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    if actual != expected:
        raise AssertionError("Saved V2 feasibility summary differs from current source audit")
    actual_overlap = pd.read_csv(outdir / "v1_v2_disease_name_overlap.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(actual_overlap, expected_overlap, check_dtype=False)
    actual_missing = pd.read_csv(outdir / "v2_missing_sequence_ids.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(actual_missing, expected_missing, check_dtype=False)
    if report_path.read_text(encoding="utf-8") != render_report(expected):
        raise AssertionError("Saved V2 feasibility report differs from current source audit")

    assert expected["v2_population"] == {
        "full_rows": EXPECTED_V2_ROWS,
        "unique_rna_ids": EXPECTED_V2_ROWS,
        "usable_sequence_rows": EXPECTED_USABLE_ROWS,
        "missing_sequence_rows": EXPECTED_V2_ROWS - EXPECTED_USABLE_ROWS,
        "sequence_feature_rows": EXPECTED_USABLE_ROWS,
    }
    assert expected["feature_contract"]["valid_sequence_features"] == EXPECTED_SEQUENCE_FEATURES
    assert expected["association_outcomes"]["v1_raw_disease_columns"] == EXPECTED_V1_ASSOCIATIONS
    assert expected["association_outcomes"]["v2_raw_disease_columns"] == EXPECTED_V2_ASSOCIATIONS
    assert expected["width_analysis"]["sequence_plus_all_v2_outcomes_width"] == 978
    assert expected["decision"]["v2_model_training_performed"] is False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate saved artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, overlap, missing = load_and_analyze()
    if args.check:
        validate_artifacts(summary, overlap, missing)
        print("V2 827 feasibility audit passed")
        return
    write_artifacts(summary, overlap, missing)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
