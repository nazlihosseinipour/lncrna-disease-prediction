#!/usr/bin/env python3
"""Prepare a diagnostic 827-column V1 input from Nazli's processed data.

This construction is inferred from the released artifact's block structure, but
it never reads or copies Felipe's feature CSV.  Every model-input value comes
from Nazli's processed V1 source: 462 DCC values and 80 normalized k-mer values
are calculated from sequence, while the middle 285 values are copied from
Nazli's RNA-disease association columns.

The association block is outcome-derived and therefore leaks target
information.  The resulting matrix is for diagnosis only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

try:
    from scripts.reconstruct_paper_sequence_features import reconstruct
except ModuleNotFoundError:
    from reconstruct_paper_sequence_features import reconstruct  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "results/nazli_inferred_827_v1_2026-10-08/inputs"
SOURCE = ROOT / "Data/output_data/sequences_for_oop.csv"
PROPERTIES = ROOT / "config/pse_in_one_rna_dcc_properties.csv"
PAPER_Y = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv"
PAPER_SPLITS = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/splits/inductive/inductive_splits.csv"
ID_REPAIRS = {"BX374997.2": "BX374997.2 "}

X_NAME = "v1_nazli_inferred_827_X.csv"
Y_NAME = "v1_paper_44_Y.csv"
SPLIT_NAME = "v1_released_folds_355.csv"
MANIFEST_NAME = "manifest.csv"
PROVENANCE_NAME = "provenance.json"


class InputError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InputError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _repair_ids(values: pd.Series) -> pd.Series:
    result = values.astype(str).copy()
    for source_id, target_id in ID_REPAIRS.items():
        require(int((result == source_id).sum()) == 1, f"expected one source ID {source_id!r}")
        require(not bool((result == target_id).any()), f"repair target already present: {target_id!r}")
        result.loc[result == source_id] = target_id
    return result


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = pd.read_csv(SOURCE, dtype={"ID": str}, keep_default_na=False)
    require(source.shape == (355, 288), f"unexpected processed source shape: {source.shape}")
    require(source.columns[:3].tolist() == ["Unnamed: 0", "ID", "seqs"], "unexpected source prefix")
    require(not source["ID"].duplicated().any(), "processed source IDs are not unique")

    association_names = source.columns[3:].astype(str).tolist()
    require(len(association_names) == 285, "processed source must have 285 association columns")
    associations = source.iloc[:, 3:].apply(pd.to_numeric, errors="raise")
    require(bool(np.isin(associations.to_numpy(), [0, 1]).all()), "association block must be binary")

    sequence_x, sequence_metadata = reconstruct(
        SOURCE,
        PROPERTIES,
        id_column="ID",
        sequence_column="seqs",
    )
    require(sequence_x.shape == (355, 542), f"unexpected sequence feature shape: {sequence_x.shape}")

    repaired_ids = _repair_ids(source["ID"])
    source = source.copy()
    source["ID"] = repaired_ids
    sequence_x = sequence_x.copy()
    sequence_x.index = _repair_ids(pd.Series(sequence_x.index, dtype=str)).to_numpy()
    sequence_x.index.name = "ID"

    y = pd.read_csv(PAPER_Y, index_col=0)
    y.index = y.index.astype(str)
    require(y.shape == (355, 44), f"unexpected paper label shape: {y.shape}")
    require(not y.index.has_duplicates, "paper label IDs are not unique")
    require(bool(np.isin(y.to_numpy(), [0, 1]).all()), "paper labels must be binary")
    require(set(source["ID"]) == set(y.index), "processed and paper label ID sets differ")

    sequence_x = sequence_x.loc[y.index]
    aligned_source = source.set_index("ID").loc[y.index]
    association_x = aligned_source.iloc[:, 2:].apply(pd.to_numeric, errors="raise")
    association_x.columns = [f"association_outcome::{name}" for name in association_names]
    require(association_x.shape == (355, 285), "aligned association block has wrong shape")

    x = pd.concat(
        [sequence_x.iloc[:, :462], association_x, sequence_x.iloc[:, 462:]],
        axis=1,
    )
    require(x.shape == (355, 827), f"inferred feature matrix has wrong shape: {x.shape}")
    require(x.index.tolist() == y.index.tolist(), "X and Y ID order differs")
    require(not x.columns.duplicated().any(), "inferred feature names are not unique")
    require(bool(np.isfinite(x.to_numpy(float)).all()), "inferred features contain non-finite values")

    facts: dict[str, object] = {
        "shape": [355, 827],
        "components": {"dcc": 462, "association": 285, "kmer2": 16, "kmer3": 64},
        "felipe_feature_csv_used": False,
        "construction": (
            "Calculated from Nazli processed sequences and copied from Nazli processed "
            "RNA-disease association columns; no Felipe feature values were read."
        ),
        "association_block": {
            "source": relative(SOURCE),
            "columns": 285,
            "positive_cells": int(association_x.to_numpy().sum()),
            "outcome_derived": True,
            "label_leakage_risk": True,
        },
        "id_repairs": ID_REPAIRS,
        "sequence_reconstruction": sequence_metadata,
        "source_hashes": {
            "nazli_processed_v1": sha256(SOURCE),
            "pse_in_one_properties": sha256(PROPERTIES),
            "paper_labels": sha256(PAPER_Y),
            "paper_splits": sha256(PAPER_SPLITS),
        },
    }
    return x, y, facts


def generate(outdir: Path) -> None:
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite existing output directory: {outdir}")
    outdir.mkdir(parents=True, exist_ok=False)
    x, y, facts = build_frames()
    x_path = outdir / X_NAME
    y_path = outdir / Y_NAME
    split_path = outdir / SPLIT_NAME
    x.to_csv(x_path)
    y.to_csv(y_path)
    shutil.copyfile(PAPER_SPLITS, split_path)

    manifest = pd.DataFrame(
        [
            {
                "version": "v1",
                "label_space": "all44",
                "population": "v1_nazli_processed_355",
                "feature_set": "nazli_inferred_827_diagnostic",
                "feature_key": "nazli_inferred_827",
                "domains": "rna",
                "components": "462 DCC + 285 outcome-derived associations + 16 kmer2 + 64 kmer3",
                "x_path": relative(x_path),
                "y_path": relative(y_path),
                "split_path": relative(split_path),
                "n_samples": 355,
                "n_features": 827,
                "n_labels": 44,
                "x_sha256": sha256(x_path),
                "y_sha256": sha256(y_path),
                "split_sha256": sha256(split_path),
                "preprocessing_mode": "code_generated_from_nazli_values_with_outcome_leakage",
                "split_method": "released_paper_ten_folds_unchanged",
            }
        ]
    )
    manifest.to_csv(outdir / MANIFEST_NAME, index=False)
    facts["outputs"] = {
        "x": {"path": relative(x_path), "sha256": sha256(x_path)},
        "y": {"path": relative(y_path), "sha256": sha256(y_path)},
        "splits": {"path": relative(split_path), "sha256": sha256(split_path)},
        "manifest": relative(outdir / MANIFEST_NAME),
    }
    (outdir / PROVENANCE_NAME).write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Nazli inferred diagnostic matrix: {x.shape[0]} rows x {x.shape[1]} columns")


def check(outdir: Path) -> None:
    for name in (X_NAME, Y_NAME, SPLIT_NAME, MANIFEST_NAME, PROVENANCE_NAME):
        require((outdir / name).is_file(), f"missing output: {outdir / name}")
    expected_x, expected_y, expected_facts = build_frames()
    stored_x = pd.read_csv(outdir / X_NAME, index_col=0)
    stored_y = pd.read_csv(outdir / Y_NAME, index_col=0)
    stored_x.index = stored_x.index.astype(str)
    stored_y.index = stored_y.index.astype(str)
    pd.testing.assert_frame_equal(stored_x, expected_x, check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(stored_y, expected_y, check_dtype=False, check_exact=True)
    require(sha256(outdir / SPLIT_NAME) == sha256(PAPER_SPLITS), "stored folds changed")
    provenance = json.loads((outdir / PROVENANCE_NAME).read_text(encoding="utf-8"))
    for key in ("shape", "components", "felipe_feature_csv_used", "association_block"):
        require(provenance[key] == expected_facts[key], f"provenance field changed: {key}")
    manifest = pd.read_csv(outdir / MANIFEST_NAME).iloc[0]
    require(int(manifest["n_features"]) == 827, "manifest feature count changed")
    require(manifest["x_sha256"] == sha256(outdir / X_NAME), "manifest X hash changed")
    require(manifest["y_sha256"] == sha256(outdir / Y_NAME), "manifest Y hash changed")
    require(manifest["split_sha256"] == sha256(outdir / SPLIT_NAME), "manifest split hash changed")
    print("Nazli inferred 827 input check passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    try:
        check(outdir) if args.check else generate(outdir)
    except (InputError, FileExistsError, FileNotFoundError, ValueError, AssertionError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
