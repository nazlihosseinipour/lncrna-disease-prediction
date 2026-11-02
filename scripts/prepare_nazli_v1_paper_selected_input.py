#!/usr/bin/env python3
"""Build the verified paper feature selection from Nazli's processed V1 sequences.

The selected feature contract is the sequence-derived portion that can be
identified in the released V1 artifact column by column: Pse-in-One DCC with
22 RNA properties at lag 1, normalized 2-mers, and normalized 3-mers (542
columns). The two RNA IDs absent from Nazli's generated feature outputs remain
excluded, and the released folds are mapped by RNA ID to the retained 353 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reconstruct_paper_sequence_features import reconstruct


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "nazli_selected_v1_felipe_2026-10-05" / "inputs"
SEQUENCES = ROOT / "Data" / "output_data" / "sequences_for_oop.csv"
GENERATED_ID_SOURCE = ROOT / "Final_output" / "V1" / "rna" / "sequences_for_oop_kmer_matrix_k2.csv"
PROPERTIES = ROOT / "config" / "pse_in_one_rna_dcc_properties.csv"
PAPER_Y = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv"
PAPER_SPLITS = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/splits/inductive/inductive_splits.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapped_splits(source: pd.DataFrame, paper_ids: list[str], kept_ids: list[str]) -> pd.DataFrame:
    new_position = {sample_id: position for position, sample_id in enumerate(kept_ids)}
    columns: dict[str, pd.Series] = {}
    for fold in range(1, 11):
        for kind in ("train", "test"):
            name = f"{kind}_fold{fold}"
            old = source[name].dropna().astype(int).tolist()
            new = [new_position[paper_ids[pos]] for pos in old if paper_ids[pos] in new_position]
            columns[name] = pd.Series(new, dtype="Int64")
    result = pd.DataFrame(columns)
    expected = set(range(len(kept_ids)))
    all_test: list[int] = []
    for fold in range(1, 11):
        train = set(result[f"train_fold{fold}"].dropna().astype(int))
        test = set(result[f"test_fold{fold}"].dropna().astype(int))
        if train & test or train | test != expected:
            raise ValueError(f"Mapped fold {fold} is not a full train/test partition")
        all_test.extend(test)
    if len(all_test) != len(kept_ids) or set(all_test) != expected:
        raise ValueError("Mapped test folds do not partition retained rows exactly once")
    return result


def run(outdir: Path) -> dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=False)
    generated_ids = pd.read_csv(GENERATED_ID_SOURCE)["sample_id"].astype(str).tolist()
    if len(generated_ids) != 353 or len(set(generated_ids)) != 353:
        raise ValueError("Expected 353 unique RNA IDs in the generated V1 feature output")

    sequence_source = pd.read_csv(SEQUENCES)
    sequence_source["ID"] = sequence_source["ID"].astype(str)
    by_id = sequence_source.set_index("ID", drop=False)
    missing = sorted(set(generated_ids) - set(by_id.index))
    if missing:
        raise ValueError(f"Generated RNA IDs missing from processed sequence source: {missing[:5]}")
    selected_sequences = by_id.loc[generated_ids, ["ID", "seqs"]].copy()
    selected_sequences_path = outdir / "v1_processed_sequences_353.csv"
    selected_sequences.to_csv(selected_sequences_path, index=False)

    x, reconstruction = reconstruct(
        selected_sequences_path,
        PROPERTIES,
        id_column="ID",
        sequence_column="seqs",
    )
    if x.shape != (353, 542) or x.index.astype(str).tolist() != generated_ids:
        raise ValueError(f"Expected selected X shape/order (353, 542), got {x.shape}")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Selected X contains non-finite values")

    paper_y = pd.read_csv(PAPER_Y, index_col=0)
    paper_y.index = paper_y.index.astype(str)
    y = paper_y.loc[generated_ids].copy()
    if y.shape != (353, 44) or not np.isin(y.to_numpy(), [0, 1]).all():
        raise ValueError(f"Expected selected Y shape (353, 44), got {y.shape}")
    splits = mapped_splits(pd.read_csv(PAPER_SPLITS), paper_y.index.tolist(), generated_ids)

    x_path = outdir / "v1_nazli_selected_paper_sequence_542_X.csv"
    y_path = outdir / "v1_paper_44_Y_for_selected_rows.csv"
    split_path = outdir / "v1_released_folds_mapped_to_353.csv"
    x.to_csv(x_path)
    y.to_csv(y_path)
    splits.to_csv(split_path, index=False)

    manifest_path = outdir / "manifest.csv"
    manifest = pd.DataFrame([{
        "version": "v1",
        "label_space": "all44",
        "population": "v1_nazli_processed_sequence_353",
        "feature_set": "paper_artifact_verified_sequence_families_542",
        "feature_key": "nazli_selected_paper_sequence_542",
        "domains": "rna",
        "components": "Pse-in-One DCC all 22 properties lag1 + normalized kmer2 + normalized kmer3",
        "x_path": x_path.relative_to(ROOT).as_posix(),
        "y_path": y_path.relative_to(ROOT).as_posix(),
        "split_path": split_path.relative_to(ROOT).as_posix(),
        "n_samples": 353,
        "n_features": 542,
        "n_labels": 44,
        "x_sha256": sha256(x_path),
        "y_sha256": sha256(y_path),
        "split_sha256": sha256(split_path),
        "preprocessing_mode": "verified_paper_sequence_family_selection_from_nazli_processed_sequences",
        "split_method": "released_paper_folds_mapped_by_rna_id_after_two_pipeline_dropped_rows",
    }])
    manifest.to_csv(manifest_path, index=False)

    dropped_ids = [sample_id for sample_id in paper_y.index if sample_id not in set(generated_ids)]
    record: dict[str, object] = {
        "purpose": "Nazli processed V1 sequences, paper-verified feature selection, Felipe MLC pipeline",
        "selected_features": {
            "dcc": 462,
            "normalized_kmer2": 16,
            "normalized_kmer3": 64,
            "total": 542,
        },
        "excluded_generated_families": [
            "kmer1", "kmer4", "reverse-complement-kmers", "ordinary-PseDNC",
            "pipeline-DAC", "pipeline-DCC", "pipeline-DACC", "composition duplicates",
            "z-curve", "k-gap features",
        ],
        "reason_for_selection": "Only DCC-22-lag1 and normalized k=2/k=3 values are identifiable in the released paper artifact.",
        "not_a_claim": "The 542-column selection excludes the released artifact's unresolved 285-column block and is not called the paper's complete 827-feature matrix.",
        "source_hashes": {
            "processed_sequences": sha256(SEQUENCES),
            "generated_id_source": sha256(GENERATED_ID_SOURCE),
            "properties": sha256(PROPERTIES),
            "paper_labels": sha256(PAPER_Y),
            "paper_splits": sha256(PAPER_SPLITS),
        },
        "dropped_paper_v1_ids_absent_from_generated_output": dropped_ids,
        "reconstruction": reconstruction,
        "outputs": {
            "x": x_path.relative_to(ROOT).as_posix(),
            "y": y_path.relative_to(ROOT).as_posix(),
            "splits": split_path.relative_to(ROOT).as_posix(),
            "manifest": manifest_path.relative_to(ROOT).as_posix(),
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
