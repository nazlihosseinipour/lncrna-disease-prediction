"""Build immutable within-version and transfer inputs for Felipe's reset analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/felipe_basics_2026-09-27/inputs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_label(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def align_labels(x: pd.DataFrame, source: pd.DataFrame, *, id_column: str) -> pd.DataFrame:
    source = source.copy()
    source[id_column] = source[id_column].astype(str)
    if source[id_column].duplicated().any():
        raise ValueError(f"Duplicate identifiers in label source column {id_column}.")
    aligned = source.set_index(id_column).loc[x.index.astype(str)]
    aligned.index.name = "ID"
    return aligned


def write_splits(y: pd.DataFrame, path: Path, *, seed: int = 0) -> None:
    splitter = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    columns: dict[str, pd.Series] = {}
    dummy = np.zeros((len(y), 1), dtype=float)
    for fold, (train, test) in enumerate(splitter.split(dummy, y.to_numpy()), start=1):
        columns[f"train_fold{fold}"] = pd.Series(train, dtype="Int64")
        columns[f"test_fold{fold}"] = pd.Series(test, dtype="Int64")
    pd.DataFrame(columns).to_csv(path, index=False)


def canonicalize_transfer_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    for column in frame.columns:
        grouped.setdefault(normalized_label(column), []).append(column)
    canonical = pd.DataFrame(
        {
            name: frame[columns].max(axis=1).astype(int)
            for name, columns in grouped.items()
        },
        index=frame.index,
    )
    return canonical, grouped


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    base = args.base.resolve()
    base.mkdir(parents=True, exist_ok=True)

    v1_x_path = base / "v1_paper_sequence_542_X.csv"
    v2_x_path = base / "v2_paper_sequence_542_X.csv"
    v1_x = pd.read_csv(v1_x_path, index_col=0)
    v2_x = pd.read_csv(v2_x_path, index_col=0)
    v1_x.index = v1_x.index.astype(str)
    v2_x.index = v2_x.index.astype(str)
    if v1_x.columns.tolist() != v2_x.columns.tolist():
        raise ValueError("V1 and V2 feature schemas are not identical.")

    v1_paper_y_source = pd.read_csv(
        ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/flat_labels/labels_flat_preprocessed.csv"
    )
    v1_within_y = align_labels(v1_x, v1_paper_y_source, id_column="ID")
    v1_within_y_path = base / "v1_paper_44_Y.csv"
    v1_within_y.to_csv(v1_within_y_path)

    v2_source = pd.read_csv(
        ROOT / "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv"
    )
    v2_within_y = align_labels(
        v2_x, v2_source.drop(columns=["seqs"]), id_column="ID"
    )
    v2_within_y_path = base / "v2_all_436_Y.csv"
    v2_within_y.to_csv(v2_within_y_path)
    v2_split_path = base / "v2_all_436_splits.csv"
    write_splits(v2_within_y, v2_split_path)

    v1_raw = pd.read_csv(
        ROOT / "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv"
    ).drop(columns=["Unnamed: 0", "seqs"], errors="ignore")
    v1_raw_y = align_labels(v1_x, v1_raw, id_column="ID")
    v2_raw_y = align_labels(
        v2_x, v2_source.drop(columns=["seqs"]), id_column="ID"
    )
    v1_canonical, v1_groups = canonicalize_transfer_labels(v1_raw_y)
    v2_canonical, v2_groups = canonicalize_transfer_labels(v2_raw_y)
    common = sorted(set(v1_canonical.columns) & set(v2_canonical.columns))
    if len(common) != 73:
        raise ValueError(f"Expected 73 normalized shared diseases, measured {len(common)}.")
    v1_transfer_y = v1_canonical[common]
    v2_transfer_y = v2_canonical[common]
    v1_transfer_path = base / "v1_shared_73_Y.csv"
    v2_transfer_path = base / "v2_shared_73_Y.csv"
    v1_transfer_y.to_csv(v1_transfer_path)
    v2_transfer_y.to_csv(v2_transfer_path)

    disease_map_path = ROOT / "config/felipe_basics_shared_73_diseases.csv"
    pd.DataFrame(
        {
            "order": range(len(common)),
            "canonical_name": common,
            "v1_source_columns": [" | ".join(v1_groups[name]) for name in common],
            "v2_source_columns": [" | ".join(v2_groups[name]) for name in common],
            "v1_positives": [int(v1_transfer_y[name].sum()) for name in common],
            "v2_positives": [int(v2_transfer_y[name].sum()) for name in common],
        }
    ).to_csv(disease_map_path, index=False)

    v1_split_path = ROOT / "lncRNA_CIBCB2025-main/create_dataset/datasets/splits/inductive/inductive_splits.csv"
    manifest_rows = []
    for version, x_path, y_path, split_path, labels in (
        ("v1", v1_x_path, v1_within_y_path, v1_split_path, 44),
        ("v2", v2_x_path, v2_within_y_path, v2_split_path, 436),
    ):
        x = pd.read_csv(x_path, index_col=0)
        manifest_rows.append(
            {
                "version": version,
                "feature_set": "paper_reconstructed_sequence_542",
                "feature_key": "paper_sequence_542",
                "domains": "rna",
                "components": "Pse-in-One DCC all 22 properties lag1 + kmer2 + kmer3",
                "x_path": relative(x_path),
                "y_path": relative(y_path),
                "split_path": relative(split_path),
                "n_samples": len(x),
                "n_features": x.shape[1],
                "n_labels": labels,
                "x_sha256": sha256_file(x_path),
                "y_sha256": sha256_file(y_path),
                "split_sha256": sha256_file(split_path),
                "preprocessing_mode": "reconstructed_from_sequence_no_label_features",
            }
        )
    within_manifest = ROOT / "config/felipe_basics_sequence_542_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(within_manifest, index=False)

    transfer_manifest = ROOT / "config/felipe_basics_transfer_73_manifest.csv"
    pd.DataFrame(
        [
            {
                "version": "v1",
                "x_path": relative(v1_x_path),
                "y_path": relative(v1_transfer_path),
                "split_path": relative(v1_split_path),
                "n_samples": len(v1_x),
                "n_features": v1_x.shape[1],
                "n_labels": len(common),
                "x_sha256": sha256_file(v1_x_path),
                "y_sha256": sha256_file(v1_transfer_path),
            },
            {
                "version": "v2",
                "x_path": relative(v2_x_path),
                "y_path": relative(v2_transfer_path),
                "split_path": relative(v2_split_path),
                "n_samples": len(v2_x),
                "n_features": v2_x.shape[1],
                "n_labels": len(common),
                "x_sha256": sha256_file(v2_x_path),
                "y_sha256": sha256_file(v2_transfer_path),
            },
        ]
    ).to_csv(transfer_manifest, index=False)

    summary = {
        "within": {
            "v1": {"rows": len(v1_x), "features": 542, "labels": 44},
            "v2": {"rows": len(v2_x), "features": 542, "labels": 436},
        },
        "transfer": {
            "features": 542,
            "shared_labels": len(common),
            "label_rule": "normalized-name intersection; duplicate spellings OR-combined",
        },
        "v2_raw_rows_without_sequence": int(len(v2_source) - len(v2_x)),
        "within_manifest": relative(within_manifest),
        "transfer_manifest": relative(transfer_manifest),
        "shared_disease_map": relative(disease_map_path),
    }
    (base / "input_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
