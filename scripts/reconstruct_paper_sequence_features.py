"""Reconstruct the sequence-only part of the released CIBCB 2025 feature file.

The released 827-column file has three contiguous blocks:

* 462 Pse-in-One RNA DCC values (all 22 properties, lag 1);
* 285 columns copied from the V1 lncRNA-disease association matrix;
* 16 dinucleotide and 64 trinucleotide occurrence frequencies.

This script deliberately emits only the 542 sequence-derived columns.  It uses
the arithmetic and column order of Pse-in-One 1.0.6.  Output columns are named
so the same schema can be generated independently for V1 and V2.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROPERTIES = PROJECT_ROOT / "config/pse_in_one_rna_dcc_properties.csv"
ALPHABET = "ACGU"
DINUCLEOTIDES = ["".join(x) for x in itertools.product(ALPHABET, repeat=2)]
TRINUCLEOTIDES = ["".join(x) for x in itertools.product(ALPHABET, repeat=3)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sequence(value: object) -> tuple[str, Counter[str]]:
    """Convert DNA spelling to RNA and remove unsupported IUPAC symbols.

    The paper artifact skipped several sequences containing ambiguous symbols.
    Removing only those symbols preserves the row as required by the corrected
    experiment contract.  Every removal is counted in the metadata sidecar.
    """
    if pd.isna(value):
        return "", Counter()
    raw = str(value).upper().replace("T", "U")
    removed = Counter(char for char in raw if char not in ALPHABET)
    return "".join(char for char in raw if char in ALPHABET), removed


def load_properties(path: Path) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path, index_col=0)
    if frame.index.tolist() != DINUCLEOTIDES:
        raise ValueError(
            f"{path} must contain Pse-in-One dinucleotides in ACGU lexical order."
        )
    values = frame.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if values.shape != (16, 22) or not np.isfinite(values).all():
        raise ValueError(f"{path} must contain a finite 16 x 22 property table.")
    return frame.columns.astype(str).tolist(), values


def dcc_vector(sequence: str, properties: np.ndarray) -> np.ndarray:
    """Return Pse-in-One RNA DCC, all properties, lag 1 (462 values)."""
    if len(sequence) < 3:
        raise ValueError("DCC requires at least three valid A/C/G/U nucleotides.")
    dinuc_to_row = {name: idx for idx, name in enumerate(DINUCLEOTIDES)}
    encoded = np.asarray(
        [properties[dinuc_to_row[sequence[i : i + 2]]] for i in range(len(sequence) - 1)],
        dtype=float,
    )
    # Pse-in-One 1.0.6 make_cc_vec uses L-k-lag+1 pairs and divides
    # each property mean by the full sequence length (rather than pair count).
    pair_count = len(sequence) - 2
    left = encoded[:pair_count]
    right = encoded[1 : pair_count + 1]
    means = left.sum(axis=0) / len(sequence)
    covariance = ((left - means).T @ (right - means)) / pair_count
    mask = ~np.eye(covariance.shape[0], dtype=bool)
    return np.round(covariance[mask], 8)


def kmer_vector(sequence: str, k: int) -> np.ndarray:
    kmers = DINUCLEOTIDES if k == 2 else TRINUCLEOTIDES
    counts = Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))
    vector = np.asarray([counts.get(kmer, 0) for kmer in kmers], dtype=float)
    total = vector.sum()
    if total == 0:
        raise ValueError(f"k={k} frequencies require at least {k} valid nucleotides.")
    return np.round(vector / total, 8)


def feature_columns(property_names: Iterable[str]) -> list[str]:
    names = list(property_names)
    dcc = [
        f"dcc_lag1::{left}=>{right}"
        for left in names
        for right in names
        if left != right
    ]
    return dcc + [f"kmer2::{x}" for x in DINUCLEOTIDES] + [
        f"kmer3::{x}" for x in TRINUCLEOTIDES
    ]


def reconstruct(
    input_csv: Path,
    properties_csv: Path,
    *,
    id_column: str,
    sequence_column: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.read_csv(input_csv)
    missing = [name for name in (id_column, sequence_column) if name not in source]
    if missing:
        raise ValueError(f"{input_csv} is missing columns: {', '.join(missing)}")
    if source[id_column].astype(str).duplicated().any():
        raise ValueError(f"{input_csv} contains duplicate {id_column} values.")

    property_names, properties = load_properties(properties_csv)
    rows: list[np.ndarray] = []
    kept_ids: list[str] = []
    removed_by_symbol: Counter[str] = Counter()
    missing_sequence_ids: list[str] = []
    cleaned_sequence_ids: list[str] = []

    for sample_id, raw_sequence in source[[id_column, sequence_column]].itertuples(index=False):
        sample_id = str(sample_id)
        sequence, removed = normalize_sequence(raw_sequence)
        if not sequence:
            missing_sequence_ids.append(sample_id)
            continue
        if removed:
            removed_by_symbol.update(removed)
            cleaned_sequence_ids.append(sample_id)
        if len(sequence) < 3:
            raise ValueError(f"{sample_id} has fewer than three valid nucleotides.")
        rows.append(
            np.concatenate(
                [dcc_vector(sequence, properties), kmer_vector(sequence, 2), kmer_vector(sequence, 3)]
            )
        )
        kept_ids.append(sample_id)

    output = pd.DataFrame(rows, columns=feature_columns(property_names), index=kept_ids)
    output.index.name = "ID"
    metadata: dict[str, object] = {
        "input_csv": input_csv.resolve().as_posix(),
        "input_sha256": sha256_file(input_csv),
        "properties_csv": properties_csv.resolve().as_posix(),
        "properties_sha256": sha256_file(properties_csv),
        "source_rows": int(len(source)),
        "output_rows": int(len(output)),
        "output_features": int(output.shape[1]),
        "missing_sequence_count": len(missing_sequence_ids),
        "missing_sequence_ids": missing_sequence_ids,
        "cleaned_sequence_count": len(cleaned_sequence_ids),
        "cleaned_sequence_ids": cleaned_sequence_ids,
        "removed_symbols": dict(sorted(removed_by_symbol.items())),
        "feature_recipe": {
            "dcc": "Pse-in-One 1.0.6 RNA DCC, 22 properties, lag=1",
            "kmer": "overlapping normalized RNA k-mers, k=2 and k=3",
            "association_derived_columns": 0,
        },
    }
    return output, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--properties", type=Path, default=DEFAULT_PROPERTIES)
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--sequence-column", default="seqs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, metadata = reconstruct(
        args.input,
        args.properties,
        id_column=args.id_column,
        sequence_column=args.sequence_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output)
    metadata["output_csv"] = args.output.resolve().as_posix()
    metadata["output_sha256"] = sha256_file(args.output)
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: {output.shape[0]} rows x {output.shape[1]} features; "
        f"missing sequences={metadata['missing_sequence_count']}"
    )


if __name__ == "__main__":
    main()
