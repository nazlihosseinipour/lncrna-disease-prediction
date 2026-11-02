"""Build the group-aware ten-fold V2 within-version split (decision D7).

Rows whose normalized transcript sequences are identical have identical 542-feature
rows, so an ordinary row-level split lets the same X appear in train and test. This
script keeps all 4,130 rows and all 436 labels, groups rows by the SHA-256 of the
normalized sequence (the exact string the features were computed from), stratifies
groups with MultilabelStratifiedKFold on the group-level OR of labels, and expands
the group folds back to rows.

The output uses the existing split format: positional row indexes into the X/Y
order in train_fold{k}/test_fold{k} columns. The historical
``v2_all_436_splits.csv`` is left untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, (ROOT / "scripts").as_posix())
from reconstruct_paper_sequence_features import normalize_sequence  # noqa: E402

INPUTS = ROOT / "results/felipe_basics_2026-09-27/inputs"
DEFAULTS = {
    "x": INPUTS / "v2_paper_sequence_542_X.csv",
    "y": INPUTS / "v2_all_436_Y.csv",
    "sequences": ROOT / "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv",
    "output": INPUTS / "v2_all_436_group_splits.csv",
}
N_SPLITS = 10
SEED = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sequence_groups(ids: list[str], sequences: pd.Series) -> pd.Series:
    """Map each ID to the SHA-256 of its normalized sequence; empty sequences are refused."""
    missing = [rna_id for rna_id in ids if rna_id not in sequences.index]
    if missing:
        raise ValueError(f"{len(missing)} X IDs have no sequence row, e.g. {missing[:3]}.")
    groups = {}
    for rna_id in ids:
        normalized, _ = normalize_sequence(sequences.loc[rna_id])
        if not normalized:
            raise ValueError(f"{rna_id} has no usable sequence but is present in X.")
        groups[rna_id] = hashlib.sha256(normalized.encode("ascii")).hexdigest()
    return pd.Series(groups, name="group")


def assign_group_folds(
    y: pd.DataFrame, groups: pd.Series, *, n_splits: int = N_SPLITS, seed: int = SEED
) -> pd.Series:
    """Return the 1-based fold of every row; all rows of a group share one fold."""
    if y.index.tolist() != groups.index.tolist():
        raise ValueError("Group IDs must match Y IDs in the same order.")
    # Groups in order of first appearance keeps the result a function of the inputs.
    group_order = list(dict.fromkeys(groups.tolist()))
    group_y = y.groupby(groups.to_numpy(), sort=False).max().loc[group_order]
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_fold: dict[str, int] = {}
    dummy = np.zeros((len(group_y), 1))
    for fold, (_, test) in enumerate(splitter.split(dummy, group_y.to_numpy()), start=1):
        for position in test:
            group_fold[group_order[position]] = fold
    if len(group_fold) != len(group_order):
        raise ValueError("Every group must be assigned to exactly one test fold.")
    return groups.map(group_fold).astype(int).rename("fold")


def split_frame(row_fold: pd.Series, *, n_splits: int = N_SPLITS) -> pd.DataFrame:
    positions = np.arange(len(row_fold))
    folds = row_fold.to_numpy()
    columns: dict[str, pd.Series] = {}
    for fold in range(1, n_splits + 1):
        columns[f"train_fold{fold}"] = pd.Series(positions[folds != fold], dtype="Int64")
        columns[f"test_fold{fold}"] = pd.Series(positions[folds == fold], dtype="Int64")
    return pd.DataFrame(columns)


def verify(
    splits: pd.DataFrame, x: pd.DataFrame, y: pd.DataFrame, groups: pd.Series, *, n_splits: int
) -> dict[str, object]:
    """Fail unless folds partition rows and no group or identical X row crosses train/test."""
    n = len(y)
    row_hash = [hashlib.sha256(row.tobytes()).hexdigest() for row in x.to_numpy(dtype=float)]
    seen_test: list[int] = []
    per_fold = []
    for fold in range(1, n_splits + 1):
        train = splits[f"train_fold{fold}"].dropna().astype(int).tolist()
        test = splits[f"test_fold{fold}"].dropna().astype(int).tolist()
        if set(train) & set(test) or len(set(train) | set(test)) != n:
            raise ValueError(f"fold {fold} is not a train/test partition.")
        shared_groups = set(groups.iloc[train]) & set(groups.iloc[test])
        shared_x = {row_hash[i] for i in train} & {row_hash[i] for i in test}
        if shared_groups or shared_x:
            raise ValueError(
                f"fold {fold}: {len(shared_groups)} sequence groups and "
                f"{len(shared_x)} identical X rows cross train/test."
            )
        seen_test.extend(test)
        test_y = y.iloc[test]
        per_fold.append(
            {
                "fold": fold,
                "n_test": len(test),
                "n_train": len(train),
                "test_positives": int(test_y.to_numpy().sum()),
                "labels_with_test_positive": int((test_y.sum() > 0).sum()),
            }
        )
    if sorted(seen_test) != list(range(n)):
        raise ValueError("Test folds do not cover every row exactly once.")
    return {"per_fold": per_fold, "identical_sequence_train_test_overlap": 0,
            "identical_x_row_train_test_overlap": 0}


def build(
    x_path: Path, y_path: Path, sequences_path: Path, *, n_splits: int = N_SPLITS, seed: int = SEED
) -> tuple[pd.DataFrame, dict[str, object]]:
    x = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    x.index = x.index.astype(str)
    y.index = y.index.astype(str)
    if x.index.has_duplicates or x.index.tolist() != y.index.tolist():
        raise ValueError("X and Y must have unique, identical IDs in identical order.")
    raw = pd.read_csv(sequences_path, usecols=["ID", "seqs"], dtype=str)
    if raw["ID"].duplicated().any():
        raise ValueError("Sequence source has duplicate IDs.")
    ids = x.index.tolist()
    groups = sequence_groups(ids, raw.set_index("ID")["seqs"])
    row_fold = assign_group_folds(y.astype(int), groups, n_splits=n_splits, seed=seed)
    splits = split_frame(row_fold, n_splits=n_splits)
    checks = verify(splits, x, y.astype(int), groups, n_splits=n_splits)

    sizes = groups.value_counts()
    duplicate_groups = sizes[sizes > 1]
    conflicting = sum(
        len(y.loc[groups.index[groups == group]].drop_duplicates()) > 1
        for group in duplicate_groups.index
    )
    assignment = "\n".join(f"{rna_id}\t{groups[rna_id]}\t{row_fold[rna_id]}" for rna_id in ids)
    metadata: dict[str, object] = {
        "method": "group-level MultilabelStratifiedKFold on OR of member labels; "
                  "groups = SHA-256 of normalized sequence",
        "population": "v2_sequence_available_4130",
        "n_splits": n_splits,
        "random_state": seed,
        "n_rows": len(ids),
        "n_labels": y.shape[1],
        "n_groups": int(len(sizes)),
        "n_duplicate_groups": int(len(duplicate_groups)),
        "rows_in_duplicate_groups": int(duplicate_groups.sum()),
        "largest_group": int(sizes.max()),
        "duplicate_groups_with_conflicting_labels": int(conflicting),
        "labels_with_zero_positives": int((y.sum() == 0).sum()),
        "assignment_sha256": hashlib.sha256(assignment.encode("utf-8")).hexdigest(),
        "duplicate_group_members": {
            group: sorted(groups.index[groups == group].tolist())
            for group in duplicate_groups.index
        },
        "inputs": {
            "x": {"path": relative(x_path), "sha256": sha256_file(x_path)},
            "y": {"path": relative(y_path), "sha256": sha256_file(y_path)},
            "sequences": {"path": relative(sequences_path), "sha256": sha256_file(sequences_path)},
        },
        "script_sha256": sha256_file(Path(__file__)),
        **checks,
    }
    return splits, metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=Path, default=DEFAULTS["x"])
    parser.add_argument("--y", type=Path, default=DEFAULTS["y"])
    parser.add_argument("--sequences", type=Path, default=DEFAULTS["sequences"])
    parser.add_argument("--output", type=Path, default=DEFAULTS["output"])
    args = parser.parse_args(argv)
    metadata_path = args.output.with_name(args.output.name + ".metadata.json")
    for path in (args.output, metadata_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; the split is immutable once written.")

    splits, metadata = build(args.x, args.y, args.sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as handle:
        splits.to_csv(handle, index=False)
    metadata["output"] = {"path": relative(args.output), "sha256": sha256_file(args.output)}
    with metadata_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, indent=2) + "\n")
    summary = {k: v for k, v in metadata.items() if k not in {"duplicate_group_members", "per_fold"}}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
