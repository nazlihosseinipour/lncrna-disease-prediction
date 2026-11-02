"""Validate immutable inputs for within-version lncRNA experiments.

The historical runners trusted manifest dimensions and split files.  That made it
possible to label a filtered or stale artifact as a different experiment.  This
module validates the files actually read by a run and returns their measured
metadata for provenance records.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidatedExperiment:
    x_path: Path
    y_path: Path
    split_path: Path
    x: pd.DataFrame
    y: pd.DataFrame
    splits: pd.DataFrame
    metadata: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact_path(project_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _required_int(row: Mapping[str, object], field: str) -> int:
    if field not in row or pd.isna(row[field]):
        raise ValueError(f"Manifest is missing required field {field!r}.")
    value = float(row[field])
    if not value.is_integer():
        raise ValueError(f"Manifest field {field!r} must be an integer, got {row[field]!r}.")
    return int(value)


def _validate_declared_hash(
    row: Mapping[str, object], field: str, path: Path, measured: str
) -> None:
    declared = row.get(field)
    if declared is None or pd.isna(declared) or not str(declared).strip():
        return
    if str(declared).strip().lower() != measured:
        raise ValueError(
            f"{field} does not match {path}: declared {declared}, measured {measured}."
        )


def _split_indexes(series: pd.Series, *, column: str, n_samples: int) -> list[int]:
    numeric = pd.to_numeric(series.dropna(), errors="raise").to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"Split column {column!r} contains non-integer indexes.")
    indexes = numeric.astype(int).tolist()
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"Split column {column!r} contains duplicate indexes.")
    invalid = [index for index in indexes if index < 0 or index >= n_samples]
    if invalid:
        raise ValueError(
            f"Split column {column!r} contains indexes outside 0..{n_samples - 1}: "
            f"{invalid[:5]}."
        )
    return indexes


def _validate_splits(splits: pd.DataFrame, *, n_samples: int, n_splits: int) -> None:
    all_rows = set(range(n_samples))
    all_test_indexes: list[int] = []
    for fold in range(1, n_splits + 1):
        train_column = f"train_fold{fold}"
        test_column = f"test_fold{fold}"
        missing = [name for name in (train_column, test_column) if name not in splits]
        if missing:
            raise ValueError(f"Split file is missing columns: {', '.join(missing)}.")
        train = _split_indexes(
            splits[train_column], column=train_column, n_samples=n_samples
        )
        test = _split_indexes(splits[test_column], column=test_column, n_samples=n_samples)
        train_set = set(train)
        test_set = set(test)
        overlap = train_set & test_set
        if overlap:
            raise ValueError(f"Fold {fold} has train/test overlap: {sorted(overlap)[:5]}.")
        if train_set | test_set != all_rows:
            missing_rows = sorted(all_rows - (train_set | test_set))
            raise ValueError(
                f"Fold {fold} does not cover every sample; missing {missing_rows[:5]}."
            )
        all_test_indexes.extend(test)

    if len(all_test_indexes) != n_samples or set(all_test_indexes) != all_rows:
        raise ValueError(
            "Outer test folds must partition every sample exactly once across the run."
        )


def validate_manifest_row(
    row: Mapping[str, object],
    *,
    project_root: Path,
    n_splits: int,
) -> ValidatedExperiment:
    """Load and validate one manifest row without changing its data."""
    paths = {
        name: resolve_artifact_path(project_root, row[name])
        for name in ("x_path", "y_path", "split_path")
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Manifest {name} does not exist: {path}")

    x = pd.read_csv(paths["x_path"], index_col=0)
    y = pd.read_csv(paths["y_path"], index_col=0)
    splits = pd.read_csv(paths["split_path"])
    x.index = x.index.astype(str)
    y.index = y.index.astype(str)

    if x.index.has_duplicates or y.index.has_duplicates:
        raise ValueError("X and Y sample identifiers must be unique.")
    if x.index.tolist() != y.index.tolist():
        raise ValueError("X and Y identifiers must match in the same order; no join is allowed.")
    if x.empty or y.empty:
        raise ValueError("X and Y must each contain at least one row and one data column.")

    x_numeric = x.apply(pd.to_numeric, errors="raise")
    y_numeric = y.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x_numeric.to_numpy(dtype=float)).all():
        raise ValueError("X contains missing or non-finite values.")
    y_values = y_numeric.to_numpy(dtype=float)
    if not np.isfinite(y_values).all() or not np.isin(y_values, [0, 1]).all():
        raise ValueError("Y must be a finite binary indicator matrix.")

    measured_shape = {
        "n_samples": x.shape[0],
        "n_features": x.shape[1],
        "n_labels": y.shape[1],
    }
    for field, measured in measured_shape.items():
        declared = _required_int(row, field)
        if declared != measured:
            raise ValueError(
                f"Manifest {field}={declared} does not match the file value {measured}."
            )

    _validate_splits(splits, n_samples=x.shape[0], n_splits=n_splits)

    hashes = {
        "x_sha256": sha256_file(paths["x_path"]),
        "y_sha256": sha256_file(paths["y_path"]),
        "split_sha256": sha256_file(paths["split_path"]),
    }
    for field, path_name in (
        ("x_sha256", "x_path"),
        ("y_sha256", "y_path"),
        ("split_sha256", "split_path"),
    ):
        _validate_declared_hash(row, field, paths[path_name], hashes[field])

    metadata: dict[str, object] = {
        **measured_shape,
        **hashes,
        "x_path": paths["x_path"].as_posix(),
        "y_path": paths["y_path"].as_posix(),
        "split_path": paths["split_path"].as_posix(),
    }
    return ValidatedExperiment(
        x_path=paths["x_path"],
        y_path=paths["y_path"],
        split_path=paths["split_path"],
        x=x_numeric,
        y=y_numeric.astype(int),
        splits=splits,
        metadata=metadata,
    )
