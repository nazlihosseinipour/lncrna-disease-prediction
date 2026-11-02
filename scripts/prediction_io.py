"""Atomic, hash-verified prediction cells for the Felipe V1/V2 runners.

A cell is one fitted model's raw scores for one evaluation population:
within = (version, label_space, model, fold); transfer = (source, target, model).

Each cell is three files in one directory, written in this order:

    <stem>.predictions.csv   ID column, then label columns in manifest order (raw scores)
    <stem>.contract.json     provenance, including prediction_file_sha256
    <stem>.done              completion marker; written last

A cell counts as complete only if all three exist, the CSV hash matches the
contract, the row/label hashes match the CSV, and every caller-supplied expected
contract field matches. Invalid cells are never edited in place; callers move
them aside with ``quarantine_cell`` before redoing the work.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
PREDICTIONS_SUFFIX = ".predictions.csv"
CONTRACT_SUFFIX = ".contract.json"
DONE_SUFFIX = ".done"
CELL_SUFFIXES = (PREDICTIONS_SUFFIX, CONTRACT_SUFFIX, DONE_SUFFIX)
# Filled by write_prediction_cell; callers must not supply them.
DERIVED_FIELDS = (
    "schema_version",
    "row_ids_sha256",
    "label_names_sha256",
    "n_rows",
    "n_labels",
    "prediction_file_sha256",
    "created_at_utc",
)


class PredictionCellError(ValueError):
    """A prediction cell is missing, incomplete, corrupted, or from another run."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(items: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(item) for item in items).encode("utf-8")).hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write via a same-directory temp file, fsync, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def cell_paths(outdir: Path, stem: str) -> dict[str, Path]:
    outdir = Path(outdir)
    return {
        "predictions": outdir / f"{stem}{PREDICTIONS_SUFFIX}",
        "contract": outdir / f"{stem}{CONTRACT_SUFFIX}",
        "done": outdir / f"{stem}{DONE_SUFFIX}",
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _validate_frame(predictions: pd.DataFrame) -> tuple[list[str], list[str]]:
    if not isinstance(predictions, pd.DataFrame):
        raise PredictionCellError("predictions must be a pandas DataFrame.")
    ids = [str(item) for item in predictions.index]
    labels = [str(item) for item in predictions.columns]
    if not ids or not labels:
        raise PredictionCellError("predictions must have at least one row and one label.")
    if len(set(ids)) != len(ids):
        raise PredictionCellError("predictions contain duplicate RNA IDs.")
    if len(set(labels)) != len(labels):
        raise PredictionCellError("predictions contain duplicate label names.")
    if "ID" in labels:
        raise PredictionCellError("'ID' is reserved for the identifier column.")
    values = predictions.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise PredictionCellError("predictions contain missing or non-finite values.")
    return ids, labels


def write_prediction_cell(
    outdir: Path,
    stem: str,
    predictions: pd.DataFrame,
    contract: dict,
) -> Path:
    """Write CSV, contract, and .done atomically in that order; return the CSV path."""
    paths = cell_paths(outdir, stem)
    reserved = sorted(set(contract) & set(DERIVED_FIELDS))
    if reserved:
        raise PredictionCellError(f"contract must not set derived fields: {reserved}")
    ids, labels = _validate_frame(predictions)
    # A stale marker must never describe new bytes, even transiently.
    if paths["done"].exists():
        raise PredictionCellError(
            f"{paths['done']} already exists; quarantine the cell before rewriting it."
        )

    frame = predictions.copy()
    frame.index = pd.Index(ids, name="ID")
    frame.columns = labels
    buffer = io.StringIO()
    frame.to_csv(buffer, float_format="%.17g", lineterminator="\n")
    data = buffer.getvalue().encode("utf-8")
    atomic_write_bytes(paths["predictions"], data)

    full = dict(_jsonable(contract))
    full.update(
        {
            "schema_version": SCHEMA_VERSION,
            "row_ids_sha256": sha256_lines(ids),
            "label_names_sha256": sha256_lines(labels),
            "n_rows": len(ids),
            "n_labels": len(labels),
            "prediction_file_sha256": hashlib.sha256(data).hexdigest(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    atomic_write_text(paths["contract"], json.dumps(full, indent=2, sort_keys=True) + "\n")
    atomic_write_text(paths["done"], full["prediction_file_sha256"] + "\n")
    return paths["predictions"]


def _values_equal(actual, expected) -> bool:
    actual = _jsonable(actual)
    expected = _jsonable(expected)
    if isinstance(actual, float) and isinstance(expected, float):
        if math.isnan(actual) and math.isnan(expected):
            return True
    return actual == expected


def read_prediction_cell(
    outdir: Path,
    stem: str,
    expected: Mapping[str, object] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load a complete cell, raising PredictionCellError on any integrity failure."""
    paths = cell_paths(outdir, stem)
    for name in ("done", "contract", "predictions"):
        if not paths[name].is_file():
            raise PredictionCellError(f"cell {stem!r} is incomplete: missing {paths[name].name}")
    try:
        contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PredictionCellError(f"cell {stem!r} has an unreadable contract: {error}") from error
    if not isinstance(contract, dict):
        raise PredictionCellError(f"cell {stem!r} contract is not a JSON object.")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise PredictionCellError(f"cell {stem!r} has schema {contract.get('schema_version')!r}.")

    measured = sha256_file(paths["predictions"])
    if measured != contract.get("prediction_file_sha256"):
        raise PredictionCellError(f"cell {stem!r} prediction hash does not match its contract.")
    marker = paths["done"].read_text(encoding="utf-8").strip()
    if marker != measured:
        raise PredictionCellError(f"cell {stem!r} .done marker does not match its predictions.")

    frame = pd.read_csv(paths["predictions"], dtype={"ID": str}, keep_default_na=False)
    if not len(frame.columns) or frame.columns[0] != "ID":
        raise PredictionCellError(f"cell {stem!r} predictions must start with an ID column.")
    frame = frame.set_index("ID")
    frame.index = frame.index.astype(str)
    ids = frame.index.tolist()
    labels = [str(column) for column in frame.columns]
    if len(set(ids)) != len(ids):
        raise PredictionCellError(f"cell {stem!r} contains duplicate RNA IDs.")
    try:
        frame = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise PredictionCellError(f"cell {stem!r} contains non-numeric scores.") from error
    if not np.isfinite(frame.to_numpy()).all():
        raise PredictionCellError(f"cell {stem!r} contains missing or non-finite scores.")
    if sha256_lines(ids) != contract.get("row_ids_sha256") or len(ids) != contract.get("n_rows"):
        raise PredictionCellError(f"cell {stem!r} row IDs do not match its contract.")
    if (
        sha256_lines(labels) != contract.get("label_names_sha256")
        or len(labels) != contract.get("n_labels")
    ):
        raise PredictionCellError(f"cell {stem!r} label names do not match its contract.")

    for key, value in (expected or {}).items():
        if key not in contract:
            raise PredictionCellError(f"cell {stem!r} contract lacks expected field {key!r}.")
        if not _values_equal(contract[key], value):
            raise PredictionCellError(
                f"cell {stem!r} contract field {key!r} differs: "
                f"found {contract[key]!r}, expected {value!r}."
            )
    return frame, contract


def cell_status(
    outdir: Path,
    stem: str,
    expected: Mapping[str, object] | None = None,
) -> Literal["missing", "complete", "invalid"]:
    paths = cell_paths(outdir, stem)
    if not any(path.exists() for path in paths.values()):
        return "missing"
    try:
        read_prediction_cell(outdir, stem, expected)
    except PredictionCellError:
        return "invalid"
    return "complete"


class FrozenSourceError(RuntimeError):
    """A source file differs from the frozen hash table (decision D1)."""


def verify_frozen_sources(table_path: Path, root: Path) -> dict[str, str]:
    """Refuse to run unless every listed source matches its frozen SHA-256.

    The table is a CSV with at least ``path`` (relative to ``root``) and ``sha256``.
    Returns {relative path: sha256} for recording in contracts.
    """
    table_path = Path(table_path)
    if not table_path.is_file():
        raise FrozenSourceError(f"frozen source table not found: {table_path}")
    table = pd.read_csv(table_path, dtype=str)
    missing_columns = {"path", "sha256"} - set(table.columns)
    if missing_columns or table.empty:
        raise FrozenSourceError(f"{table_path} must list path,sha256 rows.")
    if table["path"].duplicated().any():
        raise FrozenSourceError(f"{table_path} lists a path twice.")
    measured: dict[str, str] = {}
    problems: list[str] = []
    for row in table.itertuples(index=False):
        path = Path(root) / row.path
        if not path.is_file():
            problems.append(f"missing {row.path}")
            continue
        digest = sha256_file(path)
        if digest != str(row.sha256).strip().lower():
            problems.append(f"changed {row.path}")
        measured[row.path] = digest
    if problems:
        raise FrozenSourceError(
            "source files differ from the frozen table; refusing to run: " + "; ".join(problems)
        )
    return measured


def library_versions() -> dict[str, str]:
    import platform

    import sklearn

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }
    try:
        from importlib.metadata import version

        versions["iterative-stratification"] = version("iterative-stratification")
    except Exception:  # pragma: no cover - optional dependency metadata
        versions["iterative-stratification"] = "unknown"
    return versions


def quarantine_cell(outdir: Path, stem: str) -> Path | None:
    """Move any existing cell files to <outdir>/_rejected/<timestamp>/; never delete."""
    outdir = Path(outdir)
    present = [path for path in cell_paths(outdir, stem).values() if path.exists()]
    if not present:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = outdir / "_rejected" / stamp
    target.mkdir(parents=True, exist_ok=False)
    for path in present:
        shutil.move(os.fspath(path), os.fspath(target / path.name))
    return target
