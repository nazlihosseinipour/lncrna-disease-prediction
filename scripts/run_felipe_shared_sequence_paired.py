"""Fit V1 and V2 training populations and score the same 48 shared-sequence test rows.

Paired runner of the shared-sequence comparison
(``Reports/felipe_audit_2026-09-27/SHARED_SEQUENCE_SPEC_2026-09-28.md``, SS1-SS10).

One resume unit per (train version, model) cell (SS3):

* one ``fit`` on the version's training IDs (cohort ``train_ids_v<n>.txt``, manifest
  row order), i.e. every row whose normalized sequence is one of the shared test
  sequences has been removed;
* one ``predict_proba(test_X)`` call on the 48 test rows (index ``sequence_sha256``),
  without any label of the test rows;
* raw scores written atomically with ``prediction_io.write_prediction_cell`` to
  ``<outdir>/paired/train_<v>__shared73__paper_sequence_542__paper/<model>/shared48``.

Before any fit the runner verifies the frozen-source table, the cohort manifest
hashes, the transfer-manifest input hashes, and repeats the leakage checks
independently from the raw sequence files. It computes no metric and applies no
threshold. Nothing fitted is persisted; an interrupted cell is redone whole.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import sys
import time

import numpy as np
import pandas as pd

try:
    from scripts import prediction_io
    from scripts.felipe_paper_models import (
        MODEL_SOURCES,
        build_paper_model,
        effective_params,
        fitted_selection,
    )
    from scripts.reconstruct_paper_sequence_features import normalize_sequence
except ImportError:  # run as ``python scripts/run_felipe_shared_sequence_paired.py``
    import prediction_io
    from felipe_paper_models import (
        MODEL_SOURCES,
        build_paper_model,
        effective_params,
        fitted_selection,
    )
    from reconstruct_paper_sequence_features import normalize_sequence


ROOT = Path(__file__).resolve().parents[1]
# Root the frozen-source table paths are resolved against (tests may redirect it).
FROZEN_ROOT = ROOT
PROFESSOR_DIR = ROOT / "lncRNA_CIBCB2025-main"
MODELS = ("rf", "ipcarf", "rflda")
VERSIONS = ("v1", "v2")
KIND = "shared_sequence_paired"
LABEL_SPACE = "shared73"
FEATURE_KEY = "paper_sequence_542"
N_FEATURES = 542
MODEL_PROFILE = "paper"
STEM = "shared48"
TEST_POPULATION = "shared_sequence_all48"
COHORT_SCHEMA = "shared_sequence_cohort/1"
TEST_INDEX_NAME = "sequence_sha256"
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FROZEN = (
    "scripts/prediction_io.py",
    "scripts/felipe_paper_models.py",
    "scripts/reconstruct_paper_sequence_features.py",
    "scripts/run_felipe_shared_sequence_paired.py",
    "scripts/build_shared_sequence_cohort.py",
)
# Contract fields that describe how a cell ran, not what it is; never resume keys.
NON_EXPECTED_FIELDS = ("fitted_selection", "execution")

DEFAULTS = {
    "cohort_dir": "results/felipe_shared_sequence_2026-09-28/cohort",
    "outdir": "results/felipe_shared_sequence_2026-09-28",
    "transfer_manifest": "config/felipe_basics_transfer_73_manifest.csv",
    "label_map": "config/felipe_basics_shared_73_diseases.csv",
    "v1_raw": "lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv",
    "v2_raw": "Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv",
    "frozen_sources": "config/felipe_shared_sequence_frozen_sources.csv",
}

sha256_file = prediction_io.sha256_file


class LeakageError(ValueError):
    """A training row shares a sequence or an exact feature row with the test set."""


def resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def sequence_sha256(value: object) -> str:
    """SHA-256 of the UTF-8 normalized sequence; empty normalized sequence -> ''."""
    sequence, _ = normalize_sequence(value)
    if not sequence:
        return ""
    return hashlib.sha256(sequence.encode("utf-8")).hexdigest()


def read_indexed_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV whose first column is a string ID index (never parsed as number/NaN)."""
    index_name = pd.read_csv(path, nrows=0).columns[0]
    frame = pd.read_csv(
        path, index_col=0, dtype={index_name: str}, keep_default_na=False, na_values=[""],
        **kwargs,
    )
    if frame.index.isna().any():
        raise ValueError(f"{path} has a missing ID.")
    frame.index = frame.index.astype(str)
    return frame


def read_float_csv(path: Path) -> pd.DataFrame:
    frame = read_indexed_csv(path, float_precision="round_trip")
    if not all(dtype == np.float64 for dtype in frame.dtypes):
        raise ValueError(f"{path} must contain only float64 feature columns.")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path} contains missing or non-finite values.")
    return frame


def read_id_lines(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    ids = [line for line in text.split("\n") if line != ""]
    # IDs are kept byte-exact: real V1 data has the ID "BX374997.2 " (trailing space),
    # which X, Y and the raw file all carry. Only carriage returns are rejected.
    if any("\r" in line for line in ids):
        raise ValueError(f"{path} has IDs containing a carriage return.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} repeats an ID.")
    return ids


# --------------------------------------------------------------------------- inputs

def label_map_names(path: Path) -> list[str]:
    table = pd.read_csv(path, dtype={"canonical_name": str})
    if not {"order", "canonical_name"} <= set(table.columns):
        raise ValueError(f"{path} must have order and canonical_name columns.")
    order = pd.to_numeric(table["order"], errors="raise")
    if order.duplicated().any():
        raise ValueError(f"{path} repeats an order value.")
    return table.assign(order=order).sort_values("order")["canonical_name"].tolist()


def load_version(
    manifest: pd.DataFrame, version: str, label_map_sha: str, labels: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = manifest[manifest["version"] == version]
    if len(rows) != 1:
        raise ValueError(f"Transfer manifest must have one row for {version}.")
    row = rows.iloc[0].to_dict()
    if row.get("label_space") != LABEL_SPACE:
        raise ValueError(f"{version} manifest label_space must be {LABEL_SPACE!r}.")
    if row.get("feature_key") != FEATURE_KEY:
        raise ValueError(f"{version} manifest feature_key must be {FEATURE_KEY!r}.")
    if str(row.get("label_map_sha256", "")).strip().lower() != label_map_sha:
        raise ValueError(f"{version} manifest label_map_sha256 differs from --label-map.")
    x_path, y_path = resolve(row["x_path"]), resolve(row["y_path"])
    if sha256_file(x_path) != row["x_sha256"] or sha256_file(y_path) != row["y_sha256"]:
        raise ValueError(f"{version} input hash does not match the transfer manifest.")
    x = read_float_csv(x_path)
    y = read_indexed_csv(y_path)
    if x.index.has_duplicates:
        raise ValueError(f"{version} contains duplicate RNA IDs.")
    if x.index.tolist() != y.index.tolist():
        raise ValueError(f"{version} X/Y IDs differ or are out of order.")
    if x.shape != (int(row["n_samples"]), int(row["n_features"])):
        raise ValueError(f"{version} X shape differs from its manifest.")
    if y.shape != (int(row["n_samples"]), int(row["n_labels"])):
        raise ValueError(f"{version} Y shape differs from its manifest.")
    if x.shape[1] != N_FEATURES:
        raise ValueError(f"{FEATURE_KEY} requires {N_FEATURES} features, found {x.shape[1]}.")
    if not all(dtype == np.float64 for dtype in x.dtypes):
        raise ValueError(f"{version} X must be float64.")
    if not y.isin([0, 1]).all().all():
        raise ValueError(f"{version} Y must be binary 0/1.")
    if y.columns.tolist() != labels:
        raise ValueError(f"{version} Y columns differ from the label map canonical order.")
    return x, y.astype(int), row


def load_raw_hashes(path: Path, ids: list[str], version: str) -> dict[str, str]:
    """Recompute the normalized-sequence hash of every ID from the raw file."""
    raw = pd.read_csv(
        path, dtype={"ID": str, "seqs": str}, usecols=lambda c: c in ("ID", "seqs"),
        keep_default_na=False, na_values=[""],
    )
    if list(sorted(raw.columns)) != ["ID", "seqs"]:
        raise ValueError(f"{path} must contain ID and seqs columns.")
    if raw["ID"].isna().any() or raw["ID"].duplicated().any():
        raise ValueError(f"{version} raw file has missing or duplicate IDs.")
    seqs = raw.set_index("ID")["seqs"]
    missing = [item for item in ids if item not in seqs.index]
    if missing:
        raise ValueError(f"{version} raw file lacks X IDs: {missing[:5]}")
    hashes: dict[str, str] = {}
    for item in ids:
        digest = sequence_sha256(seqs.loc[item])
        if not digest:
            raise ValueError(f"{version} X ID {item!r} has no usable raw sequence.")
        hashes[item] = digest
    return hashes


def verify_cohort(cohort_dir: Path, versions: list[str]) -> tuple[dict, str]:
    manifest_path = cohort_dir / "cohort_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cohort manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != COHORT_SCHEMA:
        raise ValueError(f"{manifest_path} schema must be {COHORT_SCHEMA!r}.")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f"{manifest_path} has no outputs table.")
    needed = ["test_X.csv"] + [
        f"{prefix}_ids_{version}.txt" for version in versions for prefix in ("train", "excluded")
    ]
    absent = [name for name in needed if name not in outputs]
    if absent:
        raise ValueError(f"{manifest_path} does not hash required outputs: {absent}")
    problems = []
    for name, digest in outputs.items():
        path = cohort_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
        elif sha256_file(path) != str(digest).strip().lower():
            problems.append(f"changed {name}")
    if problems:
        raise ValueError(f"cohort outputs differ from {manifest_path}: " + "; ".join(problems))
    if not isinstance(manifest.get("inputs"), dict):
        raise ValueError(f"{manifest_path} has no inputs table.")
    return manifest, sha256_file(manifest_path)


def require_cohort_input(manifest: dict, digest: str, what: str) -> None:
    recorded = {str(value).strip().lower() for value in manifest["inputs"].values()}
    if digest not in recorded:
        raise ValueError(f"cohort manifest was built from a different {what} (sha256 {digest}).")


def load_test_x(path: Path, columns: list[str]) -> pd.DataFrame:
    test_x = read_float_csv(path)
    if test_x.index.name != TEST_INDEX_NAME:
        raise ValueError(f"{path} index must be named {TEST_INDEX_NAME!r}.")
    ids = test_x.index.tolist()
    if not ids:
        raise ValueError(f"{path} has no rows.")
    if any(not HASH_RE.match(item) for item in ids):
        raise ValueError(f"{path} index must be 64 lowercase hex sequence hashes.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} repeats a sequence hash.")
    if ids != sorted(ids):
        raise ValueError(f"{path} rows must be sorted ascending by sequence hash.")
    if test_x.columns.tolist() != columns:
        raise ValueError(f"{path} columns differ from the training X columns.")
    if not all(dtype == np.float64 for dtype in test_x.dtypes):
        raise ValueError(f"{path} must be float64.")
    if not np.isfinite(test_x.to_numpy()).all():
        raise ValueError(f"{path} contains non-finite values.")
    return test_x


def _row_bytes(values: np.ndarray) -> list[bytes]:
    values = np.ascontiguousarray(values, dtype=np.float64)
    return [row.tobytes() for row in values]


def check_training_split(
    version: str,
    x: pd.DataFrame,
    test_x: pd.DataFrame,
    train_ids: list[str],
    excluded_ids: list[str],
    raw_hashes: dict[str, str],
) -> dict:
    """Independent leakage checks; raises on any violation, returns counts."""
    x_ids = x.index.tolist()
    x_set, train_set, excluded_set = set(x_ids), set(train_ids), set(excluded_ids)
    if not train_ids:
        raise ValueError(f"{version} has no training IDs.")
    if not train_set <= x_set:
        raise ValueError(f"{version} training IDs are not all in X: {sorted(train_set - x_set)[:5]}")
    if not excluded_set <= x_set:
        raise ValueError(f"{version} excluded IDs are not all in X.")
    if train_set & excluded_set:
        raise ValueError(f"{version} training and excluded IDs overlap.")
    if train_set | excluded_set != x_set:
        raise ValueError(f"{version} training and excluded IDs do not cover X.")
    if train_ids != [item for item in x_ids if item in train_set]:
        raise ValueError(f"{version} training IDs are not in manifest X row order.")

    test_hashes = set(test_x.index)
    leaked = [item for item in train_ids if raw_hashes[item] in test_hashes]
    if leaked:
        raise LeakageError(f"{version} training IDs share a test sequence: {leaked[:5]}")
    by_hash = [item for item in x_ids if raw_hashes[item] in test_hashes]
    if set(by_hash) != excluded_set:
        raise LeakageError(
            f"{version} excluded IDs differ from the IDs whose raw sequence hash is a test hash."
        )

    test_rows = _row_bytes(test_x.to_numpy())
    train_rows = set(_row_bytes(x.loc[train_ids].to_numpy()))
    overlap = [h for h, row in zip(test_x.index, test_rows) if row in train_rows]
    if overlap:
        raise LeakageError(f"{version} training X has exact test rows: {overlap[:5]}")

    excluded_rows = dict(zip(x_ids, _row_bytes(x.to_numpy())))
    for digest, row in zip(test_x.index, test_rows):
        candidates = [item for item in by_hash if raw_hashes[item] == digest]
        if not any(excluded_rows[item] == row for item in candidates):
            raise LeakageError(
                f"{version} test row {digest} equals no excluded {version} X row of that hash."
            )
    return {
        "n_x": len(x_ids),
        "n_train_available": len(train_ids),
        "n_excluded": len(excluded_ids),
        "n_test": len(test_rows),
    }


# --------------------------------------------------------------------------- runner

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cohort-dir", type=Path, default=Path(DEFAULTS["cohort_dir"]))
    parser.add_argument("--outdir", type=Path, default=Path(DEFAULTS["outdir"]))
    parser.add_argument("--transfer-manifest", default=DEFAULTS["transfer_manifest"])
    parser.add_argument("--label-map", default=DEFAULTS["label_map"])
    parser.add_argument("--v1-raw", default=DEFAULTS["v1_raw"])
    parser.add_argument("--v2-raw", default=DEFAULTS["v2_raw"])
    parser.add_argument("--frozen-sources", default=DEFAULTS["frozen_sources"])
    parser.add_argument("--train-versions", nargs="+", choices=VERSIONS, default=list(VERSIONS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--model-n-jobs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--probe-max-train-rows", type=int, default=None,
        help="fit on the first N training IDs only (manifest order); outdir must contain 'probe'",
    )
    return parser.parse_args(argv)


def write_run_record(outdir: Path, record: dict) -> Path:
    body = json.dumps(record, indent=2, sort_keys=True, default=str)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = outdir / "run_records" / f"{stamp}_{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(body + "\n")
    return path


def cell_dir_for(outdir: Path, version: str, model_key: str) -> Path:
    return (
        outdir / "paired"
        / f"train_{version}__{LABEL_SPACE}__{FEATURE_KEY}__{MODEL_PROFILE}"
        / model_key
    )


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models lists a model twice.")
    if len(set(args.train_versions)) != len(args.train_versions):
        raise ValueError("--train-versions lists a version twice.")
    outdir = resolve(args.outdir).resolve()
    probe = None
    if args.probe_max_train_rows is not None:
        if args.probe_max_train_rows < 1:
            raise ValueError("--probe-max-train-rows must be positive.")
        if not any("probe" in part for part in outdir.parts):
            raise ValueError(f"probe runs must write to an outdir containing 'probe': {outdir}")
        probe = {"max_train_rows": int(args.probe_max_train_rows)}

    # 1. Frozen sources.
    frozen_path = resolve(args.frozen_sources)
    code_hashes = prediction_io.verify_frozen_sources(frozen_path, FROZEN_ROOT)
    required = list(REQUIRED_FROZEN) + [MODEL_SOURCES[key] for key in args.models]
    unfrozen = [path for path in required if path not in code_hashes]
    if unfrozen:
        raise prediction_io.FrozenSourceError(
            f"{frozen_path} does not freeze required sources: {unfrozen}"
        )

    # 2. Cohort manifest and test rows.
    cohort_dir = resolve(args.cohort_dir)
    cohort_manifest, cohort_manifest_sha = verify_cohort(cohort_dir, args.train_versions)

    label_map_path = resolve(args.label_map)
    label_map_sha = sha256_file(label_map_path)
    labels = label_map_names(label_map_path)
    manifest_path = resolve(args.transfer_manifest)
    manifest_sha = sha256_file(manifest_path)
    manifest = pd.read_csv(manifest_path, dtype=str)
    require_cohort_input(cohort_manifest, manifest_sha, "transfer manifest")
    require_cohort_input(cohort_manifest, label_map_sha, "label map")

    test_x_path = cohort_dir / "test_X.csv"
    test_x_sha = sha256_file(test_x_path)
    test_x = None
    test_ids: list[str] = []

    # 3. Training data and independent leakage checks, for every version before any fit.
    raw_paths = {"v1": resolve(args.v1_raw), "v2": resolve(args.v2_raw)}
    training: dict[str, dict] = {}
    for version in args.train_versions:
        x, y, row = load_version(manifest, version, label_map_sha, labels)
        require_cohort_input(cohort_manifest, row["x_sha256"], f"{version} X")
        require_cohort_input(cohort_manifest, row["y_sha256"], f"{version} Y")
        raw_sha = sha256_file(raw_paths[version])
        require_cohort_input(cohort_manifest, raw_sha, f"{version} raw sequence file")
        if test_x is None:
            test_x = load_test_x(test_x_path, x.columns.tolist())
            test_ids = test_x.index.tolist()
        elif x.columns.tolist() != test_x.columns.tolist():
            raise ValueError("Feature columns differ between V1 and V2.")
        train_ids = read_id_lines(cohort_dir / f"train_ids_{version}.txt")
        excluded_ids = read_id_lines(cohort_dir / f"excluded_ids_{version}.txt")
        raw_hashes = load_raw_hashes(raw_paths[version], x.index.tolist(), version)
        checks = check_training_split(version, x, test_x, train_ids, excluded_ids, raw_hashes)
        fitted_ids = train_ids if probe is None else train_ids[: probe["max_train_rows"]]
        training[version] = {
            "x": x.loc[fitted_ids],
            "y": y.loc[fitted_ids],
            "ids": fitted_ids,
            "row": row,
            "raw_sha256": raw_sha,
            "checks": {**checks, "n_train_fitted": len(fitted_ids)},
        }

    # 4. Refuse to touch existing cells without --resume.
    cell_dirs = {
        (version, key): cell_dir_for(outdir, version, key)
        for version in args.train_versions for key in args.models
    }
    if not args.resume:
        existing = [
            f"{version}/{key}" for (version, key), cell_dir in cell_dirs.items()
            if any(path.exists() for path in prediction_io.cell_paths(cell_dir, STEM).values())
        ]
        if existing:
            raise FileExistsError(
                f"cells already exist for {existing} under {outdir / 'paired'}; "
                "pass --resume to reuse or redo them."
            )

    if PROFESSOR_DIR.as_posix() not in sys.path:
        sys.path.insert(0, PROFESSOR_DIR.as_posix())

    libraries = prediction_io.library_versions()
    record = {
        "kind": KIND,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "args": {key: (value.as_posix() if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "frozen_sources": display_path(frozen_path),
        "code_hashes": code_hashes,
        "cohort_dir": display_path(cohort_dir),
        "cohort_manifest_sha256": cohort_manifest_sha,
        "test_x_sha256": test_x_sha,
        "n_test": len(test_ids),
        "probe": probe,
        "raw_sha256": {v: training[v]["raw_sha256"] for v in training},
        "leakage_checks": {v: training[v]["checks"] for v in training},
        "library_versions": libraries,
        "cells": [],
        "outcome": "incomplete",
    }
    try:
        for version in args.train_versions:
            data = training[version]
            row = data["row"]
            for model_key in args.models:
                cell_dir = cell_dirs[(version, model_key)]
                model = build_paper_model(model_key, n_jobs=args.model_n_jobs)
                params = effective_params(model_key, model)
                contract = {
                    "kind": KIND,
                    "train_version": version,
                    "label_space": LABEL_SPACE,
                    "feature_key": FEATURE_KEY,
                    "model": model_key,
                    "model_profile": MODEL_PROFILE,
                    "effective_params": params,
                    "seed": int(params["random_state"]),
                    "test_population": TEST_POPULATION,
                    "test_x_sha256": test_x_sha,
                    "test_row_ids_sha256": prediction_io.sha256_lines(test_ids),
                    "train_ids_sha256": prediction_io.sha256_lines(data["ids"]),
                    "n_train": len(data["ids"]),
                    "cohort_manifest_sha256": cohort_manifest_sha,
                    "input_hashes": {
                        "train_x": row["x_sha256"],
                        "train_y": row["y_sha256"],
                        "transfer_manifest": manifest_sha,
                        "label_map": label_map_sha,
                        "test_x": test_x_sha,
                    },
                    "code_hashes": code_hashes,
                    "library_versions": libraries,
                    "probe": probe,
                }
                expected = {k: v for k, v in contract.items() if k not in NON_EXPECTED_FIELDS}
                cell = {
                    "train_version": version,
                    "model": model_key,
                    "path": display_path(cell_dir / STEM),
                    "status": "started",
                }
                record["cells"].append(cell)
                status = prediction_io.cell_status(cell_dir, STEM, expected)
                if status == "complete":
                    cell["status"] = "reused"
                    print(f"reused {model_key}: train {version}", flush=True)
                    continue
                if status == "invalid":
                    quarantined = prediction_io.quarantine_cell(cell_dir, STEM)
                    cell["quarantined_to"] = display_path(quarantined)
                started = time.perf_counter()
                model.fit(data["x"], data["y"])
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                scores = np.asarray(model.predict_proba(test_x), dtype=float)
                predict_seconds = time.perf_counter() - started
                if scores.shape != (len(test_ids), len(labels)):
                    raise ValueError(f"{model_key} returned predictions of shape {scores.shape}.")
                predictions = pd.DataFrame(scores, index=test_ids, columns=labels)
                contract["fitted_selection"] = fitted_selection(model_key, model)
                contract["execution"] = {
                    "n_jobs": args.model_n_jobs,
                    "hostname": socket.gethostname(),
                    "fit_seconds": fit_seconds,
                }
                prediction_io.write_prediction_cell(cell_dir, STEM, predictions, contract)
                cell.update(
                    status="fitted", fit_seconds=fit_seconds, predict_seconds=predict_seconds
                )
                print(f"fitted {model_key}: train {version} ({fit_seconds:.1f}s)", flush=True)
        record["outcome"] = "complete"
    except BaseException as error:
        record["outcome"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        record_path = write_run_record(outdir, record)
        print(f"run record {display_path(record_path)}", flush=True)
    record["record_path"] = record_path
    return record


if __name__ == "__main__":
    main()
