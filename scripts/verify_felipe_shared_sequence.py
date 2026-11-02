"""Independent read-only verifier for the shared-sequence paired comparison.

Spec: Reports/felipe_audit_2026-09-27/SHARED_SEQUENCE_SPEC_2026-09-28.md (SS1-SS14,
Interfaces 1-3). The verifier deliberately does NOT import the cohort builder, the paired
runner or the scorer; it re-derives the cohort, the cell contract fields and the summary
metrics with its own code (pandas / numpy / sklearn plus the frozen ``prediction_io`` and
``normalize_sequence`` helpers).

Checks (``--check``, one or more): frozen, cohort, cells, scores, protected (default: all).
Two frozen tables (SS15): ``--frozen-sources`` = the scoring table, hashed against the current
files; ``--fit-frozen-sources`` = the fit-time table, read as data only. They must differ
only in the scorer row, and every cell's contract ``code_hashes`` must equal the fit-time table. Prints one
PASS/FAIL line per check and exits 1 on any failure. Nothing is written except an optional
``--json-out`` file, which is never overwritten.

IDs are kept byte-exact (a real V1 ID ends with a space); ID lists are split on "\\n" only.
Every X / test_X read uses ``float_precision="round_trip"`` (SS12).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, hamming_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prediction_io import (  # noqa: E402
    FrozenSourceError,
    PredictionCellError,
    read_prediction_cell,
    sha256_file,
    sha256_lines,
    verify_frozen_sources,
)
from scripts.reconstruct_paper_sequence_features import normalize_sequence  # noqa: E402

import hashlib  # noqa: E402

DEFAULT_ROOT = Path("results/felipe_shared_sequence_2026-09-28")
DEFAULT_FROZEN = Path("config/felipe_shared_sequence_scoring_frozen_sources.csv")  # current (scoring) table
DEFAULT_FIT_FROZEN = Path("config/felipe_shared_sequence_frozen_sources.csv")  # fit-time table (SS15)
SCORER_PATH = "scripts/score_felipe_shared_sequence.py"
BASE_FROZEN = Path("config/felipe_frozen_sources.csv")
TRANSFER_MANIFEST = Path("config/felipe_basics_transfer_73_manifest.csv")
LABEL_MAP = Path("config/felipe_basics_shared_73_diseases.csv")
RAW_V1 = Path("lncRNA_CIBCB2025-main/create_dataset/raw_data/sequences.csv")
RAW_V2 = Path("Data/corrected_inputs/v2_transcript_sequences/website_full_matrix.csv")
PROTECTED_DIR = Path("results/felipe_final_2026-09-28")
NEW_SCRIPTS = (
    "scripts/build_shared_sequence_cohort.py",
    "scripts/audit_shared_disease_map.py",
    "scripts/run_felipe_shared_sequence_paired.py",
    "scripts/score_felipe_shared_sequence.py",
)

MODELS = ("rf", "ipcarf", "rflda")
BASELINE = "prevalence_baseline"
TRAIN_VERSIONS = ("v1", "v2")
COHORTS = ("primary", "sensitivity")
LABEL_VERSIONS = ("v1", "v2")
LABEL_SPACE = "shared73"
FEATURE_KEY = "paper_sequence_542"
CELL_STEM = "shared48"
SEED = 20260928
N_BOOTSTRAP = 2000
TOL = 1e-12
CI_TOL = 1e-9
LABEL_FILE = {
    ("v1", "primary"): "labels_v1.csv",
    ("v1", "sensitivity"): "labels_v1.csv",
    ("v2", "primary"): "labels_v2_primary.csv",
    ("v2", "sensitivity"): "labels_v2_or.csv",
}
CHECKS = ("frozen", "cohort", "cells", "scores", "protected")


class VerifyError(RuntimeError):
    """A verification failure."""


def fail_if(problems: list[str], title: str) -> None:
    if problems:
        shown = problems[:40]
        more = f"\n  ... and {len(problems) - 40} more" if len(problems) > 40 else ""
        raise VerifyError(title + ":\n  " + "\n  ".join(shown) + more)


# --------------------------------------------------------------------------- readers


def read_id_list(path: Path) -> list[str]:
    """One ID per line, byte-exact. Only a single trailing newline is tolerated."""
    text = Path(path).read_bytes().decode("utf-8")
    if "\r" in text:
        raise VerifyError(f"{path} contains a carriage return")
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
    if any(part == "" for part in parts):
        raise VerifyError(f"{path} contains an empty line")
    if len(set(parts)) != len(parts):
        raise VerifyError(f"{path} contains duplicate IDs")
    return parts


def read_x(path: Path) -> pd.DataFrame:
    first = pd.read_csv(path, nrows=0).columns[0]
    frame = pd.read_csv(
        path, index_col=0, dtype={first: str}, keep_default_na=False, float_precision="round_trip"
    )
    frame.index = pd.Index([str(i) for i in frame.index], name=first)
    if frame.index.duplicated().any():
        raise VerifyError(f"{path} has duplicate row IDs")
    return frame.astype(np.float64)


def read_int_frame(path: Path) -> pd.DataFrame:
    first = pd.read_csv(path, nrows=0).columns[0]
    frame = pd.read_csv(path, index_col=0, dtype={first: str}, keep_default_na=False)
    frame.index = pd.Index([str(i) for i in frame.index], name=first)
    if frame.index.duplicated().any():
        raise VerifyError(f"{path} has duplicate row IDs")
    values = frame.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise VerifyError(f"{path} labels must be 0/1")
    return frame.astype(np.int64)


def read_predictions_exact(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"ID": str}, keep_default_na=False, float_precision="round_trip")
    frame = frame.set_index("ID")
    frame.index = pd.Index([str(i) for i in frame.index])
    return frame.astype(np.float64)


def label_order(label_map: Path) -> list[str]:
    frame = pd.read_csv(label_map, dtype={"canonical_name": str}, keep_default_na=False)
    frame["order"] = frame["order"].astype(int)
    if sorted(frame["order"]) != list(range(len(frame))):
        raise VerifyError(f"{label_map} 'order' is not 0..n-1")
    return frame.sort_values("order")["canonical_name"].tolist()


def to_bool(series: pd.Series) -> pd.Series:
    mapping = {"true": True, "false": False}
    out = series.astype(str).str.lower().map(mapping)
    if out.isna().any():
        raise VerifyError(f"column {series.name!r} is not True/False")
    return out.astype(bool)


def resolve(path, base: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def transfer_rows(manifest: Path, project_root: Path) -> dict:
    table = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    out = {}
    for version in TRAIN_VERSIONS:
        rows = table[
            (table["version"] == version)
            & (table["label_space"] == LABEL_SPACE)
            & (table["feature_key"] == FEATURE_KEY)
        ]
        if len(rows) != 1:
            raise VerifyError(f"transfer manifest needs one {version}/{LABEL_SPACE}/{FEATURE_KEY} row")
        row = rows.iloc[0].to_dict()
        for kind in ("x", "y"):
            path = resolve(row[f"{kind}_path"], project_root)
            if not path.is_file():
                raise VerifyError(f"transfer {kind.upper()} {version} missing: {path}")
            measured = sha256_file(path)
            if measured != row[f"{kind}_sha256"].lower():
                raise VerifyError(f"transfer {kind.upper()} {version} hash differs from the manifest")
            row[f"{kind}_abs"] = path
        out[version] = row
    out["manifest_sha256"] = sha256_file(manifest)
    return out


def seq_hash(value) -> str:
    seq, _ = normalize_sequence(value)
    return hashlib.sha256(seq.encode("utf-8")).hexdigest() if seq else ""


def raw_hashes(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, usecols=["ID", "seqs"], dtype=str, keep_default_na=False)
    if frame["ID"].duplicated().any():
        raise VerifyError(f"{path} has duplicate IDs")
    out = {}
    for sample_id, seq in zip(frame["ID"].tolist(), frame["seqs"].tolist()):
        digest = seq_hash(seq)
        if digest:
            out[sample_id] = digest
    return out


def cell_dir(results_root: Path, train_version: str, model: str) -> Path:
    group = f"train_{train_version}__{LABEL_SPACE}__{FEATURE_KEY}__paper"
    return Path(results_root) / "paired" / group / model


def row_bytes(frame: pd.DataFrame) -> list[bytes]:
    values = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
    return [values[i].tobytes() for i in range(values.shape[0])]


# --------------------------------------------------------------------------- frozen


def read_table(path: Path) -> dict[str, str]:
    """{path: sha256} of a frozen table, read as data only (no file hashing)."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if {"path", "sha256"} - set(frame.columns) or frame.empty:
        raise VerifyError(f"{path} must list path,sha256 rows")
    if frame["path"].duplicated().any():
        raise VerifyError(f"{path} lists a path twice")
    return dict(zip(frame["path"], frame["sha256"].str.strip().str.lower()))


def check_frozen(table: Path, base_table: Path, project_root: Path, new_scripts=NEW_SCRIPTS,
                 fit_table: Path | None = None, scorer_path: str = SCORER_PATH) -> dict:
    """``table`` = scoring table (hashed against current files); ``fit_table`` = fit-time table (data only)."""
    try:
        measured = verify_frozen_sources(table, project_root)
    except FrozenSourceError as error:
        raise VerifyError(str(error)) from error
    frozen = pd.read_csv(table, dtype=str, keep_default_na=False)
    base = pd.read_csv(base_table, dtype=str, keep_default_na=False)
    lookup = dict(zip(frozen["path"], frozen["sha256"].str.lower()))
    problems = []
    for path, digest in zip(base["path"], base["sha256"].str.lower()):
        if path not in lookup:
            problems.append(f"base row missing from table: {path}")
        elif lookup[path] != digest:
            problems.append(f"hash differs from base table: {path}")
    for path in new_scripts:
        if path not in lookup:
            problems.append(f"new script missing from table: {path}")
    extra = sorted(set(lookup) - set(base["path"]) - set(new_scripts))
    if extra:
        problems.append(f"unexpected extra rows: {extra}")
    fit_info = None
    if fit_table is not None:
        fit = read_table(fit_table)
        if set(fit) != set(lookup):
            problems.append(f"fit-time vs scoring table path sets differ: only fit {sorted(set(fit) - set(lookup))}, "
                            f"only scoring {sorted(set(lookup) - set(fit))}")
        differing = sorted(p for p in set(fit) & set(lookup) if fit[p] != lookup[p])
        bad = [p for p in differing if p != scorer_path]
        if bad:
            problems.append(f"fit-time and scoring tables differ outside {scorer_path}: {bad}")
        fit_info = {"differing_rows": differing}
    fail_if(problems, "frozen source table")
    return {"n_rows": len(measured), "n_base_rows": len(base), "new_scripts": list(new_scripts),
            "fit_table": fit_info}


# --------------------------------------------------------------------------- cohort


def check_cohort(
    cohort_dir: Path,
    project_root: Path,
    raw_v1: Path,
    raw_v2: Path,
    transfer_manifest: Path,
    label_map: Path,
    expected: dict | None = None,
) -> dict:
    expected = expected or {"shared": 48, "one_to_one": 44, "multi": 4, "train_v1": 307, "train_v2": 4078,
                            "v2_shared_rows": 52}
    cohort_dir = Path(cohort_dir)
    problems: list[str] = []
    manifest = json.loads((cohort_dir / "cohort_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.get("outputs", {}).items():
        path = cohort_dir / name
        if not path.is_file() or sha256_file(path) != str(digest).lower():
            problems.append(f"manifest output hash mismatch: {name}")
    for name, digest in manifest.get("inputs", {}).items():
        path = resolve(name, project_root)
        if not path.is_file() or sha256_file(path) != str(digest).lower():
            problems.append(f"manifest input hash mismatch: {name}")
    fail_if(problems, "cohort manifest")

    labels = label_order(label_map)
    transfer = transfer_rows(transfer_manifest, project_root)
    raw = {"v1": raw_hashes(raw_v1), "v2": raw_hashes(raw_v2)}
    x = {v: read_x(transfer[v]["x_abs"]) for v in TRAIN_VERSIONS}
    y = {v: read_int_frame(transfer[v]["y_abs"]) for v in TRAIN_VERSIONS}
    for v in TRAIN_VERSIONS:
        if y[v].index.tolist() != x[v].index.tolist():
            problems.append(f"{v} Y index differs from X index")
        if y[v].columns.tolist() != labels:
            problems.append(f"{v} Y columns differ from label map order")
        missing = [i for i in x[v].index if i not in raw[v]]
        if missing:
            problems.append(f"{v}: {len(missing)} X IDs have no raw sequence, e.g. {missing[:3]}")
    if x["v1"].columns.tolist() != x["v2"].columns.tolist():
        problems.append("V1/V2 X columns differ")
    fail_if(problems, "cohort inputs")

    by_hash: dict[str, dict[str, list[str]]] = {}
    for v in TRAIN_VERSIONS:
        for sample_id, digest in raw[v].items():
            by_hash.setdefault(digest, {"v1": [], "v2": []})[v].append(sample_id)
    shared = sorted(h for h, ids in by_hash.items() if ids["v1"] and ids["v2"])
    one_to_one = [h for h in shared if len(by_hash[h]["v1"]) == 1 and len(by_hash[h]["v2"]) == 1]
    multi = {h: sorted(by_hash[h]["v2"]) for h in shared if len(by_hash[h]["v2"]) > 1}
    multi_v1 = [h for h in shared if len(by_hash[h]["v1"]) > 1]
    n_v2_shared_rows = sum(len(by_hash[h]["v2"]) for h in shared)
    if len(shared) != expected["shared"]:
        problems.append(f"{len(shared)} shared hashes, expected {expected['shared']}")
    if len(one_to_one) != expected["one_to_one"]:
        problems.append(f"{len(one_to_one)} one-to-one, expected {expected['one_to_one']}")
    if len(multi) != expected["multi"]:
        problems.append(f"{len(multi)} multi-V2-ID hashes, expected {expected['multi']}")
    if multi_v1:
        problems.append(f"{len(multi_v1)} shared hashes have >1 V1 ID")
    if n_v2_shared_rows != expected["v2_shared_rows"]:
        problems.append(f"{n_v2_shared_rows} V2 rows shared, expected {expected['v2_shared_rows']}")
    for h in shared:
        for v in TRAIN_VERSIONS:
            for sample_id in by_hash[h][v]:
                if sample_id not in x[v].index:
                    problems.append(f"shared {v} ID {sample_id!r} not in X")
    fail_if(problems, "cohort shared set")

    corr = pd.read_csv(cohort_dir / "correspondence.csv", dtype=str, keep_default_na=False)
    if corr["sequence_sha256"].tolist() != shared:
        problems.append("correspondence.csv hashes differ from recomputed sorted shared set")
    else:
        v1_pos = {i: k for k, i in enumerate(x["v1"].index)}
        v2_pos = {i: k for k, i in enumerate(x["v2"].index)}
        for row in corr.to_dict("records"):
            h = row["sequence_sha256"]
            v2_ids = row["v2_ids"].split(" | ")
            if row["v1_id"] != by_hash[h]["v1"][0]:
                problems.append(f"{h[:12]} v1_id {row['v1_id']!r} != {by_hash[h]['v1'][0]!r}")
            if int(row["v1_x_row_index"]) != v1_pos[by_hash[h]["v1"][0]]:
                problems.append(f"{h[:12]} v1_x_row_index differs")
            if sorted(v2_ids) != sorted(by_hash[h]["v2"]) or int(row["n_v2_ids"]) != len(v2_ids):
                problems.append(f"{h[:12]} v2_ids {v2_ids!r} != {sorted(by_hash[h]['v2'])!r}")
            else:
                idx = [int(k) for k in row["v2_x_row_indices"].split(" | ")]
                if idx != [v2_pos[i] for i in v2_ids]:
                    problems.append(f"{h[:12]} v2_x_row_indices differ")
            status = "one_to_one" if h in one_to_one else "ambiguous_multi_v2_id"
            if row["status"] != status:
                problems.append(f"{h[:12]} status {row['status']} != {status}")
            if row["in_primary"] != str(h in one_to_one) or row["in_sensitivity"] != "True":
                problems.append(f"{h[:12]} cohort flags differ")
        corr_multi = {r["sequence_sha256"]: sorted(r["v2_ids"].split(" | "))
                      for r in corr.to_dict("records") if int(r["n_v2_ids"]) > 1}
        if corr_multi != multi:
            problems.append(f"multi-ID groups differ: recomputed {multi} vs correspondence {corr_multi}")
    fail_if(problems, "correspondence.csv")

    # test_X
    test_x = read_x(cohort_dir / "test_X.csv")
    if test_x.index.tolist() != shared:
        problems.append("test_X index != sorted shared hashes")
    if test_x.columns.tolist() != x["v1"].columns.tolist():
        problems.append("test_X columns differ from X columns")
    fail_if(problems, "test_X.csv")
    tv = test_x.to_numpy()
    for k, h in enumerate(shared):
        target = tv[k].tobytes()
        refs = [("v1", by_hash[h]["v1"][0])] + [("v2", i) for i in by_hash[h]["v2"]]
        for v, sample_id in refs:
            if x[v].loc[sample_id].to_numpy(dtype=np.float64).tobytes() != target:
                problems.append(f"test_X row {h[:12]} not bit-identical to {v} X row {sample_id!r}")
    fail_if(problems, "test_X.csv values")

    # train IDs, exclusions, overlap
    shared_set = set(shared)
    info_train = {}
    test_rows = set(row_bytes(test_x))
    for v in TRAIN_VERSIONS:
        want_train = [i for i in x[v].index if raw[v][i] not in shared_set]
        want_excl = [i for i in x[v].index if raw[v][i] in shared_set]
        got_train = read_id_list(cohort_dir / f"train_ids_{v}.txt")
        got_excl = read_id_list(cohort_dir / f"excluded_ids_{v}.txt")
        if got_train != want_train:
            problems.append(f"train_ids_{v}.txt differs from recomputed (manifest order)")
        if len(got_train) != expected[f"train_{v}"]:
            problems.append(f"train_ids_{v}: {len(got_train)} rows, expected {expected[f'train_{v}']}")
        if sorted(got_excl) != sorted(want_excl):
            problems.append(f"excluded_ids_{v}.txt differs from recomputed set")
        overlap = sum(1 for b in row_bytes(x[v].loc[want_train]) if b in test_rows)
        if overlap:
            problems.append(f"{v}: {overlap} training rows are exact float64 copies of test rows")
        info_train[v] = {"n_train": len(got_train), "n_excluded": len(got_excl), "row_overlap": overlap}
    fail_if(problems, "train/excluded IDs")

    # labels
    lab_v1 = y["v1"].loc[[by_hash[h]["v1"][0] for h in shared]].set_axis(shared, axis=0)
    lab_v2p = y["v2"].loc[[by_hash[h]["v2"][0] for h in one_to_one]].set_axis(one_to_one, axis=0)
    lab_v2or = pd.DataFrame(
        [y["v2"].loc[by_hash[h]["v2"]].max(axis=0).to_numpy() for h in shared], index=shared, columns=labels
    )
    for name, want in (("labels_v1.csv", lab_v1), ("labels_v2_primary.csv", lab_v2p), ("labels_v2_or.csv", lab_v2or)):
        got = read_int_frame(cohort_dir / name)
        if got.index.tolist() != want.index.tolist():
            problems.append(f"{name} rows/order differ")
        elif got.columns.tolist() != labels:
            problems.append(f"{name} columns differ from label map order")
        elif not np.array_equal(got.to_numpy(), want.to_numpy()):
            problems.append(f"{name}: {int((got.to_numpy() != want.to_numpy()).sum())} cells differ")
    cohorts = pd.read_csv(cohort_dir / "cohorts.csv", dtype=str, keep_default_na=False)
    prim = to_bool(cohorts["in_primary"])
    sens = to_bool(cohorts["in_sensitivity"])
    if cohorts["sequence_sha256"].tolist() != shared:
        problems.append("cohorts.csv rows differ from shared set")
    elif [h for h, p in zip(shared, prim) if p] != one_to_one or not sens.all():
        problems.append("cohorts.csv flags differ from recomputed one-to-one / all")
    if int(prim.sum()) != expected["one_to_one"] or int(sens.sum()) != expected["shared"]:
        problems.append(f"cohorts.csv counts {int(prim.sum())}/{int(sens.sum())}")
    fail_if(problems, "labels / cohorts")
    return {
        "n_shared": len(shared),
        "n_one_to_one": len(one_to_one),
        "multi_v2_groups": {h[:12]: ids for h, ids in multi.items()},
        "train": info_train,
        "n_primary": int(prim.sum()),
        "n_sensitivity": int(sens.sum()),
    }


# --------------------------------------------------------------------------- cells


def load_cohort_basics(cohort_dir: Path) -> dict:
    cohort_dir = Path(cohort_dir)
    test_ids = pd.read_csv(cohort_dir / "test_X.csv", usecols=[0], dtype=str, keep_default_na=False)
    if test_ids.columns[0] != "sequence_sha256":
        raise VerifyError("test_X.csv index column must be sequence_sha256")
    ids = test_ids.iloc[:, 0].tolist()
    cohorts = pd.read_csv(cohort_dir / "cohorts.csv", dtype=str, keep_default_na=False)
    if cohorts["sequence_sha256"].tolist() != ids:
        # order may differ; membership must not
        if sorted(cohorts["sequence_sha256"]) != sorted(ids):
            raise VerifyError("cohorts.csv rows differ from test_X rows")
    flags = cohorts.set_index("sequence_sha256")
    prim = to_bool(flags["in_primary"])
    sens = to_bool(flags["in_sensitivity"])
    return {
        "dir": cohort_dir,
        "test_ids": ids,
        "test_x_sha256": sha256_file(cohort_dir / "test_X.csv"),
        "manifest_sha256": sha256_file(cohort_dir / "cohort_manifest.json"),
        "cohort_ids": {
            "primary": [i for i in ids if prim.loc[i]],
            "sensitivity": [i for i in ids if sens.loc[i]],
        },
        "train_ids": {v: read_id_list(cohort_dir / f"train_ids_{v}.txt") for v in TRAIN_VERSIONS},
    }


def fitted_train_ids(contract: dict, train_ids: list[str]) -> list[str]:
    probe = contract.get("probe")
    if probe is None:
        return train_ids
    limit = probe.get("max_train_rows") if isinstance(probe, dict) else None
    if not isinstance(limit, int) or limit < 1:
        raise VerifyError(f"malformed probe {probe!r}")
    return train_ids[:limit]


def scan_cells(results_root: Path) -> tuple[list[Path], list[Path]]:
    contracts, rejected = [], []
    for dirpath, dirnames, filenames in os.walk(results_root):
        parts = Path(dirpath).relative_to(results_root).parts
        if "_rejected" in dirnames:
            rejected.append(Path(dirpath) / "_rejected")
        if "_rejected" in parts:
            continue
        for name in filenames:
            if name.endswith(".contract.json"):
                contracts.append(Path(dirpath) / name)
    return sorted(contracts), sorted(rejected)


def check_cells(
    results_root: Path,
    cohort_dir: Path,
    frozen_table: Path,
    label_map: Path,
    transfer_manifest: Path | None,
    project_root: Path,
    allow_probe: bool,
    models=MODELS,
) -> dict:
    results_root = Path(results_root)
    cohort = load_cohort_basics(cohort_dir)
    labels = label_order(label_map)
    frozen_hashes = read_table(frozen_table)  # the FIT-TIME table: what the cells were fitted under
    transfer = transfer_rows(transfer_manifest, project_root) if transfer_manifest else None
    test_row_ids_sha = sha256_lines(cohort["test_ids"])
    problems: list[str] = []
    info: dict = {"cells": {}}
    expected_paths = set()
    for v in TRAIN_VERSIONS:
        for model in models:
            label = f"train_{v}/{model}"
            directory = cell_dir(results_root, v, model)
            expected_paths.add((directory / f"{CELL_STEM}.contract.json").resolve())
            try:
                frame, contract = read_prediction_cell(directory, CELL_STEM)
            except PredictionCellError as error:
                problems.append(f"{label}: {error}")
                continue
            want = {
                "kind": "shared_sequence_paired",
                "train_version": v,
                "label_space": LABEL_SPACE,
                "feature_key": FEATURE_KEY,
                "model": model,
                "model_profile": "paper",
                "test_population": "shared_sequence_all48",
                "test_x_sha256": cohort["test_x_sha256"],
                "test_row_ids_sha256": test_row_ids_sha,
                "cohort_manifest_sha256": cohort["manifest_sha256"],
            }
            for key, value in want.items():
                if contract.get(key) != value:
                    problems.append(f"{label}: {key} = {contract.get(key)!r}, expected {value!r}")
            if "probe" not in contract:
                problems.append(f"{label}: contract lacks 'probe'")
                continue
            if contract["probe"] is not None and not allow_probe:
                problems.append(f"{label}: probe cell {contract['probe']!r} without --allow-probe")
            try:
                fitted = fitted_train_ids(contract, cohort["train_ids"][v])
            except VerifyError as error:
                problems.append(f"{label}: {error}")
                continue
            if contract.get("train_ids_sha256") != sha256_lines(fitted):
                problems.append(f"{label}: train_ids_sha256 differs from train_ids_{v}.txt")
            if contract.get("n_train") != len(fitted):
                problems.append(f"{label}: n_train {contract.get('n_train')!r} != {len(fitted)}")
            if frame.index.tolist() != cohort["test_ids"]:
                problems.append(f"{label}: prediction index != test_X index order")
            if frame.columns.tolist() != labels:
                problems.append(f"{label}: prediction columns != label map order")
            if contract.get("code_hashes") != frozen_hashes:
                problems.append(f"{label}: code_hashes differ from the fit-time frozen table")
            hashes = contract.get("input_hashes") or {}
            if hashes.get("test_x") != cohort["test_x_sha256"]:
                problems.append(f"{label}: input_hashes.test_x differs")
            if transfer:
                row = transfer[v]
                want_in = {
                    "train_x": row["x_sha256"].lower(),
                    "train_y": row["y_sha256"].lower(),
                    "transfer_manifest": transfer["manifest_sha256"],
                    "label_map": row["label_map_sha256"].lower(),
                }
                bad = [k for k, val in want_in.items() if hashes.get(k) != val]
                if bad:
                    problems.append(f"{label}: input_hashes differ for {bad}")
            values = read_predictions_exact(directory / f"{CELL_STEM}.predictions.csv").to_numpy()
            if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                problems.append(f"{label}: scores not finite in [0, 1]")
            info["cells"][label] = {
                "prediction_sha256": contract.get("prediction_file_sha256"),
                "n_train": contract.get("n_train"),
                "probe": contract.get("probe"),
                "fitted_selection": contract.get("fitted_selection"),
                "fit_seconds": (contract.get("execution") or {}).get("fit_seconds"),
                "score_range": [float(values.min()), float(values.max())],
            }
    test_x_values = {c.get("test_x_sha256") for c in
                     (json.loads((cell_dir(results_root, v, m) / f"{CELL_STEM}.contract.json").read_text())
                      for v in TRAIN_VERSIONS for m in models
                      if (cell_dir(results_root, v, m) / f"{CELL_STEM}.contract.json").is_file())}
    if len(test_x_values) > 1:
        problems.append(f"cells disagree on test_x_sha256: {sorted(map(str, test_x_values))}")

    contracts, rejected = scan_cells(results_root)
    extras = []
    for path in contracts:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            problems.append(f"unreadable contract {path}")
            continue
        if path.resolve() not in expected_paths:
            extras.append(str(path.relative_to(results_root)))
            if (results_root / "paired") in path.parents:
                problems.append(f"unexpected extra cell under paired/: {path.relative_to(results_root)}")
        if isinstance(data, dict) and "test_x_sha256" in data and data["test_x_sha256"] != cohort["test_x_sha256"]:
            problems.append(f"mixed cell (different test_x_sha256): {path.relative_to(results_root)}")
    info["extra_contracts"] = extras
    info["rejected_dirs"] = [str(p.relative_to(results_root)) for p in rejected]
    info["test_x_sha256"] = cohort["test_x_sha256"]
    fail_if(problems, "prediction cells")
    return info


# --------------------------------------------------------------------------- scores


def metrics(y: np.ndarray, s: np.ndarray) -> dict:
    fy, fs = y.ravel(), s.ravel()
    pred = (s >= 0.5).astype(int)
    positives = y.sum(axis=0)
    both = (positives > 0) & (positives < y.shape[0])
    aurocs = [roc_auc_score(y[:, j], s[:, j]) for j in np.flatnonzero(both)]
    return {
        "micro_roc": float(roc_auc_score(fy, fs)) if np.unique(fy).size == 2 else float("nan"),
        "micro_auprc": float(average_precision_score(fy, fs)) if fy.sum() > 0 else float("nan"),
        "micro_f1": float(f1_score(fy, pred.ravel(), zero_division=0)),
        "hamming": float(hamming_loss(y, pred)),
        "n_sequences": int(y.shape[0]),
        "n_positive_associations": int(y.sum()),
        "n_diseases_both_classes": int(both.sum()),
        "per_disease_auroc_median": float(np.median(aurocs)) if aurocs else float("nan"),
    }


def _same(a: float, b: float, tol: float) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return abs(float(a) - float(b)) <= tol


SUMMARY_FLOAT_COLUMNS = {
    "micro_roc": "micro_roc",
    "sklearn_micro_roc": "micro_roc",
    "micro_auprc": "micro_auprc",
    "sklearn_micro_auprc": "micro_auprc",
    "micro_f1": "micro_f1",
    "fscore": "micro_f1",
    "sklearn_micro_f1": "micro_f1",
    "hamming": "hamming",
    "per_disease_auroc_median": "per_disease_auroc_median",
}
SUMMARY_COUNT_COLUMNS = {
    "n_sequences": "n_sequences",
    "n_positive_associations": "n_positive_associations",
    "n_diseases_both_classes": "n_diseases_both_classes",
    "n_per_disease_defined": "n_diseases_both_classes",
}


def bootstrap_ci(y: np.ndarray, s: np.ndarray, idx: np.ndarray, metric: str) -> tuple[float, float, int]:
    values = []
    for row in idx:
        yy, ss = y[row].ravel(), s[row].ravel()
        if metric == "micro_roc":
            if np.unique(yy).size < 2:
                continue
            values.append(roc_auc_score(yy, ss))
        elif metric == "micro_auprc":
            if yy.sum() == 0:
                continue
            values.append(average_precision_score(yy, ss))
        else:
            raise VerifyError(f"unsupported bootstrap metric {metric}")
    if not values:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(np.asarray(values), [2.5, 97.5])
    return float(low), float(high), len(values)


def check_scores(
    results_root: Path,
    cohort_dir: Path,
    scores_dir: Path,
    transfer_manifest: Path,
    project_root: Path,
    models=MODELS,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    boot_combos=(("rf", "v2", "primary", "v1", "micro_roc"), ("rf", "v1", "sensitivity", "v2", "micro_roc")),
) -> dict:
    results_root, scores_dir = Path(results_root), Path(scores_dir)
    cohort = load_cohort_basics(cohort_dir)
    transfer = transfer_rows(transfer_manifest, project_root)
    problems: list[str] = []
    label_frames = {name: read_int_frame(Path(cohort_dir) / name)
                    for name in ("labels_v1.csv", "labels_v2_primary.csv", "labels_v2_or.csv")}
    scores: dict = {}
    default_scores: dict = {}  # read_prediction_cell's default pandas float parser (diagnostic only)
    fitted: dict = {v: set() for v in TRAIN_VERSIONS}
    cell_hashes: dict = {}
    parser_diffs = {}
    for v in TRAIN_VERSIONS:
        for model in models:
            directory = cell_dir(results_root, v, model)
            try:
                default_frame, contract = read_prediction_cell(directory, CELL_STEM)
            except PredictionCellError as error:
                raise VerifyError(f"train_{v}/{model}: {error}") from error
            frame = read_predictions_exact(directory / f"{CELL_STEM}.predictions.csv")
            n_diff = int((frame.to_numpy() != default_frame.to_numpy()).sum())
            if n_diff:
                parser_diffs[f"train_{v}/{model}"] = n_diff
            scores[(model, v)] = frame
            default_scores[(model, v)] = default_frame.astype(np.float64)
            fitted[v].add(tuple(fitted_train_ids(contract, cohort["train_ids"][v])))
            cell_hashes[f"train_{v}/{model}"] = {
                "prediction_sha256": sha256_file(directory / f"{CELL_STEM}.predictions.csv"),
                "contract_sha256": sha256_file(directory / f"{CELL_STEM}.contract.json"),
            }
    labels = label_frames["labels_v1.csv"].columns.tolist()
    for v in TRAIN_VERSIONS:
        if len(fitted[v]) != 1:
            raise VerifyError(f"train_{v}: cells were fitted on different training IDs")
        ids = list(next(iter(fitted[v])))
        y_train = read_int_frame(transfer[v]["y_abs"])
        missing = [i for i in ids if i not in y_train.index]
        if missing:
            raise VerifyError(f"{len(missing)} train IDs absent from transfer Y {v}, e.g. {missing[:3]!r}")
        prevalence = y_train.loc[ids, labels].mean(axis=0).to_numpy(dtype=float)
        scores[(BASELINE, v)] = pd.DataFrame(
            np.tile(prevalence, (len(cohort["test_ids"]), 1)), index=cohort["test_ids"], columns=labels
        )

    summary = pd.read_csv(scores_dir / "summary.csv", keep_default_na=False, na_values=[""],
                          float_precision="round_trip")
    key_cols = ["model", "train_version", "cohort", "label_version"]
    if summary.duplicated(key_cols).any():
        problems.append("summary.csv has duplicate keys")
    summary = summary.set_index(key_cols)
    all_models = tuple(models) + (BASELINE,)
    want_keys = {(m, v, c, lv) for m in all_models for v in TRAIN_VERSIONS for c in COHORTS for lv in LABEL_VERSIONS}
    if set(summary.index) != want_keys:
        problems.append(f"summary.csv keys differ: missing {sorted(want_keys - set(summary.index))[:4]}, "
                        f"extra {sorted(set(summary.index) - want_keys)[:4]}")
    truths = {}
    recomputed = {}
    for c in COHORTS:
        ids = cohort["cohort_ids"][c]
        for lv in LABEL_VERSIONS:
            truth = label_frames[LABEL_FILE[(lv, c)]].loc[ids]
            truths[(c, lv)] = truth.to_numpy()
            for m in all_models:
                for v in TRAIN_VERSIONS:
                    s = scores[(m, v)].loc[ids, labels].to_numpy(dtype=float)
                    got = metrics(truth.to_numpy(), s)
                    recomputed[(m, v, c, lv)] = got
                    if (m, v, c, lv) not in summary.index:
                        continue
                    row = summary.loc[(m, v, c, lv)]
                    alt = None
                    for col, key in SUMMARY_FLOAT_COLUMNS.items():
                        if col in row.index and not _same(float(row[col]), got[key], TOL):
                            if alt is None:
                                alt_frame = default_scores.get((m, v), scores[(m, v)])
                                alt = metrics(truth.to_numpy(), alt_frame.loc[ids, labels].to_numpy(dtype=float))
                            note = ("matches the default-parser (read_prediction_cell) value"
                                    if _same(float(row[col]), alt[key], TOL) else "no parser explanation")
                            problems.append(f"{m}/train {v}/{c}/label {lv}: {col} file {float(row[col])!r} != "
                                            f"round_trip recomputed {got[key]!r} (diff {float(row[col]) - got[key]:.3g}; "
                                            f"{note})")
                    for col, key in SUMMARY_COUNT_COLUMNS.items():
                        if col in row.index and int(row[col]) != got[key]:
                            problems.append(f"{m}/train {v}/{c}/label {lv}: {col} {row[col]!r} != {got[key]!r}")
                    if m != BASELINE and row.get("prediction_sha256") != cell_hashes[f"train_{v}/{m}"]["prediction_sha256"]:
                        problems.append(f"{m}/train {v}/{c}/label {lv}: prediction_sha256 differs from cell")

    # bootstrap
    rng = np.random.default_rng(seed)
    idx = {c: rng.integers(0, len(cohort["cohort_ids"][c]), size=(n_bootstrap, len(cohort["cohort_ids"][c])),
                           dtype=np.int64) for c in COHORTS}
    index_sha = {c: hashlib.sha256(np.ascontiguousarray(idx[c], dtype="<i8").tobytes()).hexdigest() for c in COHORTS}
    ci = pd.read_csv(scores_dir / "bootstrap_ci.csv", keep_default_na=False, na_values=[""],
                     float_precision="round_trip")
    boot_info = []
    for m, v, c, lv, metric in boot_combos:
        rows = ci[(ci["model"] == m) & (ci["train_version"] == v) & (ci["cohort"] == c)
                  & (ci["label_version"] == lv) & (ci["metric"] == metric)]
        if len(rows) != 1:
            problems.append(f"bootstrap_ci.csv: {len(rows)} rows for {m}/{v}/{c}/{lv}/{metric}")
            continue
        row = rows.iloc[0]
        ids = cohort["cohort_ids"][c]
        s = scores[(m, v)].loc[ids, labels].to_numpy(dtype=float)
        low, high, n_valid = bootstrap_ci(truths[(c, lv)], s, idx[c], metric)
        if int(row["n_bootstrap"]) != n_bootstrap or int(row["seed"]) != seed:
            problems.append(f"bootstrap_ci.csv n_bootstrap/seed {row['n_bootstrap']}/{row['seed']}")
        if row["index_sha256"] != index_sha[c]:
            problems.append(f"bootstrap_ci.csv index_sha256 for {c} differs from recomputed")
        if not (_same(float(row["ci_low"]), low, CI_TOL) and _same(float(row["ci_high"]), high, CI_TOL)
                and int(row["n_valid"]) == n_valid):
            problems.append(f"bootstrap {m}/{v}/{c}/{lv}/{metric}: file ({row['ci_low']}, {row['ci_high']}, "
                            f"{row['n_valid']}) != recomputed ({low}, {high}, {n_valid})")
        if not _same(float(row["point"]), recomputed[(m, v, c, lv)][metric], TOL):
            problems.append(f"bootstrap {m}/{v}/{c}/{lv}/{metric}: point differs")
        boot_info.append({"combo": f"{m}/train {v}/{c}/label {lv}/{metric}", "ci_low": low, "ci_high": high,
                          "n_valid": n_valid})
    # all-row consistency: every bootstrap_ci row uses the recomputed index hash
    if "index_sha256" in ci.columns:
        for c in COHORTS:
            bad = ci[(ci["cohort"] == c) & (ci["index_sha256"] != index_sha[c])]
            if len(bad):
                problems.append(f"bootstrap_ci.csv: {len(bad)} {c} rows with a different index_sha256")

    # score_run.json
    run = json.loads((scores_dir / "score_run.json").read_text(encoding="utf-8"))
    for name, digest in (run.get("outputs") or {}).items():
        path = scores_dir / name
        if not path.is_file() or sha256_file(path) != digest:
            problems.append(f"score_run.json output hash mismatch: {name}")
    run_cells = (run.get("inputs") or {}).get("cells") or {}
    if set(run_cells) != set(cell_hashes):
        problems.append(f"score_run.json cells {sorted(run_cells)} != {sorted(cell_hashes)}")
    for label, measured in cell_hashes.items():
        rec = run_cells.get(label) or {}
        for key in ("prediction_sha256", "contract_sha256"):
            if rec.get(key) != measured[key]:
                problems.append(f"score_run.json {label} {key} differs from the cell on disk")
    if (run.get("inputs") or {}).get("cohort_manifest_sha256") != cohort["manifest_sha256"]:
        problems.append("score_run.json cohort_manifest_sha256 differs")
    fail_if(problems, "scores")
    headline = {
        f"{m}/train_{v}/{c}/label_{lv}": round(recomputed[(m, v, c, lv)]["micro_roc"], 4)
        for (m, v, c, lv) in sorted(recomputed) if c == "primary"
    }
    return {
        "n_summary_rows": int(len(summary)),
        "bootstrap_checked": boot_info,
        "index_sha256": index_sha,
        "prediction_parser_differences": parser_diffs,
        "primary_micro_roc": headline,
    }


# --------------------------------------------------------------------------- protected


def parse_snapshot(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        if len(line) < 66 or line[64:66] not in ("  ", " *"):
            raise VerifyError(f"malformed snapshot line: {line!r}")
        entries.append((line[:64].lower(), line[66:]))
    if not entries:
        raise VerifyError(f"{path} is empty")
    return entries


def check_protected(snapshot: Path, project_root: Path, protected_dir: Path = PROTECTED_DIR) -> dict:
    entries = parse_snapshot(snapshot)
    problems = []
    for digest, rel in entries:
        path = resolve(rel, project_root)
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"changed {rel}")
    listed = {rel for _, rel in entries}
    base = resolve(protected_dir, project_root)
    unlisted = []
    if base.is_dir():
        for path in sorted(base.rglob("*")):
            if path.is_file():
                rel = path.relative_to(project_root).as_posix()
                if rel not in listed:
                    unlisted.append(rel)
    fail_if(problems, "protected files")
    return {"n_entries": len(entries), "unlisted_under_protected_dir": unlisted}


# --------------------------------------------------------------------------- CLI


def summarize(name: str, info: dict) -> str:
    if name == "frozen":
        text = f"{info['n_rows']} rows ({info['n_base_rows']} base + {len(info['new_scripts'])} new) match current files"
        if info.get("fit_table") is not None:
            text += f"; fit-time table differs only in {info['fit_table']['differing_rows'] or 'no rows'}"
        return text
    if name == "cohort":
        return (f"shared={info['n_shared']} one_to_one={info['n_one_to_one']} "
                f"multi_v2_groups={len(info['multi_v2_groups'])} train v1/v2="
                f"{info['train']['v1']['n_train']}/{info['train']['v2']['n_train']} "
                f"row_overlap={info['train']['v1']['row_overlap']}/{info['train']['v2']['row_overlap']} "
                f"cohorts {info['n_primary']}/{info['n_sensitivity']}")
    if name == "cells":
        parts = [f"{len(info['cells'])} cells test_x_sha256={info['test_x_sha256'][:12]}"]
        for label, cell in info["cells"].items():
            parts.append(f"{label}: n_train={cell['n_train']} probe={cell['probe']} "
                         f"fit_seconds={cell['fit_seconds']} fitted_selection={json.dumps(cell['fitted_selection'])}")
        parts.append(f"rejected dirs: {info['rejected_dirs'] or 'none'}")
        if info["extra_contracts"]:
            parts.append(f"extra contracts (same test_x): {info['extra_contracts']}")
        return "\n    ".join(parts)
    if name == "scores":
        parts = [f"{info['n_summary_rows']} summary rows recomputed within {TOL:g}"]
        for b in info["bootstrap_checked"]:
            parts.append(f"bootstrap {b['combo']}: [{b['ci_low']:.6f}, {b['ci_high']:.6f}] n_valid={b['n_valid']}")
        if info["prediction_parser_differences"]:
            parts.append(f"default-vs-round_trip parser differences in predictions: {info['prediction_parser_differences']}")
        return "\n    ".join(parts)
    if name == "protected":
        extra = info["unlisted_under_protected_dir"]
        return f"{info['n_entries']} snapshot entries unchanged; unlisted files under protected dir: {len(extra)}" + (
            "\n    " + "\n    ".join(extra) if extra else "")
    return ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cohort-dir", type=Path)
    parser.add_argument("--scores-dir", type=Path)
    parser.add_argument("--frozen-sources", type=Path, default=DEFAULT_FROZEN,
                        help="scoring (current) frozen table; verified against the files on disk")
    parser.add_argument("--fit-frozen-sources", type=Path, default=DEFAULT_FIT_FROZEN,
                        help="fit-time frozen table; read as data; cell code_hashes must equal it")
    parser.add_argument("--protected-snapshot", type=Path)
    parser.add_argument("--allow-probe", action="store_true")
    parser.add_argument("--check", nargs="+", choices=("all",) + CHECKS, default=["all"])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    root = args.results_root
    cohort_dir = args.cohort_dir or root / "cohort"
    scores_dir = args.scores_dir or root / "scores"
    snapshot = args.protected_snapshot or root / "protected_before.sha256"
    if args.json_out and args.json_out.exists():
        print(f"refusing to overwrite {args.json_out}", file=sys.stderr)
        return 1
    project_root = PROJECT_ROOT
    runners = {
        "frozen": lambda: check_frozen(args.frozen_sources, project_root / BASE_FROZEN, project_root,
                                       fit_table=args.fit_frozen_sources),
        "cohort": lambda: check_cohort(cohort_dir, project_root, project_root / RAW_V1, project_root / RAW_V2,
                                       project_root / TRANSFER_MANIFEST, project_root / LABEL_MAP),
        "cells": lambda: check_cells(root, cohort_dir, args.fit_frozen_sources, project_root / LABEL_MAP,
                                     project_root / TRANSFER_MANIFEST, project_root, args.allow_probe),
        "scores": lambda: check_scores(root, cohort_dir, scores_dir, project_root / TRANSFER_MANIFEST, project_root),
        "protected": lambda: check_protected(snapshot, project_root),
    }
    selected = CHECKS if "all" in args.check else tuple(c for c in CHECKS if c in args.check)
    results = {}
    failed = False
    for name in selected:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info = runners[name]()
        except (VerifyError, FileNotFoundError, KeyError, ValueError) as error:
            failed = True
            results[name] = {"status": "FAIL", "error": f"{type(error).__name__}: {error}"}
            print(f"FAIL {name}: {error}")
            continue
        results[name] = {"status": "PASS", **info}
        print(f"PASS {name}: {summarize(name, info)}")
    if args.json_out:
        payload = {"results_root": str(root), "cohort_dir": str(cohort_dir), "scores_dir": str(scores_dir),
                   "allow_probe": args.allow_probe, "checks": results, "ok": not failed}
        with open(args.json_out, "x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
