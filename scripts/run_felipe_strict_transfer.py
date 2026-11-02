"""Run V1->V2 or V2->V1 on the identical 542-feature/73-label space.

One resume unit per (source, target, model) cell (decisions D3/D4):

* one ``fit`` on the full source population;
* one ``predict_proba(x_target)`` call, without labels, on the full target after
  every exact source ID has been removed from it;
* raw scores written atomically with ``prediction_io.write_prediction_cell`` to
  ``<outdir>/transfer/<source>_to_<target>__shared73__<feature_key>__paper/<model>/full_target``.

This runner computes no metric and applies no threshold. The deterministic
target folds are recorded only as ``evaluation_partition`` in the contract so a
separate rescoring step can report paper-style mean(std); they do not affect
fitting. Nothing fitted is persisted; an interrupted cell is redone whole.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
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
except ImportError:  # run as ``python scripts/run_felipe_strict_transfer.py``
    import prediction_io
    from felipe_paper_models import (
        MODEL_SOURCES,
        build_paper_model,
        effective_params,
        fitted_selection,
    )


ROOT = Path(__file__).resolve().parents[1]
PROFESSOR_DIR = ROOT / "lncRNA_CIBCB2025-main"
MODELS = ("rf", "ipcarf", "rflda")
LABEL_SPACE = "shared73"
MODEL_PROFILE = "paper"
STEM = "full_target"
DEFAULT_FEATURE_KEY = "paper_sequence_542"
DEFAULT_LABEL_MAP = "config/felipe_basics_shared_73_diseases.csv"
FEATURE_WIDTHS = {"paper_sequence_542": 542}
MANIFEST_CONTRACT_COLUMNS = ("label_space", "feature_key", "label_map_path", "label_map_sha256")
# Sources whose hashes must appear in the frozen table before this runner fits anything.
REQUIRED_FROZEN = (
    "scripts/run_felipe_strict_transfer.py",
    "scripts/prediction_io.py",
    "scripts/felipe_paper_models.py",
)
# Contract fields that describe how a cell ran, not what it is; never resume keys.
NON_EXPECTED_FIELDS = ("fitted_selection", "execution")


sha256_file = prediction_io.sha256_file


def resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo, absolute POSIX otherwise."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_version(manifest: pd.DataFrame, version: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = manifest[manifest["version"] == version]
    if len(rows) != 1:
        raise ValueError(f"Transfer manifest must have one row for {version}.")
    row = rows.iloc[0].to_dict()
    x_path, y_path = resolve(row["x_path"]), resolve(row["y_path"])
    if sha256_file(x_path) != row["x_sha256"] or sha256_file(y_path) != row["y_sha256"]:
        raise ValueError(f"{version} input hash does not match the transfer manifest.")
    x = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    x.index = x.index.astype(str)
    y.index = y.index.astype(str)
    if x.index.has_duplicates:
        raise ValueError(f"{version} contains duplicate RNA IDs.")
    if x.index.tolist() != y.index.tolist():
        raise ValueError(f"{version} X/Y IDs differ or are out of order.")
    if x.shape != (int(row["n_samples"]), int(row["n_features"])):
        raise ValueError(f"{version} X shape differs from its manifest.")
    if y.shape != (int(row["n_samples"]), int(row["n_labels"])):
        raise ValueError(f"{version} Y shape differs from its manifest.")
    if not y.isin([0, 1]).all().all():
        raise ValueError(f"{version} Y must be binary 0/1.")
    return x, y.astype(int), row


def resolve_label_contract(manifest: pd.DataFrame, rows: list[dict]) -> dict[str, object]:
    """Return feature_key/label_space/label map, from the manifest or the documented fallback."""
    present = [column for column in MANIFEST_CONTRACT_COLUMNS if column in manifest.columns]
    if present and len(present) != len(MANIFEST_CONTRACT_COLUMNS):
        missing = sorted(set(MANIFEST_CONTRACT_COLUMNS) - set(present))
        raise ValueError(f"Transfer manifest has only some contract columns; missing {missing}.")
    if not present:
        label_map_path = resolve(DEFAULT_LABEL_MAP)
        return {
            "label_space": LABEL_SPACE,
            "feature_key": DEFAULT_FEATURE_KEY,
            "label_map_path": label_map_path,
            "label_map_sha256": sha256_file(label_map_path),
            "label_contract_source": "runtime_default",
        }
    values: dict[str, object] = {}
    for column in MANIFEST_CONTRACT_COLUMNS:
        found = {str(row[column]) for row in rows}
        if len(found) != 1:
            raise ValueError(f"Source and target manifest rows differ in {column}: {sorted(found)}.")
        values[column] = found.pop()
    if values["label_space"] != LABEL_SPACE:
        raise ValueError(f"Transfer label_space must be {LABEL_SPACE!r}, got {values['label_space']!r}.")
    label_map_path = resolve(values["label_map_path"])
    measured = sha256_file(label_map_path)
    if measured != str(values["label_map_sha256"]).strip().lower():
        raise ValueError(f"Label map hash does not match the manifest: {label_map_path}")
    return {
        "label_space": LABEL_SPACE,
        "feature_key": values["feature_key"],
        "label_map_path": label_map_path,
        "label_map_sha256": measured,
        "label_contract_source": "manifest",
    }


def label_map_names(path: Path) -> list[str]:
    table = pd.read_csv(path, dtype={"canonical_name": str})
    if not {"order", "canonical_name"} <= set(table.columns):
        raise ValueError(f"{path} must have order and canonical_name columns.")
    order = pd.to_numeric(table["order"], errors="raise")
    if order.duplicated().any():
        raise ValueError(f"{path} repeats an order value.")
    return table.assign(order=order).sort_values("order")["canonical_name"].tolist()


def remove_exact_source_ids(
    x_source: pd.DataFrame, x_target: pd.DataFrame, y_target: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    overlap = sorted(set(x_source.index) & set(x_target.index))
    if overlap:
        x_target = x_target.drop(index=overlap)
        y_target = y_target.drop(index=overlap)
    if x_target.index.tolist() != y_target.index.tolist():
        raise ValueError("Target X/Y became misaligned after overlap removal.")
    if set(x_target.index) & set(x_source.index):
        raise ValueError("Target still shares RNA IDs with the source after removal.")
    return x_target, y_target, overlap


def evaluation_partition(y_target: pd.DataFrame, n_splits: int, random_state: int) -> dict:
    """Deterministic target folds for rescoring only; uses target labels, never fitting."""
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    ids = y_target.index.tolist()
    folds = [
        [ids[index] for index in sorted(test_indexes)]
        for _, test_indexes in splitter.split(np.zeros((len(ids), 1)), y_target.to_numpy())
    ]
    flat = [item for fold in folds for item in fold]
    if sorted(flat) != sorted(ids) or len(flat) != len(ids):
        raise ValueError("Evaluation partition does not cover each target ID exactly once.")
    return {
        "method": "iterstrat.MultilabelStratifiedKFold",
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "uses_target_labels": True,
        "folds": folds,
        "sha256": partition_sha256(folds),
    }


def partition_sha256(folds: list[list[str]]) -> str:
    return prediction_io.sha256_lines(
        [f"{number}:{','.join(fold)}" for number, fold in enumerate(folds, start=1)]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--manifest", default="config/felipe_basics_transfer_73_manifest.csv"
    )
    parser.add_argument("--frozen-sources", default="config/felipe_frozen_sources.csv")
    parser.add_argument("--source", choices=["v1", "v2"], required=True)
    parser.add_argument("--target", choices=["v1", "v2"], required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--model-n-jobs", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("results/felipe_basics_2026-09-27"))
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


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.source == args.target:
        raise ValueError("Source and target versions must differ.")
    if len(set(args.models)) != len(args.models):
        raise ValueError("--models lists a model twice.")

    frozen_path = resolve(args.frozen_sources)
    code_hashes = prediction_io.verify_frozen_sources(frozen_path, ROOT)
    required = list(REQUIRED_FROZEN) + [MODEL_SOURCES[key] for key in args.models]
    unfrozen = [path for path in required if path not in code_hashes]
    if unfrozen:
        raise prediction_io.FrozenSourceError(
            f"{frozen_path} does not freeze required sources: {unfrozen}"
        )

    manifest_path = resolve(args.manifest)
    manifest = pd.read_csv(manifest_path, dtype=str)
    x_source, y_source, source_row = load_version(manifest, args.source)
    x_target, y_target, target_row = load_version(manifest, args.target)
    if x_source.columns.tolist() != x_target.columns.tolist():
        raise ValueError("Transfer feature columns differ between V1 and V2.")
    if y_source.columns.tolist() != y_target.columns.tolist():
        raise ValueError("Transfer label columns differ between V1 and V2.")
    label_contract = resolve_label_contract(manifest, [source_row, target_row])
    feature_key = str(label_contract["feature_key"])
    width = FEATURE_WIDTHS.get(feature_key)
    if width is not None and x_source.shape[1] != width:
        raise ValueError(f"{feature_key} requires {width} features, found {x_source.shape[1]}.")
    labels = label_map_names(label_contract["label_map_path"])
    if y_source.columns.tolist() != labels:
        raise ValueError("Transfer label columns differ from the label map canonical order.")

    x_target, y_target, removed = remove_exact_source_ids(x_source, x_target, y_target)
    partition = evaluation_partition(y_target, args.n_splits, args.random_state)

    target_y_path = resolve(target_row["y_path"])
    base_contract = {
        "kind": "transfer",
        "source": args.source,
        "target": args.target,
        "label_space": LABEL_SPACE,
        "feature_key": feature_key,
        "model_profile": MODEL_PROFILE,
        "fold": None,
        "n_folds_total": None,
        "truth_y_path": display_path(target_y_path),
        "truth_y_sha256": target_row["y_sha256"],
        "input_hashes": {
            "source_x": source_row["x_sha256"],
            "source_y": source_row["y_sha256"],
            "target_x": target_row["x_sha256"],
            "target_y": target_row["y_sha256"],
            "manifest": sha256_file(manifest_path),
            "label_map": label_contract["label_map_sha256"],
        },
        "manifest_path": display_path(manifest_path),
        "label_map_path": display_path(label_contract["label_map_path"]),
        "label_contract_source": label_contract["label_contract_source"],
        "code_hashes": code_hashes,
        "library_versions": prediction_io.library_versions(),
        "removed_exact_target_ids": removed,
        "evaluation_partition": partition,
    }

    outdir = resolve(args.outdir)
    direction_dir = (
        outdir
        / "transfer"
        / f"{args.source}_to_{args.target}__{LABEL_SPACE}__{feature_key}__{MODEL_PROFILE}"
    )
    cell_dirs = {key: direction_dir / key for key in args.models}
    if not args.resume:
        existing = [
            key for key, cell_dir in cell_dirs.items()
            if any(path.exists() for path in prediction_io.cell_paths(cell_dir, STEM).values())
        ]
        if existing:
            raise FileExistsError(
                f"cells already exist for {existing} under {direction_dir}; "
                "pass --resume to reuse or redo them."
            )

    if PROFESSOR_DIR.as_posix() not in sys.path:
        sys.path.insert(0, PROFESSOR_DIR.as_posix())

    record = {
        "kind": "transfer",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "args": {key: (value.as_posix() if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "frozen_sources": display_path(frozen_path),
        "code_hashes": code_hashes,
        "n_source": len(x_source),
        "n_target": len(x_target),
        "removed_exact_target_ids": removed,
        "evaluation_partition_sha256": partition["sha256"],
        "cells": [],
        "outcome": "incomplete",
    }
    try:
        for model_key in args.models:
            cell_dir = cell_dirs[model_key]
            model = build_paper_model(model_key, n_jobs=args.model_n_jobs)
            params = effective_params(model_key, model)
            contract = {
                **base_contract,
                "model": model_key,
                "effective_params": params,
                "seed": int(params["random_state"]),
            }
            expected = {k: v for k, v in contract.items() if k not in NON_EXPECTED_FIELDS}
            cell = {"model": model_key, "path": display_path(cell_dir / STEM), "status": "started"}
            record["cells"].append(cell)
            status = prediction_io.cell_status(cell_dir, STEM, expected)
            if status == "complete":
                cell["status"] = "reused"
                print(f"reused {model_key}: {args.source} -> {args.target}", flush=True)
                continue
            if status == "invalid":
                quarantined = prediction_io.quarantine_cell(cell_dir, STEM)
                cell["quarantined_to"] = display_path(quarantined)
            model.fit(x_source, y_source)
            scores = np.asarray(model.predict_proba(x_target), dtype=float)
            if scores.shape != (len(x_target), y_source.shape[1]):
                raise ValueError(f"{model_key} returned predictions of shape {scores.shape}.")
            predictions = pd.DataFrame(scores, index=x_target.index, columns=y_source.columns)
            contract["fitted_selection"] = fitted_selection(model_key, model)
            contract["execution"] = {
                "n_jobs": args.model_n_jobs,
                "hostname": socket.gethostname(),
            }
            prediction_io.write_prediction_cell(cell_dir, STEM, predictions, contract)
            cell["status"] = "fitted"
            print(f"fitted {model_key}: {args.source} -> {args.target}", flush=True)
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
