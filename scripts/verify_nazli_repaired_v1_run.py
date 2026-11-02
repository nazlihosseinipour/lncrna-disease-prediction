#!/usr/bin/env python3
"""Verify the repaired 355-row Nazli V1 prediction cells and scores.

This verifier intentionally uses only the Python standard library so it can be
run even when the model environment is unavailable.  It verifies saved bytes
and contracts; it does not import or refit any model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/nazli_repaired_v1_felipe_2026-10-05"
MODELS = ("rf", "ipcarf", "rflda")
FEATURE_KEY = "nazli_repaired_paper_sequence_542"
METRICS = (
    "hamming",
    "label_ranking",
    "micro_roc",
    "micro_auprc",
    "precision",
    "recall",
    "fscore",
    "accuracy",
)
MODEL_CLASSES = {
    "rf": "RF.rf.RF",
    "ipcarf": "IPCARF.ipcarf.IPCARF",
    "rflda": "RFLDA.rflda.RFLDA",
}


class VerificationError(RuntimeError):
    """Raised when an artifact fails a required acceptance check."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(items: list[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"cannot read JSON file {path}: {error}") from error
    require(isinstance(value, dict), f"expected a JSON object: {path}")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    require(path.is_file(), f"missing CSV file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            require(reader.fieldnames is not None, f"CSV has no header: {path}")
            return list(reader.fieldnames), list(reader)
    except (OSError, csv.Error) as error:
        raise VerificationError(f"cannot read CSV file {path}: {error}") from error


def one_row_csv(path: Path) -> dict[str, str]:
    _, rows = read_csv(path)
    require(len(rows) == 1, f"expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def parse_int(value: str, context: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{context} is not numeric: {value!r}") from error
    require(math.isfinite(number) and number.is_integer(), f"{context} is not an integer: {value!r}")
    return int(number)


def parse_float(value: str, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise VerificationError(f"{context} is not numeric: {value!r}") from error
    require(math.isfinite(number), f"{context} is not finite: {value!r}")
    return number


def is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
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


def update_record(base: Path, check: str, details: dict[str, Any]) -> None:
    path = base / "report/verification.json"
    if path.is_file():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            record = {}
    else:
        record = {}
    if not isinstance(record, dict) or record.get("schema_version") != 1:
        record = {"schema_version": 1, "base": base.relative_to(ROOT).as_posix(), "checks": {}}
    checks = record.setdefault("checks", {})
    require(isinstance(checks, dict), "verification record has invalid checks object")
    checks[check] = {"passed": True, **details}
    record["passed"] = all(
        name in checks and isinstance(checks[name], dict) and checks[name].get("passed") is True
        for name in ("cells", "scores")
    )
    atomic_write_json(path, record)


def load_inputs(base: Path) -> dict[str, Any]:
    inputs = base / "inputs"
    manifest_path = inputs / "manifest.csv"
    manifest = one_row_csv(manifest_path)
    expected_manifest = {
        "version": "v1",
        "label_space": "all44",
        "population": "v1_nazli_processed_sequence_355",
        "feature_key": FEATURE_KEY,
        "n_samples": "355",
        "n_features": "542",
        "n_labels": "44",
        "split_method": "released_paper_ten_folds_unchanged",
    }
    for key, expected in expected_manifest.items():
        require(manifest.get(key) == expected, f"manifest {key} differs: {manifest.get(key)!r}")

    x_path = resolve_project_path(manifest["x_path"])
    y_path = resolve_project_path(manifest["y_path"])
    split_path = resolve_project_path(manifest["split_path"])
    x_header, x_rows = read_csv(x_path)
    y_header, y_rows = read_csv(y_path)
    split_header, split_rows = read_csv(split_path)
    require(len(x_rows) == 355 and len(x_header) - 1 == 542, f"X shape is not 355x542")
    require(len(y_rows) == 355 and len(y_header) - 1 == 44, f"Y shape is not 355x44")
    require(x_header[0] == "ID" and y_header[0] == "ID", "X and Y must start with ID")
    x_ids = [row["ID"] for row in x_rows]
    y_ids = [row["ID"] for row in y_rows]
    require(x_ids == y_ids, "X and Y RNA ID order differs")
    require(len(set(x_ids)) == 355, "X/Y contain duplicate RNA IDs")
    labels = y_header[1:]
    require(len(set(labels)) == 44, "Y contains duplicate label names")

    hashes = {
        "manifest": sha256_file(manifest_path),
        "x": sha256_file(x_path),
        "y": sha256_file(y_path),
        "split": sha256_file(split_path),
    }
    for key in ("x", "y", "split"):
        require(hashes[key] == manifest[f"{key}_sha256"], f"{key.upper()} hash differs from manifest")

    prep = read_json(inputs / "preparation_record.json")
    released = prep.get("released_folds", {})
    require(isinstance(released, dict) and released.get("copied_byte_for_byte") is True,
            "preparation record does not identify a byte-for-byte released split")
    source_split = resolve_project_path(str(released.get("source_path", "")))
    require(source_split.is_file(), f"released split source is missing: {source_split}")
    require(sha256_file(source_split) == hashes["split"], "copied splits differ from released paper splits")

    expected_split_header = [name for fold in range(1, 11) for name in (f"train_fold{fold}", f"test_fold{fold}")]
    require(split_header == expected_split_header, "released split column names/order differ")
    fold_positions: dict[int, list[int]] = {}
    all_test_positions: list[int] = []
    universe = set(range(355))
    for fold in range(1, 11):
        train = [parse_int(row[f"train_fold{fold}"], f"train_fold{fold}")
                 for row in split_rows if row[f"train_fold{fold}"].strip()]
        test = [parse_int(row[f"test_fold{fold}"], f"test_fold{fold}")
                for row in split_rows if row[f"test_fold{fold}"].strip()]
        require(len(train) == len(set(train)) and len(test) == len(set(test)), f"fold {fold} has duplicate positions")
        require(set(train).isdisjoint(test), f"fold {fold} train/test overlap")
        require(set(train) | set(test) == universe, f"fold {fold} does not partition all 355 rows")
        fold_positions[fold] = test
        all_test_positions.extend(test)
    coverage = Counter(all_test_positions)
    require(set(coverage) == universe and set(coverage.values()) == {1},
            "released test folds do not cover each row exactly once")
    return {
        "manifest": manifest,
        "hashes": hashes,
        "x_ids": x_ids,
        "labels": labels,
        "fold_positions": fold_positions,
    }


def verify_cells(base: Path) -> dict[str, Any]:
    inputs = load_inputs(base)
    run_root = base / "run"
    run_config = read_json(run_root / "run_config.json")
    require(run_config.get("completed") is True, "run_config is not complete")
    require(run_config.get("model_profile") == "paper", "run_config model_profile is not paper")
    require(run_config.get("random_state") == 0, "run_config random_state is not 0")
    require(run_config.get("models") == list(MODELS), "run_config model list/order differs")
    require(run_config.get("versions") == ["v1"] and run_config.get("label_spaces") == ["all44"],
            "run_config population selection differs")
    require(run_config.get("n_splits") == 10, "run_config does not specify ten folds")
    expected_cells = {(model, fold) for model in MODELS for fold in range(1, 11)}
    configured_cells = run_config.get("cells")
    require(isinstance(configured_cells, list) and len(configured_cells) == 30,
            "run_config does not contain 30 cells")
    configured_keys = {(cell.get("model"), cell.get("fold")) for cell in configured_cells}
    require(configured_keys == expected_cells, "run_config cell keys differ from the 3x10 design")
    for cell in configured_cells:
        require(cell.get("feature_key") == FEATURE_KEY, "run_config cell feature_key differs")
        require(cell.get("version") == "v1" and cell.get("label_space") == "all44",
                "run_config cell population differs")
        require(cell.get("outcome") in {"fitted", "reused"}, "run_config cell was neither fitted nor reused")

    prediction_root = run_root / f"within/v1__all44__{FEATURE_KEY}__paper"
    suffixes = {"predictions": ".predictions.csv", "contract": ".contract.json", "done": ".done"}
    for name, suffix in suffixes.items():
        found = list(prediction_root.rglob(f"*{suffix}"))
        require(len(found) == 30, f"expected 30 {name} files, found {len(found)}")

    expected_params = {
        "binary_mode": False,
        "n_estimators": 150,
        "random_state": 0,
    }
    cell_records: list[dict[str, Any]] = []
    for model in MODELS:
        model_ids: list[str] = []
        for fold in range(1, 11):
            stem = f"fold{fold:02d}"
            directory = prediction_root / model
            prediction_path = directory / f"{stem}.predictions.csv"
            contract_path = directory / f"{stem}.contract.json"
            done_path = directory / f"{stem}.done"
            require(prediction_path.is_file() and contract_path.is_file() and done_path.is_file(),
                    f"incomplete cell: {model}/{stem}")
            contract = read_json(contract_path)
            measured_hash = sha256_file(prediction_path)
            require(contract.get("prediction_file_sha256") == measured_hash,
                    f"prediction hash mismatch: {model}/{stem}")
            require(done_path.read_text(encoding="utf-8").strip() == measured_hash,
                    f"done marker mismatch: {model}/{stem}")
            expected_contract = {
                "schema_version": 1,
                "kind": "within",
                "version": "v1",
                "label_space": "all44",
                "feature_key": FEATURE_KEY,
                "model_profile": "paper",
                "model": model,
                "fold": fold,
                "n_folds_total": 10,
                "n_labels": 44,
                "seed": 0,
                "truth_y_sha256": inputs["hashes"]["y"],
            }
            for key, expected in expected_contract.items():
                require(contract.get(key) == expected, f"contract {key} differs: {model}/{stem}")
            require(contract.get("input_hashes") == inputs["hashes"],
                    f"contract input hashes differ: {model}/{stem}")
            require(contract.get("code_hashes") == run_config.get("code_hashes"),
                    f"contract code hashes differ from run_config: {model}/{stem}")
            params = contract.get("effective_params")
            require(isinstance(params, dict), f"missing effective params: {model}/{stem}")
            for key, expected in expected_params.items():
                require(params.get(key) == expected, f"effective {key} differs: {model}/{stem}")
            require(params.get("class") == MODEL_CLASSES[model], f"effective model class differs: {model}/{stem}")

            header, rows = read_csv(prediction_path)
            require(header and header[0] == "ID", f"prediction ID column missing: {model}/{stem}")
            require(header[1:] == inputs["labels"], f"prediction label order differs: {model}/{stem}")
            expected_ids = [inputs["x_ids"][position] for position in inputs["fold_positions"][fold]]
            row_ids = [row["ID"] for row in rows]
            require(row_ids == expected_ids, f"prediction row IDs differ from released fold: {model}/{stem}")
            require(len(set(row_ids)) == len(row_ids), f"duplicate prediction IDs: {model}/{stem}")
            require(contract.get("n_rows") == len(rows), f"contract row count differs: {model}/{stem}")
            require(contract.get("row_ids_sha256") == sha256_lines(row_ids),
                    f"row ID hash differs: {model}/{stem}")
            require(contract.get("label_names_sha256") == sha256_lines(header[1:]),
                    f"label-name hash differs: {model}/{stem}")
            for row_number, row in enumerate(rows, start=2):
                for label in inputs["labels"]:
                    value = parse_float(row[label], f"{model}/{stem}:{row_number}:{label}")
                    require(0.0 <= value <= 1.0, f"prediction outside [0,1]: {model}/{stem}:{row_number}:{label}")
            model_ids.extend(row_ids)
            cell_records.append({
                "model": model,
                "fold": fold,
                "rows": len(rows),
                "prediction_sha256": measured_hash,
                "contract_sha256": sha256_file(contract_path),
            })
        model_coverage = Counter(model_ids)
        require(set(model_coverage) == set(inputs["x_ids"]) and set(model_coverage.values()) == {1},
                f"{model} test rows do not cover all 355 IDs exactly once")
    return {
        "cell_count": len(cell_records),
        "models": list(MODELS),
        "folds_per_model": 10,
        "input_shapes": {"x": [355, 542], "y": [355, 44]},
        "input_hashes": inputs["hashes"],
        "released_split_sha256": inputs["hashes"]["split"],
        "feature_key": FEATURE_KEY,
        "model_profile": "paper",
        "effective_model_classes": MODEL_CLASSES,
        "cells": cell_records,
    }


def population_stats(values: list[float]) -> tuple[float, float]:
    require(bool(values), "cannot summarize an empty metric series")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def verify_scores(base: Path) -> dict[str, Any]:
    scores = base / "scores"
    _, summary_rows = read_csv(scores / "summary__paper_youden.csv")
    _, fold_rows = read_csv(scores / "fold_metrics__paper_youden.csv")
    _, rejected_rows = read_csv(scores / "rejected_cells.csv")
    _, incomplete_rows = read_csv(scores / "incomplete_groups.csv")
    require(len(summary_rows) == 3, f"expected 3 score groups, found {len(summary_rows)}")
    require(len(fold_rows) == 30, f"expected 30 fold rows, found {len(fold_rows)}")
    require(not rejected_rows, f"found {len(rejected_rows)} rejected cells")
    require(not incomplete_rows, f"found {len(incomplete_rows)} incomplete groups")

    common = {
        "evaluation_mode": "paper_youden",
        "lane": "paper_comparison",
        "threshold_source": "evaluated_fold_labels",
        "kind": "within",
        "version": "v1",
        "label_space": "all44",
        "feature_key": FEATURE_KEY,
        "model_profile": "paper",
    }
    for row in summary_rows + fold_rows:
        for key, expected in common.items():
            require(row.get(key) == expected, f"score field {key} differs for model {row.get('model')}")
        require(is_true(row.get("complete")), f"incomplete score row for model {row.get('model')}")
        require(row.get("n_labels") == "44" and row.get("n_folds") == "10" and row.get("n_folds_evaluated") == "10",
                f"score dimensions differ for model {row.get('model')}")

    summary_by_model = {row["model"]: row for row in summary_rows}
    require(set(summary_by_model) == set(MODELS), "summary model groups differ")
    fold_keys = {(row.get("model"), parse_int(row.get("fold", ""), "fold metric fold")) for row in fold_rows}
    require(fold_keys == {(model, fold) for model in MODELS for fold in range(1, 11)},
            "fold metric model/fold keys differ from the 3x10 design")
    require(all(row.get("cell_stem") == f"fold{parse_int(row['fold'], 'fold'):02d}" for row in fold_rows),
            "fold metric cell stems differ from fold IDs")

    recomputed: list[dict[str, Any]] = []
    for model in MODELS:
        selected = sorted((row for row in fold_rows if row["model"] == model), key=lambda row: int(row["fold"]))
        require(len(selected) == 10, f"{model} does not have ten fold rows")
        summary = summary_by_model[model]
        require(summary.get("n_rows") == "355", f"{model} summary row count is not 355")
        require(sum(parse_int(row.get("n_rows", ""), f"{model} fold row count") for row in selected) == 355,
                f"{model} fold row counts do not sum to 355")
        for metric in METRICS:
            values = [parse_float(row.get(metric, ""), f"{model} {metric} fold {row.get('fold')}") for row in selected]
            mean, std = population_stats(values)
            saved_mean = parse_float(summary.get(f"{metric}_mean", ""), f"{model} {metric} saved mean")
            saved_std = parse_float(summary.get(f"{metric}_std", ""), f"{model} {metric} saved std")
            require(math.isclose(mean, saved_mean, rel_tol=0.0, abs_tol=1e-12),
                    f"{model} {metric} mean differs by {mean - saved_mean}")
            require(math.isclose(std, saved_std, rel_tol=0.0, abs_tol=1e-12),
                    f"{model} {metric} population SD differs by {std - saved_std}")
            recomputed.append({
                "model": model,
                "metric": metric,
                "mean": mean,
                "population_sd": std,
                "saved_mean": saved_mean,
                "saved_population_sd": saved_std,
            })

    rescore = read_json(scores / "rescore_run.json")
    require(rescore.get("n_rejected_cells") == 0 and rescore.get("n_incomplete_groups") == 0,
            "rescore record reports rejected cells or incomplete groups")
    require(rescore.get("modes", {}).get("paper_youden", {}).get("lane") == "paper_comparison",
            "rescore paper_youden lane differs")
    require(rescore.get("modes", {}).get("paper_youden", {}).get("threshold_source") == "evaluated_fold_labels",
            "rescore paper_youden threshold source differs")
    groups = rescore.get("groups_scored")
    require(isinstance(groups, list) and len(groups) == 3 and all(group.get("complete") is True for group in groups),
            "rescore record does not contain three complete groups")
    return {
        "group_count": 3,
        "fold_metric_rows": 30,
        "models": list(MODELS),
        "metrics": list(METRICS),
        "evaluation_mode": "paper_youden",
        "lane": "paper_comparison",
        "threshold_source": "evaluated_fold_labels",
        "rejected_cells": 0,
        "incomplete_groups": 0,
        "arithmetic_tolerance": 1e-12,
        "recomputed": recomputed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, choices=("cells", "scores"))
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    try:
        details = verify_cells(base) if args.check == "cells" else verify_scores(base)
        update_record(base, args.check, details)
    except VerificationError as error:
        parser.exit(1, f"verification failed: {error}\n")
    if args.check == "cells":
        print("repaired v1 cell verification passed (30 cells)")
    else:
        print("repaired v1 score verification passed (3 groups, 30 folds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
