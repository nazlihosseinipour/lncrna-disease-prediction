"""Score saved Felipe prediction cells without refitting, in two separated modes.

Reads cells written by ``scripts/prediction_io.write_prediction_cell`` (discovered
recursively by ``*.contract.json``), verifies them, joins them to their truth Y by
explicit RNA ID, and scores each evaluation fold with the paper's own ``Evaluator``
(``lncRNA_CIBCB2025-main/parse_results/evaluate.py``):

    fixed         Evaluator(threshold_mode="fixed", threshold=0.5)
                  lane=primary_clean, threshold_source=constant_0.5
    paper_youden  Evaluator(threshold_mode="youden")  (per-label Youden cut fitted on
                  the evaluated fold's own labels, i.e. the paper's procedure)
                  lane=paper_comparison, threshold_source=evaluated_fold_labels

Within groups = (kind, version, label_space, feature_key, model_profile, model) and
must hold folds 1..n_folds_total once each; the fold IDs must partition the truth Y
index. A transfer group is one full-target cell; its evaluation folds are the
contract's ``evaluation_partition.folds``.

One-class / zero-positive labels are never dropped (decision D6): per-label AUROC and
AUPRC are NaN with ``one_class=true``; micro metrics pool every label.

No model is imported or fitted here. Cells that fail integrity checks are listed in
``rejected_cells.csv`` and excluded, never repaired. Outputs are written atomically
and never overwrite an existing file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import sys
import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prediction_io import (  # noqa: E402
    CONTRACT_SUFFIX,
    PredictionCellError,
    atomic_write_text,
    library_versions,
    read_prediction_cell,
    sha256_file,
    sha256_lines,
)


PAPER_DIR_NAME = "lncRNA_CIBCB2025-main"
MEASURES = [
    "hamming",
    "label_ranking",
    "micro_roc",
    "micro_auprc",
    "precision",
    "recall",
    "fscore",
    "accuracy",
]
MODES = {
    "fixed": {
        "lane": "primary_clean",
        "threshold_source": "constant_0.5",
        "evaluator_kwargs": {"threshold_mode": "fixed", "threshold": 0.5},
    },
    "paper_youden": {
        "lane": "paper_comparison",
        "threshold_source": "evaluated_fold_labels",
        "evaluator_kwargs": {"threshold_mode": "youden"},
    },
}
GROUP_FIELDS = (
    "kind",
    "version",
    "source",
    "target",
    "label_space",
    "feature_key",
    "model_profile",
    "model",
)
COMMON_REQUIRED = (
    "kind",
    "label_space",
    "feature_key",
    "model_profile",
    "model",
    "truth_y_path",
    "truth_y_sha256",
)
WITHIN_CONSISTENT = (
    "input_hashes",
    "code_hashes",
    "effective_params",
    "truth_y_path",
    "truth_y_sha256",
    "n_folds_total",
    "label_names_sha256",
    "seed",
)
SHARED_OUTPUTS = (
    "positives.csv",
    "rejected_cells.csv",
    "incomplete_groups.csv",
    "rescore_run.json",
)


BASE_COLUMNS = [
    "evaluation_mode",
    "lane",
    "threshold_source",
    *GROUP_FIELDS,
    "complete",
    "n_labels",
    "n_folds",
    "n_folds_evaluated",
    "truth_y_path",
    "truth_y_sha256",
    "prediction_cells_sha256",
    "prediction_cell_hashes",
]
FOLD_COLUMNS = BASE_COLUMNS + [
    "fold",
    "cell_stem",
    "cell_prediction_sha256",
    "n_rows",
    "n_positives",
    "n_one_class_labels",
    *MEASURES,
    "warnings",
]
SUMMARY_COLUMNS = (
    BASE_COLUMNS
    + ["n_rows"]
    + [f"{measure}_{stat}" for measure in MEASURES for stat in ("mean", "std", "mean(std)")]
    + ["pooled_oof_micro_roc", "pooled_oof_micro_auprc", "n_folds_with_warnings"]
)
PER_LABEL_COLUMNS = BASE_COLUMNS + [
    "population",
    "label",
    "n_rows",
    "positives",
    "one_class",
    "auroc",
    "auprc",
    "threshold",
    "f1_thresholded",
]
POSITIVES_COLUMNS = [
    "evaluation_mode",
    *GROUP_FIELDS,
    "complete",
    "truth_y_sha256",
    "population",
    "label",
    "n_rows",
    "positives",
    "one_class",
]


class RescoreError(RuntimeError):
    """A hard integrity failure that must stop the whole rescoring run."""


def mode_outputs(mode: str) -> tuple[str, ...]:
    return (
        f"fold_metrics__{mode}.csv",
        f"summary__{mode}.csv",
        f"per_label__{mode}.csv",
    )


def all_output_names() -> list[str]:
    names = [name for mode in MODES for name in mode_outputs(mode)]
    return names + list(SHARED_OUTPUTS)


# --------------------------------------------------------------------------- evaluator


_EVALUATOR_CACHE: dict[Path, tuple[type, Path]] = {}


def load_evaluator_class(project_root: Path):
    """Import the paper Evaluator from ``<project_root>/lncRNA_CIBCB2025-main``.

    ``lncRNA_CIBCB2025-main`` and its ``parse_results`` are put on ``sys.path``. The
    paper code imports ``utils.transductive_converter``; any other ``utils`` package
    already in ``sys.modules`` is set aside during the import and restored after it,
    so the paper module always binds the paper's own Converter.
    """
    paper_dir = (Path(project_root) / PAPER_DIR_NAME).resolve()
    if paper_dir in _EVALUATOR_CACHE:
        return _EVALUATOR_CACHE[paper_dir]
    evaluate_file = paper_dir / "parse_results" / "evaluate.py"
    if not evaluate_file.is_file():
        raise RescoreError(f"paper Evaluator not found: {evaluate_file}")
    for path in (paper_dir / "parse_results", paper_dir):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))

    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == "utils" or name.startswith("utils.")
    }
    for name in saved:
        del sys.modules[name]
    try:
        spec = importlib.util.spec_from_file_location("felipe_paper_evaluate", evaluate_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        converter_file = Path(sys.modules[module.Converter.__module__].__file__).resolve()
    finally:
        for name in [n for n in sys.modules if n == "utils" or n.startswith("utils.")]:
            del sys.modules[name]
        sys.modules.update(saved)
    if not converter_file.is_relative_to(paper_dir):
        raise RescoreError(f"paper Converter resolved outside the paper code: {converter_file}")
    _EVALUATOR_CACHE[paper_dir] = (module.Evaluator, evaluate_file)
    return _EVALUATOR_CACHE[paper_dir]


# --------------------------------------------------------------------------- discovery


def discover_contracts(input_root: Path) -> tuple[list[Path], list[Path]]:
    """Return (candidate contracts, contracts inside ``_rejected`` quarantine dirs)."""
    found = sorted(Path(input_root).rglob(f"*{CONTRACT_SUFFIX}"))
    quarantined = [path for path in found if "_rejected" in path.relative_to(input_root).parts]
    active = [path for path in found if path not in quarantined]
    return active, quarantined


def stem_of(contract_path: Path) -> str:
    return contract_path.name[: -len(CONTRACT_SUFFIX)]


def group_key(contract: dict) -> tuple:
    return tuple(contract.get(field) for field in GROUP_FIELDS)


def group_dict(key: tuple) -> dict:
    return {field: value for field, value in zip(GROUP_FIELDS, key)}


def group_label(key: tuple) -> str:
    return "|".join("" if value is None else str(value) for value in key)


def _peek_contract(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def partition_sha256(folds: Iterable[Iterable[str]]) -> str:
    return sha256_lines(
        [f"{index}:{','.join(str(item) for item in fold)}" for index, fold in enumerate(folds, 1)]
    )


def structural_problems(contract: dict) -> list[str]:
    problems = [f"missing field {field!r}" for field in COMMON_REQUIRED if field not in contract]
    kind = contract.get("kind")
    if kind == "within":
        for field in ("version", "fold", "n_folds_total"):
            if field not in contract:
                problems.append(f"missing field {field!r}")
        fold, total = contract.get("fold"), contract.get("n_folds_total")
        if not isinstance(fold, int) or isinstance(fold, bool):
            problems.append(f"within fold must be an int, found {fold!r}")
        if not isinstance(total, int) or isinstance(total, bool) or total < 1:
            problems.append(f"n_folds_total must be a positive int, found {total!r}")
        elif isinstance(fold, int) and not 1 <= fold <= total:
            problems.append(f"fold {fold} outside 1..{total}")
    elif kind == "transfer":
        for field in ("source", "target", "evaluation_partition"):
            if field not in contract:
                problems.append(f"missing field {field!r}")
    else:
        problems.append(f"unknown kind {kind!r}")
    return problems


def check_transfer_partition(contract: dict, prediction_ids: list[str]) -> list[list[str]]:
    """Validate ``evaluation_partition``; return its folds or raise PredictionCellError."""
    partition = contract.get("evaluation_partition")
    if not isinstance(partition, dict) or not isinstance(partition.get("folds"), list):
        raise PredictionCellError("evaluation_partition.folds is missing or not a list")
    folds = [[str(item) for item in fold] for fold in partition["folds"]]
    if not folds or any(not fold for fold in folds):
        raise PredictionCellError("evaluation_partition has an empty fold")
    if partition_sha256(folds) != partition.get("sha256"):
        raise PredictionCellError("evaluation_partition sha256 does not match its folds")
    flat = [item for fold in folds for item in fold]
    if len(set(flat)) != len(flat):
        raise PredictionCellError("evaluation_partition folds are not disjoint")
    if set(flat) != set(prediction_ids) or len(flat) != len(prediction_ids):
        raise PredictionCellError("evaluation_partition union differs from the predicted IDs")
    removed = set(str(item) for item in contract.get("removed_exact_target_ids") or [])
    if removed & set(prediction_ids):
        raise PredictionCellError("predictions include removed_exact_target_ids")
    return folds


# --------------------------------------------------------------------------- truth


class TruthCache:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}

    def resolve(self, truth_path: str) -> Path:
        path = Path(truth_path)
        return path if path.is_absolute() else self.project_root / path

    def load(self, truth_path: str, truth_sha: str) -> pd.DataFrame:
        key = (str(truth_path), str(truth_sha))
        if key in self._frames:
            return self._frames[key]
        path = self.resolve(truth_path)
        if not path.is_file():
            raise RescoreError(f"truth Y not found: {path}")
        measured = sha256_file(path)
        if measured != truth_sha:
            raise RescoreError(
                f"truth Y hash mismatch for {truth_path}: found {measured}, contract {truth_sha}"
            )
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        if raw.shape[1] < 2:
            raise RescoreError(f"truth Y {truth_path} needs an ID column and label columns")
        frame = raw.set_index(raw.columns[0])
        frame.index = frame.index.astype(str)
        frame.index.name = "ID"
        if frame.index.duplicated().any():
            raise RescoreError(f"truth Y {truth_path} has duplicate IDs")
        try:
            frame = frame.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as error:
            raise RescoreError(f"truth Y {truth_path} has non-numeric labels") from error
        values = frame.to_numpy(dtype=float)
        if not np.isin(values, (0.0, 1.0)).all():
            raise RescoreError(f"truth Y {truth_path} is not binary 0/1")
        frame = frame.astype(int)
        self._frames[key] = frame
        return frame


def join_truth(predictions: pd.DataFrame, truth: pd.DataFrame, stem: str) -> pd.DataFrame:
    """Select truth rows by explicit ID join; enforce identical label order."""
    ids = predictions.index.tolist()
    if len(set(ids)) != len(ids):
        raise RescoreError(f"cell {stem!r} has duplicate prediction IDs")
    missing = [item for item in ids if item not in truth.index]
    if missing:
        raise RescoreError(
            f"cell {stem!r}: {len(missing)} prediction IDs absent from truth Y "
            f"(first: {missing[:5]})"
        )
    if list(predictions.columns) != list(truth.columns):
        raise RescoreError(
            f"cell {stem!r}: prediction label columns differ from truth Y columns "
            "(names or order); refusing to reorder"
        )
    selected = truth.loc[ids]
    assert selected.index.tolist() == ids
    return selected


# --------------------------------------------------------------------------- metrics


def _nan_if_single_class(y: np.ndarray, fn) -> float:
    if np.unique(y).size < 2:
        return float("nan")
    return float(fn())


def evaluate_fold(evaluator, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> tuple[dict, str]:
    """Return the Evaluator's 8 measures and a warning string.

    Undefined micro metrics (a fold with a single class across all labels, or no
    positives for AUPRC) are recorded as NaN with a warning; otherwise the
    Evaluator's own values are returned unchanged.
    """
    notes: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            values = dict(zip(MEASURES, (float(v) for v in evaluator.evaluate(y_true, y_pred))))
        except ValueError as error:
            notes.append(f"evaluator_error: {error}")
            values = _fallback_measures(evaluator, y_true, y_pred, notes)
    for item in caught:
        notes.append(f"{item.category.__name__}: {item.message}")
    flat = y_true.to_numpy().ravel()
    if np.unique(flat).size < 2 and not math.isnan(values["micro_roc"]):
        values["micro_roc"] = float("nan")
    if np.unique(flat).size < 2:
        notes.append("micro_roc undefined: single class in fold")
    if flat.sum() == 0:
        values["micro_auprc"] = float("nan")
        notes.append("micro_auprc undefined: no positives in fold")
    return values, "; ".join(dict.fromkeys(notes))


def _fallback_measures(evaluator, y_true, y_pred, notes) -> dict:
    """Per-measure evaluation used only if the Evaluator raises; each measure isolated."""
    from sklearn.metrics import (
        accuracy_score,
        hamming_loss,
        label_ranking_loss,
        precision_recall_fscore_support,
    )

    thresholds = fold_thresholds(evaluator, y_true, y_pred)
    thresholded = evaluator._apply_threshold(y_pred, thresholds)
    out = {measure: float("nan") for measure in MEASURES}

    def attempt(names, fn):
        try:
            result = fn()
        except ValueError as error:
            notes.append(f"{'/'.join(names)} undefined: {error}")
            return
        for name, value in zip(names, result if len(names) > 1 else [result]):
            out[name] = float(value)

    attempt(["hamming"], lambda: hamming_loss(y_true, thresholded))
    attempt(["label_ranking"], lambda: label_ranking_loss(y_true, y_pred))
    attempt(["micro_roc"], lambda: roc_auc_score(y_true, y_pred, average="micro"))
    attempt(["micro_auprc"], lambda: average_precision_score(y_true, y_pred, average="micro"))
    attempt(
        ["precision", "recall", "fscore"],
        lambda: precision_recall_fscore_support(
            y_true, thresholded, average="micro", zero_division=0
        )[:3],
    )
    attempt(["accuracy"], lambda: accuracy_score(y_true, thresholded))
    return out


def fold_thresholds(evaluator, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> list[float]:
    """The per-label thresholds the Evaluator uses for this fold (same code path)."""
    if evaluator.threshold_mode == "fixed":
        return [float(evaluator.threshold)] * y_true.shape[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [float(t) for t in evaluator.sensivity_specifity_cutoff(y_true, y_pred)]


def pooled_micro(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> tuple[float, float]:
    y = y_true.to_numpy()
    p = y_pred.to_numpy()
    flat = y.ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        roc = _nan_if_single_class(flat, lambda: roc_auc_score(y, p, average="micro"))
        auprc = (
            float("nan")
            if flat.sum() == 0
            else float(average_precision_score(y, p, average="micro"))
        )
    return roc, auprc


def label_scores(y: np.ndarray, p: np.ndarray) -> tuple[float, float, bool]:
    if np.unique(y).size < 2:
        return float("nan"), float("nan"), True
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p)), False


# --------------------------------------------------------------------------- groups


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def build_groups(cells: list[dict], allow_partial: bool, truth_cache: TruthCache):
    """Assemble evaluation groups from accepted cells.

    Returns (groups, incomplete_rows). Each group dict holds the evaluation folds as
    (fold_index, cell, ids) plus bookkeeping.
    """
    by_key: dict[tuple, list[dict]] = {}
    for cell in cells:
        by_key.setdefault(group_key(cell["contract"]), []).append(cell)

    groups: list[dict] = []
    incomplete: list[dict] = []
    for key in sorted(by_key, key=group_label):
        members = by_key[key]
        kind = key[0]
        if kind == "transfer":
            group, problem = _transfer_group(key, members, truth_cache)
        else:
            group, problem = _within_group(key, members, truth_cache)
        if problem is None:
            groups.append(group)
            continue
        status, reason = problem
        incomplete.append(
            {
                **group_dict(key),
                "status": status,
                "reason": reason,
                "n_cells": len(members),
                "cell_stems": ";".join(sorted(cell["stem"] for cell in members)),
                "included": bool(allow_partial and status == "incomplete" and group),
            }
        )
        if allow_partial and status == "incomplete" and group:
            groups.append(group)
    return groups, incomplete


def _transfer_group(key, members, truth_cache):
    if len(members) != 1:
        return None, ("invalid", f"{len(members)} transfer cells share one group key")
    cell = members[0]
    contract = cell["contract"]
    truth = truth_cache.load(contract["truth_y_path"], contract["truth_y_sha256"])
    y_all = join_truth(cell["predictions"], truth, cell["stem"])
    folds = [
        (index, cell, fold_ids)
        for index, fold_ids in enumerate(cell["partition_folds"], 1)
    ]
    group = {
        "key": key,
        "complete": True,
        "cells": [cell],
        "folds": folds,
        "n_folds_expected": len(folds),
        "truth": truth,
        "y_all": y_all,
        "pred_all": cell["predictions"],
        "contract": contract,
    }
    return group, None


def _within_group(key, members, truth_cache):
    reference = members[0]["contract"]
    for field in WITHIN_CONSISTENT:
        values = {_canonical(cell["contract"].get(field)) for cell in members}
        if len(values) > 1:
            return None, ("invalid", f"cells disagree on {field}")
    total = reference["n_folds_total"]
    by_fold: dict[int, list[dict]] = {}
    for cell in members:
        by_fold.setdefault(cell["contract"]["fold"], []).append(cell)
    duplicates = sorted(fold for fold, items in by_fold.items() if len(items) > 1)
    if duplicates:
        return None, ("invalid", f"duplicate cells for folds {duplicates}")

    truth = truth_cache.load(reference["truth_y_path"], reference["truth_y_sha256"])
    ordered = [by_fold[fold][0] for fold in sorted(by_fold)]
    seen: set[str] = set()
    for cell in ordered:
        join_truth(cell["predictions"], truth, cell["stem"])
        ids = set(cell["predictions"].index)
        overlap = seen & ids
        if overlap:
            return None, ("invalid", f"ID in two folds (e.g. {sorted(overlap)[:3]})")
        seen |= ids

    missing = [fold for fold in range(1, total + 1) if fold not in by_fold]
    pred_all = pd.concat([cell["predictions"] for cell in ordered])
    y_all = truth.loc[pred_all.index.tolist()]
    group = {
        "key": key,
        "complete": not missing,
        "cells": ordered,
        "folds": [
            (cell["contract"]["fold"], cell, cell["predictions"].index.tolist())
            for cell in ordered
        ],
        "n_folds_expected": total,
        "truth": truth,
        "y_all": y_all,
        "pred_all": pred_all,
        "contract": reference,
    }
    if missing:
        return group, ("incomplete", f"missing folds {missing} of {total}")
    if seen != set(truth.index):
        return None, (
            "invalid",
            f"fold IDs cover {len(seen)} of {len(truth.index)} truth rows",
        )
    return group, None


# --------------------------------------------------------------------------- scoring


def _base_row(group: dict, mode: str) -> dict:
    spec = MODES[mode]
    contract = group["contract"]
    hashes = sorted(
        f"{cell['stem']}:{cell['contract']['prediction_file_sha256']}" for cell in group["cells"]
    )
    return {
        "evaluation_mode": mode,
        "lane": spec["lane"],
        "threshold_source": spec["threshold_source"],
        **group_dict(group["key"]),
        "complete": group["complete"],
        "n_labels": int(group["y_all"].shape[1]),
        "n_folds": group["n_folds_expected"],
        "n_folds_evaluated": len(group["folds"]),
        "truth_y_path": contract["truth_y_path"],
        "truth_y_sha256": contract["truth_y_sha256"],
        "prediction_cells_sha256": sha256_lines(hashes),
        "prediction_cell_hashes": ";".join(hashes),
    }


def score_group(group: dict, mode: str, evaluator_cls) -> tuple[list[dict], dict, list[dict]]:
    evaluator = evaluator_cls(**MODES[mode]["evaluator_kwargs"])
    base = _base_row(group, mode)
    labels = list(group["y_all"].columns)
    fold_rows: list[dict] = []
    per_label: list[dict] = []
    thresholded_parts: list[pd.DataFrame] = []

    for fold_index, cell, ids in group["folds"]:
        y_fold = group["y_all"].loc[ids]
        p_fold = group["pred_all"].loc[ids]
        values, note = evaluate_fold(evaluator, y_fold, p_fold)
        thresholds = fold_thresholds(evaluator, y_fold, p_fold)
        thresholded = evaluator._apply_threshold(p_fold, thresholds).astype(int)
        thresholded_parts.append(thresholded)
        one_class = int(sum(y_fold[label].nunique() < 2 for label in labels))
        fold_rows.append(
            {
                **base,
                "fold": fold_index,
                "cell_stem": cell["stem"],
                "cell_prediction_sha256": cell["contract"]["prediction_file_sha256"],
                "n_rows": len(ids),
                "n_positives": int(y_fold.to_numpy().sum()),
                "n_one_class_labels": one_class,
                **values,
                "warnings": note,
            }
        )
        population = f"fold{fold_index}"
        for position, label in enumerate(labels):
            per_label.append(
                _label_row(
                    base, population, label, y_fold[label], p_fold[label],
                    thresholded[label], thresholds[position],
                )
            )

    pooled_thresholded = pd.concat(thresholded_parts).loc[group["y_all"].index]
    pooled_threshold = 0.5 if mode == "fixed" else float("nan")
    for label in labels:
        per_label.append(
            _label_row(
                base, "pooled", label, group["y_all"][label], group["pred_all"][label],
                pooled_thresholded[label], pooled_threshold,
            )
        )

    frame = pd.DataFrame(fold_rows)
    pooled_roc, pooled_auprc = pooled_micro(group["y_all"], group["pred_all"])
    summary = {**base, "n_rows": int(group["y_all"].shape[0])}
    for measure in MEASURES:
        column = frame[measure].to_numpy(dtype=float)
        summary[f"{measure}_mean"] = float(np.mean(column))
        summary[f"{measure}_std"] = float(np.std(column))
        summary[f"{measure}_mean(std)"] = f"{np.mean(column)}({np.std(column)})"
    summary["pooled_oof_micro_roc"] = pooled_roc
    summary["pooled_oof_micro_auprc"] = pooled_auprc
    summary["n_folds_with_warnings"] = int((frame["warnings"] != "").sum())
    return fold_rows, summary, per_label


def _label_row(base, population, label, y, p, thresholded, threshold) -> dict:
    y_values = y.to_numpy()
    auroc, auprc, one_class = label_scores(y_values, p.to_numpy())
    return {
        **base,
        "population": population,
        "label": label,
        "n_rows": int(len(y_values)),
        "positives": int(y_values.sum()),
        "one_class": one_class,
        "auroc": auroc,
        "auprc": auprc,
        "threshold": threshold,
        "f1_thresholded": float(
            f1_score(y_values, thresholded.to_numpy(), zero_division=0)
        ),
    }


def positives_rows(group: dict, modes: list[str]) -> list[dict]:
    base = {
        "evaluation_mode": "+".join(modes),
        **group_dict(group["key"]),
        "complete": group["complete"],
        "truth_y_sha256": group["contract"]["truth_y_sha256"],
    }
    rows = []
    populations = [(f"fold{index}", ids) for index, _, ids in group["folds"]]
    populations.append(("pooled", group["y_all"].index.tolist()))
    for population, ids in populations:
        y = group["y_all"].loc[ids]
        for label in y.columns:
            count = int(y[label].sum())
            rows.append(
                {
                    **base,
                    "population": population,
                    "label": label,
                    "n_rows": len(ids),
                    "positives": count,
                    "one_class": bool(y[label].nunique() < 2),
                }
            )
    return rows


# --------------------------------------------------------------------------- driver


def load_cells(contract_paths: list[Path], input_root: Path):
    accepted: list[dict] = []
    rejected: list[dict] = []
    ledger: list[dict] = []
    for contract_path in contract_paths:
        stem = stem_of(contract_path)
        outdir = contract_path.parent
        rel = contract_path.relative_to(input_root).as_posix()
        entry = {
            "stem": stem,
            "contract_path": rel,
            "contract_sha256": sha256_file(contract_path),
        }
        peek = _peek_contract(contract_path) or {}
        entry["prediction_file_sha256"] = peek.get("prediction_file_sha256")
        try:
            predictions, contract = read_prediction_cell(outdir, stem)
            problems = structural_problems(contract)
            if problems:
                raise PredictionCellError("; ".join(problems))
            partition_folds = None
            if contract["kind"] == "transfer":
                partition_folds = check_transfer_partition(
                    contract, predictions.index.tolist()
                )
        except PredictionCellError as error:
            rejected.append(
                {
                    "stem": stem,
                    "contract_path": rel,
                    "reason": str(error),
                    **{field: peek.get(field) for field in GROUP_FIELDS},
                    "fold": peek.get("fold"),
                    "_key": group_key(peek) if peek else None,
                }
            )
            ledger.append({**entry, "status": "rejected", "reason": str(error)})
            continue
        accepted.append(
            {
                "stem": stem,
                "path": rel,
                "predictions": predictions,
                "contract": contract,
                "partition_folds": partition_folds,
            }
        )
        ledger.append({**entry, "status": "accepted", "reason": ""})
    return accepted, rejected, ledger


def group_key_from_row(row: dict) -> tuple:
    return tuple(row.get(field) for field in GROUP_FIELDS)


def reconcile_rejected(groups: list[dict], incomplete: list[dict], rejected: list[dict]) -> None:
    """Make every group that lost a cell to rejection visible and never silently complete."""
    rejected_keys = {row["_key"] for row in rejected if row["_key"] is not None}
    listed = {group_key_from_row(row) for row in incomplete}
    for row in incomplete:
        if group_key_from_row(row) in rejected_keys:
            row["reason"] += "; has rejected cells"
    for group in list(groups):
        if group["key"] in rejected_keys and group["key"] not in listed:
            # Siblings still formed a full group (e.g. the rejected cell was a
            # duplicate); refuse to score it rather than guess which run is right.
            groups.remove(group)
            listed.add(group["key"])
            incomplete.append(
                {
                    **group_dict(group["key"]),
                    "status": "invalid",
                    "reason": "group has rejected cells",
                    "n_cells": len(group["cells"]),
                    "cell_stems": ";".join(cell["stem"] for cell in group["cells"]),
                    "included": False,
                }
            )
    for key in sorted(rejected_keys - listed, key=group_label):
        incomplete.append(
            {
                **group_dict(key),
                "status": "incomplete",
                "reason": "all cells rejected",
                "n_cells": 0,
                "cell_stems": "",
                "included": False,
            }
        )


def _refuse_existing(outdir: Path) -> None:
    present = [name for name in all_output_names() if (outdir / name).exists()]
    if present:
        raise RescoreError(
            f"{outdir} already contains rescoring outputs {present}; "
            "choose a new --outdir (outputs are never overwritten)"
        )


def _csv_text(rows: list[dict], columns: list[str]) -> str:
    extra = sorted({key for row in rows for key in row} - set(columns))
    if extra:
        raise RescoreError(f"internal: unexpected output columns {extra}")
    frame = pd.DataFrame(rows, columns=columns)
    return frame.to_csv(index=False, lineterminator="\n")


def run(
    input_root: Path,
    modes: list[str],
    outdir: Path,
    allow_partial: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    input_root = Path(input_root).resolve()
    outdir = Path(outdir)
    project_root = Path(project_root).resolve()
    modes = list(dict.fromkeys(modes))
    unknown = [mode for mode in modes if mode not in MODES]
    if not modes or unknown:
        raise RescoreError(f"--modes must be from {sorted(MODES)}; got {modes}")
    if not input_root.is_dir():
        raise RescoreError(f"--input-root is not a directory: {input_root}")
    _refuse_existing(outdir)

    # The Evaluator is code, not data: always the paper copy in this repository.
    evaluator_cls, evaluator_file = load_evaluator_class(PROJECT_ROOT)
    contract_paths, quarantined = discover_contracts(input_root)
    accepted, rejected, ledger = load_cells(contract_paths, input_root)
    truth_cache = TruthCache(project_root)
    groups, incomplete = build_groups(accepted, allow_partial, truth_cache)

    reconcile_rejected(groups, incomplete, rejected)

    outputs: dict[str, str] = {}
    run_modes_label = "+".join(modes)
    for mode in modes:
        fold_rows, summaries, label_rows = [], [], []
        for group in groups:
            folds, summary, per_label = score_group(group, mode, evaluator_cls)
            fold_rows.extend(folds)
            summaries.append(summary)
            label_rows.extend(per_label)
        names = mode_outputs(mode)
        outputs[names[0]] = _csv_text(fold_rows, FOLD_COLUMNS)
        outputs[names[1]] = _csv_text(summaries, SUMMARY_COLUMNS)
        outputs[names[2]] = _csv_text(label_rows, PER_LABEL_COLUMNS)

    positive_rows = [row for group in groups for row in positives_rows(group, modes)]
    outputs["positives.csv"] = _csv_text(positive_rows, POSITIVES_COLUMNS)
    rejected_columns = ["evaluation_mode", "stem", "contract_path", "reason", *GROUP_FIELDS, "fold"]
    outputs["rejected_cells.csv"] = _csv_text(
        [
            {"evaluation_mode": run_modes_label, **{k: v for k, v in row.items() if k != "_key"}}
            for row in rejected
        ],
        rejected_columns,
    )
    incomplete_columns = [
        "evaluation_mode", *GROUP_FIELDS, "status", "reason", "n_cells", "cell_stems", "included",
    ]
    outputs["incomplete_groups.csv"] = _csv_text(
        [{"evaluation_mode": run_modes_label, **row} for row in incomplete],
        incomplete_columns,
    )
    run_record = {
        "evaluation_mode": modes,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "input_root": str(input_root),
            "project_root": str(project_root),
            "modes": modes,
            "allow_partial": allow_partial,
        },
        "modes": {mode: {k: v for k, v in MODES[mode].items()} for mode in modes},
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "evaluator_path": str(evaluator_file),
        "evaluator_sha256": sha256_file(evaluator_file),
        "library_versions": library_versions(),
        "cells": ledger,
        "quarantined_contracts_skipped": [
            path.relative_to(input_root).as_posix() for path in quarantined
        ],
        "truth_files": sorted(
            {
                f"{cell['contract']['truth_y_path']}:{cell['contract']['truth_y_sha256']}"
                for cell in accepted
            }
        ),
        "groups_scored": [
            {**group_dict(group["key"]), "complete": group["complete"],
             "n_folds_evaluated": len(group["folds"])}
            for group in groups
        ],
        "n_rejected_cells": len(rejected),
        "n_incomplete_groups": len(incomplete),
        "outputs": sorted(outputs) + ["rescore_run.json"],
    }

    _refuse_existing(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, text in outputs.items():
        atomic_write_text(outdir / name, text)
    atomic_write_text(
        outdir / "rescore_run.json",
        json.dumps(run_record, indent=2, sort_keys=True, default=str) + "\n",
    )
    return run_record


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--modes", nargs="+", required=True, choices=sorted(MODES))
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Probe use only: score incomplete groups, marked complete=false.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root for resolving repo-relative truth_y_path values.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    record = run(args.input_root, args.modes, args.outdir, args.allow_partial, args.project_root)
    print(
        f"scored {len(record['groups_scored'])} groups; "
        f"rejected {record['n_rejected_cells']} cells; "
        f"{record['n_incomplete_groups']} incomplete/invalid groups -> {args.outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
