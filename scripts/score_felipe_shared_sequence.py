"""Score the shared-sequence 2x2 paired comparison (spec SHARED_SEQUENCE_SPEC_2026-09-28).

Reads the six prediction cells (train v1/v2 x rf/ipcarf/rflda) written under
``<results-root>/paired/train_<v>__shared73__paper_sequence_542__paper/<model>/shared48.*``
and the cohort directory (Interface 1), verifies every hash, and scores

    model (incl. prevalence_baseline) x train_version x cohort x label_version

with the paper ``Evaluator`` at a fixed 0.5 threshold (SS6), plus an independent
sklearn check of micro AUROC/AUPRC/F1. A paired row bootstrap (SS8) shares one index
matrix per cohort across every model / train version / label version, so paired
differences are valid.

No model is imported or fitted. Outputs (Interface 3) are written atomically into a
fresh or empty ``--outdir``; nothing is ever overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prediction_io import (  # noqa: E402
    PredictionCellError,
    atomic_write_text,
    cell_paths,
    library_versions,
    read_prediction_cell,
    sha256_file,
    sha256_lines,
)
from scripts.rescore_felipe_predictions import (  # noqa: E402
    MEASURES,
    evaluate_fold,
    load_evaluator_class,
)


DEFAULT_MODELS = ("rf", "ipcarf", "rflda")
BASELINE = "prevalence_baseline"
TRAIN_VERSIONS = ("v1", "v2")
COHORTS = ("primary", "sensitivity")  # bootstrap index generation order is fixed
LABEL_VERSIONS = ("v1", "v2")
LABEL_SPACE = "shared73"
FEATURE_KEY = "paper_sequence_542"
MODEL_PROFILE = "paper"
KIND = "shared_sequence_paired"
CELL_STEM = "shared48"
BOOT_METRICS = ("micro_roc", "micro_auprc", "micro_f1", "hamming")
SKLEARN_TOL = 1e-12
EVAL_MODE = {"evaluation_mode": "fixed", "lane": "primary_clean", "threshold_source": "constant_0.5"}
OUTPUT_FILES = (
    "summary.csv",
    "per_disease.csv",
    "bootstrap_ci.csv",
    "paired_differences.csv",
    "disease_definedness.csv",
)
LABEL_FILES = {
    ("v1", "primary"): "labels_v1.csv",
    ("v1", "sensitivity"): "labels_v1.csv",
    ("v2", "primary"): "labels_v2_primary.csv",
    ("v2", "sensitivity"): "labels_v2_or.csv",
}
COHORT_FILES = (
    "test_X.csv",
    "labels_v1.csv",
    "labels_v2_primary.csv",
    "labels_v2_or.csv",
    "cohorts.csv",
    "train_ids_v1.txt",
    "train_ids_v2.txt",
)


class ScoreError(RuntimeError):
    """A hard integrity failure; nothing is written."""


# --------------------------------------------------------------------------- helpers


def cell_dir(results_root: Path, train_version: str, model: str) -> Path:
    group = f"train_{train_version}__{LABEL_SPACE}__{FEATURE_KEY}__{MODEL_PROFILE}"
    return Path(results_root) / "paired" / group / model


def _read_ids(path: Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    ids = [line for line in text.split("\n") if line != ""]
    if len(set(ids)) != len(ids):
        raise ScoreError(f"{path} contains duplicate IDs.")
    return ids


def _bool_column(series: pd.Series, name: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    out = series.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise ScoreError(f"cohorts.csv column {name!r} is not boolean.")
    return out.astype(bool)


def _read_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sequence_sha256": str})
    if frame.columns[0] != "sequence_sha256":
        raise ScoreError(f"{path.name} must have index column sequence_sha256.")
    frame = frame.set_index("sequence_sha256")
    if frame.index.duplicated().any():
        raise ScoreError(f"{path.name} has duplicate sequence_sha256 rows.")
    values = frame.to_numpy()
    if not np.isin(values, (0, 1)).all():
        raise ScoreError(f"{path.name} labels must be 0/1.")
    return frame.astype(int)


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<i8").tobytes()).hexdigest()


def _csv_text(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, float_format="%.17g", lineterminator="\n")
    return buffer.getvalue()


def _resolve(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else Path(project_root) / path


# --------------------------------------------------------------------------- loading


def load_cohort(cohort_dir: Path) -> dict:
    cohort_dir = Path(cohort_dir)
    manifest_path = cohort_dir / "cohort_manifest.json"
    if not manifest_path.is_file():
        raise ScoreError(f"cohort manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ScoreError("cohort_manifest.json lacks an 'outputs' mapping.")
    problems = [f"{name} not listed in cohort manifest" for name in COHORT_FILES if name not in outputs]
    measured: dict[str, str] = {}
    for name, digest in outputs.items():
        path = cohort_dir / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        measured[name] = sha256_file(path)
        if measured[name] != str(digest).lower():
            problems.append(f"hash mismatch {name}")
    if problems:
        raise ScoreError("cohort directory failed verification: " + "; ".join(problems))

    test_ids = pd.read_csv(cohort_dir / "test_X.csv", usecols=[0], dtype=str)
    if test_ids.columns[0] != "sequence_sha256":
        raise ScoreError("test_X.csv must have index column sequence_sha256.")
    test_ids = test_ids.iloc[:, 0].tolist()
    if len(set(test_ids)) != len(test_ids):
        raise ScoreError("test_X.csv has duplicate sequence_sha256 rows.")

    cohorts = pd.read_csv(cohort_dir / "cohorts.csv", dtype=str).set_index("sequence_sha256")
    if cohorts.index.duplicated().any() or set(cohorts.index) != set(test_ids):
        raise ScoreError("cohorts.csv rows differ from test_X.csv rows.")
    flags = {
        "primary": _bool_column(cohorts["in_primary"], "in_primary"),
        "sensitivity": _bool_column(cohorts["in_sensitivity"], "in_sensitivity"),
    }
    cohort_ids = {name: [i for i in test_ids if flags[name].loc[i]] for name in COHORTS}
    for name, ids in cohort_ids.items():
        if not ids:
            raise ScoreError(f"cohort {name!r} is empty.")

    labels = {
        name: _read_labels(cohort_dir / name)
        for name in ("labels_v1.csv", "labels_v2_primary.csv", "labels_v2_or.csv")
    }
    label_columns = labels["labels_v1.csv"].columns.tolist()
    for name, frame in labels.items():
        if frame.columns.tolist() != label_columns:
            raise ScoreError(f"{name} column order differs from labels_v1.csv.")
    if set(labels["labels_v2_primary.csv"].index) != set(cohort_ids["primary"]):
        raise ScoreError("labels_v2_primary.csv rows differ from the primary cohort.")
    for (label_version, cohort), name in LABEL_FILES.items():
        missing = sorted(set(cohort_ids[cohort]) - set(labels[name].index))
        if missing:
            raise ScoreError(f"{name} lacks {len(missing)} {cohort} rows, e.g. {missing[:3]}")

    return {
        "dir": cohort_dir,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "file_hashes": measured,
        "test_ids": test_ids,
        "test_x_sha256": measured["test_X.csv"],
        "test_row_ids_sha256": sha256_lines(test_ids),
        "cohort_ids": cohort_ids,
        "labels": labels,
        "label_columns": label_columns,
        "train_ids": {v: _read_ids(cohort_dir / f"train_ids_{v}.txt") for v in TRAIN_VERSIONS},
    }


def load_cells(results_root: Path, cohort: dict, models, allow_probe: bool,
               transfer_hashes: dict) -> dict:
    """Return {(model, train_version): {"frame", "contract", "sha256", "contract_path"}}."""
    cells: dict = {}
    problems: list[str] = []
    for train_version in TRAIN_VERSIONS:
        for model in models:
            directory = cell_dir(results_root, train_version, model)
            label = f"train_{train_version}/{model}"
            expected = {
                "kind": KIND,
                "label_space": LABEL_SPACE,
                "feature_key": FEATURE_KEY,
                "model_profile": MODEL_PROFILE,
                "train_version": train_version,
                "model": model,
                "test_x_sha256": cohort["test_x_sha256"],
                "test_row_ids_sha256": cohort["test_row_ids_sha256"],
                "cohort_manifest_sha256": cohort["manifest_sha256"],
            }
            try:
                frame, contract = read_prediction_cell(directory, CELL_STEM, expected)
            except PredictionCellError as error:
                problems.append(f"{label}: {error}")
                continue
            # read_prediction_cell verified the file; its default pandas parser is not
            # correctly rounded on 17-digit text and can merge distinct scores into ties
            # (SS15). Re-read the verified bytes exactly and use those values.
            exact = pd.read_csv(
                cell_paths(directory, CELL_STEM)["predictions"], index_col=0,
                dtype={"ID": str}, keep_default_na=False, float_precision="round_trip",
            )
            exact.index = exact.index.astype(str)
            if (
                exact.index.tolist() != frame.index.tolist()
                or exact.columns.tolist() != frame.columns.tolist()
                or not np.allclose(exact.to_numpy(float), frame.to_numpy(float), rtol=0, atol=1e-15)
            ):
                problems.append(f"{label}: exact re-read of predictions disagrees with the verified cell")
                continue
            frame = exact.astype(float)
            if "probe" not in contract:
                problems.append(f"{label}: contract lacks 'probe'")
                continue
            if contract["probe"] is not None and not allow_probe:
                problems.append(f"{label}: probe cell {contract['probe']!r} (use --allow-probe)")
                continue
            if frame.index.tolist() != cohort["test_ids"]:
                problems.append(f"{label}: prediction row order differs from test_X.csv")
                continue
            if frame.columns.tolist() != cohort["label_columns"]:
                problems.append(f"{label}: prediction columns differ from labels_v1.csv")
                continue
            train_ids = cohort["train_ids"][train_version]
            if contract["probe"] is not None:
                limit = (contract["probe"] or {}).get("max_train_rows")
                if not isinstance(limit, int) or limit < 1:
                    problems.append(f"{label}: malformed probe {contract['probe']!r}")
                    continue
                train_ids = train_ids[:limit]  # probe fits use the first N IDs
            if contract.get("train_ids_sha256") != sha256_lines(train_ids):
                problems.append(f"{label}: train_ids_sha256 differs from cohort train_ids_{train_version}.txt")
                continue
            if contract.get("n_train") != len(train_ids):
                problems.append(f"{label}: n_train {contract.get('n_train')!r} != {len(train_ids)}")
                continue
            hashes = contract.get("input_hashes") or {}
            want = {
                "train_y": transfer_hashes[train_version]["y_sha256"],
                "transfer_manifest": transfer_hashes["manifest_sha256"],
                "test_x": cohort["test_x_sha256"],
            }
            bad = [k for k, v in want.items() if hashes.get(k) != v]
            if bad:
                problems.append(f"{label}: input_hashes differ for {bad}")
                continue
            paths = directory / f"{CELL_STEM}.contract.json"
            cells[(model, train_version)] = {
                "frame": frame,
                "contract": contract,
                "sha256": contract["prediction_file_sha256"],
                "contract_path": paths,
                "contract_sha256": sha256_file(paths),
                "fitted_train_ids": train_ids,
            }
    for train_version in TRAIN_VERSIONS:
        fitted = {tuple(c["fitted_train_ids"]) for (m, v), c in cells.items() if v == train_version}
        if len(fitted) > 1:
            problems.append(f"train_{train_version}: models were fitted on different training IDs")
    if problems:
        raise ScoreError("missing or invalid prediction cells:\n  " + "\n  ".join(problems))
    return cells


def load_transfer(manifest_path: Path, project_root: Path, label_columns: list[str]) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise ScoreError(f"transfer manifest not found: {manifest_path}")
    table = pd.read_csv(manifest_path, dtype=str)
    out: dict = {"manifest_path": manifest_path, "manifest_sha256": sha256_file(manifest_path)}
    for version in TRAIN_VERSIONS:
        rows = table[
            (table["version"] == version)
            & (table["label_space"] == LABEL_SPACE)
            & (table["feature_key"] == FEATURE_KEY)
        ]
        if len(rows) != 1:
            raise ScoreError(f"transfer manifest needs exactly one {version}/{LABEL_SPACE} row.")
        row = rows.iloc[0]
        y_path = _resolve(row["y_path"], project_root)
        if not y_path.is_file():
            raise ScoreError(f"transfer Y not found: {y_path}")
        measured = sha256_file(y_path)
        if measured != str(row["y_sha256"]).lower():
            raise ScoreError(f"transfer Y {version} hash differs from the manifest.")
        frame = pd.read_csv(y_path, dtype={"ID": str}).set_index("ID")
        if frame.columns.tolist() != label_columns:
            raise ScoreError(f"transfer Y {version} columns differ from labels_v1.csv.")
        out[version] = {"y_path": str(row["y_path"]), "y_sha256": measured, "frame": frame}
    return out


def train_prevalence(transfer: dict, version: str, train_ids: list[str]) -> pd.Series:
    frame = transfer[version]["frame"]
    if frame.index.duplicated().any():
        raise ScoreError(f"transfer Y {version} has duplicate IDs.")
    missing = [i for i in train_ids if i not in frame.index]
    if missing:
        raise ScoreError(f"{len(missing)} train IDs absent from transfer Y {version}, e.g. {missing[:3]}")
    values = frame.loc[train_ids]
    if not np.isin(values.to_numpy(), (0, 1)).all():
        raise ScoreError(f"transfer Y {version} labels must be 0/1.")
    return values.astype(float).mean(axis=0)


def truth_for(cohort: dict, cohort_name: str, label_version: str) -> pd.DataFrame:
    frame = cohort["labels"][LABEL_FILES[(label_version, cohort_name)]]
    ids = cohort["cohort_ids"][cohort_name]
    truth = frame.loc[ids]  # explicit join by sequence_sha256
    if truth.index.tolist() != ids:
        raise ScoreError("truth join failed")
    return truth


# --------------------------------------------------------------------------- metrics


def weighted_micro(y: np.ndarray, s: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Micro AUROC / AUPRC / F1@0.5 / hamming for each row-weight vector in ``counts``.

    ``y``/``s`` are (n, L); ``counts`` is (B, n) integer row multiplicities (a bootstrap
    replicate is exactly the data with row i repeated counts[b, i] times). AUROC uses the
    tie-corrected rank (Mann-Whitney) formula; AUPRC reproduces sklearn's step-wise
    average precision over distinct score thresholds. Undefined values are NaN.
    """
    n, n_labels = y.shape
    counts = np.asarray(counts, dtype=float)
    y_flat = y.ravel().astype(float)
    s_flat = s.ravel().astype(float)
    order = np.argsort(-s_flat, kind="mergesort")
    s_sorted = s_flat[order]
    y_sorted = y_flat[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s_sorted) != 0) + 1]
    row_sorted = order // n_labels

    pred = s >= 0.5
    tp_row = (pred & (y == 1)).sum(axis=1).astype(float)
    fp_row = (pred & (y == 0)).sum(axis=1).astype(float)
    fn_row = (~pred & (y == 1)).sum(axis=1).astype(float)
    mism_row = fp_row + fn_row

    out = {name: np.full(counts.shape[0], np.nan) for name in BOOT_METRICS}
    chunk = 256
    for lo in range(0, counts.shape[0], chunk):
        c = counts[lo : lo + chunk]
        w = c[:, row_sorted]
        pos = np.add.reduceat(w * y_sorted, starts, axis=1)
        neg = np.add.reduceat(w * (1.0 - y_sorted), starts, axis=1)
        tp = np.cumsum(pos, axis=1)
        fp = np.cumsum(neg, axis=1)
        tot_p = tp[:, -1]
        tot_n = fp[:, -1]
        num = (pos * (tot_n[:, None] - fp) + 0.5 * pos * neg).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            auc = num / (tot_p * tot_n)
            denom = tp + fp
            precision = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
            ap = (pos * precision).sum(axis=1) / tot_p
        auc[(tot_p == 0) | (tot_n == 0)] = np.nan
        ap[tot_p == 0] = np.nan
        out["micro_roc"][lo : lo + chunk] = auc
        out["micro_auprc"][lo : lo + chunk] = ap

        tps, fps, fns = c @ tp_row, c @ fp_row, c @ fn_row
        f_denom = 2 * tps + fps + fns
        out["micro_f1"][lo : lo + chunk] = np.divide(
            2 * tps, f_denom, out=np.zeros_like(tps), where=f_denom > 0
        )
        out["hamming"][lo : lo + chunk] = (c @ mism_row) / (c.sum(axis=1) * n_labels)
    return out


def sklearn_micro(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    flat_y, flat_s = y.ravel(), s.ravel()
    roc = roc_auc_score(flat_y, flat_s) if np.unique(flat_y).size == 2 else float("nan")
    auprc = average_precision_score(flat_y, flat_s) if flat_y.sum() > 0 else float("nan")
    f1 = f1_score(flat_y, (flat_s >= 0.5).astype(int), zero_division=0)
    return {"micro_roc": float(roc), "micro_auprc": float(auprc), "micro_f1": float(f1)}


def _close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) < SKLEARN_TOL


def per_disease_rows(truth: pd.DataFrame, scores: pd.DataFrame) -> list[dict]:
    rows = []
    for disease in truth.columns:
        y = truth[disease].to_numpy()
        s = scores[disease].to_numpy()
        positives = int(y.sum())
        defined = 0 < positives < len(y)
        rows.append(
            {
                "disease": disease,
                "n_rows": len(y),
                "positives": positives,
                "defined": bool(defined),
                "auroc": float(roc_auc_score(y, s)) if defined else float("nan"),
                "auprc": float(average_precision_score(y, s)) if defined else float("nan"),
            }
        )
    return rows


def score_one(evaluator, truth: pd.DataFrame, scores: pd.DataFrame) -> tuple[dict, list[dict]]:
    if truth.index.tolist() != scores.index.tolist() or truth.columns.tolist() != scores.columns.tolist():
        raise ScoreError("truth and scores are not aligned.")
    values, notes = evaluate_fold(evaluator, truth, scores)
    y = truth.to_numpy()
    s = scores.to_numpy(dtype=float)
    check = sklearn_micro(y, s)
    fast = {k: float(v[0]) for k, v in weighted_micro(y, s, np.ones((1, y.shape[0]))).items()}
    pairs = [
        ("micro_roc", values["micro_roc"], check["micro_roc"]),
        ("micro_auprc", values["micro_auprc"], check["micro_auprc"]),
        ("fscore", values["fscore"], check["micro_f1"]),
        ("fast micro_roc", fast["micro_roc"], check["micro_roc"]),
        ("fast micro_auprc", fast["micro_auprc"], check["micro_auprc"]),
        ("fast micro_f1", fast["micro_f1"], check["micro_f1"]),
        ("fast hamming", fast["hamming"], values["hamming"]),
    ]
    for name, a, b in pairs:
        if not _close(float(a), float(b)):
            raise ScoreError(f"{name}: Evaluator/fast value {a!r} != sklearn {b!r}")
    disease_rows = per_disease_rows(truth, scores)
    defined = [r for r in disease_rows if r["defined"]]
    aurocs = np.array([r["auroc"] for r in defined])
    positives = y.sum(axis=0)
    row = {
        "n_sequences": y.shape[0],
        "n_diseases": y.shape[1],
        "n_positive_associations": int(y.sum()),
        "n_diseases_both_classes": int(((positives > 0) & (positives < y.shape[0])).sum()),
        "n_diseases_no_positive": int((positives == 0).sum()),
        "n_diseases_all_positive": int((positives == y.shape[0]).sum()),
        **{m: values[m] for m in MEASURES},
        "micro_f1": values["fscore"],
        "sklearn_micro_roc": check["micro_roc"],
        "sklearn_micro_auprc": check["micro_auprc"],
        "sklearn_micro_f1": check["micro_f1"],
        "per_disease_auroc_median": float(np.median(aurocs)) if len(defined) else float("nan"),
        "per_disease_auprc_median": (
            float(np.median([r["auprc"] for r in defined])) if len(defined) else float("nan")
        ),
        "n_per_disease_defined": len(defined),
        "frac_defined_auroc_gt_0_5": float((aurocs > 0.5).mean()) if len(defined) else float("nan"),
        "evaluator_warnings": notes,
    }
    return row, disease_rows


# --------------------------------------------------------------------------- bootstrap


def bootstrap_indices(cohort_sizes: dict[str, int], n_bootstrap: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.integers(0, cohort_sizes[name], size=(n_bootstrap, cohort_sizes[name]), dtype=np.int64)
        for name in COHORTS
    }


def counts_from_indices(idx: np.ndarray) -> np.ndarray:
    n_boot, n = idx.shape
    counts = np.zeros((n_boot, n), dtype=np.int64)
    np.add.at(counts, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1)
    return counts


def _interval(values: np.ndarray) -> tuple[float, float, int]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan"), float("nan"), 0
    low, high = np.percentile(valid, [2.5, 97.5])
    return float(low), float(high), int(valid.size)


# --------------------------------------------------------------------------- run


def run(
    results_root: Path,
    cohort_dir: Path,
    transfer_manifest: Path,
    outdir: Path,
    models=DEFAULT_MODELS,
    n_bootstrap: int = 2000,
    seed: int = 20260928,
    allow_probe: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> dict:
    outdir = Path(outdir)
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        raise ScoreError(f"refusing to write into existing non-empty {outdir}")
    models = tuple(models)
    if BASELINE in models or len(set(models)) != len(models) or not models:
        raise ScoreError(f"invalid --models {models!r}")
    if n_bootstrap < 1:
        raise ScoreError("--n-bootstrap must be >= 1")

    cohort = load_cohort(cohort_dir)
    transfer = load_transfer(transfer_manifest, project_root, cohort["label_columns"])
    cells = load_cells(results_root, cohort, models, allow_probe, transfer)
    Evaluator, evaluator_path = load_evaluator_class(project_root)
    evaluator = Evaluator(threshold_mode="fixed", threshold=0.5)

    # SS7: prevalence of the training rows the models of that version were fitted on
    # (the full cohort train IDs for real runs; the first N IDs for --allow-probe runs).
    fitted_ids = {
        v: next(c["fitted_train_ids"] for (m, w), c in cells.items() if w == v)
        for v in TRAIN_VERSIONS
    }
    prevalence = {v: train_prevalence(transfer, v, fitted_ids[v]) for v in TRAIN_VERSIONS}
    all_models = models + (BASELINE,)
    full_scores: dict = {}
    for (model, v), cell in cells.items():
        full_scores[(model, v)] = cell["frame"]
    for v in TRAIN_VERSIONS:
        prev = prevalence[v].reindex(cohort["label_columns"]).to_numpy()
        full_scores[(BASELINE, v)] = pd.DataFrame(
            np.tile(prev, (len(cohort["test_ids"]), 1)),
            index=cohort["test_ids"],
            columns=cohort["label_columns"],
        )

    sizes = {c: len(cohort["cohort_ids"][c]) for c in COHORTS}
    indices = bootstrap_indices(sizes, n_bootstrap, seed)
    index_hashes = {c: _sha256_array(indices[c]) for c in COHORTS}
    counts = {c: counts_from_indices(indices[c]) for c in COHORTS}

    summary_rows, disease_rows, definedness_rows, ci_rows = [], [], [], []
    boot: dict = {}
    points: dict = {}
    for cohort_name in COHORTS:
        ids = cohort["cohort_ids"][cohort_name]
        for label_version in LABEL_VERSIONS:
            truth = truth_for(cohort, cohort_name, label_version)
            for disease in truth.columns:
                pos = int(truth[disease].sum())
                definedness_rows.append(
                    {
                        "cohort": cohort_name,
                        "label_version": label_version,
                        "label_file": LABEL_FILES[(label_version, cohort_name)],
                        "disease": disease,
                        "n_rows": len(truth),
                        "positives": pos,
                        "defined": bool(0 < pos < len(truth)),
                        "zero_positive": bool(pos == 0),
                        "all_positive": bool(pos == len(truth)),
                    }
                )
            for model in all_models:
                for v in TRAIN_VERSIONS:
                    scores = full_scores[(model, v)].loc[ids]
                    if scores.columns.tolist() != truth.columns.tolist():
                        raise ScoreError("prediction/truth column order differs")
                    key = {
                        "model": model,
                        "train_version": v,
                        "cohort": cohort_name,
                        "label_version": label_version,
                    }
                    row, per = score_one(evaluator, truth, scores)
                    source = "train_prevalence" if model == BASELINE else cells[(model, v)]["sha256"]
                    summary_rows.append(
                        {
                            **EVAL_MODE,
                            **key,
                            "label_file": LABEL_FILES[(label_version, cohort_name)],
                            **row,
                            "prediction_sha256": source,
                        }
                    )
                    disease_rows.extend({**EVAL_MODE, **key, **item} for item in per)
                    rep = weighted_micro(truth.to_numpy(), scores.to_numpy(dtype=float), counts[cohort_name])
                    point = {
                        "micro_roc": row["sklearn_micro_roc"],
                        "micro_auprc": row["sklearn_micro_auprc"],
                        "micro_f1": row["sklearn_micro_f1"],
                        "hamming": row["hamming"],
                    }
                    tup = (model, v, cohort_name, label_version)
                    boot[tup] = rep
                    points[tup] = point
                    for metric in BOOT_METRICS:
                        low, high, n_valid = _interval(rep[metric])
                        ci_rows.append(
                            {
                                **key,
                                "metric": metric,
                                "point": point[metric],
                                "ci_low": low,
                                "ci_high": high,
                                "n_valid": n_valid,
                                "n_invalid": n_bootstrap - n_valid,
                                "n_bootstrap": n_bootstrap,
                                "seed": seed,
                                "index_sha256": index_hashes[cohort_name],
                            }
                        )

    diff_rows = []

    def add_diff(kind, a, b):
        cohort_name = a[2]
        for metric in BOOT_METRICS:
            diffs = boot[a][metric] - boot[b][metric]
            valid = diffs[~np.isnan(diffs)]
            low, high, n_valid = _interval(diffs)
            diff_rows.append(
                {
                    "comparison": kind,
                    "cohort": cohort_name,
                    "metric": metric,
                    "a_model": a[0],
                    "a_train_version": a[1],
                    "a_label_version": a[3],
                    "b_model": b[0],
                    "b_train_version": b[1],
                    "b_label_version": b[3],
                    "point_diff": points[a][metric] - points[b][metric],
                    "ci_low": low,
                    "ci_high": high,
                    "n_valid": n_valid,
                    "n_invalid": n_bootstrap - n_valid,
                    "frac_gt_0": float((valid > 0).mean()) if n_valid else float("nan"),
                    "n_bootstrap": n_bootstrap,
                    "index_sha256": index_hashes[cohort_name],
                }
            )

    for cohort_name in COHORTS:
        for label_version in LABEL_VERSIONS:
            for model in all_models:
                add_diff(
                    "train_v2_minus_train_v1",
                    (model, "v2", cohort_name, label_version),
                    (model, "v1", cohort_name, label_version),
                )
            for v in TRAIN_VERSIONS:
                for model in models:
                    add_diff(
                        "model_minus_prevalence_baseline",
                        (model, v, cohort_name, label_version),
                        (BASELINE, v, cohort_name, label_version),
                    )
        for model in all_models:
            for v in TRAIN_VERSIONS:
                add_diff(
                    "label_v2_minus_label_v1",
                    (model, v, cohort_name, "v2"),
                    (model, v, cohort_name, "v1"),
                )

    texts = {
        "summary.csv": _csv_text(pd.DataFrame(summary_rows)),
        "per_disease.csv": _csv_text(pd.DataFrame(disease_rows)),
        "bootstrap_ci.csv": _csv_text(pd.DataFrame(ci_rows)),
        "paired_differences.csv": _csv_text(pd.DataFrame(diff_rows)),
        "disease_definedness.csv": _csv_text(pd.DataFrame(definedness_rows)),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    output_hashes = {}
    for name in OUTPUT_FILES:
        atomic_write_text(outdir / name, texts[name])
        output_hashes[name] = sha256_file(outdir / name)

    run_record = {
        "schema": "shared_sequence_scores/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "results_root": str(results_root),
            "cells": {
                f"train_{v}/{m}": {
                    "prediction_sha256": c["sha256"],
                    "contract_path": str(c["contract_path"]),
                    "contract_sha256": c["contract_sha256"],
                    "train_ids_sha256": c["contract"].get("train_ids_sha256"),
                    "probe": c["contract"].get("probe"),
                }
                for (m, v), c in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0]))
            },
            "cohort_dir": str(cohort["dir"]),
            "cohort_manifest_sha256": cohort["manifest_sha256"],
            "cohort_files_sha256": cohort["file_hashes"],
            "label_files_sha256": {
                name: cohort["file_hashes"][name]
                for name in ("labels_v1.csv", "labels_v2_primary.csv", "labels_v2_or.csv")
            },
            "transfer_manifest": {
                "path": str(transfer["manifest_path"]),
                "sha256": transfer["manifest_sha256"],
                "y": {v: {"path": transfer[v]["y_path"], "sha256": transfer[v]["y_sha256"]} for v in TRAIN_VERSIONS},
            },
            "train_ids_sha256": {v: sha256_lines(cohort["train_ids"][v]) for v in TRAIN_VERSIONS},
            "baseline_train_ids_sha256": {v: sha256_lines(fitted_ids[v]) for v in TRAIN_VERSIONS},
            "baseline_n_train": {v: len(fitted_ids[v]) for v in TRAIN_VERSIONS},
        },
        "outputs": output_hashes,
        "parameters": {
            "models": list(models),
            "baseline": BASELINE,
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "index_generation": "numpy.random.default_rng(seed).integers(0, n, (B, n)); primary then sensitivity",
            "index_dtype": "<i8",
            "index_sha256": index_hashes,
            "cohort_sizes": sizes,
            "allow_probe": allow_probe,
            "threshold": 0.5,
            **EVAL_MODE,
            "ci": "percentile 2.5/97.5; undefined replicates dropped",
        },
        "code": {
            "scorer": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
            "evaluator": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
        },
        "library_versions": library_versions(),
    }
    atomic_write_text(outdir / "score_run.json", json.dumps(run_record, indent=2, sort_keys=True) + "\n")
    return run_record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument(
        "--transfer-manifest",
        type=Path,
        default=PROJECT_ROOT / "config" / "felipe_basics_transfer_73_manifest.csv",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--allow-probe", action="store_true")
    args = parser.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        record = run(
            args.results_root,
            args.cohort_dir,
            args.transfer_manifest,
            args.outdir,
            models=args.models,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            allow_probe=args.allow_probe,
        )
    print(json.dumps(record["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
