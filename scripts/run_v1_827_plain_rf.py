"""V1->V1 plain RF on the released 355 x 827 matrix with the paper's evaluation.

Paper-faithful MLC construction (same as the verified IPCARF/RFLDA MLC references):
released features_preprocessed.csv used as-is (all 827 columns), 44 flat labels,
released ten inductive folds sliced by position with the authors'
``iterator_cross_validation``, the authors' ``RF`` class (150 trees, random_state 0,
binary_mode=False, transductive_mode=False), and the authors' first-commit Evaluator
(73b21ee; per-disease Youden threshold fitted on the evaluated test fold).

The released matrix contains a 285-column binary block that numerically matches the
raw association matrix, but its provenance conflicts with the paper's description
and remains unresolved; it is kept unchanged here.

Writes fold-level probabilities (with RNA ID and test position), per-fold
thresholds, the eight measures per fold, mean and population SD, and a run
contract with source hashes. Refuses to write into an existing output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import socket
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "lncRNA_CIBCB2025-main"
DS = PAPER / "create_dataset/datasets"
X_PATH = DS / "features/features_preprocessed.csv"
Y_PATH = DS / "flat_labels/labels_flat_preprocessed.csv"
SPLIT_PATH = DS / "splits/inductive/inductive_splits.csv"
EVALUATOR_PATH = (
    ROOT / "results/paper_reproduction_reference_2026-09-20/table3_other_rows/code_used/evaluate_73b21ee.py"
)
EXPECTED = {
    X_PATH: "b264dd75545ca424f1db96622555066791c52ba6105001770ae869e211428887",
    Y_PATH: "53c3573b72be6a893eba2a31e59b29899fba837986c9545e3115133048817e9e",
    SPLIT_PATH: "e9fe2eb5fc4ec928a005b4d000dc3ba8623868498b157b21cfb03579356e1e5a",
}
MEASURES = ["hamming", "label_ranking", "micro_roc", "micro_auprc", "precision", "recall", "fscore", "accuracy"]
N_FOLDS = 10


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def load_evaluator():
    """Import the 73b21ee Evaluator, binding the paper's own utils.transductive_converter."""
    for path in (PAPER,):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("evaluate_73b21ee", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Evaluator


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", type=Path,
                        default=ROOT / "results/felipe_paper_exact_2026-09-29T0014/v1_rf_827")
    args = parser.parse_args(argv)
    outdir = args.outdir if args.outdir.is_absolute() else ROOT / args.outdir
    if outdir.exists():
        raise FileExistsError(f"refusing to write into existing {outdir}")

    for path, digest in EXPECTED.items():
        if sha256(path) != digest:
            raise ValueError(f"input hash changed: {path}")
    if str(PAPER) not in sys.path:
        sys.path.insert(0, str(PAPER))
    from RF.rf import RF  # noqa: E402
    from utils.utils import iterator_cross_validation  # noqa: E402
    Evaluator = load_evaluator()

    x = pd.read_csv(X_PATH, index_col=0)
    y = pd.read_csv(Y_PATH, index_col=0)
    indexes = pd.read_csv(SPLIT_PATH)
    if x.shape != (355, 827) or y.shape != (355, 44) or not (x.index == y.index).all():
        raise ValueError("released X/Y shape or index mismatch")
    train, test = iterator_cross_validation(indexes, x, y, n_folds=N_FOLDS)

    outdir.mkdir(parents=True)
    (outdir / "predictions").mkdir()
    fold_rows, threshold_rows, timings = [], [], []
    seen = []
    for fold in range(1, N_FOLDS + 1):
        x_train, y_train = train[fold - 1]
        x_test, y_test = test[fold - 1]
        positions = indexes[f"test_fold{fold}"].dropna().astype(int).tolist()
        train_positions = indexes[f"train_fold{fold}"].dropna().astype(int).tolist()
        assert not set(positions) & set(train_positions)
        assert x_test.index.tolist() == y.index[positions].tolist() == y_test.index.tolist()
        seen.extend(positions)
        model = RF(binary_mode=False, transductive_mode=False)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        scores = np.asarray(model.predict_proba(x_test), dtype=float)  # no labels passed
        timings.append(time.perf_counter() - started)
        assert scores.shape == y_test.shape and np.isfinite(scores).all()
        pred = pd.DataFrame(scores, index=y_test.index, columns=y.columns)
        out = pred.copy()
        out.insert(0, "test_position", positions)
        out.index.name = "ID"
        out.to_csv(outdir / "predictions" / f"rf_mlc_predictions_test_fold{fold}.csv", float_format="%.17g")

        evaluator = Evaluator()
        thresholds = evaluator.sensivity_specifity_cutoff(y_test, pred)
        values = evaluator.evaluate(y_test, pred)  # paper Evaluator (Youden on this fold)
        fold_rows.append({"fold": fold, "n_test": len(positions), **dict(zip(MEASURES, map(float, values)))})
        for label, threshold in zip(y.columns, thresholds):
            threshold_rows.append({"fold": fold, "label": label, "threshold": float(threshold)})
        print(f"fold {fold}: AUROC {values[2]:.4f} AUPRC {values[3]:.4f} F {values[6]:.4f}", flush=True)
    if sorted(seen) != list(range(len(y))):
        raise ValueError("test folds do not cover every row exactly once")

    per_fold = pd.DataFrame(fold_rows)
    per_fold.to_csv(outdir / "rf_mlc_per_fold.csv", index=False, float_format="%.17g")
    pd.DataFrame(threshold_rows).to_csv(outdir / "rf_mlc_thresholds.csv", index=False, float_format="%.17g")
    summary = pd.DataFrame({
        "metric": MEASURES,
        "mean": [per_fold[m].mean() for m in MEASURES],
        "std_population": [float(np.std(per_fold[m].to_numpy())) for m in MEASURES],
    })
    summary.to_csv(outdir / "rf_mlc_summary.csv", index=False, float_format="%.17g")

    import sklearn
    outputs = {p.relative_to(outdir).as_posix(): sha256(p) for p in sorted(outdir.rglob("*")) if p.is_file()}
    contract = {
        "kind": "v1_827_plain_rf_mlc",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {rel(p): d for p, d in EXPECTED.items()},
        "code": {rel(p): sha256(p) for p in [Path(__file__), PAPER / "RF/rf.py", PAPER / "utils/utils.py",
                                             PAPER / "utils/transductive_converter.py", EVALUATOR_PATH]},
        "model": {"class": "RF.rf.RF", "estimator": "sklearn.ensemble.RandomForestRegressor",
                  "n_estimators": RF.n_estimators, "random_state": RF.random_state, "n_jobs": RF.n_jobs,
                  "max_features": "sklearn_default", "binary_mode": False, "transductive_mode": False},
        "evaluation": {"evaluator": rel(EVALUATOR_PATH), "threshold": "per-label Youden on evaluated test fold "
                       "(argmax(tpr-fpr) if sum(tpr-fpr)>0 else 0.5)", "aggregation": "mean, np.std over 10 folds"},
        "folds": N_FOLDS, "n_rows": len(y), "n_features": x.shape[1], "n_labels": y.shape[1],
        "fit_predict_seconds": timings,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                        "hostname": socket.gethostname()},
        "outputs": outputs,
    }
    (outdir / "run_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
