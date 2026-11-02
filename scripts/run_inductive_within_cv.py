"""Run within-version 10-fold inductive CV for RFLDA / IPCARF / RF.

Self-contained: for every feature representation in the prepared manifest it runs
each model fold by fold, saves the per-fold predictions into a clean results tree,
evaluates them through the shared inductive Evaluator (fixed, legacy Youden, or
training-only OOB-calibrated threshold; micro metrics), and writes per-experiment
performance CSVs plus an aggregated summary.

Outputs (under --outdir, default results/within_version_cv):
    --model-profile current (default):
        predictions/<version>_<feature_key>/<model>/test_fold{1..10}.csv
        performance/<version>_<feature_key>_<model>_<threshold_mode>_performance.csv
        within_version_cv_summary.csv
    --model-profile paper (raw prediction cells only; no thresholded metrics):
        within/<version>__<label_space>__<feature_key>__paper/<model>/fold{NN}.*
        (predictions.csv, contract.json, .done via scripts/prediction_io.py;
        metrics come only from scripts/rescore_felipe_predictions.py)
    both:
        run_config.json (latest invocation, retained for compatibility)
        run_configs/<timestamp>_<digest>.json (immutable invocation record)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

try:
    from scripts import felipe_paper_models, prediction_io
    from scripts.experiment_contract import sha256_file, validate_manifest_row
except ModuleNotFoundError:  # Direct execution: python scripts/run_inductive_within_cv.py
    import felipe_paper_models
    import prediction_io
    from experiment_contract import sha256_file, validate_manifest_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results")

METRICS = [
    "hamming",
    "label_ranking",
    "micro_roc",
    "micro_auprc",
    "precision",
    "recall",
    "fscore",
    "accuracy",
]

# Declared label spaces and the label count each one must have (paper profile).
LABEL_SPACE_SIZES = {"all44": 44, "all436": 436, "shared73": 73}
RUNNER_SOURCE = "scripts/run_inductive_within_cv.py"
# Sources that must be pinned in the frozen table for a paper-profile run, in
# addition to the source of every selected model (decision D1).
REQUIRED_FROZEN_SOURCES = (
    RUNNER_SOURCE,
    "scripts/felipe_paper_models.py",
    "scripts/prediction_io.py",
)
# Contract fields recorded but never used to accept a cell on resume.
NON_ACCEPTANCE_FIELDS = ("fitted_selection", "execution")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Explicit path to a prepared, contamination-audited feature manifest.",
    )
    parser.add_argument("--project-dir", default="lncRNA_CIBCB2025-main")
    parser.add_argument(
        "--models", nargs="+", choices=["rflda", "ipcarf", "rf"],
        default=["rflda", "ipcarf", "rf"],
    )
    parser.add_argument("--versions", nargs="+", choices=["v1", "v2"], default=None)
    parser.add_argument(
        "--contains", default=None,
        help="Optional substring filter over feature_set.",
    )
    parser.add_argument(
        "--feature-keys", nargs="+", default=None,
        help="Optional exact filter over feature_key (sanitized feature set).",
    )
    parser.add_argument(
        "--label-spaces", nargs="+", choices=sorted(LABEL_SPACE_SIZES), default=None,
        help="Optional exact filter over the manifest label_space column.",
    )
    parser.add_argument("--n-splits", type=int, default=10)
    fold_selection = parser.add_mutually_exclusive_group()
    fold_selection.add_argument(
        "--max-folds",
        type=int,
        default=None,
        help="Run only the first N outer folds while retaining the original split definition. "
             "Intended for controlled probes; omit for the final full run.",
    )
    fold_selection.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit 1-based outer folds to run (e.g. --folds 1 2 3), so folds can be "
             "split across processes. Mutually exclusive with --max-folds.",
    )
    parser.add_argument(
        "--fold-workers",
        type=int,
        default=1,
        help="Number of CV folds to fit concurrently.",
    )
    parser.add_argument(
        "--model-n-jobs",
        type=int,
        default=None,
        help="CPU workers per model. Defaults to all cores for one fold, or "
             "CPU count divided by --fold-workers for parallel folds.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid existing fold predictions in the output directory. In the "
             "paper profile, complete cells are reused and invalid cells are moved to "
             "_rejected/ and refit.",
    )
    parser.add_argument(
        "--threshold-mode", choices=["youden", "fixed", "train_oob_f1"], default=None,
        help="Current profile only (default 'fixed'). Thresholding for the thresholded "
             "metrics. 'youden' is the legacy (test-fit) cut; 'fixed' uses 0.5; "
             "'train_oob_f1' learns one global micro-F1 threshold from Random Forest "
             "out-of-bag training scores. Rejected with --model-profile paper, which "
             "writes raw scores only.",
    )
    parser.add_argument(
        "--model-profile",
        choices=["current", "paper"],
        default="current",
        help="'current' preserves the stabilized local model behavior. 'paper' "
             "uses unscaled IPCARF without PCA substitution and RFLDA's published "
             "five-fold inner search with the authors' subset-accuracy objective.",
    )
    parser.add_argument(
        "--frozen-sources",
        default="config/felipe_frozen_sources.csv",
        help="Paper profile: path,sha256 table of pinned sources. The run refuses to "
             "start if any listed file differs.",
    )
    parser.add_argument("--outdir", default=f"{RESULTS_ROOT}/within_version_cv")
    args = parser.parse_args(argv)
    if args.model_profile == "paper":
        if args.threshold_mode is not None:
            parser.error(
                "--threshold-mode cannot be used with --model-profile paper: the paper "
                "profile writes raw scores only; thresholded metrics come from "
                "scripts/rescore_felipe_predictions.py."
            )
    elif args.threshold_mode is None:
        args.threshold_mode = "fixed"
    if args.n_splits < 1:
        parser.error("--n-splits must be positive.")
    if args.max_folds is not None and not 1 <= args.max_folds <= args.n_splits:
        parser.error("--max-folds must be between 1 and --n-splits.")
    if args.folds is not None:
        invalid = [fold for fold in args.folds if not 1 <= fold <= args.n_splits]
        if invalid:
            parser.error(f"--folds must be between 1 and --n-splits; got {invalid}.")
        if len(set(args.folds)) != len(args.folds):
            parser.error("--folds lists a fold twice.")
    return args


def selected_fold_indexes(args: argparse.Namespace) -> list[int]:
    """Return the 0-based outer folds selected by --max-folds / --folds."""
    if args.folds is not None:
        return sorted(fold - 1 for fold in args.folds)
    folds = list(range(args.n_splits))
    if args.max_folds is not None:
        folds = folds[: args.max_folds]
    return folds


def resolve_model_n_jobs(args: argparse.Namespace) -> tuple[int, int]:
    fold_workers = max(1, args.fold_workers)
    if args.model_n_jobs is not None:
        return fold_workers, args.model_n_jobs
    if fold_workers == 1:
        return fold_workers, -1
    return fold_workers, max(1, (os.cpu_count() or 1) // fold_workers)


def filter_manifest(manifest: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.versions:
        manifest = manifest[manifest["version"].isin(args.versions)]
    if args.contains:
        manifest = manifest[
            manifest["feature_set"].str.contains(args.contains, case=False, na=False)
        ]
    if args.feature_keys:
        manifest = manifest[manifest["feature_key"].isin(args.feature_keys)]
    if args.label_spaces:
        if "label_space" not in manifest.columns:
            raise ValueError("--label-spaces requires a manifest with a label_space column.")
        manifest = manifest[manifest["label_space"].isin(args.label_spaces)]
    return manifest


def repo_relative(value: object) -> str:
    """Return a manifest path as repo-relative POSIX when it lies inside the repo."""
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def assert_same_ids(x_part: pd.DataFrame, y_part: pd.DataFrame, *, what: str) -> None:
    x_ids = [str(item) for item in x_part.index]
    y_ids = [str(item) for item in y_part.index]
    if x_ids != y_ids:
        raise ValueError(f"{what}: X and Y RNA IDs differ; refusing positional alignment.")


def parse_mean_std(value: str) -> tuple[float, float]:
    left, right = value.rsplit("(", 1)
    return float(left), float(right.rstrip(")"))


def write_run_configs(
    outdir: Path,
    run_config: dict[str, object],
    *,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """Write a compatibility pointer plus a non-overwriting provenance record."""
    config_json = json.dumps(run_config, indent=2)
    latest_config_path = outdir / "run_config.json"
    latest_config_path.write_text(config_json)
    run_configs_dir = outdir / "run_configs"
    run_configs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    config_digest = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:12]
    immutable_config_path = run_configs_dir / f"{timestamp}_{config_digest}.json"
    with immutable_config_path.open("x", encoding="utf-8") as handle:
        handle.write(config_json)
    return latest_config_path, immutable_config_path


def ensure_prediction_contract(
    path: Path,
    contract: dict[str, object],
    *,
    resume: bool,
) -> None:
    """Prevent predictions from different inputs or model code sharing a directory."""
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                f"Prediction contract mismatch at {path}. Use a new output directory "
                "instead of mixing runs."
            )
        return
    if resume:
        raise ValueError(
            f"Cannot resume predictions without a provenance contract: {path}."
        )
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")


def select_global_f1_threshold(
    y_true: pd.DataFrame | np.ndarray,
    y_score: pd.DataFrame | np.ndarray,
    *,
    default: float = 0.5,
) -> float:
    """Choose one micro-F1 threshold using training-only scores.

    A single global threshold is intentionally used instead of one threshold per
    label: many disease labels are sparse, so per-label calibration would be
    unstable and would add a large number of fitted hyperparameters.
    """
    truth = np.asarray(y_true).reshape(-1)
    score = np.asarray(y_score).reshape(-1)
    if truth.shape != score.shape:
        raise ValueError("Threshold calibration requires equal truth/score shapes.")
    finite = np.isfinite(score)
    truth = truth[finite]
    score = score[finite]
    if truth.size == 0 or np.unique(truth).size < 2:
        return float(default)
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Threshold calibration requires binary targets.")

    precision, recall, thresholds = precision_recall_curve(truth, score)
    if thresholds.size == 0:
        return float(default)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > 0,
    )
    if not np.isfinite(f1).any():
        return float(default)
    best = np.flatnonzero(f1 == np.nanmax(f1))
    # Prefer the largest equally optimal threshold to avoid gratuitous positives.
    return float(np.clip(thresholds[int(best[-1])], 0.0, 1.0))


def build_model_factory(
    model_key: str,
    *,
    oob_score: bool = False,
    model_profile: str = "current",
):
    if model_key == "rflda":
        from RFLDA.rflda import RFLDA
        if model_profile == "paper":
            return lambda: RFLDA(
                binary_mode=False,
                n_folds=5,
                selection_metric="subset_accuracy",
            )
        return lambda: RFLDA(binary_mode=False)
    if model_key == "ipcarf":
        from IPCARF.ipcarf import IPCARF
        if model_profile == "paper":
            return lambda: IPCARF(
                binary_mode=False,
                standardize=False,
                allow_pca_fallback=False,
            )
        return lambda: IPCARF(binary_mode=False)
    from RF.rf import RF
    return lambda: RF(binary_mode=False, oob_score=oob_score)


def _prepend_project_paths(project_dir: Path) -> None:
    # Put professor-code imports before PYTHONPATH=mainfolder so its top-level
    # `utils` package is not shadowed by mainfolder/utils.
    for path in (project_dir / "parse_results", project_dir):
        entry = path.as_posix()
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


@dataclass
class PaperCell:
    """One (version, label_space, feature_key, model, fold) prediction cell."""

    cell_dir: Path
    stem: str
    model_key: str
    fold: int  # 0-based
    contract: dict[str, object]
    expected: dict[str, object]
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame
    status: str = "missing"
    outcome: str = ""
    quarantined_to: str | None = None
    fitted_selection: dict[str, object] = field(default_factory=dict)


def _expected_fields(contract: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in contract.items() if key not in NON_ACCEPTANCE_FIELDS
    }


def plan_paper_cells(
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    *,
    manifest_sha256: str,
    code_hashes: dict[str, str],
    outdir: Path,
    model_n_jobs: int,
    iterator_cross_validation,
) -> list[PaperCell]:
    """Validate every selected manifest row and build the cells, fitting nothing."""
    if "label_space" not in manifest.columns:
        raise ValueError(
            "The paper profile requires a manifest label_space column "
            f"({'/'.join(sorted(LABEL_SPACE_SIZES))})."
        )
    library_versions = prediction_io.library_versions()
    selected_folds = selected_fold_indexes(args)
    cells: list[PaperCell] = []
    for _, row in manifest.iterrows():
        version = str(row["version"])
        feature_key = str(row["feature_key"])
        label_space = row["label_space"]
        if pd.isna(label_space) or str(label_space).strip() not in LABEL_SPACE_SIZES:
            raise ValueError(
                f"Manifest row {version}/{feature_key} has label_space={label_space!r}; "
                f"expected one of {sorted(LABEL_SPACE_SIZES)}."
            )
        label_space = str(label_space).strip()
        validated = validate_manifest_row(
            row.to_dict(),
            project_root=PROJECT_ROOT,
            n_splits=args.n_splits,
        )
        x = validated.x
        y = validated.y
        n_labels = int(validated.metadata["n_labels"])
        if n_labels != LABEL_SPACE_SIZES[label_space]:
            raise ValueError(
                f"{version}/{feature_key}: label_space {label_space} requires "
                f"{LABEL_SPACE_SIZES[label_space]} labels, Y has {n_labels}."
            )
        label_names = [str(column) for column in y.columns]
        train, test = iterator_cross_validation(
            validated.splits, x, y, n_folds=args.n_splits
        )
        input_hashes = {
            "x": validated.metadata["x_sha256"],
            "y": validated.metadata["y_sha256"],
            "split": validated.metadata["split_sha256"],
            "manifest": manifest_sha256,
        }
        run_dir = outdir / "within" / f"{version}__{label_space}__{feature_key}__paper"
        for model_key in args.models:
            probe = felipe_paper_models.build_paper_model(model_key, n_jobs=model_n_jobs)
            params = felipe_paper_models.effective_params(model_key, probe)
            for fold in selected_folds:
                x_train, y_train = train[fold]
                x_test, y_test = test[fold]
                assert_same_ids(x_train, y_train, what=f"{version} fold {fold + 1} train")
                assert_same_ids(x_test, y_test, what=f"{version} fold {fold + 1} test")
                if [str(column) for column in y_test.columns] != label_names:
                    raise ValueError("Fold Y columns differ from the manifest Y order.")
                contract: dict[str, object] = {
                    "kind": "within",
                    "version": version,
                    "label_space": label_space,
                    "feature_key": feature_key,
                    "model": model_key,
                    "model_profile": "paper",
                    "effective_params": params,
                    "seed": 0,
                    "fold": fold + 1,
                    "n_folds_total": args.n_splits,
                    "truth_y_path": repo_relative(row["y_path"]),
                    "truth_y_sha256": validated.metadata["y_sha256"],
                    "input_hashes": input_hashes,
                    "code_hashes": code_hashes,
                    "library_versions": library_versions,
                }
                cells.append(
                    PaperCell(
                        cell_dir=run_dir / model_key,
                        stem=f"fold{fold + 1:02d}",
                        model_key=model_key,
                        fold=fold,
                        contract=contract,
                        expected=_expected_fields(contract),
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                    )
                )
    return cells


def fit_paper_cell(cell: PaperCell, *, model_n_jobs: int, fold_workers: int) -> PaperCell:
    """Fit one fold, predict without labels, and write its cell atomically."""
    model = felipe_paper_models.build_paper_model(cell.model_key, n_jobs=model_n_jobs)
    params = felipe_paper_models.effective_params(cell.model_key, model)
    if params != cell.contract["effective_params"]:
        raise RuntimeError(
            f"{cell.cell_dir}/{cell.stem}: model parameters changed after planning."
        )
    model.fit(cell.x_train, cell.y_train)
    selection = felipe_paper_models.fitted_selection(cell.model_key, model)
    scores = np.asarray(model.predict_proba(cell.x_test), dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    expected_shape = (len(cell.y_test.index), len(cell.y_test.columns))
    if scores.shape != expected_shape:
        raise ValueError(
            f"{cell.cell_dir}/{cell.stem}: predictions have shape {scores.shape}, "
            f"expected {expected_shape}."
        )
    # Rows are the test fold in split order; X and Y test IDs were asserted equal.
    predictions = pd.DataFrame(
        scores,
        index=pd.Index([str(item) for item in cell.x_test.index], name="ID"),
        columns=[str(column) for column in cell.y_test.columns],
    )
    contract = dict(cell.contract)
    contract["fitted_selection"] = selection
    contract["execution"] = {
        "n_jobs": model_n_jobs,
        "fold_workers": fold_workers,
        "hostname": socket.gethostname(),
    }
    prediction_io.write_prediction_cell(cell.cell_dir, cell.stem, predictions, contract)
    # Read back through the same checks a resume or verifier would apply.
    prediction_io.read_prediction_cell(cell.cell_dir, cell.stem, cell.expected)
    cell.fitted_selection = selection
    cell.outcome = "fitted"
    return cell


def run_paper_profile(
    args: argparse.Namespace,
    *,
    argv: list[str] | None,
    manifest_path: Path,
    project_dir: Path,
    outdir: Path,
) -> None:
    """Paper profile: write raw per-fold prediction cells; compute no metrics."""
    frozen_path = (PROJECT_ROOT / args.frozen_sources).resolve()
    code_hashes = prediction_io.verify_frozen_sources(frozen_path, PROJECT_ROOT)
    required = list(REQUIRED_FROZEN_SOURCES) + [
        felipe_paper_models.MODEL_SOURCES[model_key] for model_key in args.models
    ]
    unpinned = [path for path in required if path not in code_hashes]
    if unpinned:
        raise prediction_io.FrozenSourceError(
            f"{frozen_path} does not pin required sources: {unpinned}; refusing to run."
        )

    _prepend_project_paths(project_dir)
    from utils.utils import iterator_cross_validation

    manifest_sha256 = sha256_file(manifest_path)
    manifest = filter_manifest(pd.read_csv(manifest_path), args)
    if manifest.empty:
        raise ValueError("No manifest rows match the requested filters.")
    fold_workers, model_n_jobs = resolve_model_n_jobs(args)
    cells = plan_paper_cells(
        args,
        manifest,
        manifest_sha256=manifest_sha256,
        code_hashes=code_hashes,
        outdir=outdir,
        model_n_jobs=model_n_jobs,
        iterator_cross_validation=iterator_cross_validation,
    )

    # Decide every cell's fate before fitting anything.
    existing = [
        cell for cell in cells
        if any(path.exists() for path in prediction_io.cell_paths(cell.cell_dir, cell.stem).values())
    ]
    if existing and not args.resume:
        listing = ", ".join(f"{cell.cell_dir}/{cell.stem}" for cell in existing[:5])
        raise FileExistsError(
            f"{len(existing)} selected prediction cell(s) already exist (e.g. {listing}); "
            "use --resume or a fresh --outdir."
        )
    for cell in cells:
        cell.status = prediction_io.cell_status(cell.cell_dir, cell.stem, cell.expected)

    run_config: dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv[1:] if argv is None else argv),
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": manifest_sha256,
        "frozen_sources": frozen_path.as_posix(),
        "code_hashes": code_hashes,
        "models": args.models,
        "model_profile": "paper",
        "versions": args.versions,
        "label_spaces": args.label_spaces,
        "contains": args.contains,
        "feature_keys": args.feature_keys,
        "n_splits": args.n_splits,
        "max_folds": args.max_folds,
        "folds": args.folds,
        "fold_workers": fold_workers,
        "model_n_jobs": model_n_jobs,
        "resume": args.resume,
        "random_state": 0,
        "threshold_mode": None,
        "metrics_written": False,
        "outdir": outdir.as_posix(),
        "completed": False,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        to_fit: list[PaperCell] = []
        for cell in cells:
            if cell.status == "complete":
                cell.outcome = "reused"
                print(
                    f"  {cell.cell_dir.relative_to(outdir)}/{cell.stem} (reused)", flush=True
                )
                continue
            if cell.status == "invalid":
                moved = prediction_io.quarantine_cell(cell.cell_dir, cell.stem)
                cell.quarantined_to = None if moved is None else moved.as_posix()
                print(
                    f"  {cell.cell_dir.relative_to(outdir)}/{cell.stem} invalid; "
                    f"moved to {moved}",
                    flush=True,
                )
            to_fit.append(cell)

        def report(cell: PaperCell) -> None:
            print(f"  {cell.cell_dir.relative_to(outdir)}/{cell.stem} (fitted)", flush=True)

        if fold_workers == 1:
            for cell in to_fit:
                report(fit_paper_cell(cell, model_n_jobs=model_n_jobs, fold_workers=1))
        else:
            with ThreadPoolExecutor(max_workers=fold_workers) as executor:
                futures = [
                    executor.submit(
                        fit_paper_cell,
                        cell,
                        model_n_jobs=model_n_jobs,
                        fold_workers=fold_workers,
                    )
                    for cell in to_fit
                ]
                try:
                    for future in as_completed(futures):
                        report(future.result())
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        run_config["completed"] = True
    finally:
        run_config["cells"] = [
            {
                "dir": cell.cell_dir.as_posix(),
                "stem": cell.stem,
                "version": cell.contract["version"],
                "label_space": cell.contract["label_space"],
                "feature_key": cell.contract["feature_key"],
                "model": cell.model_key,
                "fold": cell.fold + 1,
                "status_at_start": cell.status,
                "outcome": cell.outcome or "not_run",
                "quarantined_to": cell.quarantined_to,
                "fitted_selection": cell.fitted_selection,
            }
            for cell in cells
        ]
        latest_config_path, immutable_config_path = write_run_configs(outdir, run_config)
        print(f"Saved latest run config: {latest_config_path}")
        print(f"Saved immutable run config: {immutable_config_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    project_dir = PROJECT_ROOT / args.project_dir
    outdir = PROJECT_ROOT / args.outdir
    if args.model_profile == "paper":
        run_paper_profile(
            args,
            argv=argv,
            manifest_path=manifest_path,
            project_dir=project_dir,
            outdir=outdir,
        )
        return

    (outdir / "predictions").mkdir(parents=True, exist_ok=True)
    (outdir / "performance").mkdir(parents=True, exist_ok=True)
    _prepend_project_paths(project_dir)

    from parse_results.evaluate import Evaluator
    from utils.utils import iterator_cross_validation

    manifest = filter_manifest(pd.read_csv(manifest_path), args)

    summary_rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        version = row["version"]
        feature_key = row["feature_key"]
        validated = validate_manifest_row(
            row.to_dict(),
            project_root=PROJECT_ROOT,
            n_splits=args.n_splits,
        )
        x = validated.x
        y = validated.y
        splits = validated.splits
        train, test = iterator_cross_validation(splits, x, y, n_folds=args.n_splits)
        selected_folds = selected_fold_indexes(args)

        for model_key in args.models:
            if args.threshold_mode == "train_oob_f1" and model_key != "rf":
                raise ValueError(
                    "--threshold-mode train_oob_f1 currently supports only --models rf."
                )
            uses_oob_threshold = args.threshold_mode == "train_oob_f1"
            factory = build_model_factory(
                model_key,
                oob_score=uses_oob_threshold,
                model_profile=args.model_profile,
            )
            evaluator = Evaluator(
                threshold_mode="fixed" if uses_oob_threshold else args.threshold_mode
            )
            experiment_key = f"{version}_{feature_key}"
            if args.model_profile != "current":
                experiment_key = f"{experiment_key}_{args.model_profile}"
            pred_dir = outdir / "predictions" / experiment_key / model_key
            pred_dir.mkdir(parents=True, exist_ok=True)

            model_source = project_dir / {
                "rf": "RF/rf.py",
                "ipcarf": "IPCARF/ipcarf.py",
                "rflda": "RFLDA/rflda.py",
            }[model_key]
            prediction_contract = {
                "version": version,
                "feature_key": feature_key,
                "model": model_key,
                "model_profile": args.model_profile,
                "n_splits": args.n_splits,
                "x_sha256": validated.metadata["x_sha256"],
                "y_sha256": validated.metadata["y_sha256"],
                "split_sha256": validated.metadata["split_sha256"],
                "model_source_sha256": sha256_file(model_source),
            }
            ensure_prediction_contract(
                pred_dir / "prediction_contract.json",
                prediction_contract,
                resume=args.resume,
            )

            fold_workers, model_n_jobs = resolve_model_n_jobs(args)

            def load_existing_prediction(fold: int, y_test: pd.DataFrame):
                pred_path = pred_dir / f"test_fold{fold + 1}.csv"
                # OOB calibration is fold-specific. A prediction alone is not
                # enough to reproduce the fitted threshold, so refit this mode.
                if uses_oob_threshold or not args.resume or not pred_path.exists():
                    return None
                pred = pd.read_csv(pred_path, index_col=0)
                pred.index = pred.index.astype(str)
                y_index = y_test.index.astype(str)
                if list(pred.columns) != list(y_test.columns) or set(pred.index) != set(y_index):
                    return None
                return pred.reindex(y_index)

            def fit_fold(fold: int):
                x_train, y_train = train[fold]
                x_test, y_test = test[fold]
                existing = load_existing_prediction(fold, y_test)
                if existing is not None:
                    return fold, evaluator.evaluate(y_test, existing), None, True, np.nan
                model = factory()
                if hasattr(model, "n_jobs"):
                    model.n_jobs = model_n_jobs
                model.fit(x_train, y_train)
                pred = pd.DataFrame(
                    model.predict_proba(x_test, y_test),
                    columns=y_test.columns,
                    index=y_test.index,
                )
                pred.to_csv(pred_dir / f"test_fold{fold + 1}.csv")
                if uses_oob_threshold:
                    threshold = select_global_f1_threshold(
                        y_train, model.get_oob_predictions()
                    )
                    fold_evaluator = Evaluator(
                        threshold=threshold,
                        threshold_mode="fixed",
                    )
                    fixed_baseline_perf = Evaluator(
                        threshold=0.5,
                        threshold_mode="fixed",
                    ).evaluate(y_test, pred)
                else:
                    threshold = 0.5 if args.threshold_mode == "fixed" else np.nan
                    fold_evaluator = evaluator
                    fixed_baseline_perf = None
                return (
                    fold,
                    fold_evaluator.evaluate(y_test, pred),
                    fixed_baseline_perf,
                    False,
                    threshold,
                )

            perf_by_fold: dict[int, tuple[float, ...]] = {}
            fixed_perf_by_fold: dict[int, tuple[float, ...]] = {}
            threshold_by_fold: dict[int, float] = {}
            if fold_workers == 1:
                fold_results = (fit_fold(fold) for fold in selected_folds)
            else:
                executor = ThreadPoolExecutor(max_workers=fold_workers)
                futures = {
                    executor.submit(fit_fold, fold): fold
                    for fold in selected_folds
                }
                fold_results = (future.result() for future in as_completed(futures))

            try:
                for fold, fold_perf, fixed_fold_perf, reused, threshold in fold_results:
                    perf_by_fold[fold] = fold_perf
                    if fixed_fold_perf is not None:
                        fixed_perf_by_fold[fold] = fixed_fold_perf
                    threshold_by_fold[fold] = threshold
                    status = "reused" if reused else "fitted"
                    print(
                        f"  {version}/{feature_key} | {model_key} | "
                        f"fold {fold + 1:02d} ({status})",
                        flush=True,
                    )
            finally:
                if fold_workers > 1:
                    executor.shutdown(wait=True, cancel_futures=True)

            perf = [perf_by_fold[fold] for fold in selected_folds]

            perf_df = evaluator.create_df_inductive(
                perf,
                fold_labels=[f"fold{fold + 1}" for fold in selected_folds],
            )
            # threshold_mode is part of the name so fixed and youden runs of the
            # same experiment cannot overwrite each other.
            perf_path = (
                outdir / "performance"
                / f"{experiment_key}_{model_key}_{args.threshold_mode}_performance.csv"
            )
            perf_df.to_csv(perf_path, index=False)

            fixed_baseline_mean_row = None
            if uses_oob_threshold:
                calibration_path = (
                    outdir / "performance"
                    / f"{experiment_key}_{model_key}_oob_thresholds.csv"
                )
                pd.DataFrame(
                    {
                        "fold": [fold + 1 for fold in selected_folds],
                        "threshold": [
                            threshold_by_fold[fold] for fold in selected_folds
                        ],
                        "calibration_source": "training_fold_rf_oob_predictions",
                    }
                ).to_csv(calibration_path, index=False)
                fixed_perf = [fixed_perf_by_fold[fold] for fold in selected_folds]
                fixed_baseline_df = evaluator.create_df_inductive(
                    fixed_perf,
                    fold_labels=[f"fold{fold + 1}" for fold in selected_folds],
                )
                fixed_baseline_path = (
                    outdir / "performance"
                    / f"{experiment_key}_{model_key}_fixed_baseline_performance.csv"
                )
                fixed_baseline_df.to_csv(fixed_baseline_path, index=False)
                fixed_baseline_mean_row = fixed_baseline_df[
                    fixed_baseline_df["folds"] == "mean(std)"
                ].iloc[0].to_dict()

            mean_row = perf_df[perf_df["folds"] == "mean(std)"].iloc[0].to_dict()
            summary_row: dict[str, object] = {
                "experiment": f"within_{version}_cv",
                "version": version,
                "source_version": version,
                "target_version": version,
                "feature_set": row["feature_set"],
                "feature_key": feature_key,
                "model": model_key,
                "threshold_mode": args.threshold_mode,
                "model_profile": args.model_profile,
                "n_samples": int(validated.metadata["n_samples"]),
                "n_features": int(validated.metadata["n_features"]),
                "n_labels": int(validated.metadata["n_labels"]),
                "x_sha256": validated.metadata["x_sha256"],
                "y_sha256": validated.metadata["y_sha256"],
                "split_sha256": validated.metadata["split_sha256"],
                "model_source_sha256": prediction_contract["model_source_sha256"],
                "completed_folds": len(selected_folds),
                "performance_csv": perf_path.as_posix(),
                "prediction_dir": pred_dir.as_posix(),
            }
            for metric in METRICS:
                summary_row[f"{metric}_meanstd"] = mean_row[metric]
                mean_value, std_value = parse_mean_std(mean_row[metric])
                summary_row[f"{metric}_mean"] = mean_value
                summary_row[f"{metric}_std"] = std_value
                if fixed_baseline_mean_row is not None:
                    fixed_mean, fixed_std = parse_mean_std(
                        fixed_baseline_mean_row[metric]
                    )
                    summary_row[f"fixed_baseline_{metric}_mean"] = fixed_mean
                    summary_row[f"fixed_baseline_{metric}_std"] = fixed_std
            summary_rows.append(summary_row)
            print(f"Saved {perf_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "within_version_cv_summary.csv"
    # Merge cumulatively: a per-version invocation must not clobber rows from a
    # previous run on the other version (or other feature sets in Stage B).
    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        if not summary_df.empty:
            if "model_profile" not in existing.columns:
                existing["model_profile"] = "current"
            if "threshold_mode" not in existing.columns:
                existing["threshold_mode"] = "unknown"
            key = ["version", "feature_key", "model", "model_profile", "threshold_mode"]
            done = set(map(tuple, summary_df[key].itertuples(index=False, name=None)))
            existing = existing[
                ~existing[key].apply(tuple, axis=1).isin(done)
            ]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            ["version", "feature_set", "model_profile", "model"], kind="stable"
        )
    summary_df.to_csv(summary_path, index=False)

    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path.as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "models": args.models,
        "model_profile": args.model_profile,
        "versions": args.versions,
        "contains": args.contains,
        "feature_keys": args.feature_keys,
        "n_splits": args.n_splits,
        "max_folds": args.max_folds,
        "folds": args.folds,
        "fold_workers": max(1, args.fold_workers),
        "model_n_jobs": args.model_n_jobs,
        "resume": args.resume,
        "random_state": 0,
        "model_n_estimators": 150,
        "threshold_mode": args.threshold_mode,
        "outdir": outdir.as_posix(),
        "experiments": summary_rows,
    }
    latest_config_path, immutable_config_path = write_run_configs(outdir, run_config)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved latest run config: {latest_config_path}")
    print(f"Saved immutable run config: {immutable_config_path}")


if __name__ == "__main__":
    main()
