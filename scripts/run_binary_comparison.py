"""Binary (pairwise) model comparison alongside the structured-output runs.

In binary mode each lncRNA-disease pair is one example: the lncRNA's sequence features
are concatenated with an ontology-based disease-similarity vector for the disease, and
the model predicts the single association bit. This mirrors the original IPCARF/RFLDA
formulation. The disease-similarity matrix (Wang/BMA, derived from the disease ontology,
NOT from the association matrix) is leakage-free.

Predictions are unflattened back to (n_test_lncRNA x n_labels) and scored with the same
inductive Evaluator, so binary results sit in the same metric space as the
structured-output results.

Outputs (under --outdir, default results/binary_comparison):
    predictions/<version>_<feature_key>/<model>/test_fold{1..}.csv
    performance/<version>_<feature_key>_<model>_performance.csv
    binary_comparison_summary.csv
    run_config.json

NOTE: binary mode expands the row count to n_lncRNA x n_disease. Use --max-folds for a
cheap validation run; full 10-fold runs belong on the server.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", "results")
METRICS = ["hamming", "label_ranking", "micro_roc", "micro_auprc",
           "precision", "recall", "fscore", "accuracy"]

DISEASE_SIM_FILENAME = "disease_similarity_bma.csv"
VERSION_DIRS = {
    "v1": ("v1", "V1"),
    "v2": ("v2", "V2"),
}
DISEASE_DIRS = ("disease", "Dis")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",
                   default="inductive_inputs/feature_representations/feature_representation_manifest.csv")
    p.add_argument("--project-dir", default="lncRNA_CIBCB2025-main")
    p.add_argument("--models", nargs="+", choices=["rflda", "ipcarf", "rf"],
                   default=["rflda", "ipcarf", "rf"])
    p.add_argument("--versions", nargs="+", choices=["v1", "v2"], default=None)
    p.add_argument("--feature-keys", nargs="+", default=None)
    p.add_argument("--n-splits", type=int, default=10)
    p.add_argument("--max-folds", type=int, default=None,
                   help="Cap folds for a cheap validation run.")
    p.add_argument("--threshold-mode", choices=["youden", "fixed"], default="youden")
    p.add_argument("--outdir", default=f"{RESULTS_ROOT}/binary_comparison")
    return p.parse_args()


def parse_mean_std(value: str) -> tuple[float, float]:
    left, right = value.rsplit("(", 1)
    return float(left), float(right.rstrip(")"))


def resolve_disease_similarity_path(version: str) -> Path:
    """Find the BMA disease-similarity file across historical output layouts."""
    candidates: list[Path] = []
    for root in ("final_output", "Final_output", "final-output"):
        for version_dir in VERSION_DIRS[version]:
            for disease_dir in DISEASE_DIRS:
                candidates.append(
                    PROJECT_ROOT / root / version_dir / disease_dir / DISEASE_SIM_FILENAME
                )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n  ".join(str(p.relative_to(PROJECT_ROOT)) for p in candidates)
    raise FileNotFoundError(
        f"{version}: could not find {DISEASE_SIM_FILENAME}. Tried:\n  {tried}"
    )


def load_disease_similarity(version: str, label_cols: list[str]) -> pd.DataFrame:
    sim_path = resolve_disease_similarity_path(version)
    sim = pd.read_csv(sim_path, index_col=0)
    missing = [c for c in label_cols if c not in sim.index]
    if missing:
        raise ValueError(f"{version}: disease similarity missing {len(missing)} labels, "
                         f"e.g. {missing[:3]} in {sim_path}")
    return sim.loc[label_cols, label_cols]


def build_model_factory(model_key: str, disease_similarities):
    if model_key == "rflda":
        from RFLDA.rflda import RFLDA
        return lambda: RFLDA(binary_mode=True, disease_similarities=disease_similarities)
    if model_key == "ipcarf":
        from IPCARF.ipcarf import IPCARF
        return lambda: IPCARF(binary_mode=True, disease_similarities=disease_similarities)
    from RF.rf import RF
    return lambda: RF(binary_mode=True, disease_similarities=disease_similarities)


def main() -> None:
    args = parse_args()
    project_dir = PROJECT_ROOT / args.project_dir
    outdir = PROJECT_ROOT / args.outdir
    (outdir / "predictions").mkdir(parents=True, exist_ok=True)
    (outdir / "performance").mkdir(parents=True, exist_ok=True)

    # Put professor-code imports before PYTHONPATH=mainfolder so its top-level
    # `utils` package is not shadowed by mainfolder/utils.
    sys.path.insert(0, (project_dir / "parse_results").as_posix())
    sys.path.insert(0, project_dir.as_posix())
    from parse_results.evaluate import Evaluator
    from utils.utils import iterator_cross_validation

    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)
    if args.versions:
        manifest = manifest[manifest["version"].isin(args.versions)]
    if args.feature_keys:
        manifest = manifest[manifest["feature_key"].isin(args.feature_keys)]

    n_folds = args.max_folds or args.n_splits
    summary_rows: list[dict[str, object]] = []

    for _, row in manifest.iterrows():
        version, feature_key = row["version"], row["feature_key"]
        x = pd.read_csv(Path(row["x_path"]), index_col=0)
        y = pd.read_csv(Path(row["y_path"]), index_col=0)
        splits = pd.read_csv(Path(row["split_path"]))
        disease_sim = load_disease_similarity(version, list(y.columns)).to_numpy()
        train, test = iterator_cross_validation(splits, x, y, n_folds=args.n_splits)

        for model_key in args.models:
            factory = build_model_factory(model_key, disease_sim)
            evaluator = Evaluator(threshold_mode=args.threshold_mode)
            pred_dir = outdir / "predictions" / f"{version}_{feature_key}" / model_key
            pred_dir.mkdir(parents=True, exist_ok=True)

            perf = []
            for fold in range(n_folds):
                x_tr, y_tr = train[fold]
                x_te, y_te = test[fold]
                model = factory()
                model.fit(x_tr, y_tr)
                # binary-mode predict_proba returns a DataFrame with its own index;
                # take values before re-wrapping to avoid index-alignment NaNs.
                raw = model.predict_proba(x_te, y_te)
                arr = raw.to_numpy() if isinstance(raw, pd.DataFrame) else raw
                pred = pd.DataFrame(arr, columns=y_te.columns, index=y_te.index)
                pred.to_csv(pred_dir / f"test_fold{fold + 1}.csv")
                perf.append(evaluator.evaluate(y_te, pred))
                print(f"  [binary] {version}/{feature_key} | {model_key} | fold {fold + 1:02d}")

            # pad to 10 rows for the inductive df helper if folds were capped
            df_perf = pd.DataFrame(perf, columns=METRICS)
            df_perf.insert(0, "folds", [f"fold{i+1}" for i in range(len(perf))])
            mean = df_perf[METRICS].mean()
            std = df_perf[METRICS].std(ddof=0)
            mean_row = {m: f"{mean[m]}({std[m]})" for m in METRICS}
            perf_path = outdir / "performance" / f"{version}_{feature_key}_{model_key}_performance.csv"
            df_perf.to_csv(perf_path, index=False)

            summary = {"experiment": f"binary_{version}_cv", "mode": "binary",
                       "version": version, "feature_set": row["feature_set"],
                       "feature_key": feature_key, "model": model_key,
                       "threshold_mode": args.threshold_mode,
                       "n_folds_run": len(perf), "performance_csv": perf_path.as_posix()}
            for m in METRICS:
                summary[f"{m}_meanstd"] = mean_row[m]
                summary[f"{m}_mean"] = float(mean[m])
                summary[f"{m}_std"] = float(std[m])
            summary_rows.append(summary)
            print(f"Saved {perf_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "binary_comparison_summary.csv"
    if summary_path.exists() and not summary_df.empty:
        existing = pd.read_csv(summary_path)
        key = ["version", "feature_key", "model"]
        done = set(map(tuple, summary_df[key].itertuples(index=False, name=None)))
        existing = existing[~existing[key].apply(tuple, axis=1).isin(done)]
        summary_df = pd.concat([existing, summary_df], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    (outdir / "run_config.json").write_text(json.dumps({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "binary", "models": args.models, "versions": args.versions,
        "n_splits": args.n_splits, "max_folds": args.max_folds,
        "threshold_mode": args.threshold_mode, "random_state": 0,
        "disease_similarity_filename": DISEASE_SIM_FILENAME,
    }, indent=2))
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
