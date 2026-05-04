from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def find_project_root(marker_rel: Path = Path("Data/output_data/website_full_matrix.csv")) -> Path:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        if (base / marker_rel).exists():
            return base
    raise FileNotFoundError(f"Could not locate project root containing {marker_rel}")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mainfolder.utils.inductive_models import (  # noqa: E402
    build_rflda_param_grid,
    make_binary_dataset,
    make_multilabel_dataset,
    run_nested_cv,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run nested-CV inductive models for lncRNA-disease prediction. "
            "Supports multi-label structured output and binary classification."
        )
    )
    p.add_argument("--task", choices=["multilabel", "binary"], default="multilabel")
    p.add_argument("--dataset-name", required=True, help="Short dataset label used in saved reports")
    p.add_argument("--x", nargs="+", required=True, help="One or more feature CSV paths")
    p.add_argument("--y", required=True, help="Label CSV path")
    p.add_argument("--models", nargs="+", choices=["rflda", "ipcarf"], default=["rflda", "ipcarf"])
    p.add_argument("--binary-label-col", help="For binary task: explicit label column name in --y")
    p.add_argument("--outdir", default="final_output/inductive_models", help="Directory to save outputs")
    p.add_argument("--outer-splits", type=int, default=10)
    p.add_argument("--inner-splits", type=int, default=5)
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--class-weight", default="balanced", help="RF class_weight; use 'none' to disable")
    p.add_argument(
        "--threshold-mode",
        choices=["per-label", "global"],
        default="per-label",
        help="For multi-label task: per-label or one global Youden threshold",
    )
    p.add_argument(
        "--rflda-step",
        type=int,
        default=50,
        help="RFLDA grid: max_features in steps of this size up to total feature count",
    )
    p.add_argument(
        "--ipca-components",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32, 64, 128],
        help="IPCARF grid: IncrementalPCA component counts",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    class_weight = None if str(args.class_weight).lower() == "none" else args.class_weight

    if args.task == "binary":
        bundle = make_binary_dataset(args.x, args.y, label_col=args.binary_label_col)
    else:
        bundle = make_multilabel_dataset(args.x, args.y)

    print(
        f"[info] dataset={args.dataset_name} task={args.task} "
        f"samples={len(bundle.ids)} features={bundle.X.shape[1]} "
        f"labels={bundle.Y.shape[1] if args.task != 'binary' else 1} "
        f"positive_rate={bundle.positive_rate:.6f}"
    )

    all_summary = []
    all_fold_metrics = []
    all_inner = []
    all_thresholds = []
    dataset_info_written = False

    for model_name in args.models:
        if model_name == "rflda":
            param_grid = build_rflda_param_grid(bundle.X.shape[1], step=args.rflda_step)
            print(f"[info] model=rflda grid=max_features {param_grid}")
        else:
            param_grid = sorted(set(int(v) for v in args.ipca_components if int(v) >= 1 and int(v) <= bundle.X.shape[1]))
            if not param_grid:
                raise ValueError("IPCARF grid became empty after filtering against the number of features.")
            print(f"[info] model=ipcarf grid=n_components {param_grid}")

        reports = run_nested_cv(
            dataset_name=args.dataset_name,
            task=args.task,
            model_name=model_name,
            bundle=bundle,
            param_grid=param_grid,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            class_weight=class_weight,
            threshold_mode=args.threshold_mode,
        )

        summary_path = outdir / f"{args.dataset_name}_{model_name}_summary.csv"
        folds_path = outdir / f"{args.dataset_name}_{model_name}_fold_metrics.csv"
        inner_path = outdir / f"{args.dataset_name}_{model_name}_inner_search.csv"
        thresholds_path = outdir / f"{args.dataset_name}_{model_name}_thresholds.csv"
        info_path = outdir / f"{args.dataset_name}_dataset_info.csv"

        reports["summary"].to_csv(summary_path, index=False)
        reports["fold_metrics"].to_csv(folds_path, index=False)
        reports["inner_search"].to_csv(inner_path, index=False)
        reports["thresholds"].to_csv(thresholds_path, index=False)
        if not dataset_info_written:
            reports["dataset_info"].to_csv(info_path, index=False)
            dataset_info_written = True

        all_summary.append(reports["summary"])
        all_fold_metrics.append(reports["fold_metrics"])
        all_inner.append(reports["inner_search"])
        all_thresholds.append(reports["thresholds"])

        print(f"[saved] {summary_path}")
        print(f"[saved] {folds_path}")
        print(f"[saved] {inner_path}")
        print(f"[saved] {thresholds_path}")

    summary_all = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    folds_all = pd.concat(all_fold_metrics, ignore_index=True) if all_fold_metrics else pd.DataFrame()
    inner_all = pd.concat(all_inner, ignore_index=True) if all_inner else pd.DataFrame()
    thresholds_all = pd.concat(all_thresholds, ignore_index=True) if all_thresholds else pd.DataFrame()

    all_summary_path = outdir / f"{args.dataset_name}_all_models_summary.csv"
    all_folds_path = outdir / f"{args.dataset_name}_all_models_fold_metrics.csv"
    all_inner_path = outdir / f"{args.dataset_name}_all_models_inner_search.csv"
    all_thresholds_path = outdir / f"{args.dataset_name}_all_models_thresholds.csv"

    summary_all.to_csv(all_summary_path, index=False)
    folds_all.to_csv(all_folds_path, index=False)
    inner_all.to_csv(all_inner_path, index=False)
    thresholds_all.to_csv(all_thresholds_path, index=False)

    print(f"[saved] {all_summary_path}")
    print(f"[saved] {all_folds_path}")
    print(f"[saved] {all_inner_path}")
    print(f"[saved] {all_thresholds_path}")

    if not summary_all.empty:
        printable = summary_all.copy()
        printable["mean±std"] = printable.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
        print("\n=== Summary ===")
        print(printable[["dataset", "model", "metric", "mean±std"]].to_string(index=False))


if __name__ == "__main__":
    main()
