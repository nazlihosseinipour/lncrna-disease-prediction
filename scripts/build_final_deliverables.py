"""Tasks 5/6/7 — aggregate every experiment into the final deliverables.

Reads the within-version CV summary and the cross-dataset transfer summary, then
writes a single consistent results tree:

    results/final_comparison.csv            one row per experiment, all 8 metrics
    results/feature_comparison.csv          Task 2 (within-version feature sweep)
    results/model_comparison.csv            Task 3 (RF vs RFLDA vs IPCARF)
    results/cross_dataset_generalization.csv Task 4 (transfer + common-CV)
    results/final_report.md                 Task 6 conclusions (data-driven)
    results/results_summary.md              Task 7 index of every experiment

At the end it prints the results/ tree and an explicit "where everything is saved" list.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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
# (metric, higher_is_better)
METRIC_DIR = {
    "hamming": False,
    "label_ranking": False,
    "micro_roc": True,
    "micro_auprc": True,
    "precision": True,
    "recall": True,
    "fscore": True,
    "accuracy": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--within-dir", default="results/within_version_cv",
        help="Directory holding within_version_cv_summary*.csv files.",
    )
    parser.add_argument(
        "--transfer-summary",
        default="results/transfer/transfer_feature_representation_summary.csv",
    )
    parser.add_argument("--outdir", default="results")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def load_within(within_dir: Path) -> pd.DataFrame:
    """Concatenate every within_version_cv_summary*.csv and de-duplicate."""
    frames = [load_csv(p) for p in sorted(within_dir.glob("within_version_cv_summary*.csv"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["version", "feature_key", "model"], keep="last")
    return df


def to_unified(within: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in within.iterrows():
        row = {
            "experiment": r["experiment"],
            "train_dataset": r["version"],
            "test_dataset": r["version"],
            "evaluation": "within_version_cv",
            "feature_set": r["feature_set"],
            "model": r["model"],
            "n_labels": r["n_labels"],
            "n_train_samples": r["n_samples"],
            "n_test_samples": r["n_samples"],
            "label_match": "",
            "performance_csv": r.get("performance_csv", ""),
        }
        for m in METRICS:
            row[f"{m}_mean"] = r[f"{m}_mean"]
            row[f"{m}_std"] = r[f"{m}_std"]
        rows.append(row)

    for _, r in transfer.iterrows():
        is_transfer = str(r["experiment"]).endswith("_transfer")
        row = {
            "experiment": r["experiment"],
            "train_dataset": r["source_version"] if is_transfer else r["target_version"],
            "test_dataset": r["target_version"],
            "evaluation": "cross_version_transfer" if is_transfer else "target_common_cv",
            "feature_set": r["feature_set"],
            "model": r["model"],
            "n_labels": r["n_labels"],
            "n_train_samples": r.get("n_source_samples", "") if is_transfer else r.get("n_target_samples", ""),
            "n_test_samples": r.get("n_target_samples", ""),
            "label_match": r.get("label_match", ""),
            "performance_csv": r.get("performance_csv", ""),
        }
        for m in METRICS:
            row[f"{m}_mean"] = r[f"{m}_mean"]
            row[f"{m}_std"] = r[f"{m}_std"]
        rows.append(row)

    return pd.DataFrame(rows)


def best_by(df: pd.DataFrame, metric: str) -> pd.Series:
    asc = not METRIC_DIR[metric]
    return df.sort_values(f"{metric}_mean", ascending=asc).iloc[0]


def fmt(df: pd.DataFrame, cols: list[str]) -> str:
    """Render a markdown table without requiring the optional `tabulate` dep."""
    def cell(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(cell(r[c]) for c in cols) + " |"
        for _, r in df[cols].iterrows()
    ]
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    within_dir = PROJECT_ROOT / args.within_dir
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    within = load_within(within_dir)
    transfer = load_csv(PROJECT_ROOT / args.transfer_summary)
    if within.empty and transfer.empty:
        raise FileNotFoundError("No within-version or transfer summaries found.")

    unified = to_unified(within, transfer)
    unified = unified.sort_values(
        ["evaluation", "train_dataset", "test_dataset", "feature_set", "model"],
        kind="stable",
    )
    final_csv = outdir / "final_comparison.csv"
    unified.to_csv(final_csv, index=False)

    # Task 2: feature comparison (within-version only)
    feat_cols = ["train_dataset", "feature_set", "model", "micro_roc_mean",
                 "micro_auprc_mean", "fscore_mean", "hamming_mean",
                 "label_ranking_mean", "accuracy_mean"]
    feature_tbl = unified[unified["evaluation"] == "within_version_cv"][feat_cols]
    feature_csv = outdir / "feature_comparison.csv"
    feature_tbl.to_csv(feature_csv, index=False)

    # Task 3: model comparison (within-version, averaged over feature sets)
    model_tbl = (
        unified[unified["evaluation"] == "within_version_cv"]
        .groupby(["train_dataset", "model"])[[f"{m}_mean" for m in METRICS]]
        .mean()
        .reset_index()
    )
    model_csv = outdir / "model_comparison.csv"
    model_tbl.to_csv(model_csv, index=False)

    # Task 4: cross-dataset generalization
    cross_tbl = unified[unified["evaluation"].isin(["cross_version_transfer", "target_common_cv"])]
    cross_csv = outdir / "cross_dataset_generalization.csv"
    cross_tbl.to_csv(cross_csv, index=False)

    # --- Task 6 final report (data-driven) ---
    lines: list[str] = ["# Task 6 — Final Report\n"]
    within_only = unified[unified["evaluation"] == "within_version_cv"]

    if not within_only.empty:
        lines.append("## Best within-version configurations\n")
        for ds in sorted(within_only["train_dataset"].unique()):
            sub = within_only[within_only["train_dataset"] == ds]
            b = best_by(sub, "micro_auprc")
            lines.append(
                f"- **{ds.upper()}** best by micro-AUPRC: `{b.feature_set}` + "
                f"`{b.model}` (AUPRC={b.micro_auprc_mean:.4f}, AUROC={b.micro_roc_mean:.4f}, "
                f"F1={b.fscore_mean:.4f})"
            )
        lines.append("")
        lines.append("### Feature comparison (within-version)\n")
        lines.append(fmt(feature_tbl.sort_values(
            ["train_dataset", "micro_auprc_mean"], ascending=[True, False]), feat_cols))
        lines.append("")
        lines.append("### Model comparison (mean over feature sets)\n")
        lines.append(fmt(model_tbl, ["train_dataset", "model", "micro_roc_mean",
                                     "micro_auprc_mean", "fscore_mean", "hamming_mean"]))
        lines.append("")

        # Task 5 — V1 vs V2 deltas (matched feature+model)
        piv = within_only.pivot_table(
            index=["feature_set", "model"], columns="train_dataset",
            values=[f"{m}_mean" for m in ["micro_roc", "micro_auprc", "fscore"]],
        )
        if {"v1", "v2"}.issubset(set(within_only["train_dataset"])):
            lines.append("## Task 5 — V1 → V2 change (matched feature+model)\n")
            for m in ["micro_roc", "micro_auprc", "fscore"]:
                col = f"{m}_mean"
                if (col, "v1") in piv.columns and (col, "v2") in piv.columns:
                    d = (piv[(col, "v2")] - piv[(col, "v1")]).mean()
                    lines.append(f"- mean Δ({m}) V2−V1 = {d:+.4f}")
            lines.append("")

    if not cross_tbl.empty:
        lines.append("## Task 4 — Cross-dataset generalization\n")
        lines.append(fmt(cross_tbl.sort_values(["evaluation", "feature_set", "model"]),
                         ["experiment", "train_dataset", "test_dataset", "feature_set",
                          "model", "n_labels", "micro_roc_mean", "micro_auprc_mean",
                          "fscore_mean"]))
        lines.append("")
        tr = cross_tbl[cross_tbl["evaluation"] == "cross_version_transfer"]
        if not tr.empty:
            bt = best_by(tr, "micro_auprc")
            lines.append(
                f"\nBest transfer config (micro-AUPRC): `{bt.feature_set}` + `{bt.model}` "
                f"({bt.train_dataset}→{bt.test_dataset}, AUPRC={bt.micro_auprc_mean:.4f}, "
                f"AUROC={bt.micro_roc_mean:.4f}) over {bt.n_labels} shared diseases.\n"
            )

    lines.append("## Recommendation for V3\n")
    if not within_only.empty:
        gb = best_by(within_only, "micro_auprc")
        lines.append(
            f"Train on **{gb.train_dataset.upper()}** using **`{gb.feature_set}`** with "
            f"**`{gb.model}`** — best within-version micro-AUPRC ({gb.micro_auprc_mean:.4f}). "
            "See the dataset-size/sparsity discussion in this report and the transfer table "
            "above for generalization evidence."
        )
    (outdir / "final_report.md").write_text("\n".join(lines))

    # --- Task 7 results index ---
    idx = ["# Task 7 — Results Summary (index)\n",
           "| experiment | train | test | feature_set | model | metrics file | predictions |",
           "|---|---|---|---|---|---|---|"]
    for _, r in within.iterrows():
        idx.append(
            f"| {r['experiment']} | {r['version']} | {r['version']} | {r['feature_set']} | "
            f"{r['model']} | {r.get('performance_csv','')} | {r.get('prediction_dir','')} |"
        )
    for _, r in transfer.iterrows():
        idx.append(
            f"| {r['experiment']} | {r['source_version']} | {r['target_version']} | "
            f"{r['feature_set']} | {r['model']} | {r.get('performance_csv','')} | (transfer) |"
        )
    idx += [
        "\n## Key files",
        f"- Final unified comparison: `{final_csv.relative_to(PROJECT_ROOT)}`",
        f"- Feature comparison (Task 2): `{feature_csv.relative_to(PROJECT_ROOT)}`",
        f"- Model comparison (Task 3): `{model_csv.relative_to(PROJECT_ROOT)}`",
        f"- Cross-dataset (Task 4): `{cross_csv.relative_to(PROJECT_ROOT)}`",
        "- IPCARF reproduction (Task 1): `results/ipcarf_reproduction_report.md`",
        "- Final report (Task 6): `results/final_report.md`",
        "- Manifest / X / Y / splits: `inductive_inputs/feature_representations/`",
        "- Fold-level predictions: `results/within_version_cv/predictions/`",
        "- Run configs / seeds: `results/within_version_cv/run_config.json`",
    ]
    (outdir / "results_summary.md").write_text("\n".join(idx))

    print("Wrote deliverables:")
    for p in [final_csv, feature_csv, model_csv, cross_csv,
              outdir / "final_report.md", outdir / "results_summary.md"]:
        print(f"  - {p.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
