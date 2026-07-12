#!/usr/bin/env python3
"""Generate publication figures and tables from existing experiment summaries only."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / os.environ.get("RESULTS_ROOT", "results")
DEFAULT_BINARY = DEFAULT_RESULTS / "binary_comparison"

FEATURE_ORDER = [
    "kmer_matrix_k4",
    "rc_kmer_matrix_k4",
    "psednc_matrix",
    "kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix",
]
FEATURE_LABELS = {
    "kmer_matrix_k4": "k-mer\n(k=4)",
    "rc_kmer_matrix_k4": "RC k-mer\n(k=4)",
    "psednc_matrix": "PseDNC",
    "kmer_matrix_k4+rc_kmer_matrix_k4+psednc_matrix": "Combined",
    "kmer_matrix_k4__rc_kmer_matrix_k4__psednc_matrix": "Combined",
}
MODEL_ORDER = ["rflda", "ipcarf", "rf"]
MODEL_LABELS = {"rflda": "RFLDA", "ipcarf": "IPCARF", "rf": "RF"}
MODEL_COLORS = {"rflda": "#0072B2", "ipcarf": "#D55E00", "rf": "#009E73"}
VERSION_COLORS = {"v1": "#0072B2", "v2": "#E69F00"}
SCENARIO_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]

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
METRIC_LABELS = {
    "hamming": "Hamming loss",
    "label_ranking": "Ranking error",
    "micro_roc": "Micro AUROC",
    "micro_auprc": "Micro AUPRC",
    "precision": "Precision",
    "recall": "Recall",
    "fscore": "F1 score",
    "accuracy": "Accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--binary-dir", type=Path, default=DEFAULT_BINARY)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required existing result file is missing: {path}")
    return path


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_table(df: pd.DataFrame, output_base: Path, caption: str, label: str) -> None:
    df.to_csv(output_base.with_suffix(".csv"), index=False, float_format="%.6f")
    latex = df.to_latex(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        caption=caption,
        label=label,
        escape=True,
    )
    output_base.with_suffix(".tex").write_text(latex, encoding="utf-8")


def normalize_binary_feature(value: str) -> str:
    return value.replace("__", "+")


def validate_feature_summary(
    within: pd.DataFrame, published_feature_summary: pd.DataFrame
) -> pd.DataFrame:
    selected = within[within["feature_set"].isin(FEATURE_ORDER)].copy()
    expected = 2 * len(FEATURE_ORDER) * len(MODEL_ORDER)
    if len(selected) != expected:
        raise ValueError(
            f"Expected {expected} clean within-version feature rows; found {len(selected)}."
        )

    keys = ["version", "feature_set", "model"]
    summary = published_feature_summary.rename(columns={"train_dataset": "version"})
    merged = selected.merge(summary, on=keys, suffixes=("_within", "_published"), validate="one_to_one")
    for metric in ["hamming", "label_ranking", "micro_roc", "micro_auprc", "fscore", "accuracy"]:
        difference = (
            merged[f"{metric}_mean_within"] - merged[f"{metric}_mean_published"]
        ).abs()
        if difference.max() > 1e-12:
            raise ValueError(f"Feature summary mismatch detected for {metric}.")
    return selected


def publication_feature_table(feature_rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "version",
        "feature_set",
        "model",
        "n_samples",
        "n_features",
        "n_labels",
    ]
    for metric in METRICS:
        columns.extend([f"{metric}_mean", f"{metric}_std"])
    table = feature_rows[columns].copy()
    table["version"] = table["version"].str.upper()
    table["feature_set"] = table["feature_set"].map(FEATURE_LABELS).str.replace("\n", " ")
    table["model"] = table["model"].map(MODEL_LABELS)
    table = table.rename(
        columns={
            "version": "Dataset",
            "feature_set": "Feature representation",
            "model": "Model",
            "n_samples": "Samples",
            "n_features": "Features",
            "n_labels": "Labels",
            **{
                f"{metric}_{stat}": f"{METRIC_LABELS[metric]} {stat}"
                for metric in METRICS
                for stat in ("mean", "std")
            },
        }
    )
    return table.sort_values(
        ["Dataset", "Feature representation", "Model"], kind="stable"
    )


def plot_feature_comparison(feature_rows: pd.DataFrame, output_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    metrics = [("micro_auprc", "Micro AUPRC"), ("fscore", "F1 score")]
    width = 0.24
    x = np.arange(len(FEATURE_ORDER))

    for row_index, version in enumerate(["v1", "v2"]):
        version_rows = feature_rows[feature_rows["version"] == version]
        for col_index, (metric, title) in enumerate(metrics):
            ax = axes[row_index, col_index]
            best_patch = None
            best_value = -np.inf
            for model_index, model in enumerate(MODEL_ORDER):
                model_rows = (
                    version_rows[version_rows["model"] == model]
                    .set_index("feature_set")
                    .reindex(FEATURE_ORDER)
                )
                values = model_rows[f"{metric}_mean"].to_numpy()
                errors = model_rows[f"{metric}_std"].to_numpy()
                positions = x + (model_index - 1) * width
                bars = ax.bar(
                    positions,
                    values,
                    width=width,
                    yerr=errors,
                    capsize=2.5,
                    color=MODEL_COLORS[model],
                    edgecolor="white",
                    linewidth=0.7,
                    label=MODEL_LABELS[model],
                    error_kw={"elinewidth": 0.8, "alpha": 0.8},
                )
                local_index = int(np.nanargmax(values))
                if values[local_index] > best_value:
                    best_value = values[local_index]
                    best_patch = bars[local_index]

            if best_patch is not None:
                ax.annotate(
                    f"{best_value:.3f}",
                    (
                        best_patch.get_x() + best_patch.get_width() / 2,
                        best_patch.get_height(),
                    ),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            ax.set_title(f"{version.upper()} · {title}")
            ax.set_ylabel(title)
            ax.set_ylim(bottom=0)
            ax.set_xticks(x, [FEATURE_LABELS[item] for item in FEATURE_ORDER])
            ax.grid(axis="x", visible=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "Within-version feature comparison",
        fontsize=16,
        fontweight="bold",
        y=1.06,
    )
    fig.text(
        0.5,
        0.005,
        "Bars show 10-fold means; error bars show fold standard deviations. "
        "The highest mean in each panel is labelled.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(fig, output_base)


def publication_model_table(model_summary: pd.DataFrame) -> pd.DataFrame:
    table = model_summary.copy()
    table.insert(2, "n_feature_sets", 4)
    table["train_dataset"] = table["train_dataset"].str.upper()
    table["model"] = table["model"].map(MODEL_LABELS)
    table = table.rename(
        columns={
            "train_dataset": "Dataset",
            "model": "Model",
            "n_feature_sets": "Feature sets averaged",
            **{f"{metric}_mean": METRIC_LABELS[metric] for metric in METRICS},
        }
    )
    return table


def plot_model_comparison(model_summary: pd.DataFrame, output_base: Path) -> None:
    metrics = [
        ("micro_roc_mean", "Micro AUROC"),
        ("micro_auprc_mean", "Micro AUPRC"),
        ("fscore_mean", "F1 score"),
        ("hamming_mean", "Hamming loss"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(len(MODEL_ORDER))
    width = 0.34

    for ax, (metric, title) in zip(axes.flat, metrics):
        for version_index, version in enumerate(["v1", "v2"]):
            rows = (
                model_summary[model_summary["train_dataset"] == version]
                .set_index("model")
                .reindex(MODEL_ORDER)
            )
            values = rows[metric].to_numpy()
            positions = x + (version_index - 0.5) * width
            bars = ax.bar(
                positions,
                values,
                width=width,
                color=VERSION_COLORS[version],
                edgecolor="white",
                linewidth=0.8,
                label=version.upper(),
            )
            ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.set_xticks(x, [MODEL_LABELS[model] for model in MODEL_ORDER])
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "Model comparison averaged across four leakage-free feature sets",
        fontsize=15,
        fontweight="bold",
        y=1.06,
    )
    fig.text(
        0.5,
        0.005,
        "Higher is better except for Hamming loss. Values are the existing "
        "means reported in results/model_comparison.csv.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(fig, output_base)


def binary_tables(binary_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    binary = binary_summary.copy()
    binary["feature_key"] = binary["feature_key"].map(normalize_binary_feature)
    binary["status"] = "measured"

    expected = pd.MultiIndex.from_product(
        [["v1", "v2"], FEATURE_ORDER, MODEL_ORDER],
        names=["version", "feature_key", "model"],
    ).to_frame(index=False)
    observed = binary[["version", "feature_key", "model"]]
    missing = expected.merge(observed, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    missing["status"] = "missing"
    missing["reason"] = (
        "No row in results/binary_comparison/"
        "binary_comparison_summary_partial.csv"
    )

    output = binary.rename(
        columns={
            "version": "Dataset",
            "feature_key": "Feature representation",
            "model": "Model",
            "status": "Status",
            **{f"{metric}_mean": METRIC_LABELS[metric] for metric in METRICS},
        }
    )
    output["Dataset"] = output["Dataset"].str.upper()
    output["Feature representation"] = (
        output["Feature representation"].map(FEATURE_LABELS).str.replace("\n", " ")
    )
    output["Model"] = output["Model"].map(MODEL_LABELS)

    missing = missing.rename(
        columns={
            "version": "Dataset",
            "feature_key": "Feature representation",
            "model": "Model",
            "status": "Status",
            "reason": "Reason",
        }
    )
    missing["Dataset"] = missing["Dataset"].str.upper()
    missing["Feature representation"] = (
        missing["Feature representation"].map(FEATURE_LABELS).str.replace("\n", " ")
    )
    missing["Model"] = missing["Model"].map(MODEL_LABELS)
    return output, missing


def plot_binary_comparison(binary_summary: pd.DataFrame, output_base: Path) -> None:
    binary = binary_summary.copy()
    binary["feature_key"] = binary["feature_key"].map(normalize_binary_feature)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    metrics = [("micro_auprc_mean", "Micro AUPRC"), ("fscore_mean", "F1 score")]
    width = 0.24
    x = np.arange(len(FEATURE_ORDER))

    for row_index, version in enumerate(["v1", "v2"]):
        version_rows = binary[binary["version"] == version]
        for col_index, (metric, title) in enumerate(metrics):
            ax = axes[row_index, col_index]
            best_patch = None
            best_value = -np.inf
            for model_index, model in enumerate(MODEL_ORDER):
                model_rows = (
                    version_rows[version_rows["model"] == model]
                    .set_index("feature_key")
                    .reindex(FEATURE_ORDER)
                )
                values = model_rows[metric].to_numpy(dtype=float)
                positions = x + (model_index - 1) * width
                bars = ax.bar(
                    positions,
                    values,
                    width=width,
                    color=MODEL_COLORS[model],
                    edgecolor="white",
                    linewidth=0.7,
                    label=MODEL_LABELS[model],
                )
                if np.isfinite(values).any():
                    local_index = int(np.nanargmax(values))
                    if values[local_index] > best_value:
                        best_value = values[local_index]
                        best_patch = bars[local_index]
            if best_patch is not None:
                ax.annotate(
                    f"{best_value:.3f}",
                    (
                        best_patch.get_x() + best_patch.get_width() / 2,
                        best_patch.get_height(),
                    ),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )
            ax.set_title(f"{version.upper()} · {title}")
            ax.set_ylabel(title)
            ax.set_ylim(bottom=0)
            ax.set_xticks(x, [FEATURE_LABELS[item] for item in FEATURE_ORDER])
            ax.grid(axis="x", visible=False)
            if version == "v2":
                ax.axvspan(2.5, 3.5, color="#D9D9D9", alpha=0.35, zorder=0)
                ax.text(
                    3,
                    ax.get_ylim()[1] * 0.88,
                    "not available",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#555555",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "Binary pairwise comparison (partial completed results)",
        fontsize=16,
        fontweight="bold",
        y=1.06,
    )
    fig.text(
        0.5,
        0.005,
        "21 of 24 model-feature-dataset cells are available. "
        "All V2 combined-feature cells are missing; no values were estimated.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(fig, output_base)


def scenario_label(row: pd.Series) -> str:
    if row["evaluation"] == "within_version_cv":
        return f"Within {row['train_dataset'].upper()} CV"
    if row["evaluation"] == "target_common_cv":
        return f"{row['test_dataset'].upper()} common-label CV"
    return f"{row['train_dataset'].upper()}→{row['test_dataset'].upper()} transfer"


def combined_summary(final_comparison: pd.DataFrame) -> pd.DataFrame:
    scenario_columns = ["evaluation", "experiment", "train_dataset", "test_dataset"]
    selected_indices = final_comparison.groupby(scenario_columns, sort=False)[
        "micro_auprc_mean"
    ].idxmax()
    selected = final_comparison.loc[selected_indices].copy()
    selected["Scenario"] = selected.apply(scenario_label, axis=1)
    selected["Selection criterion"] = "Highest micro AUPRC within scenario"
    selected["Feature representation"] = (
        selected["feature_set"].map(FEATURE_LABELS).str.replace("\n", " ")
    )
    selected["Model"] = selected["model"].map(MODEL_LABELS)
    selected = selected.rename(
        columns={
            "n_labels": "Labels",
            "n_train_samples": "Training samples",
            "n_test_samples": "Test samples",
            **{
                f"{metric}_{stat}": f"{METRIC_LABELS[metric]} {stat}"
                for metric in METRICS
                for stat in ("mean", "std")
            },
        }
    )
    columns = [
        "Scenario",
        "Selection criterion",
        "Feature representation",
        "Model",
        "Labels",
        "Training samples",
        "Test samples",
    ]
    for metric in METRICS:
        columns.extend([f"{METRIC_LABELS[metric]} mean", f"{METRIC_LABELS[metric]} std"])
    return selected[columns].reset_index(drop=True)


def plot_combined_summary(combined: pd.DataFrame, output_base: Path) -> None:
    metrics = ["Micro AUROC", "Micro AUPRC", "F1 score", "Hamming loss"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    labels = combined["Scenario"].tolist()
    x = np.arange(len(labels))

    for ax, metric in zip(axes.flat, metrics):
        values = combined[f"{metric} mean"].to_numpy()
        errors = combined[f"{metric} std"].to_numpy()
        bars = ax.bar(
            x,
            values,
            yerr=errors,
            capsize=3,
            color=SCENARIO_COLORS[: len(values)],
            edgecolor="white",
            linewidth=0.8,
            error_kw={"elinewidth": 0.8, "alpha": 0.8},
        )
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.set_xticks(x, labels, rotation=18, ha="right")
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)

    fig.suptitle(
        "Combined summary of best AUPRC configuration per evaluation scenario",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.005,
        "Selection uses only existing micro-AUPRC means within each scenario; "
        "error bars are reported fold standard deviations.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(fig, output_base)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    binary_dir = args.binary_dir.resolve()
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    within_path = require_file(
        results_dir / "within_version_cv" / "within_version_cv_summary.csv"
    )
    feature_path = require_file(results_dir / "feature_comparison.csv")
    model_path = require_file(results_dir / "model_comparison.csv")
    final_path = require_file(results_dir / "final_comparison.csv")
    binary_path = require_file(binary_dir / "binary_comparison_summary_partial.csv")

    within = pd.read_csv(within_path)
    feature_summary = pd.read_csv(feature_path)
    model_summary = pd.read_csv(model_path)
    final_comparison = pd.read_csv(final_path)
    binary_summary = pd.read_csv(binary_path)

    feature_rows = validate_feature_summary(within, feature_summary)
    feature_table = publication_feature_table(feature_rows)
    model_table = publication_model_table(model_summary)
    binary_table, binary_missing = binary_tables(binary_summary)
    combined_table = combined_summary(final_comparison)

    save_table(
        feature_table,
        tables_dir / "feature_comparison",
        "Within-version feature comparison. Values are existing 10-fold means and standard deviations.",
        "tab:feature-comparison",
    )
    save_table(
        model_table,
        tables_dir / "model_comparison",
        "Model comparison averaged across four leakage-free feature representations.",
        "tab:model-comparison",
    )
    save_table(
        binary_table,
        tables_dir / "binary_comparison_partial",
        "Available binary pairwise comparison results (21 of 24 cells).",
        "tab:binary-comparison-partial",
    )
    save_table(
        binary_missing,
        tables_dir / "binary_comparison_missing",
        "Missing binary comparison cells. No values were estimated.",
        "tab:binary-comparison-missing",
    )
    save_table(
        combined_table,
        tables_dir / "combined_summary",
        "Best observed micro-AUPRC configuration in each evaluation scenario.",
        "tab:combined-summary",
    )

    source_manifest = pd.DataFrame(
        [
            {
                "Output": "Feature comparison",
                "Source file": str(within_path.relative_to(PROJECT_ROOT)),
                "Rows used": len(feature_rows),
                "Completeness": "complete",
                "Notes": "Validated against results/feature_comparison.csv",
            },
            {
                "Output": "Model comparison",
                "Source file": str(model_path.relative_to(PROJECT_ROOT)),
                "Rows used": len(model_summary),
                "Completeness": "complete",
                "Notes": "Existing means across four clean feature sets",
            },
            {
                "Output": "Binary comparison",
                "Source file": str(binary_path.relative_to(PROJECT_ROOT)),
                "Rows used": len(binary_summary),
                "Completeness": "partial (21/24)",
                "Notes": "V2 combined-feature results missing for all three models",
            },
            {
                "Output": "Combined summary",
                "Source file": str(final_path.relative_to(PROJECT_ROOT)),
                "Rows used": len(final_comparison),
                "Completeness": "complete for structured-output scenarios",
                "Notes": "Best configuration selected by existing micro-AUPRC mean",
            },
        ]
    )
    source_manifest.to_csv(tables_dir / "source_manifest.csv", index=False)

    configure_plotting()
    plot_feature_comparison(feature_rows, figures_dir / "feature_comparison")
    plot_model_comparison(model_summary, figures_dir / "model_comparison")
    plot_binary_comparison(binary_summary, figures_dir / "binary_comparison")
    plot_combined_summary(combined_table, figures_dir / "combined_summary")

    print(f"Saved tables to {tables_dir}")
    print(f"Saved figures to {figures_dir}")
    if not binary_missing.empty:
        print(f"Binary comparison remains partial: {len(binary_missing)} cells are missing.")


if __name__ == "__main__":
    main()
