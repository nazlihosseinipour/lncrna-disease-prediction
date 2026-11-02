"""Create clearly labelled plots from completed corrected-validation folds."""
from __future__ import annotations

import argparse
import re
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "inductive_inputs/corrected_v2_transcripts_10fold_2026-08-17/"
            "feature_representation_manifest.csv"
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        default=(
            "results/corrected_v2_transcript_validation_2026-08-17/predictions/"
            "v2_all_safe_concatenated/rf"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="results/corrected_v2_transcript_validation_2026-08-17/presentation",
    )
    parser.add_argument("--expected-folds", type=int, default=10)
    parser.add_argument("--dataset-label", default="V2")
    parser.add_argument("--filename-prefix", default="corrected_v2_rf")
    parser.add_argument(
        "--footer",
        default=(
            "Canonical transcripts; exact-sequence groups collapsed; "
            "labels ≥10 positives; fixed 0.5 threshold."
        ),
    )
    return parser.parse_args()


def fold_number(path: Path) -> int:
    match = re.search(r"fold(\d+)$", path.stem)
    if not match:
        raise ValueError(f"Could not identify fold number from {path.name}")
    return int(match.group(1))


def calculate_metrics(manifest_path: Path, prediction_dir: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path).iloc[0]
    y = pd.read_csv(manifest["y_path"], index_col=0)
    y.index = y.index.astype(str)
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(prediction_dir.glob("test_fold*.csv"), key=fold_number):
        predictions = pd.read_csv(path, index_col=0)
        predictions.index = predictions.index.astype(str)
        truth = y.loc[predictions.index]
        y_true = truth.to_numpy().ravel()
        y_score = predictions.to_numpy().ravel()
        y_binary = y_score >= 0.5
        rows.append(
            {
                "fold": f"Fold {fold_number(path)}",
                "auroc": roc_auc_score(y_true, y_score),
                "auprc": average_precision_score(y_true, y_score),
                "f1_fixed_0.5": f1_score(y_true, y_binary, zero_division=0),
                "prevalence": float(y_true.mean()),
                "predicted_positive_rate": float(y_binary.mean()),
            }
        )
    if not rows:
        raise ValueError(f"No completed fold predictions found under {prediction_dir}")
    return pd.DataFrame(rows)


def make_plot(
    metrics: pd.DataFrame,
    outdir: Path,
    expected_folds: int,
    dataset_label: str,
    filename_prefix: str,
    footer: str,
) -> Path:
    completed = len(metrics)
    display = pd.concat(
        [
            metrics[["fold", "auroc", "auprc", "f1_fixed_0.5"]],
            pd.DataFrame(
                [
                    {
                        "fold": "Current mean",
                        "auroc": metrics["auroc"].mean(),
                        "auprc": metrics["auprc"].mean(),
                        "f1_fixed_0.5": metrics["f1_fixed_0.5"].mean(),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    width, height = 1100, 650
    left, right, top, bottom = 90, 35, 85, 150
    plot_width = width - left - right
    plot_height = height - top - bottom
    group_width = plot_width / len(display)
    bar_width = min(64, group_width * 0.22)
    state = "FINAL" if completed == expected_folds else "PRELIMINARY"
    filename = (
        f"{filename_prefix}_final_metrics.svg"
        if completed == expected_folds
        else f"{filename_prefix}_preliminary_metrics_{completed}_of_{expected_folds}.svg"
    )
    path = outdir / filename
    series = [
        ("auroc", "Micro-AUROC", "#2563eb"),
        ("auprc", "Micro-AUPRC", "#ea580c"),
        ("f1_fixed_0.5", "Micro-F1 (threshold 0.5)", "#16a34a"),
    ]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827}</style>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-size="22" font-weight="700">Corrected {escape(dataset_label)} features — RF validation</text>',
        f'<text x="{width / 2}" y="62" text-anchor="middle" font-size="15" fill="#4b5563">{state}: {completed}/{expected_folds} folds completed</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#9ca3af"/>',
    ]
    for tick in np.linspace(0, 1, 6):
        y = top + plot_height * (1 - tick)
        svg.extend(
            [
                f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="12">{tick:.1f}</text>',
            ]
        )
    svg.append(
        f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-size="13" transform="rotate(-90 24 {top + plot_height / 2})">Metric value</text>'
    )
    offsets = (-bar_width, 0, bar_width)
    for group_index, row in display.iterrows():
        center = left + group_width * (group_index + 0.5)
        for (column, _, color), offset in zip(series, offsets):
            value = float(row[column])
            bar_height = value * plot_height
            x = center + offset - bar_width * 0.42
            y = top + plot_height - bar_height
            svg.extend(
                [
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.84:.1f}" height="{bar_height:.1f}" fill="{color}"/>',
                    f'<text x="{x + bar_width * 0.42:.1f}" y="{max(top + 13, y - 7):.1f}" text-anchor="middle" font-size="12" font-weight="600">{value:.3f}</text>',
                ]
            )
        svg.append(
            f'<text x="{center:.1f}" y="{top + plot_height + 28}" text-anchor="middle" font-size="13">{escape(str(row["fold"]))}</text>'
        )
    legend_y = height - 78
    legend_width = 275
    legend_start = (width - legend_width * len(series)) / 2
    for index, (_, label, color) in enumerate(series):
        x = legend_start + index * legend_width
        svg.extend(
            [
                f'<rect x="{x:.1f}" y="{legend_y}" width="15" height="15" fill="{color}"/>',
                f'<text x="{x + 23:.1f}" y="{legend_y + 12}" font-size="13">{escape(label)}</text>',
            ]
        )
    svg.append(
        f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-size="12" fill="#4b5563">{escape(footer)}</text>'
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    manifest_path = PROJECT_ROOT / args.manifest
    prediction_dir = PROJECT_ROOT / args.prediction_dir
    outdir = PROJECT_ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = calculate_metrics(manifest_path, prediction_dir)
    metrics_path = outdir / f"{args.filename_prefix}_completed_fold_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    plot_path = make_plot(
        metrics,
        outdir,
        args.expected_folds,
        args.dataset_label,
        args.filename_prefix,
        args.footer,
    )
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
