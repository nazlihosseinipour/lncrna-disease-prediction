"""Build one presentation plot from the final clean V1 and V2 RF summaries."""
from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "results" / "clean_corrected_presentation_2026-08-18"
SOURCES = [
    (
        "V1 (strict)",
        ROOT / "results" / "corrected_validation_2026-08-17" / "within_version_cv_summary.csv",
        "353 sequences · 2,023 features · 24 labels · 613 positives",
    ),
    (
        "V2 (transcripts)",
        ROOT
        / "results"
        / "corrected_v2_transcript_validation_2026-08-17"
        / "within_version_cv_summary.csv",
        "3,849 sequences · 583 features · 86 labels · 7,806 positives",
    ),
]


def read_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for label, path, details in SOURCES:
        row = pd.read_csv(path).iloc[0]
        if row["threshold_mode"] != "fixed":
            raise ValueError(f"Expected fixed threshold in {path}")
        rows.append(
            {
                "dataset": label,
                "details": details,
                "AUROC": float(row["micro_roc_mean"]),
                "AUPRC": float(row["micro_auprc_mean"]),
                "F1": float(row["fscore_mean"]),
            }
        )
    return rows


def write_plot(rows: list[dict[str, float | str]]) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "clean_v1_v2_rf_final_comparison.svg"
    width, height = 1120, 690
    left, top, plot_width, plot_height = 100, 120, 960, 390
    group_centres = [355, 805]
    bar_width = 95
    series = [("AUROC", "#2563eb"), ("AUPRC", "#ea580c"), ("F1", "#16a34a")]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#111827}</style>',
        '<text x="560" y="42" text-anchor="middle" font-size="24" font-weight="700">Final corrected RF validation</text>',
        '<text x="560" y="72" text-anchor="middle" font-size="15" fill="#4b5563">Clean sequence-derived inputs · 10-fold inductive CV · fixed 0.5 threshold</text>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#9ca3af"/>',
    ]
    for tick in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = top + plot_height * (1 - tick)
        svg.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        svg.append(f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-size="13">{tick:.1f}</text>')
    svg.append(f'<text x="30" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 30 {top + plot_height / 2})">Mean metric value</text>')
    offsets = [-bar_width, 0, bar_width]
    for centre, row in zip(group_centres, rows):
        for (metric, colour), offset in zip(series, offsets):
            value = float(row[metric])
            bar_height = value * plot_height
            x = centre + offset - bar_width * 0.42
            y = top + plot_height - bar_height
            svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width * 0.84:.1f}" height="{bar_height:.1f}" fill="{colour}"/>')
            svg.append(f'<text x="{x + bar_width * 0.42:.1f}" y="{max(top + 16, y - 8):.1f}" text-anchor="middle" font-size="15" font-weight="700">{value:.3f}</text>')
        svg.append(f'<text x="{centre}" y="548" text-anchor="middle" font-size="17" font-weight="700">{escape(str(row["dataset"]))}</text>')
        svg.append(f'<text x="{centre}" y="574" text-anchor="middle" font-size="12" fill="#4b5563">{escape(str(row["details"]))}</text>')
    legend_x = [350, 530, 700]
    for (metric, colour), x in zip(series, legend_x):
        svg.append(f'<rect x="{x}" y="615" width="18" height="18" fill="{colour}"/>')
        svg.append(f'<text x="{x + 27}" y="629" font-size="14">Micro-{metric}</text>')
    svg.append('<text x="560" y="666" text-anchor="middle" font-size="12" fill="#4b5563">The quarantined 827-column paper artifact is excluded from every value shown.</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")
    return path


def main() -> None:
    rows = read_rows()
    pd.DataFrame(rows).to_csv(OUTDIR / "clean_v1_v2_rf_final_summary.csv", index=False)
    path = write_plot(rows)
    print(path)


if __name__ == "__main__":
    main()
