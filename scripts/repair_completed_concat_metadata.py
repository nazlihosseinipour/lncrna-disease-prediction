#!/usr/bin/env python3
"""Mark validated all-safe concatenation summaries as fixed-threshold results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from validate_performance import validate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary_path = ROOT / "results/within_version_cv/within_version_cv_summary.csv"
    summary = pd.read_csv(summary_path)
    if "threshold_mode" not in summary:
        summary["threshold_mode"] = pd.NA
    repaired = 0
    mask = summary["feature_key"].astype(str).eq("all_safe_concatenated")
    for index, row in summary[mask].iterrows():
        performance = ROOT / "results/within_version_cv/performance" / (
            f"{row['version']}_all_safe_concatenated_{row['model']}_performance.csv"
        )
        if not performance.exists():
            continue
        validate(performance)
        summary.loc[index, "threshold_mode"] = "fixed"
        summary.loc[index, "result_type"] = "all_safe_concatenation"
        repaired += 1
    summary.to_csv(summary_path, index=False)
    print(f"Validated and labelled {repaired} completed all-safe concatenation rows.")


if __name__ == "__main__":
    main()
