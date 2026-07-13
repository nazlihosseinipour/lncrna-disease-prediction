#!/usr/bin/env python3
"""Validate a complete ten-fold performance CSV while allowing summary rows."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [
    "hamming", "label_ranking", "micro_roc", "micro_auprc",
    "precision", "recall", "fscore", "accuracy",
]


def validate(path: Path) -> None:
    frame = pd.read_csv(path)
    required = {"folds", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns: {missing}")
    labels = frame["folds"].astype(str)
    fold_rows = frame[labels.str.fullmatch(r"fold(?:[1-9]|10)")].copy()
    expected = {f"fold{i}" for i in range(1, 11)}
    actual = set(fold_rows["folds"].astype(str))
    if len(fold_rows) != 10 or actual != expected:
        raise ValueError(
            f"{path}: expected exactly fold1..fold10 once each; "
            f"found {fold_rows['folds'].astype(str).tolist()}"
        )
    values = fold_rows[METRICS].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{path}: fold metrics contain NaN or non-finite values")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("performance_csv", type=Path)
    args = parser.parse_args()
    validate(args.performance_csv)


if __name__ == "__main__":
    main()
