from pathlib import Path

import pandas as pd
import pytest

from scripts.validate_performance import METRICS, validate


def performance_frame() -> pd.DataFrame:
    rows = []
    for fold in range(1, 11):
        rows.append({"folds": f"fold{fold}", **{metric: 0.5 for metric in METRICS}})
    rows.append({"folds": "mean(std)", **{metric: "0.5(0.0)" for metric in METRICS}})
    return pd.DataFrame(rows)


def test_validator_accepts_ten_folds_plus_summary(tmp_path: Path):
    path = tmp_path / "performance.csv"
    performance_frame().to_csv(path, index=False)
    validate(path)


def test_validator_rejects_missing_or_duplicate_fold(tmp_path: Path):
    frame = performance_frame().iloc[:-2].copy()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "performance.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="fold1..fold10"):
        validate(path)
