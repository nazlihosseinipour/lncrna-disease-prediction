from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from scripts.experiment_contract import validate_manifest_row
from scripts.run_inductive_within_cv import ensure_prediction_contract, write_run_configs


def _write_valid_experiment(root: Path) -> dict[str, object]:
    x_path = root / "X.csv"
    y_path = root / "Y.csv"
    split_path = root / "splits.csv"
    pd.DataFrame(
        {"ID": ["a", "b", "c", "d"], "f1": [0.1, 0.2, 0.3, 0.4], "f2": [1, 0, 1, 0]}
    ).to_csv(x_path, index=False)
    pd.DataFrame(
        {"ID": ["a", "b", "c", "d"], "d1": [1, 0, 1, 0], "d2": [0, 1, 0, 1]}
    ).to_csv(y_path, index=False)
    pd.DataFrame(
        {
            "train_fold1": [2, 3],
            "test_fold1": [0, 1],
            "train_fold2": [0, 1],
            "test_fold2": [2, 3],
        }
    ).to_csv(split_path, index=False)
    return {
        "x_path": x_path.name,
        "y_path": y_path.name,
        "split_path": split_path.name,
        "n_samples": 4,
        "n_features": 2,
        "n_labels": 2,
        "x_sha256": hashlib.sha256(x_path.read_bytes()).hexdigest(),
        "y_sha256": hashlib.sha256(y_path.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }


def test_manifest_contract_accepts_exact_aligned_inputs(tmp_path):
    row = _write_valid_experiment(tmp_path)

    experiment = validate_manifest_row(row, project_root=tmp_path, n_splits=2)

    assert experiment.x.shape == (4, 2)
    assert experiment.y.shape == (4, 2)
    assert experiment.metadata["n_samples"] == 4


def test_manifest_contract_rejects_row_reordering(tmp_path):
    row = _write_valid_experiment(tmp_path)
    y_path = tmp_path / "Y.csv"
    y = pd.read_csv(y_path).iloc[[1, 0, 2, 3]]
    y.to_csv(y_path, index=False)
    row.pop("y_sha256")

    with pytest.raises(ValueError, match="same order"):
        validate_manifest_row(row, project_root=tmp_path, n_splits=2)


def test_manifest_contract_rejects_stale_dimensions(tmp_path):
    row = _write_valid_experiment(tmp_path)
    row["n_features"] = 827

    with pytest.raises(ValueError, match="does not match"):
        validate_manifest_row(row, project_root=tmp_path, n_splits=2)


def test_manifest_contract_rejects_nonpartitioning_test_folds(tmp_path):
    row = _write_valid_experiment(tmp_path)
    split_path = tmp_path / "splits.csv"
    splits = pd.read_csv(split_path)
    splits["test_fold2"] = [1, 3]
    splits["train_fold2"] = [0, 2]
    splits.to_csv(split_path, index=False)
    row.pop("split_sha256")

    with pytest.raises(ValueError, match="partition every sample exactly once"):
        validate_manifest_row(row, project_root=tmp_path, n_splits=2)


def test_run_configs_keep_an_immutable_record(tmp_path):
    latest, immutable = write_run_configs(
        tmp_path,
        {"models": ["rf"], "manifest_sha256": "abc"},
        timestamp="20260927T000000000000Z",
    )

    assert latest.is_file()
    assert immutable.is_file()
    assert latest.read_text() == immutable.read_text()

    with pytest.raises(FileExistsError):
        write_run_configs(
            tmp_path,
            {"models": ["rf"], "manifest_sha256": "abc"},
            timestamp="20260927T000000000000Z",
        )


def test_resume_requires_the_same_prediction_contract(tmp_path):
    path = tmp_path / "prediction_contract.json"
    contract = {"x_sha256": "x", "model_source_sha256": "model"}
    ensure_prediction_contract(path, contract, resume=False)
    ensure_prediction_contract(path, contract, resume=True)

    with pytest.raises(ValueError, match="contract mismatch"):
        ensure_prediction_contract(
            path,
            {"x_sha256": "different", "model_source_sha256": "model"},
            resume=True,
        )


def test_resume_rejects_legacy_predictions_without_contract(tmp_path):
    with pytest.raises(ValueError, match="without a provenance contract"):
        ensure_prediction_contract(
            tmp_path / "prediction_contract.json",
            {"x_sha256": "x"},
            resume=True,
        )
