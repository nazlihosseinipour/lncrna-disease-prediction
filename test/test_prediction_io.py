from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from scripts import prediction_io
from scripts.prediction_io import (
    PredictionCellError,
    atomic_write_bytes,
    cell_paths,
    cell_status,
    quarantine_cell,
    read_prediction_cell,
    sha256_lines,
    write_prediction_cell,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"disease b": [0.1, 1 / 3, 0.9], "disease a": [0.0, 0.5, 2 / 7]},
        index=["NR_1", "007", "ABC"],
    )


def _contract() -> dict:
    return {"kind": "within", "version": "v1", "model": "rf", "fold": 1, "seed": 0}


def test_round_trip_preserves_ids_label_order_and_exact_values(tmp_path):
    frame = _frame()
    write_prediction_cell(tmp_path, "cell", frame, _contract())

    loaded, contract = read_prediction_cell(tmp_path, "cell", {"model": "rf", "fold": 1})

    assert loaded.index.tolist() == ["NR_1", "007", "ABC"]  # leading zero kept as string
    assert loaded.columns.tolist() == ["disease b", "disease a"]
    np.testing.assert_array_equal(loaded.to_numpy(), frame.to_numpy())
    assert contract["n_rows"] == 3 and contract["n_labels"] == 2
    assert contract["row_ids_sha256"] == sha256_lines(["NR_1", "007", "ABC"])
    assert contract["label_names_sha256"] == sha256_lines(["disease b", "disease a"])
    assert cell_status(tmp_path, "cell") == "complete"


def test_missing_cell_and_missing_done_marker(tmp_path):
    assert cell_status(tmp_path, "cell") == "missing"
    write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    cell_paths(tmp_path, "cell")["done"].unlink()
    assert cell_status(tmp_path, "cell") == "invalid"
    with pytest.raises(PredictionCellError, match="incomplete"):
        read_prediction_cell(tmp_path, "cell")


def test_corrupted_predictions_are_rejected(tmp_path):
    write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    path = cell_paths(tmp_path, "cell")["predictions"]
    text = path.read_text()
    path.write_text(text.replace("0.9", "0.8"))
    with pytest.raises(PredictionCellError, match="hash"):
        read_prediction_cell(tmp_path, "cell")
    assert cell_status(tmp_path, "cell") == "invalid"


def test_truncated_predictions_are_rejected(tmp_path):
    write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    path = cell_paths(tmp_path, "cell")["predictions"]
    path.write_bytes(path.read_bytes()[:-10])
    assert cell_status(tmp_path, "cell") == "invalid"


def test_expected_contract_mismatch_is_rejected(tmp_path):
    write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    with pytest.raises(PredictionCellError, match="'model' differs"):
        read_prediction_cell(tmp_path, "cell", {"model": "rflda"})
    with pytest.raises(PredictionCellError, match="lacks expected field"):
        read_prediction_cell(tmp_path, "cell", {"x_sha256": "abc"})
    assert cell_status(tmp_path, "cell", {"seed": 1}) == "invalid"


def test_contract_row_hash_tampering_is_rejected(tmp_path):
    write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    path = cell_paths(tmp_path, "cell")["contract"]
    data = json.loads(path.read_text())
    data["row_ids_sha256"] = sha256_lines(["x"])
    path.write_text(json.dumps(data))
    with pytest.raises(PredictionCellError, match="row IDs"):
        read_prediction_cell(tmp_path, "cell")


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"a": [0.1, np.nan]}, index=["r1", "r2"]),
        pd.DataFrame({"a": [0.1, 0.2]}, index=["r1", "r1"]),
        pd.DataFrame({"a": [0.1, np.inf]}, index=["r1", "r2"]),
    ],
)
def test_invalid_frames_are_refused_before_writing(tmp_path, frame):
    with pytest.raises(PredictionCellError):
        write_prediction_cell(tmp_path, "cell", frame, _contract())
    assert cell_status(tmp_path, "cell") == "missing"


def test_derived_fields_cannot_be_supplied(tmp_path):
    with pytest.raises(PredictionCellError, match="derived"):
        write_prediction_cell(
            tmp_path, "cell", _frame(), {**_contract(), "prediction_file_sha256": "x"}
        )


def test_existing_done_marker_blocks_overwrite(tmp_path):
    write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    with pytest.raises(PredictionCellError, match="quarantine"):
        write_prediction_cell(tmp_path, "cell", _frame(), _contract())


def test_quarantine_moves_files_without_deleting(tmp_path):
    write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    before = {p.name: p.read_bytes() for p in cell_paths(tmp_path, "cell").values()}
    target = quarantine_cell(tmp_path, "cell")
    assert target is not None and target.parent.name == "_rejected"
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert cell_status(tmp_path, "cell") == "missing"
    assert quarantine_cell(tmp_path, "cell") is None
    write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    assert cell_status(tmp_path, "cell") == "complete"


def test_crash_before_replace_leaves_no_partial_final_file(tmp_path, monkeypatch):
    final = tmp_path / "x.csv"

    def boom(src, dst):
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(prediction_io.os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(final, b"data")
    assert not final.exists()
    assert os.listdir(tmp_path) == []


def test_crash_between_csv_and_done_is_incomplete(tmp_path, monkeypatch):
    real = prediction_io.atomic_write_text
    calls = {"n": 0}

    def fail_on_done(path, text):
        calls["n"] += 1
        if str(path).endswith(".done"):
            raise OSError("simulated crash before .done")
        return real(path, text)

    monkeypatch.setattr(prediction_io, "atomic_write_text", fail_on_done)
    with pytest.raises(OSError):
        write_prediction_cell(tmp_path, "cell", _frame(), _contract())
    assert cell_status(tmp_path, "cell") == "invalid"


def test_frozen_sources_accept_exact_hashes_and_refuse_changes(tmp_path):
    from scripts.prediction_io import FrozenSourceError, sha256_file, verify_frozen_sources

    source = tmp_path / "src" / "model.py"
    source.parent.mkdir()
    source.write_text("x = 1\n")
    table = tmp_path / "frozen.csv"
    pd.DataFrame({"path": ["src/model.py"], "sha256": [sha256_file(source)]}).to_csv(
        table, index=False
    )
    assert verify_frozen_sources(table, tmp_path) == {"src/model.py": sha256_file(source)}

    source.write_text("x = 2\n")
    with pytest.raises(FrozenSourceError, match="changed src/model.py"):
        verify_frozen_sources(table, tmp_path)
    source.unlink()
    with pytest.raises(FrozenSourceError, match="missing src/model.py"):
        verify_frozen_sources(table, tmp_path)
    with pytest.raises(FrozenSourceError, match="not found"):
        verify_frozen_sources(tmp_path / "absent.csv", tmp_path)


def test_library_versions_reports_core_packages():
    from scripts.prediction_io import library_versions

    versions = library_versions()
    assert {"python", "numpy", "pandas", "scikit-learn"} <= set(versions)
