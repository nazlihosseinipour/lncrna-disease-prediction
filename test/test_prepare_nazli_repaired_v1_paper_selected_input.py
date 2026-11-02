import json
from pathlib import Path
import tempfile

import pandas as pd
import pytest

from scripts import prepare_nazli_repaired_v1_paper_selected_input as preparer


def test_processed_sequences_repair_to_exact_paper_ids_and_keep_all_rows():
    sequences, labels, repairs = preparer.repaired_sequences_and_labels()

    assert sequences.shape == (355, 2)
    assert labels.shape == (355, 44)
    assert sequences["ID"].tolist() == labels.index.tolist()
    assert repairs == [
        {"source_id": "BX374997.2", "paper_exact_id": "BX374997.2 "}
    ]
    assert preparer.removed_symbol_counts(sequences) == {
        "AK153778.1": {"N": 1},
        "BX374997.2 ": {"D": 1, "K": 3, "Y": 3},
    }


def test_released_folds_partition_every_row_once_and_reject_duplicates():
    splits = pd.read_csv(preparer.PAPER_SPLITS)
    proof = preparer.validate_released_folds(splits)

    assert proof["test_rows"] == 355
    assert proof["test_occurrence_min"] == 1
    assert proof["test_occurrence_max"] == 1
    assert proof["each_row_in_exactly_one_test_fold"] is True

    broken = splits.copy()
    broken.loc[1, "test_fold1"] = broken.loc[0, "test_fold1"]
    with pytest.raises(ValueError, match="duplicate indices"):
        preparer.validate_released_folds(broken)


def test_run_writes_complete_355_row_contract_matching_independent_matrix():
    results_dir = preparer.ROOT / "results"
    with tempfile.TemporaryDirectory(prefix="test_nazli_v1_repair_", dir=results_dir) as parent:
        outdir = Path(parent) / "inputs"
        record = preparer.run(outdir)

        expected_files = {
            "v1_processed_sequences_355.csv",
            "v1_nazli_repaired_paper_sequence_542_X.csv",
            "v1_paper_44_Y.csv",
            "v1_released_folds_355.csv",
            "manifest.csv",
            "preparation_record.json",
        }
        assert {path.name for path in outdir.iterdir()} == expected_files

        x = pd.read_csv(outdir / "v1_nazli_repaired_paper_sequence_542_X.csv", index_col=0)
        y = pd.read_csv(outdir / "v1_paper_44_Y.csv", index_col=0)
        manifest = pd.read_csv(outdir / "manifest.csv").iloc[0]
        saved_record = json.loads((outdir / "preparation_record.json").read_text())

        assert x.shape == (355, 542)
        assert y.shape == (355, 44)
        assert x.index.tolist() == y.index.tolist()
        assert manifest["version"] == "v1"
        assert manifest["label_space"] == "all44"
        assert manifest["population"] == "v1_nazli_processed_sequence_355"
        assert manifest["feature_key"] == "nazli_repaired_paper_sequence_542"
        assert manifest["split_method"] == "released_paper_ten_folds_unchanged"
        assert int(manifest["n_samples"]) == 355
        assert int(manifest["n_features"]) == 542
        assert int(manifest["n_labels"]) == 44
        assert not Path(manifest["x_path"]).is_absolute()
        assert not Path(manifest["y_path"]).is_absolute()
        assert not Path(manifest["split_path"]).is_absolute()
        assert record["dropped_rows"] == 0
        assert saved_record == record
        assert record["accepted_353_x_schema"]["column_names_and_order_match"] is True
        assert all(record["independent_recovered_355_comparison"][key] for key in (
            "hashes_match",
            "shape_match",
            "id_order_match",
            "column_order_match",
            "values_match_exactly",
        ))
        assert preparer.sha256(outdir / "v1_released_folds_355.csv") == preparer.sha256(
            preparer.PAPER_SPLITS
        )


def test_run_rejects_an_existing_output_directory_before_writing():
    results_dir = preparer.ROOT / "results"
    with tempfile.TemporaryDirectory(prefix="test_nazli_v1_existing_", dir=results_dir) as parent:
        with pytest.raises(FileExistsError):
            preparer.run(Path(parent))
