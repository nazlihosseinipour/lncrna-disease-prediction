from pathlib import Path

import pandas as pd

from mainfolder.utils.compare_datasets import (
    compare_matrix_changes,
    load_matrix,
    matrix_version_summary,
)


def test_load_matrix_drops_unnamed_and_sequence_columns(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "ID": ["a", "b"],
            "seqs": ["ACGU", "UGCA"],
            "Disease1": [1, 0],
            "Disease2": [0, 1],
        }
    ).to_csv(path, index=False)

    df = load_matrix(path)

    assert list(df.columns) == ["ID", "Disease1", "Disease2"]
    assert df.to_dict("records") == [
        {"ID": "a", "Disease1": 1, "Disease2": 0},
        {"ID": "b", "Disease1": 0, "Disease2": 1},
    ]


def test_compare_matrix_changes_counts_added_removed_and_new_diseases() -> None:
    v1 = pd.DataFrame(
        {
            "ID": ["a", "b", "c"],
            "D1": [1, 0, 1],
            "D2": [0, 1, 0],
        }
    )
    v2 = pd.DataFrame(
        {
            "ID": ["a", "b", "d"],
            "D2": [1, 0, 1],
            "D3": [0, 1, 1],
        }
    )

    version_summary = matrix_version_summary(v1, v2)
    change_summary, disease_rank, new_diseases, dropped_diseases = compare_matrix_changes(v1, v2)

    assert version_summary.loc[version_summary["version"] == "v1", "num_sequences"].item() == 3
    assert version_summary.loc[version_summary["version"] == "v2", "num_diseases"].item() == 2

    row = change_summary.iloc[0].to_dict()
    assert row["common_sequences"] == 2
    assert row["new_sequences_in_v2"] == 1
    assert row["dropped_sequences_from_v1"] == 1
    assert row["common_diseases"] == 1
    assert row["new_diseases_in_v2"] == 1
    assert row["dropped_diseases_from_v1"] == 1
    assert row["added_interactions_0_to_1"] == 1
    assert row["removed_interactions_1_to_0"] == 1

    assert new_diseases["disease"].tolist() == ["D3"]
    assert dropped_diseases["disease"].tolist() == ["D1"]
    assert disease_rank.loc[0, "disease"] == "D2"
    assert disease_rank.loc[0, "zero_to_one_count"] == 1
    assert disease_rank.loc[0, "one_to_zero_count"] == 1
