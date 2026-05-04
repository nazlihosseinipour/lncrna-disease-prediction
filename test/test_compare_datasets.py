from pathlib import Path

import pandas as pd

from mainfolder.utils.compare_datasets import (
    build_feature_sanity_report,
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


def test_build_feature_sanity_report_handles_version_casing_and_token_sample_check(tmp_path: Path) -> None:
    v1_matrix = tmp_path / "sequences_for_oop.csv"
    v2_matrix = tmp_path / "website_full_matrix.csv"
    pd.DataFrame(
        {
            "ID": ["a", "b"],
            "seqs": ["ACGU", "UGCA"],
            "D1": [1, 0],
            "D2": [0, 1],
        }
    ).to_csv(v1_matrix, index=False)
    pd.DataFrame(
        {
            "ID": ["x", "y", "z"],
            "seqs": ["AAAA", "CCCC", "GGGG"],
            "D1": [1, 0, 0],
        }
    ).to_csv(v2_matrix, index=False)

    feature_base = tmp_path / "final_output"
    v1_rna_dir = feature_base / "V1" / "rna"
    v1_nn_dir = feature_base / "V1" / "nn"
    v2_cross_dir = feature_base / "V2" / "Cross"
    v1_rna_dir.mkdir(parents=True)
    v1_nn_dir.mkdir(parents=True)
    v2_cross_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                "feature_group": "rna",
                "feature_name": "kmer_matrix_k1",
                "method_id": 1,
                "source_name": "sequences_for_oop",
                "rows": 2,
                "cols": 5,
                "output_path": "final_output/v1/rna/sequences_for_oop_kmer_matrix_k1.csv",
            }
        ]
    ).to_csv(v1_rna_dir / "rna_feature_shapes.csv", index=False)

    nn_output = v1_nn_dir / "sequences_for_oop_mp_tokens.csv"
    pd.DataFrame(
        {
            "sample_id": ["a", "a", "b"],
            "token_index": [0, 1, 0],
            "f0": [0.1, 0.2, 0.3],
        }
    ).to_csv(nn_output, index=False)
    pd.DataFrame(
        [
            {
                "feature_group": "nn",
                "feature_name": "mp_tokens",
                "method_id": 101,
                "source_name": "sequences_for_oop",
                "rows": 3,
                "cols": 3,
                "output_path": "final_output/v1/nn/sequences_for_oop_mp_tokens.csv",
            }
        ]
    ).to_csv(v1_nn_dir / "nn_feature_shapes.csv", index=False)

    pd.DataFrame(
        [
            {
                "feature_group": "cross",
                "feature_name": "gip_lncRNA",
                "method_id": 16,
                "source_name": "website_full_matrix",
                "rows": 3,
                "cols": 3,
                "output_path": "final_output/v2/cross/website_full_matrix_gip_lncRNA.csv",
            }
        ]
    ).to_csv(v2_cross_dir / "cross_feature_shapes.csv", index=False)

    report = build_feature_sanity_report(
        v1_matrix_path=v1_matrix,
        v2_matrix_path=v2_matrix,
        feature_base=feature_base,
    )

    rna_row = report.loc[report["feature_name"] == "kmer_matrix_k1"].iloc[0]
    assert rna_row["version"] == "v1"
    assert rna_row["row_status"] == "ok"
    assert rna_row["original_sequence_rows"] == 2

    nn_row = report.loc[report["feature_name"] == "mp_tokens"].iloc[0]
    assert nn_row["sample_status"] == "ok"
    assert nn_row["observed_unique_samples"] == 2
    assert nn_row["row_status"] == "not_checked"

    cross_row = report.loc[report["feature_name"] == "gip_lncRNA"].iloc[0]
    assert cross_row["version"] == "v2"
    assert cross_row["row_status"] == "ok"
    assert cross_row["col_status"] == "ok"
