import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipelines import run_all_rna_features as pipeline
from scripts.reconstruct_paper_sequence_features import load_properties


ROOT = Path(__file__).resolve().parents[1]
V1_SOURCE = ROOT / "Data/raw/sequences.csv"
REFERENCE_X = ROOT / "results/felipe_basics_2026-09-27/inputs/v1_paper_sequence_542_X.csv"


def test_paper_sequence_542_mode_matches_verified_v1_matrix(tmp_path):
    outputs = pipeline.run_paper_sequence_542(
        [V1_SOURCE],
        outdir=tmp_path,
        version_name="V1",
    )

    assert outputs == [tmp_path / "V1/rna/sequences_paper_sequence_542.csv"]
    actual = pd.read_csv(outputs[0], index_col=0)
    expected = pd.read_csv(REFERENCE_X, index_col=0)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    metadata = json.loads(
        outputs[0].with_suffix(outputs[0].suffix + ".metadata.json").read_text()
    )
    assert actual.shape == (355, 542)
    assert metadata["pipeline_feature_set"] == "paper_sequence_542"
    assert metadata["feature_recipe"] == {
        "dcc": "Pse-in-One 1.0.6 RNA DCC, 22 properties, lag=1",
        "kmer": "overlapping normalized RNA k-mers, k=2 and k=3",
        "association_derived_columns": 0,
    }
    assert metadata["source_rows"] == 355
    assert metadata["output_rows"] == 355
    assert metadata["output_features"] == 542
    assert metadata["missing_sequence_count"] == 0
    assert metadata["cleaned_sequence_count"] == 2
    assert metadata["cleaned_sequence_ids"] == ["AK153778.1", "BX374997.2 "]
    assert metadata["removed_symbols"] == {"D": 1, "K": 3, "N": 1, "Y": 3}


def test_paper_sequence_542_rejects_raw_count_mode():
    args = argparse.Namespace(no_normalize=True, feature_set="paper_sequence_542")

    with pytest.raises(ValueError, match="fixed Felipe feature contracts"):
        pipeline.validate_paper_mode_args(args)


def test_paper_sequence_542_rejects_historical_two_property_table(tmp_path):
    with pytest.raises(ValueError, match="16 x 22 property table"):
        pipeline.run_paper_sequence_542(
            [V1_SOURCE],
            outdir=tmp_path,
            version_name="V1",
            properties_csv=ROOT / "Data/output_data/dinuc_props.csv",
        )


def test_detect_columns_preserves_exact_identifier_whitespace():
    id_column, sequence_column = pipeline.detect_id_sequence_columns(V1_SOURCE)
    source = pd.read_csv(V1_SOURCE, dtype=str)

    assert (id_column, sequence_column) == ("ID", "seqs")
    assert "BX374997.2 " in source[id_column].tolist()


def test_default_property_table_is_the_verified_22_property_table():
    names, values = load_properties(Path(pipeline.DEFAULT_PROPERTIES))

    assert len(names) == 22
    assert values.shape == (16, 22)
    assert np.isfinite(values).all()


def test_felipe_released_827_mode_copies_verified_artifact_exactly(tmp_path):
    outputs = pipeline.run_felipe_released_827(
        [V1_SOURCE],
        outdir=tmp_path,
        version_name="V1",
    )

    output = tmp_path / "V1/rna/sequences_felipe_released_827.csv"
    assert outputs == [output]
    assert output.read_bytes() == pipeline.DEFAULT_RELEASED_FEATURES.read_bytes()
    features = pd.read_csv(output, index_col=0)
    metadata = json.loads(output.with_suffix(output.suffix + ".metadata.json").read_text())

    assert features.shape == (355, 827)
    assert features.columns.astype(str).tolist() == [str(index) for index in range(827)]
    assert metadata["pipeline_feature_set"] == "felipe_released_827"
    assert metadata["generation_method"] == "validated_byte_copy_of_felipe_released_artifact"
    assert metadata["rna_id_order_match"] is True
    assert metadata["released_artifact_columns"] == 827
    assert metadata["sequence_generated_columns"] == 0
    assert metadata["sequence_reconstructible_columns"] == 542
    assert metadata["unresolved_binary_block_columns"] == 285
    assert metadata["output_sha256"] == pipeline.EXPECTED_RELEASED_FEATURES_SHA256


def test_felipe_released_827_rejects_a_different_rna_order(tmp_path):
    source = pd.read_csv(V1_SOURCE, dtype=str, keep_default_na=False)
    source.iloc[[0, 1]] = source.iloc[[1, 0]].to_numpy()
    changed = tmp_path / "reordered.csv"
    source[["ID", "seqs"]].to_csv(changed, index=False)

    with pytest.raises(ValueError, match="IDs/order"):
        pipeline.run_felipe_released_827(
            [changed],
            outdir=tmp_path / "output",
            version_name="V1",
        )


def test_felipe_released_827_rejects_a_changed_artifact_hash(tmp_path):
    changed = tmp_path / "changed_features.csv"
    changed.write_bytes(pipeline.DEFAULT_RELEASED_FEATURES.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="hash differs"):
        pipeline.run_felipe_released_827(
            [V1_SOURCE],
            outdir=tmp_path / "output",
            version_name="V1",
            released_features_csv=changed,
        )
