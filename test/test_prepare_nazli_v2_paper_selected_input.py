from __future__ import annotations

import pandas as pd
import pytest

from scripts import prepare_nazli_v2_paper_selected_input as prepare


def test_align_labels_uses_feature_id_order_and_preserves_all_labels(monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_LABELS", 2)
    source = pd.DataFrame(
        {
            "ID": ["rna-b", "rna-a", "unused"],
            "seqs": ["ACGU", "UGCA", None],
            "disease-1": [0, 1, 1],
            "disease-2": [1, 0, 1],
        }
    )

    result = prepare.align_labels(source, ["rna-a", "rna-b"])

    assert result.index.tolist() == ["rna-a", "rna-b"]
    assert result.columns.tolist() == ["disease-1", "disease-2"]
    assert result.to_numpy().tolist() == [[1, 0], [0, 1]]


def test_align_labels_refuses_duplicate_ids_and_nonbinary_labels(monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_LABELS", 1)
    duplicate = pd.DataFrame(
        {"ID": ["rna-a", "rna-a"], "seqs": ["ACG", "UGC"], "disease": [0, 1]}
    )
    with pytest.raises(ValueError, match="duplicate ID"):
        prepare.align_labels(duplicate, ["rna-a"])

    nonbinary = pd.DataFrame({"ID": ["rna-a"], "seqs": ["ACG"], "disease": [2]})
    with pytest.raises(ValueError, match="binary"):
        prepare.align_labels(nonbinary, ["rna-a"])


def test_v2_schema_must_exactly_match_v1_names_and_order(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "EXPECTED_FEATURES", 3)
    reference = tmp_path / "accepted_v1.csv"
    pd.DataFrame([[1, 2, 3]], index=["v1"], columns=["dcc-a", "k2-a", "k3-a"]).to_csv(
        reference
    )
    same = pd.DataFrame([[4, 5, 6]], index=["v2"], columns=["dcc-a", "k2-a", "k3-a"])
    assert prepare.require_accepted_v1_schema(same, reference) == prepare.sha256(reference)

    reordered = same[["k2-a", "dcc-a", "k3-a"]]
    with pytest.raises(ValueError, match="differs from the accepted V1"):
        prepare.require_accepted_v1_schema(reordered, reference)


def test_exclusive_writers_refuse_overwrite(tmp_path):
    csv_path = tmp_path / "matrix.csv"
    json_path = tmp_path / "record.json"
    frame = pd.DataFrame({"value": [1]})
    prepare.write_csv_exclusive(frame, csv_path, index=False)
    prepare.write_json_exclusive({"value": 1}, json_path)

    with pytest.raises(FileExistsError):
        prepare.write_csv_exclusive(frame, csv_path, index=False)
    with pytest.raises(FileExistsError):
        prepare.write_json_exclusive({"value": 2}, json_path)


def test_independent_comparison_is_hash_only_and_detects_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    generated = tmp_path / "generated.csv"
    recovered = tmp_path / "recovered.csv"
    generated.write_bytes(b"ID,value\nrna,1\n")
    recovered.write_bytes(generated.read_bytes())

    comparison = prepare.independent_comparison(generated, recovered)

    assert comparison["byte_identical"] is True
    assert comparison["generated_sha256"] == comparison["recovered_sha256"]
    assert comparison["generated_path"] == "generated.csv"
    assert comparison["recovered_path"] == "recovered.csv"
