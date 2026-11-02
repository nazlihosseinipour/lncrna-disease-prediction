import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_shared_sequence_cohort import (
    CohortAssertionError,
    CohortRefusal,
    build_cohort,
    hash_raw,
    sequence_hash,
)
from scripts.prediction_io import sha256_file
from scripts.reconstruct_paper_sequence_features import normalize_sequence

FEATURES = ["fA", "fC", "fG", "fU", "flen"]
LABELS = ["d0", "d1"]


def toy_features(sequence: str) -> np.ndarray:
    n = len(sequence)
    return np.asarray([sequence.count(c) / n for c in "ACGU"] + [n / 7.0], dtype=np.float64)


V1_RAW = [("a1", "acgt"), ("a2", "GGGAAA"), ("a3", "CCCUUU"), ("a4", "AAACCC")]
# b1 == a1 after normalization (case, T->U, removed N); b2/b2x both == a2; b3 missing
V2_RAW = [
    ("b1", "ACNGU"),
    ("b2", "GGGAAA"),
    ("b2x", "gggaaa"),
    ("b3", ""),
    ("b4", "UUUGGG"),
    ("b5", "CAGCAG"),
]
V1_Y = {"a1": [1, 0], "a2": [0, 1], "a3": [1, 1], "a4": [0, 0]}
V2_Y = {"b1": [1, 1], "b2": [1, 0], "b2x": [0, 0], "b4": [0, 1], "b5": [1, 0]}
EXPECTED = {"n_shared_hashes": 2, "n_v2_rows_shared": 3, "n_one_to_one": 1, "n_train_v1": 2, "n_train_v2": 2}


def write_raw(path: Path, rows):
    frame = pd.DataFrame(rows, columns=["ID", "seqs"])
    frame["other disease"] = 0
    frame.to_csv(path, index=False)


def write_x(path: Path, ids, sequences, overrides=None):
    rows = []
    for sample_id in ids:
        seq, _ = normalize_sequence(sequences[sample_id])
        rows.append(toy_features(seq))
    frame = pd.DataFrame(rows, columns=FEATURES, index=pd.Index(ids, name="ID"))
    for sample_id, vector in (overrides or {}).items():
        frame.loc[sample_id] = vector
    frame.to_csv(path, float_format="%.17g")


def write_y(path: Path, ids, labels):
    frame = pd.DataFrame([labels[i] for i in ids], columns=LABELS, index=pd.Index(ids, name="ID"))
    frame.to_csv(path)


def make_fixture(tmp_path: Path, *, v1_x_overrides=None, v2_x_overrides=None):
    data = tmp_path / "data"
    data.mkdir()
    write_raw(data / "v1_raw.csv", V1_RAW)
    write_raw(data / "v2_raw.csv", V2_RAW)
    v1_seq = dict(V1_RAW)
    v2_seq = dict(V2_RAW)
    # X rows deliberately in a different order from the raw files (mapping must use IDs)
    v1_ids = ["a4", "a2", "a1", "a3"]
    v2_ids = ["b5", "b2x", "b4", "b1", "b2"]
    write_x(data / "v1_X.csv", v1_ids, v1_seq, v1_x_overrides)
    write_x(data / "v2_X.csv", v2_ids, v2_seq, v2_x_overrides)
    write_y(data / "v1_Y.csv", v1_ids, V1_Y)
    write_y(data / "v2_Y.csv", v2_ids, V2_Y)
    label_map = data / "labels.csv"
    pd.DataFrame({"order": [0, 1], "canonical_name": LABELS}).to_csv(label_map, index=False)
    manifest = data / "manifest.csv"
    pd.DataFrame(
        [
            {
                "version": v,
                "x_path": str(data / f"{v}_X.csv"),
                "y_path": str(data / f"{v}_Y.csv"),
                "x_sha256": sha256_file(data / f"{v}_X.csv"),
                "y_sha256": sha256_file(data / f"{v}_Y.csv"),
                "label_map_path": str(label_map),
                "label_map_sha256": sha256_file(label_map),
            }
            for v in ("v1", "v2")
        ]
    ).to_csv(manifest, index=False)
    fromserver = tmp_path / "fromserver"
    fromserver.mkdir()
    pd.DataFrame({"v1_id": ["a1", "a3"], "v2_id": ["b1", "b4"]}).to_csv(
        fromserver / "overlap_map.csv", index=False
    )
    return dict(
        repo_root=tmp_path,
        out_root=tmp_path / "out",
        v1_raw=data / "v1_raw.csv",
        v2_raw=data / "v2_raw.csv",
        transfer_manifest=manifest,
        label_map=label_map,
        fromserver=fromserver,
        expected=EXPECTED,
        reconstruct_fn=toy_features,
        expected_feature_columns=FEATURES,
    )


def test_hash_identical_after_normalization():
    frame = pd.DataFrame({"ID": ["x", "y", "z", "m"], "seqs": ["acgt", "ACGU", "AcNg-Y u", ""]})
    hashed = hash_raw(frame)
    assert "m" not in hashed.index  # empty sequence = missing, not hashed
    assert hashed.at["x", "hash"] == hashed.at["y", "hash"] == hashed.at["z", "hash"]
    assert hashed.at["x", "hash"] == sequence_hash("ACGU")
    assert hashed.at["z", "removed"] == " :1;-:1;N:1;Y:1"


def test_build_cohort_synthetic(tmp_path):
    kwargs = make_fixture(tmp_path)
    manifest = build_cohort(**kwargs)
    cohort = tmp_path / "out" / "cohort"
    assert all(manifest["assertions"].values())
    corr = pd.read_csv(cohort / "correspondence.csv", keep_default_na=False)
    h1 = sequence_hash("ACGU")
    h2 = sequence_hash("GGGAAA")
    assert corr["sequence_sha256"].tolist() == sorted([h1, h2])
    row1 = corr.set_index("sequence_sha256").loc[h1]
    row2 = corr.set_index("sequence_sha256").loc[h2]
    assert row1["status"] == "one_to_one" and row1["v2_ids"] == "b1"
    assert row1["v1_x_row_index"] == 2 and row1["v2_x_row_indices"] == "3"
    assert row1["v2_ambiguity_removed"] == "N:1"
    assert row2["status"] == "ambiguous_multi_v2_id"
    assert row2["v2_ids"] == "b2 | b2x" and row2["v2_x_row_indices"] == "4 | 1"
    assert row2["n_v2_label_conflicts"] == 1 and row2["v2_conflict_diseases"] == "d0"
    assert bool(row2["in_primary"]) is False and bool(row2["in_sensitivity"]) is True

    or_labels = pd.read_csv(cohort / "labels_v2_or.csv", index_col=0)
    assert or_labels.loc[h2].tolist() == [1, 0]
    primary = pd.read_csv(cohort / "labels_v2_primary.csv", index_col=0)
    assert primary.index.tolist() == [h1] and primary.loc[h1].tolist() == [1, 1]
    v1_labels = pd.read_csv(cohort / "labels_v1.csv", index_col=0)
    assert v1_labels.loc[h1].tolist() == [1, 0]
    # V1 [1,0] vs V2 [1,1] -> 1 disagreement; V1 [0,1] vs OR [1,0] -> 2
    assert row1["n_v1_v2_label_disagreements_primary"] == 1
    assert row2["n_v1_v2_label_disagreements_primary"] == 2

    conflicts = pd.read_csv(cohort / "v2_label_conflicts.csv")
    assert conflicts.values.tolist() == [[h2, "d0", "b2", 1], [h2, "d0", "b2x", 0]]

    assert (cohort / "train_ids_v1.txt").read_text().split() == ["a4", "a3"]
    assert (cohort / "train_ids_v2.txt").read_text().split() == ["b5", "b4"]  # b3 missing ignored
    assert (cohort / "excluded_ids_v2.txt").read_text().split() == ["b1", "b2", "b2x"]
    test_x = pd.read_csv(cohort / "test_X.csv", index_col=0, float_precision="round_trip")
    assert test_x.columns.tolist() == FEATURES
    assert np.array_equal(test_x.loc[h1].to_numpy(), toy_features("ACGU"))

    written = json.loads((cohort / "cohort_manifest.json").read_text())
    assert written["schema"] == "shared_sequence_cohort/1"
    for name, digest in written["outputs"].items():
        assert sha256_file(cohort / name) == digest
    check = json.loads((cohort / "fromserver_check.json").read_text())
    [candidate] = check["rna_pair_mapping_files"]
    assert candidate["n_pairs_agree_with_new_correspondence"] == 1
    assert candidate["pairs_not_in_new_correspondence"] == [["a3", "b4"]]


def test_refuses_non_empty_cohort_dir(tmp_path):
    kwargs = make_fixture(tmp_path)
    cohort = tmp_path / "out" / "cohort"
    cohort.mkdir(parents=True)
    (cohort / "existing.txt").write_text("x")
    with pytest.raises(CohortRefusal):
        build_cohort(**kwargs)


def test_feature_mismatch_between_versions_raises(tmp_path):
    vector = toy_features("ACGU")
    vector[0] += 1e-9
    kwargs = make_fixture(tmp_path, v2_x_overrides={"b1": vector})
    with pytest.raises(CohortAssertionError, match="features_v1_equal_v2_all"):
        build_cohort(**kwargs)
    assert not (tmp_path / "out" / "cohort").exists()


def test_reconstruction_mismatch_raises(tmp_path):
    vector = toy_features("ACGU")
    vector[4] += 1e-6
    kwargs = make_fixture(tmp_path, v1_x_overrides={"a1": vector}, v2_x_overrides={"b1": vector})
    with pytest.raises(CohortAssertionError, match="features_equal_reconstruction_all"):
        build_cohort(**kwargs)


def test_exact_x_row_overlap_between_train_and_test_raises(tmp_path):
    # training row a3 has a different sequence but a byte-identical X row to test row a1
    kwargs = make_fixture(tmp_path, v1_x_overrides={"a3": toy_features("ACGU")})
    with pytest.raises(CohortAssertionError, match="no_exact_x_row_overlap_v1"):
        build_cohort(**kwargs)


def test_wrong_expected_count_raises(tmp_path):
    kwargs = make_fixture(tmp_path)
    kwargs["expected"] = {**EXPECTED, "n_one_to_one": 2}
    with pytest.raises(CohortAssertionError, match="n_one_to_one"):
        build_cohort(**kwargs)
