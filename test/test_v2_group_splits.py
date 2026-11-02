from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.experiment_contract import _validate_splits
from scripts.make_v2_group_splits import build, main


def _write_inputs(tmp_path, *, n=60, labels=4, seed=0):
    rng = np.random.RandomState(seed)
    ids = [f"R{i:03d}" for i in range(n)]
    bases = ["".join(rng.choice(list("ACGT"), 30)) for _ in range(n)]
    # Five duplicate groups, one with a different spelling of the same RNA sequence.
    for group, members in enumerate([(0, 1, 2), (3, 4), (5, 6), (7, 8, 9, 10), (11, 12)]):
        for member in members:
            bases[member] = bases[members[0]]
    bases[12] = bases[11].lower().replace("t", "u") + "N"  # normalizes to the same sequence
    x = pd.DataFrame(rng.rand(n, 5), index=ids)
    for members in [(0, 1, 2), (3, 4), (5, 6), (7, 8, 9, 10), (11, 12)]:
        for member in members:
            x.iloc[member] = x.iloc[members[0]].to_numpy()
    y = pd.DataFrame(rng.randint(0, 2, size=(n, labels)), index=ids,
                     columns=[f"d{j}" for j in range(labels)])
    y.iloc[1, 0] = 1 - y.iloc[0, 0]  # conflicting labels inside a duplicate group
    y["empty"] = 0
    x.index.name = y.index.name = "ID"
    x.to_csv(tmp_path / "X.csv")
    y.to_csv(tmp_path / "Y.csv")
    pd.DataFrame({"ID": ids, "seqs": bases}).to_csv(tmp_path / "seq.csv", index=False)
    return tmp_path / "X.csv", tmp_path / "Y.csv", tmp_path / "seq.csv", ids


def test_groups_never_cross_train_and_test_and_rows_are_kept(tmp_path):
    x_path, y_path, seq_path, ids = _write_inputs(tmp_path)
    splits, meta = build(x_path, y_path, seq_path)
    _validate_splits(splits, n_samples=len(ids), n_splits=10)
    assert meta["n_rows"] == 60 and meta["n_labels"] == 5  # all-zero label kept
    assert meta["n_duplicate_groups"] == 5 and meta["rows_in_duplicate_groups"] == 13
    assert meta["duplicate_groups_with_conflicting_labels"] >= 1
    fold_of = {}
    for fold in range(1, 11):
        for position in splits[f"test_fold{fold}"].dropna().astype(int):
            fold_of[ids[position]] = fold
    for members in [(0, 1, 2), (3, 4), (5, 6), (7, 8, 9, 10), (11, 12)]:
        assert len({fold_of[ids[m]] for m in members}) == 1


def test_split_is_deterministic(tmp_path):
    x_path, y_path, seq_path, _ = _write_inputs(tmp_path)
    first, meta1 = build(x_path, y_path, seq_path)
    second, meta2 = build(x_path, y_path, seq_path)
    pd.testing.assert_frame_equal(first, second)
    assert meta1["assignment_sha256"] == meta2["assignment_sha256"]


def test_missing_sequence_for_an_x_row_is_refused(tmp_path):
    x_path, y_path, seq_path, _ = _write_inputs(tmp_path)
    seqs = pd.read_csv(seq_path)
    seqs.loc[5, "seqs"] = np.nan
    seqs.to_csv(seq_path, index=False)
    with pytest.raises(ValueError, match="no usable sequence"):
        build(x_path, y_path, seq_path)


def test_xy_misalignment_is_refused(tmp_path):
    x_path, y_path, seq_path, _ = _write_inputs(tmp_path)
    y = pd.read_csv(y_path).iloc[::-1]
    y.to_csv(y_path, index=False)
    with pytest.raises(ValueError, match="identical order"):
        build(x_path, y_path, seq_path)


def test_output_is_immutable(tmp_path):
    x_path, y_path, seq_path, _ = _write_inputs(tmp_path)
    out = tmp_path / "splits.csv"
    argv = ["--x", str(x_path), "--y", str(y_path), "--sequences", str(seq_path), "--output", str(out)]
    main(argv)
    meta = json.loads((tmp_path / "splits.csv.metadata.json").read_text())
    assert meta["identical_x_row_train_test_overlap"] == 0
    with pytest.raises(FileExistsError):
        main(argv)
