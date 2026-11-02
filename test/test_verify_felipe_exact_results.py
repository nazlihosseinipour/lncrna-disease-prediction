from __future__ import annotations

import json
import time

import pandas as pd
import pytest

from scripts import verify_felipe_exact_results as verifier
from scripts.prediction_io import sha256_file, sha256_lines, write_prediction_cell

ROOT = verifier.ROOT
FAKE_FROZEN = {"lncRNA_CIBCB2025-main/RF/rf.py": "0" * 64}


@pytest.fixture
def frozen(monkeypatch):
    """The real frozen table is created last; tests pin a stand-in."""
    monkeypatch.setattr(verifier, "verify_frozen_sources", lambda table, root: dict(FAKE_FROZEN))


@pytest.fixture(scope="module")
def v1():
    within = pd.read_csv(ROOT / verifier.WITHIN_MANIFEST).set_index(["version", "label_space"])
    row = within.loc[("v1", "all44")]
    y = pd.read_csv(ROOT / row["y_path"], index_col=0)
    y.index = y.index.astype(str)
    split = pd.read_csv(ROOT / row["split_path"])
    return row, y, split


def _within_cell(tmp_path, v1, *, fold=1, model="rf", mutate=None):
    row, y, split = v1
    test_ids = [y.index[i] for i in split[f"test_fold{fold}"].dropna().astype(int)]
    predictions = pd.DataFrame(0.25, index=test_ids, columns=y.columns)
    contract = {
        "kind": "within",
        "version": "v1",
        "label_space": "all44",
        "feature_key": "paper_sequence_542",
        "model": model,
        "model_profile": "paper",
        "effective_params": verifier._expected_effective_params()[model],
        "fitted_selection": {},
        "seed": 0,
        "fold": fold,
        "n_folds_total": 10,
        "truth_y_path": row["y_path"],
        "truth_y_sha256": row["y_sha256"],
        "input_hashes": {
            "x": row["x_sha256"],
            "y": row["y_sha256"],
            "split": row["split_sha256"],
            "manifest": sha256_file(ROOT / verifier.WITHIN_MANIFEST),
        },
        "code_hashes": dict(FAKE_FROZEN),
        "library_versions": {},
        "execution": {},
    }
    if mutate:
        mutate(predictions, contract)
    outdir = tmp_path / "within" / "v1__all44__paper_sequence_542__paper" / model
    write_prediction_cell(outdir, f"fold{fold:02d}", predictions, contract)
    return outdir


def test_real_manifests_pass_every_input_check(frozen, capsys):
    start = time.perf_counter()
    result = verifier.check_manifests()
    elapsed = time.perf_counter() - start
    out = capsys.readouterr().out
    assert "manifest verification passed" in out
    assert len(result["loaded"]) == 4 and result["overlap"] == ["NR_024015"]
    assert result["loaded"][("v2", "all436")].x.shape == (4130, 542)
    assert elapsed > 0  # recorded for the report


def test_valid_within_cell_is_accepted(frozen, tmp_path, v1):
    _within_cell(tmp_path, v1)
    status = verifier.check_cells(tmp_path, quiet=True)
    assert status == {("within", "v1", "all44", "rf", 1): "accepted"}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p, c: c["effective_params"].update(n_estimators=100), "effective_params"),
        (lambda p, c: c.update(code_hashes={"x": "1" * 64}), "code_hashes"),
        (lambda p, c: c["input_hashes"].update(split="2" * 64), "input_hashes"),
        (lambda p, c: c.update(model_profile="current"), "model_profile"),
    ],
)
def test_contract_drift_is_rejected(frozen, tmp_path, v1, mutate, message):
    _within_cell(tmp_path, v1, mutate=mutate)
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.check_cells(tmp_path, quiet=True)


def test_wrong_fold_rows_are_rejected(frozen, tmp_path, v1):
    def shift(predictions, contract):
        contract["fold"] = 2  # rows are fold 1's

    _within_cell(tmp_path, v1, mutate=shift)
    with pytest.raises(verifier.VerificationError, match="fold 2 IDs differ"):
        verifier.check_cells(tmp_path, quiet=True)


def test_corrupted_cell_is_rejected(frozen, tmp_path, v1):
    outdir = _within_cell(tmp_path, v1)
    path = outdir / "fold01.predictions.csv"
    path.write_text(path.read_text().replace("0.25", "0.35", 1))
    with pytest.raises(verifier.VerificationError, match="hash"):
        verifier.check_cells(tmp_path, quiet=True)


def test_quarantined_cells_are_ignored(frozen, tmp_path, v1):
    outdir = _within_cell(tmp_path, v1)
    rejected = outdir / "_rejected" / "x"
    rejected.mkdir(parents=True)
    (rejected / "fold01.contract.json").write_text("{}")
    assert verifier.check_cells(tmp_path, quiet=True)


def test_matrix_reports_missing_cells(frozen, tmp_path, v1):
    _within_cell(tmp_path, v1)
    with pytest.raises(verifier.VerificationError, match="59 of 60 expected cells"):
        verifier.check_matrix(tmp_path, "within6")
    assert len(verifier.expected_cells("all")) == 4 * 3 * 10 + 2 * 3


def test_report_check_requires_facts_and_rejects_stale_claims(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(" ".join(verifier.REPORT_REQUIRED))
    verifier.check_report(report)
    report.write_text(" ".join(verifier.REPORT_REQUIRED) + " V2 must be higher")
    with pytest.raises(verifier.VerificationError, match="stale"):
        verifier.check_report(report)
    report.write_text("542 only")
    with pytest.raises(verifier.VerificationError, match="missing"):
        verifier.check_report(report)


def test_transfer_partition_hash_formula_matches_addendum():
    folds = [["a", "b"], ["c"]]
    assert sha256_lines(["1:a,b", "2:c"]) == sha256_lines(
        [f"{n}:" + ",".join(ids) for n, ids in enumerate(folds, start=1)]
    )
