"""Resume, provenance and leakage tests for scripts/run_felipe_strict_transfer.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lncRNA_CIBCB2025-main"))

from scripts import prediction_io  # noqa: E402
from scripts import run_felipe_strict_transfer as M  # noqa: E402
from scripts.felipe_paper_models import MODEL_SOURCES  # noqa: E402


LABELS = ["disease a", "disease b", "disease c", "never positive"]
FEATURES = [f"f{i}" for i in range(8)]
OVERLAP_ID = "SHARED1"


# --------------------------------------------------------------------------- fixtures

def _frame(ids, columns, values):
    return pd.DataFrame(values, index=pd.Index(ids, name="ID"), columns=columns)


def _make_version(rng, ids):
    x = _frame(ids, FEATURES, rng.normal(size=(len(ids), len(FEATURES))))
    y = np.zeros((len(ids), len(LABELS)), dtype=int)
    y[:, 0] = rng.random(len(ids)) < 0.4
    y[:, 1] = rng.random(len(ids)) < 0.3
    y[:, 2] = rng.random(len(ids)) < 0.5
    # last label stays all-zero on purpose (never filtered)
    return x, _frame(ids, LABELS, y)


def _write_manifest(tmp, frames, *, full=True, label_map=None, label_map_sha=None):
    rows = []
    for version, (x, y) in frames.items():
        x_path, y_path = tmp / f"{version}_X.csv", tmp / f"{version}_Y.csv"
        x.to_csv(x_path)
        y.to_csv(y_path)
        row = {
            "version": version,
            "x_path": x_path.as_posix(),
            "y_path": y_path.as_posix(),
            "split_path": "",
            "n_samples": len(x),
            "n_features": x.shape[1],
            "n_labels": y.shape[1],
            "x_sha256": prediction_io.sha256_file(x_path),
            "y_sha256": prediction_io.sha256_file(y_path),
        }
        if full:
            row.update(
                {
                    "label_space": "shared73",
                    "feature_key": "toy_sequence_8",
                    "label_map_path": label_map.as_posix(),
                    "label_map_sha256": label_map_sha or prediction_io.sha256_file(label_map),
                }
            )
        rows.append(row)
    path = tmp / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_frozen(tmp, *, corrupt=None):
    paths = list(M.REQUIRED_FROZEN) + list(MODEL_SOURCES.values())
    rows = []
    for path in paths:
        digest = prediction_io.sha256_file(ROOT / path)
        if path == corrupt:
            digest = "0" * 64
        rows.append({"path": path, "sha256": digest, "role": "test"})
    table = tmp / ("frozen_corrupt.csv" if corrupt else "frozen.csv")
    pd.DataFrame(rows).to_csv(table, index=False)
    return table


@pytest.fixture
def toy(tmp_path):
    rng = np.random.default_rng(7)
    source_ids = [f"S{i}" for i in range(39)] + [OVERLAP_ID]
    target_ids = [f"T{i}" for i in range(29)] + [OVERLAP_ID]
    frames = {"v1": _make_version(rng, source_ids), "v2": _make_version(rng, target_ids)}
    label_map = tmp_path / "label_map.csv"
    pd.DataFrame({"order": range(len(LABELS)), "canonical_name": LABELS}).to_csv(
        label_map, index=False
    )
    manifest = _write_manifest(tmp_path, frames, label_map=label_map)
    frozen = _write_frozen(tmp_path)
    return {
        "tmp": tmp_path,
        "frames": frames,
        "manifest": manifest,
        "frozen": frozen,
        "label_map": label_map,
    }


def _argv(toy, outdir, *extra, models=("rf", "ipcarf", "rflda")):
    return [
        "--manifest", str(toy["manifest"]),
        "--frozen-sources", str(toy["frozen"]),
        "--source", "v1", "--target", "v2",
        "--models", *models,
        "--n-splits", "3",
        "--model-n-jobs", "1",
        "--outdir", str(outdir),
        *extra,
    ]


class Spy:
    """Deterministic stand-in model that records every fit/predict call."""

    def __init__(self, fail_on_fit=None):
        self.fits: list[dict] = []
        self.predicts: list[dict] = []
        self.fail_on_fit = set(fail_on_fit or ())

    def install(self, monkeypatch):
        spy = self

        class FakeModel:
            random_state = 0

            def __init__(self, key, n_jobs):
                self.key, self.n_jobs = key, n_jobs

            def fit(self, x, y):
                spy.fits.append({"model": self.key, "x_ids": list(x.index),
                                 "x_cols": list(x.columns), "y_ids": list(y.index)})
                if self.key in spy.fail_on_fit:
                    raise RuntimeError(f"simulated interruption in {self.key}")
                self.weights = y.mean(axis=0).to_numpy()

            def predict_proba(self, x, y=None):
                spy.predicts.append({"model": self.key, "x_ids": list(x.index), "y": y})
                assert y is None, "target labels must never reach predict_proba"
                base = 1.0 / (1.0 + np.exp(-x.to_numpy().mean(axis=1)))
                return base[:, None] * self.weights[None, :]

        monkeypatch.setattr(M, "build_paper_model", lambda key, n_jobs=None: FakeModel(key, n_jobs))
        monkeypatch.setattr(M, "effective_params",
                            lambda key, model: {"class": "FakeModel", "random_state": 0, "key": key})
        monkeypatch.setattr(M, "fitted_selection", lambda key, model: {"fake": key})
        return self


def _cell_dir(outdir, model, feature_key="toy_sequence_8"):
    return outdir / "transfer" / f"v1_to_v2__shared73__{feature_key}__paper" / model


# --------------------------------------------------------------------------- tests

def test_one_fit_and_one_predict_per_model_with_raw_scores(toy, monkeypatch):
    spy = Spy().install(monkeypatch)
    outdir = toy["tmp"] / "out"
    record = M.main(_argv(toy, outdir))
    assert [fit["model"] for fit in spy.fits] == ["rf", "ipcarf", "rflda"]
    assert [call["model"] for call in spy.predicts] == ["rf", "ipcarf", "rflda"]
    assert [cell["status"] for cell in record["cells"]] == ["fitted"] * 3
    x_target = toy["frames"]["v2"][0]
    for model in ("rf", "ipcarf", "rflda"):
        cell_dir = _cell_dir(outdir, model)
        frame, contract = prediction_io.read_prediction_cell(cell_dir, "full_target")
        assert frame.columns.tolist() == LABELS
        assert contract["kind"] == "transfer" and contract["model"] == model
        assert contract["fold"] is None and contract["n_folds_total"] is None
        assert contract["seed"] == 0 and contract["model_profile"] == "paper"
        assert contract["label_space"] == "shared73"
        assert contract["fitted_selection"] == {"fake": model}
        assert contract["execution"]["n_jobs"] == 1
        assert set(contract["input_hashes"]) == {
            "source_x", "source_y", "target_x", "target_y", "manifest", "label_map"
        }
        assert contract["truth_y_sha256"] == prediction_io.sha256_file(toy["tmp"] / "v2_Y.csv")
        assert "threshold_mode" not in contract
        assert len(frame) == len(x_target) - 1
    # no metrics, thresholds, shards or persisted models
    written = sorted(p.name for p in (outdir / "transfer").rglob("*") if p.is_file())
    assert all(name.startswith("full_target.") for name in written)
    records = list((outdir / "run_records").glob("*.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text())
    assert saved["outcome"] == "complete" and len(saved["cells"]) == 3


def test_interrupted_run_resumes_only_remaining_models(toy, monkeypatch):
    reference = toy["tmp"] / "reference"
    Spy().install(monkeypatch)
    M.main(_argv(toy, reference))

    outdir = toy["tmp"] / "interrupted"
    Spy(fail_on_fit={"ipcarf"}).install(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        M.main(_argv(toy, outdir))
    assert prediction_io.cell_status(_cell_dir(outdir, "rf"), "full_target") == "complete"
    assert prediction_io.cell_status(_cell_dir(outdir, "ipcarf"), "full_target") == "missing"

    spy = Spy().install(monkeypatch)
    record = M.main(_argv(toy, outdir, "--resume"))
    assert [fit["model"] for fit in spy.fits] == ["ipcarf", "rflda"]
    assert [cell["status"] for cell in record["cells"]] == ["reused", "fitted", "fitted"]
    for model in ("rf", "ipcarf", "rflda"):
        resumed, _ = prediction_io.read_prediction_cell(_cell_dir(outdir, model), "full_target")
        clean, _ = prediction_io.read_prediction_cell(_cell_dir(reference, model), "full_target")
        pd.testing.assert_frame_equal(resumed, clean)
    outcomes = sorted(json.loads(p.read_text())["outcome"]
                      for p in (outdir / "run_records").glob("*.json"))
    assert outcomes == ["complete", "failed"]


def test_corrupted_cell_is_quarantined_and_redone(toy, monkeypatch):
    Spy().install(monkeypatch)
    outdir = toy["tmp"] / "out"
    M.main(_argv(toy, outdir, models=("rf", "ipcarf")))
    cell_dir = _cell_dir(outdir, "rf")
    csv = cell_dir / "full_target.predictions.csv"
    corrupted = csv.read_text().replace("0.", "9.", 1)
    csv.write_text(corrupted)

    spy = Spy().install(monkeypatch)
    record = M.main(_argv(toy, outdir, "--resume", models=("rf", "ipcarf")))
    assert [fit["model"] for fit in spy.fits] == ["rf"]
    assert [cell["status"] for cell in record["cells"]] == ["fitted", "reused"]
    assert prediction_io.cell_status(cell_dir, "full_target") == "complete"
    rejected = list((cell_dir / "_rejected").rglob("full_target.predictions.csv"))
    assert len(rejected) == 1 and rejected[0].read_text() == corrupted
    assert "quarantined_to" in record["cells"][0]


def test_existing_cell_without_resume_aborts_before_fitting(toy, monkeypatch):
    Spy().install(monkeypatch)
    outdir = toy["tmp"] / "out"
    M.main(_argv(toy, outdir, models=("rf",)))
    before = prediction_io.sha256_file(_cell_dir(outdir, "rf") / "full_target.predictions.csv")
    spy = Spy().install(monkeypatch)
    with pytest.raises(FileExistsError, match="--resume"):
        M.main(_argv(toy, outdir, models=("ipcarf", "rf")))
    assert spy.fits == []
    assert prediction_io.sha256_file(_cell_dir(outdir, "rf") / "full_target.predictions.csv") == before
    # a partial cell (no .done) also blocks a non-resume run
    partial = _cell_dir(outdir, "rflda")
    partial.mkdir(parents=True)
    (partial / "full_target.predictions.csv").write_text("ID\n")
    with pytest.raises(FileExistsError):
        M.main(_argv(toy, outdir, models=("rflda",)))
    assert spy.fits == []


def test_overlap_id_removed_recorded_and_never_predicted(toy, monkeypatch):
    spy = Spy().install(monkeypatch)
    outdir = toy["tmp"] / "out"
    M.main(_argv(toy, outdir, models=("rf",)))
    frame, contract = prediction_io.read_prediction_cell(_cell_dir(outdir, "rf"), "full_target")
    assert contract["removed_exact_target_ids"] == [OVERLAP_ID]
    assert OVERLAP_ID not in frame.index
    assert OVERLAP_ID not in spy.predicts[0]["x_ids"]
    expected_ids = [i for i in toy["frames"]["v2"][0].index if i != OVERLAP_ID]
    assert frame.index.tolist() == expected_ids
    assert spy.predicts[0]["x_ids"] == expected_ids
    all_partition_ids = [i for fold in contract["evaluation_partition"]["folds"] for i in fold]
    assert OVERLAP_ID not in all_partition_ids
    # the source keeps its copy of the shared ID
    assert OVERLAP_ID in spy.fits[0]["x_ids"]


def test_target_labels_never_reach_fit_or_predict(toy, monkeypatch):
    spy = Spy().install(monkeypatch)
    M.main(_argv(toy, toy["tmp"] / "out"))
    source_ids = toy["frames"]["v1"][0].index.tolist()
    for fit in spy.fits:
        assert fit["x_ids"] == source_ids and fit["y_ids"] == source_ids
        assert fit["x_cols"] == FEATURES  # no labels appended to X
    assert len(spy.predicts) == 3
    assert all(call["y"] is None for call in spy.predicts)


def test_label_order_mismatch_between_source_and_target_aborts(toy, monkeypatch):
    spy = Spy().install(monkeypatch)
    x2, y2 = toy["frames"]["v2"]
    swapped = {"v1": toy["frames"]["v1"], "v2": (x2, y2[[LABELS[1], LABELS[0], *LABELS[2:]]])}
    manifest = _write_manifest(toy["tmp"], swapped, label_map=toy["label_map"])
    argv = _argv({**toy, "manifest": manifest}, toy["tmp"] / "out")
    with pytest.raises(ValueError, match="label columns differ"):
        M.main(argv)
    assert spy.fits == []


def test_label_map_order_and_hash_are_enforced(toy, monkeypatch):
    spy = Spy().install(monkeypatch)
    bad_hash = _write_manifest(toy["tmp"], toy["frames"], label_map=toy["label_map"],
                               label_map_sha="f" * 64)
    with pytest.raises(ValueError, match="Label map hash"):
        M.main(_argv({**toy, "manifest": bad_hash}, toy["tmp"] / "out"))
    reordered = toy["tmp"] / "reordered_map.csv"
    pd.DataFrame({"order": [1, 0, 2, 3], "canonical_name": LABELS}).to_csv(reordered, index=False)
    manifest = _write_manifest(toy["tmp"], toy["frames"], label_map=reordered)
    with pytest.raises(ValueError, match="label map canonical order"):
        M.main(_argv({**toy, "manifest": manifest}, toy["tmp"] / "out"))
    assert spy.fits == []


def test_manifest_without_contract_columns_uses_recorded_fallback(toy, monkeypatch):
    Spy().install(monkeypatch)
    manifest = _write_manifest(toy["tmp"], toy["frames"], full=False)
    monkeypatch.setattr(M, "DEFAULT_LABEL_MAP", toy["label_map"].as_posix())
    monkeypatch.setattr(M, "FEATURE_WIDTHS", {"paper_sequence_542": 8})
    outdir = toy["tmp"] / "out"
    M.main(_argv({**toy, "manifest": manifest}, outdir, models=("rf",)))
    _, contract = prediction_io.read_prediction_cell(
        _cell_dir(outdir, "rf", "paper_sequence_542"), "full_target"
    )
    assert contract["feature_key"] == "paper_sequence_542"
    assert contract["label_contract_source"] == "runtime_default"
    assert contract["input_hashes"]["label_map"] == prediction_io.sha256_file(toy["label_map"])
    # the paper feature key still enforces its width
    monkeypatch.setattr(M, "FEATURE_WIDTHS", {"paper_sequence_542": 542})
    with pytest.raises(ValueError, match="requires 542 features"):
        M.main(_argv({**toy, "manifest": manifest}, toy["tmp"] / "out2", models=("rf",)))


def test_frozen_source_change_aborts(toy, monkeypatch):
    spy = Spy().install(monkeypatch)
    frozen = _write_frozen(toy["tmp"], corrupt="lncRNA_CIBCB2025-main/RF/rf.py")
    with pytest.raises(prediction_io.FrozenSourceError, match="RF/rf.py"):
        M.main(_argv({**toy, "frozen": frozen}, toy["tmp"] / "out"))
    missing = toy["tmp"] / "no_such_table.csv"
    with pytest.raises(prediction_io.FrozenSourceError):
        M.main(_argv({**toy, "frozen": missing}, toy["tmp"] / "out"))
    # a table that does not freeze the runner itself is refused too
    partial = toy["tmp"] / "partial.csv"
    pd.read_csv(toy["frozen"]).query("path != 'scripts/run_felipe_strict_transfer.py'").to_csv(
        partial, index=False
    )
    with pytest.raises(prediction_io.FrozenSourceError, match="run_felipe_strict_transfer"):
        M.main(_argv({**toy, "frozen": partial}, toy["tmp"] / "out"))
    assert spy.fits == []
    assert not (toy["tmp"] / "out").exists()


def test_evaluation_partition_covers_targets_once_and_is_reproducible(toy, monkeypatch):
    Spy().install(monkeypatch)
    contracts = []
    for name in ("a", "b"):
        outdir = toy["tmp"] / name
        M.main(_argv(toy, outdir, models=("rf",)))
        contracts.append(
            prediction_io.read_prediction_cell(_cell_dir(outdir, "rf"), "full_target")[1]
        )
    partition = contracts[0]["evaluation_partition"]
    assert partition["method"] == "iterstrat.MultilabelStratifiedKFold"
    assert partition["n_splits"] == 3 and partition["random_state"] == 0
    assert partition["uses_target_labels"] is True
    flat = [i for fold in partition["folds"] for i in fold]
    target_ids = [i for i in toy["frames"]["v2"][0].index if i != OVERLAP_ID]
    assert sorted(flat) == sorted(target_ids) and len(flat) == len(set(flat))
    assert len(partition["folds"]) == 3
    recomputed = prediction_io.sha256_lines(
        [f"{n}:{','.join(fold)}" for n, fold in enumerate(partition["folds"], start=1)]
    )
    assert partition["sha256"] == recomputed
    assert contracts[1]["evaluation_partition"] == partition


def test_threshold_mode_option_is_rejected(toy):
    with pytest.raises(SystemExit):
        M.main(_argv(toy, toy["tmp"] / "out", "--threshold-mode", "fixed"))


def test_real_rf_cell_is_deterministic_and_reused(toy):
    first, second = toy["tmp"] / "first", toy["tmp"] / "second"
    M.main(_argv(toy, first, models=("rf",)))
    M.main(_argv(toy, second, models=("rf",)))
    a, contract = prediction_io.read_prediction_cell(_cell_dir(first, "rf"), "full_target")
    b, _ = prediction_io.read_prediction_cell(_cell_dir(second, "rf"), "full_target")
    pd.testing.assert_frame_equal(a, b)
    assert contract["effective_params"]["n_estimators"] == 150
    assert contract["effective_params"]["random_state"] == 0 and contract["seed"] == 0
    assert (a.to_numpy() >= 0).all() and (a.to_numpy() <= 1).all()
    assert (a["never positive"] == 0).all()  # all-zero label kept, not filtered
    record = M.main(_argv(toy, first, "--resume", models=("rf",)))
    assert record["cells"][0]["status"] == "reused"
