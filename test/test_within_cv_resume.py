"""Paper-profile prediction cells and resume semantics of the within-version runner."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lncRNA_CIBCB2025-main"))

from scripts import felipe_paper_models, prediction_io  # noqa: E402
from scripts import run_inductive_within_cv as runner  # noqa: E402

# Captured once so monkeypatched counters never wrap each other.
_ORIGINAL_BUILD = felipe_paper_models.build_paper_model
N_ROWS = 40
N_FEATURES = 8
N_LABELS = 44  # all44 label space; the size check is exercised, not bypassed
N_SPLITS = 3
FROZEN = [
    runner.RUNNER_SOURCE,
    "scripts/felipe_paper_models.py",
    "scripts/prediction_io.py",
    "scripts/experiment_contract.py",
    *felipe_paper_models.MODEL_SOURCES.values(),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frozen(path: Path, *, corrupt: str | None = None) -> Path:
    rows = []
    for rel in FROZEN:
        digest = _sha(ROOT / rel)
        if rel == corrupt:
            digest = "0" * 64
        rows.append({"path": rel, "sha256": digest, "role": "test"})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_inputs(root: Path, *, label_space: str = "all44", extra_row: bool = False) -> Path:
    rng = np.random.RandomState(0)
    ids = [f"RNA{i:03d}" for i in range(N_ROWS)]
    x = pd.DataFrame(rng.rand(N_ROWS, N_FEATURES), index=ids,
                     columns=[f"f{j}" for j in range(N_FEATURES)])
    # Deliberately non-alphabetical label order, including an all-zero label.
    labels = [f"disease_{(7 * j) % N_LABELS:02d}" for j in range(N_LABELS)]
    y = pd.DataFrame(rng.randint(0, 2, size=(N_ROWS, N_LABELS)), index=ids, columns=labels)
    y[labels[3]] = 0
    x.index.name = y.index.name = "ID"
    x.to_csv(root / "X.csv")
    y.to_csv(root / "Y.csv")

    order = rng.permutation(N_ROWS)
    tests = np.array_split(order, N_SPLITS)
    split_cols = {}
    for fold, test in enumerate(tests, start=1):
        train = sorted(set(range(N_ROWS)) - set(test))
        split_cols[f"train_fold{fold}"] = pd.Series(train)
        split_cols[f"test_fold{fold}"] = pd.Series(test)
    pd.DataFrame(split_cols).to_csv(root / "splits.csv", index=False)

    row = {
        "version": "v1",
        "feature_set": "toy",
        "feature_key": "toy_seq",
        "label_space": label_space,
        "x_path": (root / "X.csv").as_posix(),
        "y_path": (root / "Y.csv").as_posix(),
        "split_path": (root / "splits.csv").as_posix(),
        "n_samples": N_ROWS,
        "n_features": N_FEATURES,
        "n_labels": N_LABELS,
        "x_sha256": _sha(root / "X.csv"),
        "y_sha256": _sha(root / "Y.csv"),
        "split_sha256": _sha(root / "splits.csv"),
    }
    rows = [row]
    if extra_row:  # a second, unselected row changes the manifest bytes only
        rows.append({**row, "version": "v2"})
    manifest = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


class SimulatedCrash(RuntimeError):
    """Stands in for a killed process part-way through a run."""


class FitCounter:
    """Shrink paper models for speed and count / optionally interrupt fits."""

    def __init__(self, fail_on_fit: int | None = None):
        self.fits: list[int] = []
        self.fail_on_fit = fail_on_fit

    def __call__(self, key, *, n_jobs=None):
        model = _ORIGINAL_BUILD(key, n_jobs=n_jobs)
        model.n_estimators = 5
        original_fit = model.fit

        def fit(x, y):
            if self.fail_on_fit is not None and len(self.fits) + 1 == self.fail_on_fit:
                raise SimulatedCrash("simulated interruption")
            self.fits.append(len(x))
            return original_fit(x, y)

        model.fit = fit
        return model


@pytest.fixture
def env(tmp_path, monkeypatch):
    counter = FitCounter()
    monkeypatch.setattr(runner.felipe_paper_models, "build_paper_model", counter)
    manifest = _write_inputs(tmp_path)
    frozen = _write_frozen(tmp_path / "frozen.csv")
    return {"tmp": tmp_path, "manifest": manifest, "frozen": frozen, "counter": counter,
            "monkeypatch": monkeypatch}


def _argv(env, outdir: Path, *extra: str) -> list[str]:
    return [
        "--manifest", env["manifest"].as_posix(),
        "--model-profile", "paper",
        "--models", "rf",
        "--n-splits", str(N_SPLITS),
        "--frozen-sources", env["frozen"].as_posix(),
        "--model-n-jobs", "1",
        "--outdir", outdir.as_posix(),
        *extra,
    ]


def _cell_dir(outdir: Path, model: str = "rf") -> Path:
    return outdir / "within" / "v1__all44__toy_seq__paper" / model


def _read_all(outdir: Path) -> dict[str, pd.DataFrame]:
    cells = {}
    for fold in range(1, N_SPLITS + 1):
        frame, _ = prediction_io.read_prediction_cell(_cell_dir(outdir), f"fold{fold:02d}")
        cells[f"fold{fold:02d}"] = frame
    return cells


def test_cells_have_ids_label_order_contract_and_no_metrics(env):
    outdir = env["tmp"] / "out"
    runner.main(_argv(env, outdir))

    y = pd.read_csv(env["tmp"] / "Y.csv", index_col=0)
    y.index = y.index.astype(str)
    splits = pd.read_csv(env["tmp"] / "splits.csv")
    seen: list[str] = []
    for fold in range(1, N_SPLITS + 1):
        frame, contract = prediction_io.read_prediction_cell(_cell_dir(outdir), f"fold{fold:02d}")
        test_idx = splits[f"test_fold{fold}"].dropna().astype(int).tolist()
        assert frame.index.tolist() == y.index[test_idx].tolist()
        assert list(frame.columns) == list(y.columns)
        seen.extend(frame.index)
        assert contract["kind"] == "within" and contract["version"] == "v1"
        assert contract["label_space"] == "all44" and contract["feature_key"] == "toy_seq"
        assert contract["model"] == "rf" and contract["model_profile"] == "paper"
        assert contract["seed"] == 0 and contract["fold"] == fold
        assert contract["n_folds_total"] == N_SPLITS
        assert contract["effective_params"]["n_estimators"] == 5
        assert contract["fitted_selection"] == {}
        assert set(contract["input_hashes"]) == {"x", "y", "split", "manifest"}
        assert contract["input_hashes"]["manifest"] == _sha(env["manifest"])
        assert contract["truth_y_sha256"] == _sha(env["tmp"] / "Y.csv")
        assert contract["truth_y_path"] == (env["tmp"] / "Y.csv").resolve().as_posix()
        assert set(contract["code_hashes"]) == set(FROZEN)
        assert set(contract["execution"]) == {"n_jobs", "fold_workers", "hostname"}
        assert "library_versions" in contract
    assert sorted(seen) == sorted(y.index)
    assert len(env["counter"].fits) == N_SPLITS

    produced = {p.name for p in outdir.rglob("*") if p.is_file()}
    assert not any("performance" in name or "summary" in name for name in produced)
    assert not (outdir / "performance").exists() and not (outdir / "predictions").exists()
    records = list((outdir / "run_configs").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["completed"] is True and record["metrics_written"] is False
    assert [cell["outcome"] for cell in record["cells"]] == ["fitted"] * N_SPLITS


def test_interrupted_run_resumes_only_missing_folds(env):
    reference = env["tmp"] / "reference"
    runner.main(_argv(env, reference))

    interrupted = env["tmp"] / "interrupted"
    failing = FitCounter(fail_on_fit=2)
    env["monkeypatch"].setattr(runner.felipe_paper_models, "build_paper_model", failing)
    with pytest.raises(SimulatedCrash):
        runner.main(_argv(env, interrupted))
    assert failing.fits and len(failing.fits) == 1
    assert prediction_io.cell_status(_cell_dir(interrupted), "fold01") == "complete"
    assert prediction_io.cell_status(_cell_dir(interrupted), "fold02") == "missing"

    resumed = FitCounter()
    env["monkeypatch"].setattr(runner.felipe_paper_models, "build_paper_model", resumed)
    runner.main(_argv(env, interrupted, "--resume"))
    assert len(resumed.fits) == N_SPLITS - 1

    ref, got = _read_all(reference), _read_all(interrupted)
    for stem in ref:
        pd.testing.assert_frame_equal(ref[stem], got[stem])

    # A second resume reuses everything and fits nothing.
    again = FitCounter()
    env["monkeypatch"].setattr(runner.felipe_paper_models, "build_paper_model", again)
    runner.main(_argv(env, interrupted, "--resume"))
    assert again.fits == []


def test_corrupted_cell_is_quarantined_and_refit(env):
    outdir = env["tmp"] / "out"
    runner.main(_argv(env, outdir))
    csv = _cell_dir(outdir) / "fold02.predictions.csv"
    original = csv.read_bytes()
    csv.write_bytes(original.replace(b"RNA", b"RNX", 1))

    counter = FitCounter()
    env["monkeypatch"].setattr(runner.felipe_paper_models, "build_paper_model", counter)
    runner.main(_argv(env, outdir, "--resume"))
    assert len(counter.fits) == 1
    rejected = list((_cell_dir(outdir) / "_rejected").glob("*/fold02.predictions.csv"))
    assert len(rejected) == 1 and rejected[0].read_bytes() != original
    assert prediction_io.cell_status(_cell_dir(outdir), "fold02") == "complete"
    assert csv.read_bytes() == original  # deterministic refit


def test_existing_cell_without_resume_aborts_before_fitting(env):
    outdir = env["tmp"] / "out"
    runner.main(_argv(env, outdir, "--folds", "2"))
    counter = FitCounter()
    env["monkeypatch"].setattr(runner.felipe_paper_models, "build_paper_model", counter)
    with pytest.raises(FileExistsError, match="--resume or a fresh --outdir"):
        runner.main(_argv(env, outdir))
    assert counter.fits == []
    # Selecting only folds with no files is allowed without --resume.
    runner.main(_argv(env, outdir, "--folds", "1", "3"))
    assert len(counter.fits) == 2


def test_changed_frozen_source_refuses_to_run(env):
    outdir = env["tmp"] / "out"
    _write_frozen(env["frozen"], corrupt="lncRNA_CIBCB2025-main/RF/rf.py")
    with pytest.raises(prediction_io.FrozenSourceError, match="RF/rf.py"):
        runner.main(_argv(env, outdir))
    assert env["counter"].fits == [] and not outdir.exists()


def test_unpinned_runner_refuses_to_run(env):
    outdir = env["tmp"] / "out"
    table = pd.read_csv(env["frozen"])
    table[table["path"] != runner.RUNNER_SOURCE].to_csv(env["frozen"], index=False)
    with pytest.raises(prediction_io.FrozenSourceError, match="does not pin"):
        runner.main(_argv(env, outdir))


def test_threshold_mode_is_rejected_with_paper_profile(env):
    with pytest.raises(SystemExit):
        runner.main(_argv(env, env["tmp"] / "out", "--threshold-mode", "fixed"))
    with pytest.raises(SystemExit):
        runner.main(_argv(env, env["tmp"] / "out", "--folds", "1", "--max-folds", "1"))


def test_manifest_hash_change_invalidates_and_refits(env):
    outdir = env["tmp"] / "out"
    runner.main(_argv(env, outdir))
    _write_inputs(env["tmp"], extra_row=True)  # same data, different manifest bytes

    counter = FitCounter()
    env["monkeypatch"].setattr(runner.felipe_paper_models, "build_paper_model", counter)
    runner.main(_argv(env, outdir, "--resume", "--versions", "v1"))
    assert len(counter.fits) == N_SPLITS
    assert len(list((_cell_dir(outdir) / "_rejected").glob("*/fold*.done"))) == N_SPLITS
    for fold in range(1, N_SPLITS + 1):
        _, contract = prediction_io.read_prediction_cell(_cell_dir(outdir), f"fold{fold:02d}")
        assert contract["input_hashes"]["manifest"] == _sha(env["manifest"])


def test_label_space_is_required_and_size_checked(env):
    manifest = pd.read_csv(env["manifest"])
    manifest.drop(columns="label_space").to_csv(env["manifest"], index=False)
    with pytest.raises(ValueError, match="label_space"):
        runner.main(_argv(env, env["tmp"] / "a"))
    _write_inputs(env["tmp"], label_space="shared73")
    with pytest.raises(ValueError, match="requires 73 labels"):
        runner.main(_argv(env, env["tmp"] / "b"))
    assert env["counter"].fits == []


def test_fold_workers_write_every_cell(env):
    outdir = env["tmp"] / "out"
    runner.main(_argv(env, outdir, "--fold-workers", "3"))
    serial = env["tmp"] / "serial"
    runner.main(_argv(env, serial))
    ref, got = _read_all(serial), _read_all(outdir)
    for stem in ref:
        pd.testing.assert_frame_equal(ref[stem], got[stem])


def test_current_profile_keys_include_threshold_mode(env):
    outdir = env["tmp"] / "current"
    base = [
        "--manifest", env["manifest"].as_posix(),
        "--models", "rf",
        "--n-splits", str(N_SPLITS),
        "--model-n-jobs", "1",
        "--outdir", outdir.as_posix(),
    ]
    runner.main(base)  # default resolves to fixed
    runner.main(base + ["--threshold-mode", "youden", "--resume"])
    perf = outdir / "performance"
    assert (perf / "v1_toy_seq_rf_fixed_performance.csv").is_file()
    assert (perf / "v1_toy_seq_rf_youden_performance.csv").is_file()
    summary = pd.read_csv(outdir / "within_version_cv_summary.csv")
    assert sorted(summary["threshold_mode"]) == ["fixed", "youden"]
    assert not (outdir / "within").exists()
