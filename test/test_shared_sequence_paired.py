"""Synthetic tests for scripts/run_felipe_shared_sequence_paired.py (no real data, no real fits
except one tiny RF)."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lncRNA_CIBCB2025-main"))

from scripts import prediction_io  # noqa: E402
from scripts import run_felipe_shared_sequence_paired as M  # noqa: E402
from scripts.felipe_paper_models import MODEL_SOURCES  # noqa: E402


LABELS = ["disease a", "disease b", "disease c", "never positive"]
FEATURES = [f"feat_{i}" for i in range(M.N_FEATURES)]
SHARED_SEQS = ["ACGUACGUAAGG", "GGGCCCAUAUAU", "UUUUACGACGAC"]


def _seq_hash(seq: str) -> str:
    return hashlib.sha256(seq.encode("utf-8")).hexdigest()


def _rand_seq(rng, n=14):
    return "".join(rng.choice(list("ACGU"), size=n))


# --------------------------------------------------------------------------- fixture builder

class Toy:
    """Synthetic V1/V2 inputs + cohort dir; mutate ``self.*`` then call ``write()``."""

    def __init__(self, tmp: Path):
        self.tmp = tmp
        rng = np.random.default_rng(11)
        shared_rows = rng.normal(size=(len(SHARED_SEQS), M.N_FEATURES))
        # V1: 3 shared + 8 train; V2: 3 shared (+1 duplicate ID of shared 0) + 10 train.
        self.v1_ids = [f"V1_S{i}" for i in range(3)] + [f"V1_T{i}" for i in range(8)]
        self.v1_seqs = list(SHARED_SEQS) + [_rand_seq(rng) for _ in range(8)]
        self.v2_ids = [f"V2_T{i}" for i in range(5)] + ["V2_S0", "V2_S0dup", "V2_S1", "V2_S2"] + [
            f"V2_T{i}" for i in range(5, 10)
        ]
        v2_train_seqs = [_rand_seq(rng) for _ in range(10)]
        self.v2_seqs = v2_train_seqs[:5] + [
            SHARED_SEQS[0], SHARED_SEQS[0].lower().replace("u", "t"), SHARED_SEQS[1], SHARED_SEQS[2]
        ] + v2_train_seqs[5:]
        x1 = rng.normal(size=(len(self.v1_ids), M.N_FEATURES))
        x1[:3] = shared_rows
        x2 = rng.normal(size=(len(self.v2_ids), M.N_FEATURES))
        x2[5], x2[6], x2[7], x2[8] = shared_rows[0], shared_rows[0], shared_rows[1], shared_rows[2]
        self.x = {
            "v1": pd.DataFrame(x1, index=pd.Index(self.v1_ids, name="ID"), columns=FEATURES),
            "v2": pd.DataFrame(x2, index=pd.Index(self.v2_ids, name="ID"), columns=FEATURES),
        }
        self.y = {}
        for version, x in self.x.items():
            y = np.zeros((len(x), len(LABELS)), dtype=int)
            y[:, 0] = rng.random(len(x)) < 0.5
            y[:, 1] = rng.random(len(x)) < 0.4
            y[:, 2] = rng.random(len(x)) < 0.3
            self.y[version] = pd.DataFrame(y, index=x.index, columns=LABELS)
        hashes = [_seq_hash(s) for s in SHARED_SEQS]
        order = np.argsort(hashes)
        self.test_x = pd.DataFrame(
            shared_rows[order], index=pd.Index([hashes[i] for i in order], name="sequence_sha256"),
            columns=FEATURES,
        )
        self.train_ids = {
            "v1": [f"V1_T{i}" for i in range(8)],
            "v2": [f"V2_T{i}" for i in range(10)],
        }
        self.excluded_ids = {
            "v1": ["V1_S0", "V1_S1", "V1_S2"],
            "v2": ["V2_S0", "V2_S0dup", "V2_S1", "V2_S2"],
        }
        self.test_float_format = "%.17g"
        self.schema = M.COHORT_SCHEMA
        self.frozen_root = tmp / "frozen_root"
        self.cohort = tmp / "cohort"
        self.write()

    def write(self):
        tmp = self.tmp
        inputs = tmp / "inputs"
        inputs.mkdir(exist_ok=True)
        label_map = inputs / "label_map.csv"
        pd.DataFrame({"order": range(len(LABELS)), "canonical_name": LABELS}).to_csv(
            label_map, index=False
        )
        rows = []
        for version in ("v1", "v2"):
            x_path, y_path = inputs / f"{version}_X.csv", inputs / f"{version}_Y.csv"
            self.x[version].to_csv(x_path, float_format="%.17g")
            self.y[version].to_csv(y_path)
            rows.append({
                "version": version, "label_space": "shared73", "feature_key": M.FEATURE_KEY,
                "x_path": x_path.as_posix(), "y_path": y_path.as_posix(),
                "n_samples": len(self.x[version]), "n_features": M.N_FEATURES,
                "n_labels": len(LABELS),
                "x_sha256": prediction_io.sha256_file(x_path),
                "y_sha256": prediction_io.sha256_file(y_path),
                "label_map_path": label_map.as_posix(),
                "label_map_sha256": prediction_io.sha256_file(label_map),
            })
        manifest = inputs / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        v1_raw, v2_raw = inputs / "v1_raw.csv", inputs / "v2_raw.csv"
        pd.DataFrame({"ID": self.v1_ids, "seqs": self.v1_seqs}).to_csv(v1_raw)  # Unnamed: 0
        pd.DataFrame({
            "ID": self.v2_ids + ["V2_NOSEQ"], "seqs": self.v2_seqs + [None]
        }).to_csv(v2_raw, index=False)
        self.paths = {"label_map": label_map, "manifest": manifest,
                      "v1_raw": v1_raw, "v2_raw": v2_raw}

        cohort = self.cohort
        cohort.mkdir(exist_ok=True)
        self.test_x.to_csv(cohort / "test_X.csv", float_format=self.test_float_format)
        for version in ("v1", "v2"):
            (cohort / f"train_ids_{version}.txt").write_text(
                "\n".join(self.train_ids[version]) + "\n", encoding="utf-8")
            (cohort / f"excluded_ids_{version}.txt").write_text(
                "\n".join(self.excluded_ids[version]) + "\n", encoding="utf-8")
        outputs = {p.name: prediction_io.sha256_file(p) for p in sorted(cohort.iterdir())
                   if p.name != "cohort_manifest.json"}
        input_files = [label_map, manifest, v1_raw, v2_raw] + [
            Path(r[k]) for r in rows for k in ("x_path", "y_path")
        ]
        body = {
            "schema": self.schema,
            "inputs": {p.as_posix(): prediction_io.sha256_file(p) for p in input_files},
            "outputs": outputs,
        }
        (cohort / "cohort_manifest.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
        self.write_frozen()

    def write_frozen(self, *, corrupt=None, omit=None, extra_missing=False):
        paths = list(M.REQUIRED_FROZEN) + list(MODEL_SOURCES.values())
        rows = []
        for rel in paths:
            target = self.frozen_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            source = ROOT / rel
            if source.is_file():
                shutil.copyfile(source, target)
            elif not target.exists():
                target.write_text("# stand-in for a not-yet-written script\n", encoding="utf-8")
            if rel == omit:
                continue
            digest = prediction_io.sha256_file(target)
            if rel == corrupt:
                digest = "0" * 64
            rows.append({"path": rel, "sha256": digest, "role": "test"})
        if extra_missing:
            rows.append({"path": "scripts/does_not_exist.py", "sha256": "0" * 64, "role": "test"})
        self.frozen = self.tmp / "frozen.csv"
        pd.DataFrame(rows).to_csv(self.frozen, index=False)

    def argv(self, outdir, *extra, models=("rf", "ipcarf")):
        return [
            "--cohort-dir", str(self.cohort),
            "--outdir", str(outdir),
            "--transfer-manifest", str(self.paths["manifest"]),
            "--label-map", str(self.paths["label_map"]),
            "--v1-raw", str(self.paths["v1_raw"]),
            "--v2-raw", str(self.paths["v2_raw"]),
            "--frozen-sources", str(self.frozen),
            "--models", *models,
            "--model-n-jobs", "1",
            *extra,
        ]


class Spy:
    """Deterministic stand-in model recording every fit/predict call."""

    def __init__(self):
        self.fits: list[dict] = []
        self.predicts: list[dict] = []

    def install(self, monkeypatch):
        spy = self

        class FakeModel:
            random_state = 0

            def __init__(self, key, n_jobs):
                self.key, self.n_jobs = key, n_jobs

            def fit(self, x, y):
                spy.fits.append({"model": self.key, "x_ids": list(x.index),
                                 "y_ids": list(y.index), "x_cols": list(x.columns)})
                self.prevalence = y.to_numpy(dtype=float).mean(axis=0)
                return self

            def predict_proba(self, x):  # only X; any label argument would be a TypeError
                assert isinstance(x, pd.DataFrame)
                assert x.columns.tolist() == FEATURES
                spy.predicts.append({"model": self.key, "x_ids": list(x.index)})
                bump = 0.01 * np.tanh(x.to_numpy()[:, : len(self.prevalence)])
                return np.clip(self.prevalence + bump, 0.0, 1.0)

        monkeypatch.setattr(M, "build_paper_model", lambda key, n_jobs=None: FakeModel(key, n_jobs))
        monkeypatch.setattr(M, "effective_params",
                            lambda key, model: {"class": "Fake", "random_state": 0, "key": key})
        monkeypatch.setattr(M, "fitted_selection", lambda key, model: {"fake": key})
        return self


@pytest.fixture
def toy(tmp_path, monkeypatch):
    t = Toy(tmp_path)
    monkeypatch.setattr(M, "FROZEN_ROOT", t.frozen_root)
    return t


@pytest.fixture
def spy(monkeypatch):
    return Spy().install(monkeypatch)


def _cell(outdir, version, model):
    return M.cell_dir_for(Path(outdir), version, model)


# --------------------------------------------------------------------------- happy path

def test_writes_cells_with_contract_and_order(toy, spy):
    outdir = toy.tmp / "out"
    record = M.main(toy.argv(outdir))
    assert record["outcome"] == "complete"
    # sequential order: v1 rf, v1 ipcarf, v2 rf, v2 ipcarf
    assert [(c["train_version"], c["model"], c["status"]) for c in record["cells"]] == [
        ("v1", "rf", "fitted"), ("v1", "ipcarf", "fitted"),
        ("v2", "rf", "fitted"), ("v2", "ipcarf", "fitted"),
    ]
    test_ids = toy.test_x.index.tolist()
    for fit, version in zip(spy.fits, ["v1", "v1", "v2", "v2"]):
        assert fit["x_ids"] == toy.train_ids[version] == fit["y_ids"]
        assert not set(fit["x_ids"]) & (set(test_ids) | set(toy.excluded_ids[version]))
    assert all(p["x_ids"] == test_ids for p in spy.predicts)

    frame, contract = prediction_io.read_prediction_cell(_cell(outdir, "v2", "rf"), M.STEM)
    assert frame.index.tolist() == test_ids == sorted(test_ids)
    assert frame.columns.tolist() == LABELS
    caller_fields = set(contract) - set(prediction_io.DERIVED_FIELDS)
    assert caller_fields == {
        "kind", "train_version", "label_space", "feature_key", "model", "model_profile",
        "effective_params", "seed", "test_population", "test_x_sha256", "test_row_ids_sha256",
        "train_ids_sha256", "n_train", "cohort_manifest_sha256", "input_hashes", "code_hashes",
        "library_versions", "probe", "fitted_selection", "execution",
    }
    assert contract["kind"] == "shared_sequence_paired"
    assert contract["train_version"] == "v2" and contract["model"] == "rf"
    assert contract["test_population"] == "shared_sequence_all48"
    assert contract["probe"] is None
    assert contract["n_train"] == 10
    assert contract["train_ids_sha256"] == prediction_io.sha256_lines(toy.train_ids["v2"])
    assert contract["test_row_ids_sha256"] == prediction_io.sha256_lines(test_ids)
    assert contract["test_x_sha256"] == prediction_io.sha256_file(toy.cohort / "test_X.csv")
    assert contract["cohort_manifest_sha256"] == prediction_io.sha256_file(
        toy.cohort / "cohort_manifest.json")
    assert set(contract["input_hashes"]) == {
        "train_x", "train_y", "transfer_manifest", "label_map", "test_x"}
    assert set(contract["execution"]) == {"n_jobs", "hostname", "fit_seconds"}
    assert "scripts/build_shared_sequence_cohort.py" in contract["code_hashes"]
    records = list((outdir / "run_records").glob("*.json"))
    assert len(records) == 1
    body = json.loads(records[0].read_text())
    assert body["leakage_checks"]["v2"]["n_excluded"] == 4


def test_real_rf_tiny_fit(toy):
    outdir = toy.tmp / "out"
    M.main(toy.argv(outdir, "--train-versions", "v1", models=("rf",)))
    frame, contract = prediction_io.read_prediction_cell(_cell(outdir, "v1", "rf"), M.STEM)
    assert frame.shape == (3, len(LABELS))
    assert ((frame.to_numpy() >= 0) & (frame.to_numpy() <= 1)).all()
    assert contract["effective_params"]["class"].endswith("RF")


# --------------------------------------------------------------------------- refusals

@pytest.mark.parametrize("kind", ["missing", "changed", "lacking"])
def test_frozen_table_refusals(toy, spy, kind):
    if kind == "missing":
        toy.write_frozen(extra_missing=True)
    elif kind == "changed":
        toy.write_frozen(corrupt="scripts/prediction_io.py")
    else:
        toy.write_frozen(omit="scripts/build_shared_sequence_cohort.py")
    with pytest.raises(prediction_io.FrozenSourceError):
        M.main(toy.argv(toy.tmp / "out"))
    assert spy.fits == []


def test_cohort_output_hash_mismatch_refused(toy, spy):
    path = toy.cohort / "train_ids_v1.txt"
    path.write_text(path.read_text() + "V1_S0\n")
    with pytest.raises(ValueError, match="changed train_ids_v1.txt"):
        M.main(toy.argv(toy.tmp / "out"))
    assert spy.fits == []


def test_cohort_schema_refused(toy, spy):
    toy.schema = "shared_sequence_cohort/0"
    toy.write()
    with pytest.raises(ValueError, match="schema"):
        M.main(toy.argv(toy.tmp / "out"))


def test_cohort_from_other_inputs_refused(toy, spy):
    body = json.loads((toy.cohort / "cohort_manifest.json").read_text())
    body["inputs"] = {k: v for k, v in body["inputs"].items() if not k.endswith("manifest.csv")}
    (toy.cohort / "cohort_manifest.json").write_text(json.dumps(body))
    with pytest.raises(ValueError, match="different transfer manifest"):
        M.main(toy.argv(toy.tmp / "out"))


def test_training_id_with_test_sequence_refused(toy, spy):
    # V2_T0 stays a training ID but its raw sequence (DNA spelling) is a shared test sequence.
    toy.v2_seqs[0] = SHARED_SEQS[2].replace("U", "T")
    toy.write()
    with pytest.raises(M.LeakageError, match="share a test sequence"):
        M.main(toy.argv(toy.tmp / "out"))
    assert spy.fits == []


def test_excluded_list_must_match_raw_hashes(toy, spy):
    toy.excluded_ids["v2"].remove("V2_S0dup")
    toy.train_ids["v2"].insert(5, "V2_S0dup")
    toy.write()
    with pytest.raises(M.LeakageError):
        M.main(toy.argv(toy.tmp / "out"))
    assert spy.fits == []


def test_exact_x_row_overlap_refused(toy, spy):
    toy.x["v1"].loc["V1_T3"] = toy.test_x.iloc[1].to_numpy()
    toy.write()
    with pytest.raises(M.LeakageError, match="exact test rows"):
        M.main(toy.argv(toy.tmp / "out"))
    assert spy.fits == []


def test_test_row_must_equal_an_excluded_row(toy, spy):
    toy.test_x.iloc[0, 7] += 1e-9
    toy.write()
    with pytest.raises(M.LeakageError, match="equals no excluded"):
        M.main(toy.argv(toy.tmp / "out"))


def test_unsorted_test_x_refused(toy, spy):
    toy.test_x = toy.test_x.iloc[::-1]
    toy.write()
    with pytest.raises(ValueError, match="sorted ascending"):
        M.main(toy.argv(toy.tmp / "out"))


def test_training_ids_out_of_manifest_order_refused(toy, spy):
    toy.train_ids["v1"] = toy.train_ids["v1"][::-1]
    toy.write()
    with pytest.raises(ValueError, match="manifest X row order"):
        M.main(toy.argv(toy.tmp / "out"))


# --------------------------------------------------------------------------- resume

def test_resume_reuses_without_fit(toy, spy):
    outdir = toy.tmp / "out"
    M.main(toy.argv(outdir))
    assert len(spy.fits) == 4
    record = M.main(toy.argv(outdir, "--resume"))
    assert len(spy.fits) == 4 and len(spy.predicts) == 4
    assert {c["status"] for c in record["cells"]} == {"reused"}
    assert len(list((outdir / "run_records").glob("*.json"))) == 2


def test_existing_cells_refused_without_resume(toy, spy):
    outdir = toy.tmp / "out"
    M.main(toy.argv(outdir, "--train-versions", "v1", models=("rf",)))
    with pytest.raises(FileExistsError):
        M.main(toy.argv(outdir))
    assert len(spy.fits) == 1


def test_corrupted_predictions_quarantined_and_redone(toy, spy):
    outdir = toy.tmp / "out"
    M.main(toy.argv(outdir))
    cell_dir = _cell(outdir, "v1", "ipcarf")
    csv = prediction_io.cell_paths(cell_dir, M.STEM)["predictions"]
    csv.write_text(csv.read_text().replace("0.", "0.9", 1))
    record = M.main(toy.argv(outdir, "--resume"))
    statuses = {(c["train_version"], c["model"]): c["status"] for c in record["cells"]}
    assert statuses[("v1", "ipcarf")] == "fitted"
    assert [s for k, s in statuses.items() if k != ("v1", "ipcarf")] == ["reused"] * 3
    assert len(spy.fits) == 5
    assert list((cell_dir / "_rejected").iterdir())
    prediction_io.read_prediction_cell(cell_dir, M.STEM)


def test_changed_cohort_invalidates_old_cells(toy, spy):
    outdir = toy.tmp / "out"
    M.main(toy.argv(outdir, "--train-versions", "v1", models=("rf",)))
    old_sha = prediction_io.sha256_file(toy.cohort / "test_X.csv")
    toy.test_float_format = "%.20g"  # same values, different test_X bytes
    toy.write()
    assert prediction_io.sha256_file(toy.cohort / "test_X.csv") != old_sha
    record = M.main(toy.argv(outdir, "--train-versions", "v1", "--resume", models=("rf",)))
    assert record["cells"][0]["status"] == "fitted"
    assert "quarantined_to" in record["cells"][0]
    _, contract = prediction_io.read_prediction_cell(_cell(outdir, "v1", "rf"), M.STEM)
    assert contract["test_x_sha256"] != old_sha
    assert len(spy.fits) == 2


# --------------------------------------------------------------------------- limited-row runs

def test_limited_rows_refused_outside_marked_outdir(toy, spy):
    outdir = toy.tmp / "out"
    assert "probe" not in outdir.as_posix()
    with pytest.raises(ValueError, match="containing 'probe'"):
        M.main(toy.argv(outdir, "--probe-max-train-rows", "3"))
    assert spy.fits == [] and not outdir.exists()


def test_probe_recorded(toy, spy):
    outdir = toy.tmp / "felipe_probes" / "shared_sequence"
    record = M.main(toy.argv(outdir, "--probe-max-train-rows", "3", models=("rf",)))
    assert record["probe"] == {"max_train_rows": 3}
    assert [f["x_ids"] for f in spy.fits] == [toy.train_ids["v1"][:3], toy.train_ids["v2"][:3]]
    _, contract = prediction_io.read_prediction_cell(_cell(outdir, "v2", "rf"), M.STEM)
    assert contract["probe"] == {"max_train_rows": 3}
    assert contract["n_train"] == 3
    assert contract["train_ids_sha256"] == prediction_io.sha256_lines(toy.train_ids["v2"][:3])
