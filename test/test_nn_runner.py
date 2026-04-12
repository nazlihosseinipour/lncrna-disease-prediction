import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import run_all_nn_features as runner


def test_nn_runner_continues_after_method_failure(monkeypatch, tmp_path):
    seqs_csv = tmp_path / "seqs.csv"
    pd.DataFrame(
        {
            "id": ["s1", "s2"],
            "seq": ["AUGC", "UUAA"],
        }
    ).to_csv(seqs_csv, index=False)

    outdir = tmp_path / "out"
    monkeypatch.setattr(runner.NNFeatures, "METHOD_MAP", {100: "mp_sequence", 103: "aido_sequence"})

    def fake_run(group, method_id, seqs, **kwargs):
        assert group == "nn"
        if method_id == 103:
            raise RuntimeError("mock aido failure")
        return ["f0"], pd.DataFrame({"sample_id": kwargs["sample_ids"], "f0": [1.0] * len(seqs)})

    monkeypatch.setattr(runner.FeatureExtractor, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_nn_features.py",
            "--seqs_csv",
            str(seqs_csv),
            "--outdir",
            str(outdir),
            "--version_name",
            "vtest",
        ],
    )

    runner.main()

    nn_dir = outdir / "vtest" / "nn"
    assert (nn_dir / "seqs_mp_sequence.csv").exists()
    fail_path = nn_dir / "seqs_nn_failures.csv"
    assert fail_path.exists()
    failures = pd.read_csv(fail_path)
    assert failures.shape[0] == 1
    assert int(failures.loc[0, "method_id"]) == 103
    assert failures.loc[0, "method_name"] == "aido_sequence"
    assert "mock aido failure" in failures.loc[0, "error"]


def test_nn_runner_can_filter_methods(monkeypatch, tmp_path):
    seqs_csv = tmp_path / "seqs.csv"
    pd.DataFrame(
        {
            "id": ["s1", "s2"],
            "seq": ["AUGC", "UUAA"],
        }
    ).to_csv(seqs_csv, index=False)

    outdir = tmp_path / "out"
    monkeypatch.setattr(runner.NNFeatures, "METHOD_MAP", {100: "mp_sequence", 103: "aido_sequence"})

    seen = []

    def fake_run(group, method_id, seqs, **kwargs):
        seen.append(method_id)
        return ["f0"], pd.DataFrame({"sample_id": kwargs["sample_ids"], "f0": [1.0] * len(seqs)})

    monkeypatch.setattr(runner.FeatureExtractor, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_all_nn_features.py",
            "--seqs_csv",
            str(seqs_csv),
            "--outdir",
            str(outdir),
            "--version_name",
            "vtest",
            "--methods",
            "100",
        ],
    )

    runner.main()

    assert seen == [100]
    nn_dir = outdir / "vtest" / "nn"
    assert (nn_dir / "seqs_mp_sequence.csv").exists()
    assert not (nn_dir / "seqs_aido_sequence.csv").exists()
