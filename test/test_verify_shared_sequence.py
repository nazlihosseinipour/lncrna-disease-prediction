"""Tests for scripts/verify_felipe_shared_sequence.py on tiny synthetic data (tmp_path)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.prediction_io import sha256_file, sha256_lines, verify_frozen_sources, write_prediction_cell
from scripts import verify_felipe_shared_sequence as vf

LABELS = ["disease a", "disease b", "disease c", "disease d"]
N_BOOT = 40
MODELS = vf.MODELS


def _hash(i: int) -> str:
    return hashlib.sha256(f"seq{i}".encode()).hexdigest()


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, float_format="%.17g", lineterminator="\n")


def build(tmp_path: Path) -> dict:
    """Synthetic project: frozen table, transfer manifest, cohort, 6 cells, scorer outputs."""
    rng = np.random.default_rng(0)
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    for name in ("a.py", "b.py", "c.py"):
        (proj / "src" / name).write_text(f"# {name}\n")
    base = pd.DataFrame({"path": ["src/a.py", "src/b.py"],
                         "sha256": [sha256_file(proj / "src" / n) for n in ("a.py", "b.py")]})
    base_path = proj / "base_frozen.csv"
    base.to_csv(base_path, index=False)
    table = pd.concat([base, pd.DataFrame({"path": ["src/c.py"], "sha256": [sha256_file(proj / "src/c.py")]})])
    table_path = proj / "frozen.csv"
    table.to_csv(table_path, index=False)
    code_hashes = verify_frozen_sources(table_path, proj)

    label_map = proj / "label_map.csv"
    pd.DataFrame({"order": [1, 0, 2, 3], "canonical_name": [LABELS[1], LABELS[0], LABELS[2], LABELS[3]]}).to_csv(
        label_map, index=False)

    # training data: v1 includes a byte-exact trailing-space ID
    inputs = proj / "inputs"
    inputs.mkdir()
    train_ids = {
        "v1": ["BX374997.2 ", "NR_1", "NR_2", "NR_3", "NR_4", "NR_5", "NR_6", "NR_7"],
        "v2": [f"G{i}" for i in range(10)],
    }
    rows = []
    for v in ("v1", "v2"):
        ids = train_ids[v]
        x = pd.DataFrame(rng.random((len(ids), 3)), index=pd.Index(ids, name="ID"), columns=["f1", "f2", "f3"])
        y = pd.DataFrame(rng.integers(0, 2, (len(ids), len(LABELS))), index=pd.Index(ids, name="ID"), columns=LABELS)
        _write_csv(x, inputs / f"{v}_X.csv")
        _write_csv(y, inputs / f"{v}_Y.csv")
        rows.append({"version": v, "label_space": "shared73", "feature_key": "paper_sequence_542",
                     "x_path": str(inputs / f"{v}_X.csv"), "y_path": str(inputs / f"{v}_Y.csv"),
                     "x_sha256": sha256_file(inputs / f"{v}_X.csv"), "y_sha256": sha256_file(inputs / f"{v}_Y.csv"),
                     "label_map_sha256": sha256_file(label_map)})
    manifest = proj / "transfer.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    # cohort: 12 test rows, 10 primary
    cohort = proj / "cohort"
    cohort.mkdir()
    test_ids = sorted(_hash(i) for i in range(12))
    primary = test_ids[:10]
    test_x = pd.DataFrame(rng.random((12, 3)), index=pd.Index(test_ids, name="sequence_sha256"),
                          columns=["f1", "f2", "f3"])
    _write_csv(test_x, cohort / "test_X.csv")
    lab_v1 = pd.DataFrame(rng.integers(0, 2, (12, 4)), index=pd.Index(test_ids, name="sequence_sha256"), columns=LABELS)
    lab_or = pd.DataFrame(rng.integers(0, 2, (12, 4)), index=lab_v1.index, columns=LABELS)
    lab_or.iloc[:, 0] = [0, 1] * 6
    lab_v1.iloc[:, 0] = [1, 0] * 6
    lab_or.to_csv(cohort / "labels_v2_or.csv")
    lab_or.loc[primary].to_csv(cohort / "labels_v2_primary.csv")
    lab_v1.to_csv(cohort / "labels_v1.csv")
    pd.DataFrame({"sequence_sha256": test_ids, "in_primary": [i in primary for i in test_ids],
                  "in_sensitivity": True}).to_csv(cohort / "cohorts.csv", index=False)
    for v in ("v1", "v2"):
        (cohort / f"train_ids_{v}.txt").write_bytes(("\n".join(train_ids[v]) + "\n").encode())
    outputs = {p.name: sha256_file(p) for p in sorted(cohort.iterdir())}
    (cohort / "cohort_manifest.json").write_text(json.dumps({"schema": "shared_sequence_cohort/1",
                                                            "outputs": outputs}))

    results = proj / "results"
    for v in ("v1", "v2"):
        for model in MODELS:
            scores = pd.DataFrame(rng.integers(0, 65, (12, 4)) / 64.0, index=test_ids, columns=LABELS)
            write_prediction_cell(vf.cell_dir(results, v, model), vf.CELL_STEM, scores,
                                  contract(v, model, cohort, train_ids[v], code_hashes, rows, manifest))
    return {"proj": proj, "table": table_path, "base": base_path, "label_map": label_map, "manifest": manifest,
            "cohort": cohort, "results": results, "train_ids": train_ids, "code_hashes": code_hashes,
            "rows": rows, "test_ids": test_ids}


def contract(v, model, cohort, train_ids, code_hashes, rows, manifest, **override):
    row = next(r for r in rows if r["version"] == v)
    out = {
        "kind": "shared_sequence_paired", "train_version": v, "label_space": "shared73",
        "feature_key": "paper_sequence_542", "model": model, "model_profile": "paper",
        "effective_params": {}, "seed": 0, "test_population": "shared_sequence_all48",
        "test_x_sha256": sha256_file(cohort / "test_X.csv"),
        "test_row_ids_sha256": sha256_lines(pd.read_csv(cohort / "test_X.csv", usecols=[0], dtype=str).iloc[:, 0]),
        "train_ids_sha256": sha256_lines(train_ids), "n_train": len(train_ids),
        "cohort_manifest_sha256": sha256_file(cohort / "cohort_manifest.json"),
        "input_hashes": {"train_x": row["x_sha256"], "train_y": row["y_sha256"],
                         "transfer_manifest": sha256_file(manifest), "label_map": row["label_map_sha256"],
                         "test_x": sha256_file(cohort / "test_X.csv")},
        "code_hashes": code_hashes, "library_versions": {}, "probe": None,
        "fitted_selection": {}, "execution": {"n_jobs": 1, "hostname": "t", "fit_seconds": 0.1},
    }
    out.update(override)
    return out


def run_scorer(s: dict) -> Path:
    from scripts import score_felipe_shared_sequence as scorer  # test-only: produces real scorer output

    outdir = s["proj"] / "scores"
    scorer.run(s["results"], s["cohort"], s["manifest"], outdir, n_bootstrap=N_BOOT)
    return outdir


def cells(s, allow_probe=False):
    return vf.check_cells(s["results"], s["cohort"], s["table"], s["label_map"], s["manifest"], s["proj"], allow_probe)


def scores(s, outdir):
    return vf.check_scores(s["results"], s["cohort"], outdir, s["manifest"], s["proj"], n_bootstrap=N_BOOT,
                           boot_combos=(("rf", "v2", "primary", "v1", "micro_roc"),
                                        ("rflda", "v1", "sensitivity", "v2", "micro_auprc")))


def set_aside(d: Path) -> None:
    target = d / "_rejected" / "old"
    target.mkdir(parents=True)
    for suffix in (".done", ".contract.json", ".predictions.csv"):
        (d / f"shared48{suffix}").rename(target / f"shared48{suffix}")


@pytest.fixture()
def setup(tmp_path):
    return build(tmp_path)


def test_passing_synthetic(setup):
    info = cells(setup)
    assert len(info["cells"]) == 6 and info["rejected_dirs"] == []
    out = run_scorer(setup)
    result = scores(setup, out)
    assert result["n_summary_rows"] == 4 * 2 * 2 * 2
    assert all(b["n_valid"] > 0 for b in result["bootstrap_checked"])
    frozen = vf.check_frozen(setup["table"], setup["base"], setup["proj"], new_scripts=("src/c.py",))
    assert frozen["n_rows"] == 3


def test_frozen_missing_new_script_fails(setup):
    with pytest.raises(vf.VerifyError, match="new script missing"):
        vf.check_frozen(setup["table"], setup["base"], setup["proj"], new_scripts=("src/c.py", "src/d.py"))


def test_tampered_prediction_fails(setup):
    path = vf.cell_dir(setup["results"], "v2", "ipcarf") / "shared48.predictions.csv"
    text = path.read_text()
    first_value = text.split("\n")[1].split(",")[1]
    path.write_text(text.replace(first_value, "0.99", 1))
    with pytest.raises(vf.VerifyError, match="prediction hash"):
        cells(setup)


def test_mismatched_test_x_sha_fails(setup):
    s = setup
    d = vf.cell_dir(s["results"], "v1", "rflda")
    set_aside(d)
    frame = pd.DataFrame(0.5, index=s["test_ids"], columns=LABELS)
    write_prediction_cell(d, "shared48", frame, contract("v1", "rflda", s["cohort"], s["train_ids"]["v1"],
                                                         s["code_hashes"], s["rows"], s["manifest"],
                                                         test_x_sha256="0" * 64))
    with pytest.raises(vf.VerifyError, match="test_x_sha256"):
        cells(s)


def test_stray_mixed_cell_fails_but_rejected_is_listed(setup):
    s = setup
    bad = contract("v1", "rf", s["cohort"], s["train_ids"]["v1"], s["code_hashes"], s["rows"], s["manifest"],
                   test_x_sha256="1" * 64)
    frame = pd.DataFrame(0.5, index=s["test_ids"], columns=LABELS)
    write_prediction_cell(vf.cell_dir(s["results"], "v1", "rf") / "_rejected" / "t0", "shared48", frame, bad)
    info = cells(s)
    assert info["rejected_dirs"] == ["paired/train_v1__shared73__paper_sequence_542__paper/rf/_rejected"]
    write_prediction_cell(s["results"] / "elsewhere", "stray", frame, bad)
    with pytest.raises(vf.VerifyError, match="mixed cell"):
        cells(s)


def test_probe_cells_need_allow_probe(setup):
    s = setup
    for v in ("v1", "v2"):
        for model in MODELS:
            d = vf.cell_dir(s["results"], v, model)
            set_aside(d)
            ids = s["train_ids"][v][:5]
            write_prediction_cell(d, "shared48", pd.DataFrame(0.25, index=s["test_ids"], columns=LABELS),
                                  contract(v, model, s["cohort"], ids, s["code_hashes"], s["rows"], s["manifest"],
                                           probe={"max_train_rows": 5}))
    with pytest.raises(vf.VerifyError, match="without --allow-probe"):
        cells(s)
    assert len(cells(s, allow_probe=True)["cells"]) == 6


def test_wrong_summary_metric_fails(setup):
    out = run_scorer(setup)
    summary = pd.read_csv(out / "summary.csv", keep_default_na=False, float_precision="round_trip")
    summary.loc[3, "micro_roc"] = float(summary.loc[3, "micro_roc"]) + 1e-9
    summary.to_csv(out / "summary.csv", index=False, float_format="%.17g", lineterminator="\n")
    with pytest.raises(vf.VerifyError, match="micro_roc") as err:
        scores(setup, out)
    assert "score_run.json output hash mismatch: summary.csv" in str(err.value)


def test_wrong_bootstrap_ci_fails(setup):
    out = run_scorer(setup)
    ci = pd.read_csv(out / "bootstrap_ci.csv", keep_default_na=False, float_precision="round_trip")
    mask = (ci["model"] == "rf") & (ci["train_version"] == "v2") & (ci["cohort"] == "primary") & (
        ci["label_version"] == "v1") & (ci["metric"] == "micro_roc")
    ci.loc[mask, "ci_high"] = ci.loc[mask, "ci_high"] + 1e-6
    ci.to_csv(out / "bootstrap_ci.csv", index=False, float_format="%.17g", lineterminator="\n")
    with pytest.raises(vf.VerifyError, match="bootstrap rf/v2/primary/v1/micro_roc"):
        scores(setup, out)


def test_changed_protected_file_fails(tmp_path):
    proj = tmp_path
    prot = proj / "results" / "final"
    prot.mkdir(parents=True)
    (prot / "a b.csv").write_text("x\n")
    (prot / "c.csv").write_text("y\n")
    snap = proj / "snap.sha256"
    snap.write_text("".join(f"{sha256_file(prot / n)}  results/final/{n}\n" for n in ("a b.csv", "c.csv")))
    info = vf.check_protected(snap, proj, Path("results/final"))
    assert info["n_entries"] == 2 and info["unlisted_under_protected_dir"] == []
    (prot / "new.csv").write_text("z\n")
    assert vf.check_protected(snap, proj, Path("results/final"))["unlisted_under_protected_dir"] == [
        "results/final/new.csv"]
    (prot / "a b.csv").write_text("changed\n")
    with pytest.raises(vf.VerifyError, match="changed results/final/a b.csv"):
        vf.check_protected(snap, proj, Path("results/final"))


def test_trailing_space_ids_preserved(setup, tmp_path):
    path = tmp_path / "ids.txt"
    path.write_bytes(b"BX374997.2 \nNR_1\n")
    assert vf.read_id_list(path) == ["BX374997.2 ", "NR_1"]
    s = setup
    assert vf.load_cohort_basics(s["cohort"])["train_ids"]["v1"][0] == "BX374997.2 "
    cells(s)  # contract hashes were computed with the trailing space
    d = vf.cell_dir(s["results"], "v1", "rf")
    set_aside(d)
    stripped = [i.strip() for i in s["train_ids"]["v1"]]
    write_prediction_cell(d, "shared48", pd.DataFrame(0.5, index=s["test_ids"], columns=LABELS),
                          contract("v1", "rf", s["cohort"], stripped, s["code_hashes"], s["rows"], s["manifest"]))
    with pytest.raises(vf.VerifyError, match="train_ids_sha256"):
        cells(s)


def test_id_list_rejects_blank_and_cr(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_bytes(b"A\n\nB\n")
    with pytest.raises(vf.VerifyError, match="empty line"):
        vf.read_id_list(path)
    path.write_bytes(b"A\r\nB\n")
    with pytest.raises(vf.VerifyError, match="carriage return"):
        vf.read_id_list(path)


def _scorer_fix(s: dict) -> Path:
    """Simulate SS15: the scorer (src/c.py) changes after fitting; write a scoring table."""
    (s["proj"] / "src" / "c.py").write_text("# c.py fixed\n")
    table = pd.read_csv(s["table"], dtype=str)
    table.loc[table["path"] == "src/c.py", "sha256"] = sha256_file(s["proj"] / "src" / "c.py")
    scoring = s["proj"] / "scoring_frozen.csv"
    table.to_csv(scoring, index=False)
    return scoring


def test_two_tables_scorer_row_only_passes(setup):
    s = setup
    # identical tables are allowed
    info = vf.check_frozen(s["table"], s["base"], s["proj"], new_scripts=("src/c.py",), fit_table=s["table"],
                           scorer_path="src/c.py")
    assert info["fit_table"]["differing_rows"] == []
    scoring = _scorer_fix(s)
    info = vf.check_frozen(scoring, s["base"], s["proj"], new_scripts=("src/c.py",), fit_table=s["table"],
                           scorer_path="src/c.py")
    assert info["fit_table"]["differing_rows"] == ["src/c.py"]
    # the fit-time table no longer matches the files, but is only read as data
    with pytest.raises(vf.VerifyError, match="changed src/c.py"):
        vf.check_frozen(s["table"], s["base"], s["proj"], new_scripts=("src/c.py",))


def test_two_tables_other_row_differs_fails(setup):
    s = setup
    fit = pd.read_csv(s["table"], dtype=str)
    fit.loc[fit["path"] == "src/a.py", "sha256"] = "0" * 64
    fit_path = s["proj"] / "fit_bad.csv"
    fit.to_csv(fit_path, index=False)
    with pytest.raises(vf.VerifyError, match=r"differ outside src/c.py: \['src/a.py'\]"):
        vf.check_frozen(s["table"], s["base"], s["proj"], new_scripts=("src/c.py",), fit_table=fit_path,
                        scorer_path="src/c.py")
    fit = pd.read_csv(s["table"], dtype=str)
    fit = pd.concat([fit, pd.DataFrame({"path": ["src/z.py"], "sha256": ["1" * 64]})])
    fit.to_csv(fit_path, index=False)
    with pytest.raises(vf.VerifyError, match="path sets differ"):
        vf.check_frozen(s["table"], s["base"], s["proj"], new_scripts=("src/c.py",), fit_table=fit_path,
                        scorer_path="src/c.py")


def test_cells_code_hashes_use_fit_time_table(setup):
    s = setup
    scoring = _scorer_fix(s)
    assert len(cells(s)["cells"]) == 6  # s["table"] is the fit-time table the cells recorded
    with pytest.raises(vf.VerifyError, match="code_hashes differ from the fit-time frozen table"):
        vf.check_cells(s["results"], s["cohort"], scoring, s["label_map"], s["manifest"], s["proj"], False)


def test_cli_accepts_multiple_checks(setup, capsys):
    s = setup
    scoring = _scorer_fix(s)
    rc = vf.main(["--results-root", str(s["results"]), "--cohort-dir", str(s["cohort"]),
                  "--frozen-sources", str(scoring), "--fit-frozen-sources", str(s["table"]),
                  "--check", "cells", "protected", "--protected-snapshot", str(s["proj"] / "missing.sha256")])
    out = capsys.readouterr().out
    # CLI wiring only (the CLI uses the repo's label map / manifest, so cells fails here)
    lines = [line.split(":")[0] for line in out.splitlines() if line.startswith(("PASS", "FAIL"))]
    assert rc == 1 and [line.split()[1] for line in lines] == ["cells", "protected"]
