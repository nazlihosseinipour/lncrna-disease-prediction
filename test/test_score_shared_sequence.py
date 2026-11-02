from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from scripts.prediction_io import sha256_file, sha256_lines, write_prediction_cell
from scripts.score_felipe_shared_sequence import (
    BASELINE,
    PROJECT_ROOT,
    ScoreError,
    bootstrap_indices,
    cell_dir,
    counts_from_indices,
    load_evaluator_class,
    main,
    run,
    score_one,
    weighted_micro,
)


LABELS = ["disease b", "disease a", "never observed", "disease c", "disease d"]
MODELS = ("rf", "ipcarf", "rflda")
N_TEST = 12
N_PRIMARY = 10


def _hash(i: int) -> str:
    return hashlib.sha256(f"seq{i}".encode()).hexdigest()


def _write_csv(path, frame, index=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, float_format="%.17g", lineterminator="\n")


def build_fixture(tmp_path, train_prev=None, probe=None, bad_test_x_model=None,
                  reorder_model=None, same_scores=False):
    rng = np.random.default_rng(3)
    test_ids = sorted(_hash(i) for i in range(N_TEST))
    primary = set(test_ids[:N_PRIMARY])
    cohort = tmp_path / "cohort"
    cohort.mkdir(parents=True)

    test_x = pd.DataFrame(rng.random((N_TEST, 4)), index=pd.Index(test_ids, name="sequence_sha256"),
                          columns=[f"f{j}" for j in range(4)])
    _write_csv(cohort / "test_X.csv", test_x)

    def labels(seed, ids):
        r = np.random.default_rng(seed)
        values = (r.random((len(ids), len(LABELS))) < 0.4).astype(int)
        values[:, 2] = 0  # zero-positive disease
        values[0, 0], values[1, 0] = 1, 0
        values[0, 1], values[1, 1] = 1, 0
        return pd.DataFrame(values, index=pd.Index(ids, name="sequence_sha256"), columns=LABELS)

    lab_v1 = labels(11, test_ids)
    lab_v2_or = labels(12, test_ids)
    lab_v2_primary = labels(13, [i for i in test_ids if i in primary])
    # make primary file distinguishable from OR file on the same rows
    lab_v2_primary.iloc[:, 3] = 1 - lab_v2_or.loc[lab_v2_primary.index].iloc[:, 3]
    _write_csv(cohort / "labels_v1.csv", lab_v1)
    _write_csv(cohort / "labels_v2_or.csv", lab_v2_or)
    _write_csv(cohort / "labels_v2_primary.csv", lab_v2_primary)
    cohorts = pd.DataFrame({"sequence_sha256": test_ids,
                            "in_primary": [i in primary for i in test_ids],
                            "in_sensitivity": [True] * N_TEST})
    _write_csv(cohort / "cohorts.csv", cohorts, index=False)

    # training Y (transfer manifest); train IDs are a strict subset
    inputs = tmp_path / "inputs"
    manifest_rows = []
    train_ids = {}
    train_y = {}
    for version, n in (("v1", 15), ("v2", 25)):
        ids = [f"{version}_RNA{k:03d}" for k in range(n)]
        r = np.random.default_rng(21 if version == "v1" else 22)
        values = (r.random((n, len(LABELS))) < (0.3 if version == "v1" else 0.6)).astype(int)
        frame = pd.DataFrame(values, index=pd.Index(ids, name="ID"), columns=LABELS)
        # excluded (shared) rows: all-ones so train prevalence != full-Y prevalence
        frame.iloc[:3] = 1
        y_path = inputs / f"{version}_Y.csv"
        _write_csv(y_path, frame)
        train_ids[version] = ids[3:]
        train_y[version] = frame.loc[ids[3:]]
        (cohort / f"train_ids_{version}.txt").write_text("\n".join(ids[3:]) + "\n")
        manifest_rows.append({"version": version, "label_space": "shared73",
                              "feature_key": "paper_sequence_542", "x_path": "unused",
                              "y_path": str(y_path), "y_sha256": sha256_file(y_path)})
    transfer_manifest = tmp_path / "transfer_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(transfer_manifest, index=False)

    outputs = {name: sha256_file(cohort / name) for name in (
        "test_X.csv", "labels_v1.csv", "labels_v2_primary.csv", "labels_v2_or.csv", "cohorts.csv",
        "train_ids_v1.txt", "train_ids_v2.txt")}
    (cohort / "cohort_manifest.json").write_text(json.dumps(
        {"schema": "shared_sequence_cohort/1", "outputs": outputs}, indent=2))
    manifest_sha = sha256_file(cohort / "cohort_manifest.json")

    results = tmp_path / "results"
    base_scores = pd.DataFrame(np.random.default_rng(99).random((N_TEST, len(LABELS))),
                               index=test_ids, columns=LABELS)
    for version in ("v1", "v2"):
        for k, model in enumerate(MODELS):
            if same_scores:
                scores = base_scores.copy()
            else:
                noise = np.random.default_rng(100 + k + (10 if version == "v2" else 0)).random(
                    (N_TEST, len(LABELS)))
                scores = pd.DataFrame(np.clip(0.4 * lab_v1.to_numpy() + 0.7 * noise, 0, 1),
                                      index=test_ids, columns=LABELS)
            if model == reorder_model:
                scores = scores.iloc[::-1]
            fitted = train_ids[version] if probe is None else train_ids[version][: probe["max_train_rows"]]
            contract = {
                "kind": "shared_sequence_paired", "train_version": version,
                "label_space": "shared73", "feature_key": "paper_sequence_542",
                "model": model, "model_profile": "paper", "effective_params": {}, "seed": 0,
                "test_population": "shared_sequence_all48",
                "test_x_sha256": ("0" * 64 if model == bad_test_x_model else outputs["test_X.csv"]),
                "test_row_ids_sha256": sha256_lines(test_ids),
                "train_ids_sha256": sha256_lines(fitted),
                "n_train": len(fitted),
                "cohort_manifest_sha256": manifest_sha,
                "input_hashes": {"train_x": "x", "train_y": manifest_rows[0 if version == "v1" else 1]["y_sha256"],
                                 "transfer_manifest": sha256_file(transfer_manifest),
                                 "label_map": "m", "test_x": outputs["test_X.csv"]},
                "code_hashes": {}, "library_versions": {}, "probe": probe,
                "fitted_selection": None, "execution": {},
            }
            write_prediction_cell(cell_dir(results, version, model), "shared48", scores, contract)
    return {
        "results": results, "cohort": cohort, "transfer_manifest": transfer_manifest,
        "train_y": train_y, "labels": {"v1": lab_v1, "v2_or": lab_v2_or, "v2_primary": lab_v2_primary},
        "test_ids": test_ids, "primary": [i for i in test_ids if i in primary],
    }


def _run(fx, outdir, **kwargs):
    kwargs.setdefault("n_bootstrap", 200)
    return run(fx["results"], fx["cohort"], fx["transfer_manifest"], outdir, **kwargs)


@pytest.fixture
def scored(tmp_path):
    fx = build_fixture(tmp_path)
    out = tmp_path / "scores"
    record = _run(fx, out)
    return fx, out, record


# --------------------------------------------------------------------------- metrics


def test_hand_checked_tiny_case():
    y = np.array([[1, 0], [0, 1]])
    s = np.array([[0.9, 0.2], [0.6, 0.4]])
    # flattened: y=[1,0,0,1], s=[.9,.2,.6,.4]; pos {.9,.4}, neg {.2,.6}
    # pairs pos>neg: .9>.2, .9>.6, .4>.2 -> 3/4
    # AP: ranks desc .9(P) .6(N) .4(P) .2(N): P@1=1, P@3=2/3 -> (1 + 2/3)/2 = 5/6
    # thresholded @0.5: [1,0,1,0] -> TP=1 FP=1 FN=1 -> F1 = 2/(2+1+1) = 0.5; hamming 2/4
    out = weighted_micro(y, s, np.ones((1, 2)))
    assert out["micro_roc"][0] == pytest.approx(0.75, abs=1e-15)
    assert out["micro_auprc"][0] == pytest.approx(5 / 6, abs=1e-15)
    assert out["micro_f1"][0] == pytest.approx(0.5, abs=1e-15)
    assert out["hamming"][0] == pytest.approx(0.5, abs=1e-15)


def test_fast_metrics_match_sklearn_on_replicates_with_ties():
    rng = np.random.default_rng(5)
    y = (rng.random((30, 7)) < 0.3).astype(int)
    s = np.round(rng.random((30, 7)), 1)  # heavy ties
    s[:, 0] = 0.5  # boundary value is predicted positive
    idx = bootstrap_indices({"primary": 30, "sensitivity": 30}, 25, 1)["primary"]
    fast = weighted_micro(y, s, counts_from_indices(idx))
    for b in range(idx.shape[0]):
        yb, sb = y[idx[b]].ravel(), s[idx[b]].ravel()
        assert abs(fast["micro_roc"][b] - roc_auc_score(yb, sb)) < 1e-12
        assert abs(fast["micro_auprc"][b] - average_precision_score(yb, sb)) < 1e-12
        assert abs(fast["micro_f1"][b] - f1_score(yb, (sb >= 0.5).astype(int))) < 1e-12
        assert abs(fast["hamming"][b] - np.mean(yb != (sb >= 0.5))) < 1e-12


def test_single_class_replicate_is_undefined():
    y = np.array([[1, 0], [0, 0]])
    s = np.array([[0.9, 0.1], [0.2, 0.3]])
    counts = np.array([[0, 2], [1, 1]])  # replicate 0 only has row 1 -> no positives
    out = weighted_micro(y, s, counts)
    assert np.isnan(out["micro_roc"][0]) and np.isnan(out["micro_auprc"][0])
    assert out["micro_roc"][1] == 1.0


def test_evaluator_agrees_with_sklearn_and_per_disease_nan():
    Evaluator, _ = load_evaluator_class(PROJECT_ROOT)
    rng = np.random.default_rng(8)
    ids = [f"r{i}" for i in range(20)]
    truth = pd.DataFrame((rng.random((20, 4)) < 0.3).astype(int), index=ids, columns=list("abcd"))
    truth["b"] = 0
    truth.loc["r0", "a"], truth.loc["r1", "a"] = 1, 0
    scores = pd.DataFrame(rng.random((20, 4)), index=ids, columns=list("abcd"))
    row, per = score_one(Evaluator(threshold_mode="fixed", threshold=0.5), truth, scores)
    flat_y, flat_s = truth.to_numpy().ravel(), scores.to_numpy().ravel()
    assert abs(row["micro_roc"] - roc_auc_score(flat_y, flat_s)) < 1e-12
    assert row["micro_roc"] == row["sklearn_micro_roc"]
    assert abs(row["fscore"] - f1_score(flat_y, flat_s >= 0.5)) < 1e-12
    by = {r["disease"]: r for r in per}
    assert not by["b"]["defined"] and np.isnan(by["b"]["auroc"]) and np.isnan(by["b"]["auprc"])
    assert row["n_diseases_no_positive"] >= 1
    assert row["n_per_disease_defined"] == sum(r["defined"] for r in per)


# --------------------------------------------------------------------------- end to end


def test_outputs_and_label_selection(scored):
    fx, out, record = scored
    for name in ("summary.csv", "per_disease.csv", "bootstrap_ci.csv", "paired_differences.csv",
                 "disease_definedness.csv", "score_run.json"):
        assert (out / name).is_file()
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 4 * 2 * 2 * 2
    assert set(summary["model"]) == set(MODELS) | {BASELINE}
    assert (summary["evaluation_mode"] == "fixed").all()
    assert (summary["threshold_source"] == "constant_0.5").all()
    assert (abs(summary["micro_roc"] - summary["sklearn_micro_roc"]) < 1e-12).all()
    prim = summary[summary["cohort"] == "primary"]
    sens = summary[summary["cohort"] == "sensitivity"]
    assert (prim["n_sequences"] == N_PRIMARY).all() and (sens["n_sequences"] == N_TEST).all()
    assert set(prim[prim.label_version == "v2"]["label_file"]) == {"labels_v2_primary.csv"}
    assert set(sens[sens.label_version == "v2"]["label_file"]) == {"labels_v2_or.csv"}
    # truth actually used: positives match the chosen files
    dd = pd.read_csv(out / "disease_definedness.csv")
    got = dd[(dd.cohort == "primary") & (dd.label_version == "v2")].set_index("disease")["positives"]
    want = fx["labels"]["v2_primary"].loc[fx["primary"]].sum()
    assert got.to_dict() == want.to_dict()
    got = dd[(dd.cohort == "sensitivity") & (dd.label_version == "v2")].set_index("disease")["positives"]
    assert got.to_dict() == fx["labels"]["v2_or"].sum().to_dict()
    assert dd[dd.disease == "never observed"]["zero_positive"].all()
    # model rows: independently recompute one
    row = summary[(summary.model == "rf") & (summary.train_version == "v1")
                  & (summary.cohort == "primary") & (summary.label_version == "v2")].iloc[0]
    from scripts.prediction_io import read_prediction_cell
    frame, _ = read_prediction_cell(cell_dir(fx["results"], "v1", "rf"), "shared48")
    y = fx["labels"]["v2_primary"].loc[fx["primary"]].to_numpy().ravel()
    s = frame.loc[fx["primary"]].to_numpy().ravel()
    assert abs(row["micro_roc"] - roc_auc_score(y, s)) < 1e-12
    per = pd.read_csv(out / "per_disease.csv")
    undefined = per[per.disease == "never observed"]
    assert (~undefined["defined"]).all() and undefined["auroc"].isna().all()
    rec = json.loads((out / "score_run.json").read_text())
    for name, digest in rec["outputs"].items():
        assert sha256_file(out / name) == digest
    assert set(rec["parameters"]["index_sha256"]) == {"primary", "sensitivity"}


def test_baseline_uses_train_prevalence(scored):
    fx, out, _ = scored
    per = pd.read_csv(out / "summary.csv")
    base = per[per.model == BASELINE]
    assert len(base) == 8
    # rebuild baseline scores from TRAIN prevalence and compare hamming
    for version in ("v1", "v2"):
        prev = fx["train_y"][version].mean()
        full_prev = pd.read_csv(fx["transfer_manifest"]).set_index("version").loc[version, "y_path"]
        full_prev = pd.read_csv(full_prev).set_index("ID").mean()
        assert not np.allclose(prev.to_numpy(), full_prev.to_numpy())  # excluded rows matter
        truth = fx["labels"]["v1"]
        test_prev = truth.mean()
        assert not np.allclose(prev.to_numpy(), test_prev.to_numpy())
        y = truth.to_numpy()
        s = np.tile(prev.to_numpy(), (len(truth), 1))
        row = base[(base.train_version == version) & (base.cohort == "sensitivity")
                   & (base.label_version == "v1")].iloc[0]
        assert abs(row["hamming"] - np.mean(y != (s >= 0.5))) < 1e-12
        assert abs(row["micro_auprc"] - average_precision_score(y.ravel(), s.ravel())) < 1e-12


def test_bootstrap_deterministic(tmp_path):
    fx = build_fixture(tmp_path)
    _run(fx, tmp_path / "a")
    _run(fx, tmp_path / "b")
    for name in ("bootstrap_ci.csv", "paired_differences.csv", "summary.csv", "per_disease.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    _run(fx, tmp_path / "c", seed=1)
    assert (tmp_path / "a" / "bootstrap_ci.csv").read_bytes() != (tmp_path / "c" / "bootstrap_ci.csv").read_bytes()


def test_paired_indices_shared_across_models(tmp_path):
    fx = build_fixture(tmp_path, same_scores=True)
    out = tmp_path / "scores"
    _run(fx, out, n_bootstrap=150)
    diffs = pd.read_csv(out / "paired_differences.csv")
    same = diffs[diffs.comparison == "train_v2_minus_train_v1"]
    same = same[same.a_model != BASELINE]
    assert (same["point_diff"] == 0).all()
    assert (same["ci_low"] == 0).all() and (same["ci_high"] == 0).all()
    assert (same["frac_gt_0"] == 0).all()
    assert set(diffs["comparison"]) == {
        "train_v2_minus_train_v1", "model_minus_prevalence_baseline", "label_v2_minus_label_v1"}
    ci = pd.read_csv(out / "bootstrap_ci.csv")
    for cohort, group in ci.groupby("cohort"):
        assert group["index_sha256"].nunique() == 1
    # the bootstrap distribution of a model equals itself across train versions
    a = ci[(ci.model == "rf") & (ci.train_version == "v1")].reset_index(drop=True)
    b = ci[(ci.model == "rf") & (ci.train_version == "v2")].reset_index(drop=True)
    assert (a[["ci_low", "ci_high", "n_valid"]] == b[["ci_low", "ci_high", "n_valid"]]).all().all()


# --------------------------------------------------------------------------- refusals


def test_refuses_probe_cells_without_flag(tmp_path):
    fx = build_fixture(tmp_path, probe={"max_train_rows": 5})
    with pytest.raises(ScoreError, match="probe"):
        _run(fx, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    rec = _run(fx, tmp_path / "out2", allow_probe=True, n_bootstrap=20)
    assert rec["inputs"]["baseline_n_train"] == {"v1": 5, "v2": 5}
    summary = pd.read_csv(tmp_path / "out2" / "summary.csv")
    base = summary[(summary.model == BASELINE) & (summary.train_version == "v2")
                   & (summary.cohort == "sensitivity") & (summary.label_version == "v1")].iloc[0]
    prev = fx["train_y"]["v2"].iloc[:5].mean().to_numpy()
    y = fx["labels"]["v1"].to_numpy()
    assert abs(base["hamming"] - np.mean(y != (np.tile(prev, (len(y), 1)) >= 0.5))) < 1e-12


def test_refuses_wrong_test_x_hash(tmp_path):
    fx = build_fixture(tmp_path, bad_test_x_model="ipcarf")
    with pytest.raises(ScoreError, match="test_x_sha256"):
        _run(fx, tmp_path / "out")


def test_refuses_prediction_order_mismatch(tmp_path):
    fx = build_fixture(tmp_path, reorder_model="rflda")
    with pytest.raises(ScoreError, match="order"):
        _run(fx, tmp_path / "out")


def test_refuses_missing_cell_lists_it(tmp_path):
    fx = build_fixture(tmp_path)
    with pytest.raises(ScoreError, match="train_v1/extra"):
        _run(fx, tmp_path / "out", models=MODELS + ("extra",))


def test_refuses_nonempty_outdir(tmp_path):
    fx = build_fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(ScoreError, match="non-empty"):
        _run(fx, out)
    assert (out / "keep.txt").read_text() == "x"
    empty = tmp_path / "empty"
    empty.mkdir()
    _run(fx, empty, n_bootstrap=10)


def test_refuses_tampered_cohort_file(tmp_path):
    fx = build_fixture(tmp_path)
    path = fx["cohort"] / "labels_v2_or.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ScoreError, match="labels_v2_or"):
        _run(fx, tmp_path / "out")


def test_cli(tmp_path):
    fx = build_fixture(tmp_path)
    rc = main(["--results-root", str(fx["results"]), "--cohort-dir", str(fx["cohort"]),
               "--transfer-manifest", str(fx["transfer_manifest"]), "--outdir",
               str(tmp_path / "cli"), "--n-bootstrap", "30"])
    assert rc == 0 and (tmp_path / "cli" / "score_run.json").is_file()
