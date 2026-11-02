from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from scripts.prediction_io import cell_paths, sha256_file, write_prediction_cell
from scripts.rescore_felipe_predictions import (
    MEASURES,
    PROJECT_ROOT,
    RescoreError,
    load_evaluator_class,
    main,
    partition_sha256,
    run,
)


LABELS = ["disease b", "disease a", "never observed", "disease c"]
N_FOLDS = 4


def _truth(tmp_path, ids=None, name="truth_y.csv"):
    rng = np.random.default_rng(7)
    ids = ids or [f"RNA{index:03d}" for index in range(48)]
    values = (rng.random((len(ids), len(LABELS))) < 0.35).astype(int)
    values[:, 2] = 0  # D6: zero-positive label, must be kept
    frame = pd.DataFrame(values, index=pd.Index(ids, name="ID"), columns=LABELS)
    path = tmp_path / "data" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    return frame, "data/" + name, sha256_file(path)


def _scores(truth: pd.DataFrame, ids, seed) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.random((len(ids), truth.shape[1]))
    values = np.clip(0.55 * truth.loc[ids].to_numpy() + 0.6 * noise, 0, 1)
    return pd.DataFrame(values, index=ids, columns=truth.columns)


def _within_contract(truth_rel, truth_sha, fold, model="rf"):
    return {
        "kind": "within",
        "version": "v1",
        "label_space": "all44",
        "feature_key": "paper_sequence_542",
        "model": model,
        "model_profile": "paper",
        "effective_params": {"n_estimators": 10},
        "seed": 0,
        "fold": fold,
        "n_folds_total": N_FOLDS,
        "truth_y_path": truth_rel,
        "truth_y_sha256": truth_sha,
        "input_hashes": {"x": "x" * 8, "y": truth_sha},
        "code_hashes": {"runner.py": "c" * 8},
    }


def _within_group(tmp_path, truth, truth_rel, truth_sha, skip=(), columns=None, model="rf"):
    root = tmp_path / "cells"
    ids = truth.index.tolist()
    folds = [ids[index::N_FOLDS] for index in range(N_FOLDS)]
    written = {}
    for fold, fold_ids in enumerate(folds, 1):
        if fold in skip:
            continue
        shuffled = list(reversed(fold_ids))  # rows need an ID join, not position
        preds = _scores(truth, shuffled, seed=fold)
        if columns is not None:
            preds = preds[columns]
        outdir = root / "within" / f"v1__all44__paper_sequence_542__paper/{model}/fold{fold:02d}"
        stem = f"v1_all44_{model}_fold{fold:02d}"
        write_prediction_cell(
            outdir, stem, preds, _within_contract(truth_rel, truth_sha, fold, model)
        )
        written[fold] = (outdir, stem, preds, fold_ids)
    return root, written


def _transfer_cell(tmp_path, truth, truth_rel, truth_sha, bad_sha=False):
    root = tmp_path / "cells"
    removed = truth.index[:3].tolist()
    ids = truth.index[3:].tolist()
    folds = [ids[index::3] for index in range(3)]
    partition = {
        "method": "test",
        "n_splits": 3,
        "random_state": 0,
        "uses_target_labels": True,
        "folds": folds,
        "sha256": "0" * 64 if bad_sha else partition_sha256(folds),
    }
    contract = {
        "kind": "transfer",
        "source": "v1",
        "target": "v2",
        "label_space": "shared73",
        "feature_key": "paper_sequence_542",
        "model": "rf",
        "model_profile": "paper",
        "effective_params": {},
        "seed": 0,
        "fold": None,
        "n_folds_total": None,
        "truth_y_path": truth_rel,
        "truth_y_sha256": truth_sha,
        "input_hashes": {},
        "code_hashes": {},
        "removed_exact_target_ids": sorted(removed),
        "evaluation_partition": partition,
    }
    outdir = root / "transfer" / "v1_to_v2__shared73__paper_sequence_542__paper/rf/full_target"
    preds = _scores(truth, ids, seed=99)
    write_prediction_cell(outdir, "v1_to_v2_rf", preds, contract)
    return root, preds, folds


def _read(outdir, name):
    return pd.read_csv(outdir / name, keep_default_na=True)


def _evaluator():
    return load_evaluator_class(PROJECT_ROOT)[0]


def test_modes_write_distinct_files_and_labels(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, _ = _within_group(tmp_path, truth, rel, sha)
    out = tmp_path / "out"
    run(root, ["fixed", "paper_youden"], out, project_root=tmp_path)

    expected = {
        "fixed": ("primary_clean", "constant_0.5"),
        "paper_youden": ("paper_comparison", "evaluated_fold_labels"),
    }
    for mode, (lane, source) in expected.items():
        for kind in ("fold_metrics", "summary", "per_label"):
            frame = _read(out, f"{kind}__{mode}.csv")
            assert not frame.empty
            assert set(frame["evaluation_mode"]) == {mode}
            assert set(frame["lane"]) == {lane}
            assert set(frame["threshold_source"]) == {source}
            assert set(frame["complete"]) == {True}
    for name in ("positives.csv", "rejected_cells.csv", "incomplete_groups.csv", "rescore_run.json"):
        assert (out / name).is_file()
    summary = _read(out, "summary__fixed.csv").iloc[0]
    assert summary["n_rows"] == len(truth) and summary["n_labels"] == len(LABELS)
    assert summary["n_folds"] == N_FOLDS and summary["truth_y_sha256"] == sha
    record = json.loads((out / "rescore_run.json").read_text())
    assert record["script_sha256"] and len(record["cells"]) == N_FOLDS
    assert all(cell["status"] == "accepted" for cell in record["cells"])
    # Youden and fixed differ on thresholded measures for the same predictions.
    fixed = _read(out, "fold_metrics__fixed.csv")
    youden = _read(out, "fold_metrics__paper_youden.csv")
    assert not np.allclose(fixed["fscore"], youden["fscore"])
    np.testing.assert_allclose(fixed["micro_roc"], youden["micro_roc"])


@pytest.mark.parametrize(
    "mode,kwargs",
    [("fixed", {"threshold_mode": "fixed", "threshold": 0.5}),
     ("paper_youden", {"threshold_mode": "youden"})],
)
def test_measures_equal_direct_paper_evaluator(tmp_path, mode, kwargs):
    truth, rel, sha = _truth(tmp_path)
    root, written = _within_group(tmp_path, truth, rel, sha)
    out = tmp_path / "out"
    run(root, [mode], out, project_root=tmp_path)
    folds = _read(out, f"fold_metrics__{mode}.csv").set_index("fold")
    evaluator_cls = _evaluator()
    for fold, (_, _, preds, _) in written.items():
        y_fold = truth.loc[preds.index]
        direct = evaluator_cls(**kwargs).evaluate(y_fold, preds)
        np.testing.assert_allclose(
            folds.loc[fold, MEASURES].to_numpy(dtype=float), np.array(direct, dtype=float),
            rtol=0, atol=1e-12,
        )
    summary = _read(out, f"summary__{mode}.csv").iloc[0]
    for measure in MEASURES:
        column = folds[measure].to_numpy(dtype=float)
        assert summary[f"{measure}_mean"] == pytest.approx(np.mean(column), abs=1e-12)
        assert summary[f"{measure}_std"] == pytest.approx(np.std(column), abs=1e-12)


def test_one_class_label_kept_with_nan_scores(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, _ = _within_group(tmp_path, truth, rel, sha)
    out = tmp_path / "out"
    run(root, ["fixed", "paper_youden"], out, project_root=tmp_path)
    for mode in ("fixed", "paper_youden"):
        per_label = _read(out, f"per_label__{mode}.csv")
        assert set(per_label["label"]) == set(LABELS)
        dead = per_label[per_label["label"] == "never observed"]
        assert len(dead) == N_FOLDS + 1  # every fold + pooled
        assert dead["one_class"].all()
        assert dead["auroc"].isna().all() and dead["auprc"].isna().all()
        assert (dead["positives"] == 0).all()
        live = per_label[(per_label["label"] == "disease a") & (per_label["population"] == "pooled")]
        assert not live["one_class"].iloc[0] and np.isfinite(live["auroc"].iloc[0])
    positives = _read(out, "positives.csv")
    pooled = positives[positives["population"] == "pooled"].set_index("label")
    assert pooled.loc["never observed", "positives"] == 0
    assert pooled.loc["disease a", "positives"] == int(truth["disease a"].sum())


def test_zero_positive_fold_records_nan_and_warning(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    ids = truth.index.tolist()
    fold_one = ids[0::N_FOLDS]
    truth.loc[fold_one, :] = 0  # fold 1 has no positives at all
    path = tmp_path / rel
    truth.to_csv(path)
    sha = sha256_file(path)
    root, _ = _within_group(tmp_path, truth, rel, sha)
    out = tmp_path / "out"
    run(root, ["fixed"], out, project_root=tmp_path)
    folds = _read(out, "fold_metrics__fixed.csv").set_index("fold")
    assert np.isnan(folds.loc[1, "micro_roc"]) and np.isnan(folds.loc[1, "micro_auprc"])
    assert "undefined" in folds.loc[1, "warnings"]
    assert np.isfinite(folds.loc[2, "micro_roc"])


def test_corrupted_cell_rejected_and_group_excluded(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, written = _within_group(tmp_path, truth, rel, sha)
    outdir, stem, _, _ = written[2]
    with cell_paths(outdir, stem)["predictions"].open("a") as handle:
        handle.write("tampered\n")
    out = tmp_path / "out"
    run(root, ["fixed"], out, project_root=tmp_path)
    rejected = _read(out, "rejected_cells.csv")
    assert rejected["stem"].tolist() == [stem]
    assert "hash" in rejected["reason"].iloc[0]
    incomplete = _read(out, "incomplete_groups.csv")
    assert len(incomplete) == 1
    assert incomplete["status"].iloc[0] == "incomplete"
    assert "missing folds [2]" in incomplete["reason"].iloc[0]
    assert not incomplete["included"].iloc[0]
    assert _read(out, "summary__fixed.csv").empty
    assert pd.read_csv(out / "fold_metrics__fixed.csv").empty


def test_missing_fold_excluded_unless_allow_partial(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, _ = _within_group(tmp_path, truth, rel, sha, skip=(3,))
    strict = tmp_path / "strict"
    run(root, ["fixed"], strict, project_root=tmp_path)
    assert pd.read_csv(strict / "fold_metrics__fixed.csv").empty
    incomplete = _read(strict, "incomplete_groups.csv")
    assert incomplete["reason"].iloc[0].startswith("missing folds [3]")

    partial = tmp_path / "partial"
    run(root, ["fixed"], partial, allow_partial=True, project_root=tmp_path)
    summary = _read(partial, "summary__fixed.csv")
    assert summary["complete"].tolist() == [False]
    assert summary["n_folds"].iloc[0] == N_FOLDS and summary["n_folds_evaluated"].iloc[0] == 3
    assert set(_read(partial, "fold_metrics__fixed.csv")["fold"]) == {1, 2, 4}
    assert _read(partial, "incomplete_groups.csv")["included"].tolist() == [True]


def test_label_order_mismatch_is_an_error(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, _ = _within_group(tmp_path, truth, rel, sha, columns=list(reversed(LABELS)))
    with pytest.raises(RescoreError, match="label columns"):
        run(root, ["fixed"], tmp_path / "out", project_root=tmp_path)
    assert not (tmp_path / "out").exists()


def test_truth_hash_mismatch_is_an_error(tmp_path):
    truth, rel, _ = _truth(tmp_path)
    root, _ = _within_group(tmp_path, truth, rel, "f" * 64)
    with pytest.raises(RescoreError, match="hash mismatch"):
        run(root, ["fixed"], tmp_path / "out", project_root=tmp_path)


def test_prediction_id_absent_from_truth_is_an_error(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root = tmp_path / "cells"
    ids = truth.index.tolist()
    for fold in range(1, N_FOLDS + 1):
        fold_ids = ids[fold - 1::N_FOLDS]
        preds = _scores(truth, fold_ids, seed=fold)
        if fold == 1:
            preds.index = ["GHOST"] + fold_ids[1:]
        write_prediction_cell(
            root / f"f{fold}", f"c{fold}", preds, _within_contract(rel, sha, fold)
        )
    with pytest.raises(RescoreError, match="absent from truth"):
        run(root, ["fixed"], tmp_path / "out", project_root=tmp_path)


def test_refuses_existing_outdir(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, _ = _within_group(tmp_path, truth, rel, sha)
    out = tmp_path / "out"
    run(root, ["fixed"], out, project_root=tmp_path)
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(RescoreError, match="already contains"):
        run(root, ["paper_youden"], out, project_root=tmp_path)
    with pytest.raises(RescoreError, match="already contains"):
        main(["--input-root", str(root), "--modes", "fixed", "--outdir", str(out),
              "--project-root", str(tmp_path)])
    assert {path.name: path.read_bytes() for path in out.iterdir()} == before


def test_transfer_uses_evaluation_partition(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, preds, folds = _transfer_cell(tmp_path, truth, rel, sha)
    out = tmp_path / "out"
    run(root, ["fixed", "paper_youden"], out, project_root=tmp_path)
    fold_rows = _read(out, "fold_metrics__paper_youden.csv")
    assert fold_rows["fold"].tolist() == [1, 2, 3]
    assert fold_rows["n_rows"].tolist() == [len(fold) for fold in folds]
    assert set(fold_rows["kind"]) == {"transfer"}
    evaluator_cls = _evaluator()
    for index, fold_ids in enumerate(folds):
        direct = evaluator_cls(threshold_mode="youden").evaluate(
            truth.loc[fold_ids], preds.loc[fold_ids]
        )
        np.testing.assert_allclose(
            fold_rows.loc[index, MEASURES].to_numpy(dtype=float), direct, atol=1e-12
        )
    summary = _read(out, "summary__fixed.csv").iloc[0]
    assert summary["n_rows"] == len(truth) - 3


def test_transfer_partition_hash_mismatch_rejected(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, _, _ = _transfer_cell(tmp_path, truth, rel, sha, bad_sha=True)
    out = tmp_path / "out"
    run(root, ["fixed"], out, project_root=tmp_path)
    rejected = _read(out, "rejected_cells.csv")
    assert rejected["stem"].tolist() == ["v1_to_v2_rf"]
    assert "evaluation_partition sha256" in rejected["reason"].iloc[0]
    assert pd.read_csv(out / "fold_metrics__fixed.csv").empty
    incomplete = _read(out, "incomplete_groups.csv")
    assert incomplete["kind"].tolist() == ["transfer"]


def test_independent_sklearn_recomputation(tmp_path):
    truth, rel, sha = _truth(tmp_path)
    root, written = _within_group(tmp_path, truth, rel, sha)
    out = tmp_path / "out"
    run(root, ["fixed"], out, project_root=tmp_path)
    summary = _read(out, "summary__fixed.csv").iloc[0]
    pred_all = pd.concat([item[2] for item in written.values()])
    y_all = truth.loc[pred_all.index]
    y_flat, p_flat = y_all.to_numpy().ravel(), pred_all.to_numpy().ravel()
    assert summary["pooled_oof_micro_roc"] == pytest.approx(roc_auc_score(y_flat, p_flat), abs=1e-12)
    assert summary["pooled_oof_micro_auprc"] == pytest.approx(
        average_precision_score(y_flat, p_flat), abs=1e-12
    )
    folds = _read(out, "fold_metrics__fixed.csv").set_index("fold")
    for fold, (_, _, preds, _) in written.items():
        y = truth.loc[preds.index].to_numpy().ravel()
        p = preds.to_numpy().ravel()
        assert folds.loc[fold, "fscore"] == pytest.approx(
            f1_score(y, (p >= 0.5).astype(int), zero_division=0), abs=1e-12
        )
        assert folds.loc[fold, "micro_roc"] == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    per_label = _read(out, "per_label__fixed.csv")
    pooled = per_label[(per_label["population"] == "pooled")].set_index("label")
    for label in ("disease a", "disease c"):
        assert pooled.loc[label, "auroc"] == pytest.approx(
            roc_auc_score(y_all[label], pred_all[label]), abs=1e-12
        )
        assert pooled.loc[label, "f1_thresholded"] == pytest.approx(
            f1_score(y_all[label], (pred_all[label] >= 0.5).astype(int), zero_division=0),
            abs=1e-12,
        )
        assert pooled.loc[label, "threshold"] == 0.5
