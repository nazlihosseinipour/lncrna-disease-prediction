#!/usr/bin/env python3
"""Verify and report Nazli's selected-feature V1 run through Felipe's pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "Hamming": "hamming",
    "RankingErr": "label_ranking",
    "AUROC": "micro_roc",
    "AUPRC": "micro_auprc",
    "Precision": "precision",
    "Recall": "recall",
    "F": "fscore",
    "Accuracy": "accuracy",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(base: Path) -> dict[str, object]:
    inputs = base / "inputs"
    run_root = base / "run"
    scores = base / "scores"
    report_dir = base / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(inputs / "manifest.csv").iloc[0]
    x = pd.read_csv(ROOT / manifest.x_path, index_col=0)
    y = pd.read_csv(ROOT / manifest.y_path, index_col=0)
    splits = pd.read_csv(ROOT / manifest.split_path)
    assert x.shape == (353, 542)
    assert y.shape == (353, 44)
    assert x.index.astype(str).tolist() == y.index.astype(str).tolist()
    assert sha256(ROOT / manifest.x_path) == manifest.x_sha256
    assert sha256(ROOT / manifest.y_path) == manifest.y_sha256
    assert sha256(ROOT / manifest.split_path) == manifest.split_sha256

    run_config = json.loads((run_root / "run_config.json").read_text())
    assert run_config["completed"] is True
    assert len(run_config["cells"]) == 30
    assert all(cell["outcome"] == "fitted" for cell in run_config["cells"])

    expected_ids = set(x.index.astype(str))
    cell_checks: list[dict[str, object]] = []
    prediction_root = run_root / "within" / "v1__all44__nazli_selected_paper_sequence_542__paper"
    for model in ("rf", "ipcarf", "rflda"):
        model_ids: list[str] = []
        for fold in range(1, 11):
            stem = f"fold{fold:02d}"
            contract_path = prediction_root / model / f"{stem}.contract.json"
            prediction_path = prediction_root / model / f"{stem}.predictions.csv"
            done_path = prediction_root / model / f"{stem}.done"
            assert contract_path.is_file() and prediction_path.is_file() and done_path.is_file()
            contract = json.loads(contract_path.read_text())
            predictions = pd.read_csv(prediction_path, index_col=0)
            assert predictions.shape[1] == 44
            assert predictions.columns.tolist() == y.columns.tolist()
            assert np.isfinite(predictions.to_numpy(dtype=float)).all()
            assert ((predictions.to_numpy() >= 0) & (predictions.to_numpy() <= 1)).all()
            assert sha256(prediction_path) == contract["prediction_file_sha256"]
            assert contract["input_hashes"]["x"] == manifest.x_sha256
            assert contract["input_hashes"]["y"] == manifest.y_sha256
            assert contract["input_hashes"]["split"] == manifest.split_sha256
            fold_ids = predictions.index.astype(str).tolist()
            expected_fold_positions = splits[f"test_fold{fold}"].dropna().astype(int).tolist()
            assert fold_ids == x.index[expected_fold_positions].astype(str).tolist()
            model_ids.extend(fold_ids)
            cell_checks.append({"model": model, "fold": fold, "rows": len(fold_ids), "sha256": sha256(prediction_path)})
        assert len(model_ids) == 353 and len(set(model_ids)) == 353 and set(model_ids) == expected_ids

    summary = pd.read_csv(scores / "summary__paper_youden.csv")
    folds = pd.read_csv(scores / "fold_metrics__paper_youden.csv")
    rejected = pd.read_csv(scores / "rejected_cells.csv")
    incomplete = pd.read_csv(scores / "incomplete_groups.csv")
    assert len(summary) == 3 and summary["complete"].all()
    assert len(folds) == 30 and rejected.empty and incomplete.empty
    for row in summary.itertuples(index=False):
        selected = folds[folds.model == row.model]
        assert len(selected) == 10
        for key in METRICS.values():
            measured_mean = float(selected[key].mean())
            measured_std = float(selected[key].std(ddof=0))
            assert np.isclose(measured_mean, getattr(row, f"{key}_mean"), atol=1e-12)
            assert np.isclose(measured_std, getattr(row, f"{key}_std"), atol=1e-12)

    reference_path = ROOT / "Reports/felipe_paper_exact_2026-09-29T0014/V1_V1_AND_V2_V2_RESULTS.csv"
    reference = pd.read_csv(reference_path)
    paper = reference[(reference.direction == "V1->V1") & reference.model.str.contains("IPCARF|RFLDA", na=False)]
    comparison_rows: list[dict[str, object]] = []
    for row in summary.itertuples(index=False):
        model_name = str(row.model).upper()
        for display, key in METRICS.items():
            reference_model = paper.model.astype(str).str.split().str[0]
            ref = paper[(reference_model == model_name) & (paper.metric == display)]
            paper_mean = float(ref.paper_mean.iloc[0]) if len(ref) else np.nan
            paper_std = float(ref.paper_std.iloc[0]) if len(ref) else np.nan
            measured_mean = float(getattr(row, f"{key}_mean"))
            measured_std = float(getattr(row, f"{key}_std"))
            comparison_rows.append({
                "model": model_name,
                "metric": display,
                "nazli_353x542_mean": measured_mean,
                "nazli_353x542_std": measured_std,
                "paper_355x827_mean": paper_mean,
                "paper_355x827_std": paper_std,
                "mean_difference": measured_mean - paper_mean if np.isfinite(paper_mean) else np.nan,
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison_path = report_dir / "V1_SELECTED_FEATURES_VS_PAPER.csv"
    comparison.to_csv(comparison_path, index=False)

    metric_order = list(METRICS)
    compact = comparison.pivot(index="model", columns="metric", values="nazli_353x542_mean").reindex(columns=metric_order)
    lines = [
        "# V1 selected paper features through Felipe's MLC pipeline",
        "",
        "## What was actually run",
        "",
        "- Input rows: 353 RNAs retained by Nazli's generated V1 pipeline.",
        "- Input columns: 542 sequence features rebuilt from Nazli's processed sequences: 462 DCC (22 properties, lag 1), 16 normalized 2-mers and 64 normalized 3-mers.",
        "- Labels: the paper's 44 V1 disease columns, aligned by RNA ID.",
        "- Splits: the released ten folds, mapped by RNA ID after the two absent feature rows were removed.",
        "- Models: Felipe RF, IPCARF and RFLDA MLC paper profile; seed 0.",
        "- Evaluation: the paper's per-label Youden threshold on each evaluated fold and all eight MLC measures.",
        "",
        "## Measured results (mean over ten folds)",
        "",
        "| Model | Hamming | Ranking error | AUROC | AUPRC | Precision | Recall | F | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("RF", "IPCARF", "RFLDA"):
        values = compact.loc[model]
        lines.append("| " + model + " | " + " | ".join(f"{values[name]:.3f}" for name in metric_order) + " |")
    lines += [
        "",
        "## Interpretation",
        "",
        "These are measured outputs from Nazli's processed V1 sequences passed through Felipe's model and evaluation code. They are not copied paper values.",
        "",
        "IPCARF and RFLDA do not match Table III: their AUROC values are about 0.64 instead of 0.82/0.81, and their AUPRC values are about 0.10/0.09 instead of 0.44/0.41. This is expected to be a feature-input difference: this run uses 542 verified sequence columns and 353 RNAs, while Table III used the released 827-column matrix and 355 RNAs.",
        "",
        "The 285 omitted columns are the unresolved binary block in the released artifact. They were not appended because no sequence construction for them is documented and they numerically match the V1 association matrix. Therefore this run is the defensible sequence-only comparison, but it is not an exact full-827 reproduction.",
        "",
        "## Verification",
        "",
        "- 30/30 prediction cells exist and match their recorded SHA-256 hashes.",
        "- For each model, the ten test folds cover all 353 RNA IDs exactly once.",
        "- Prediction labels and truth labels match in the same 44-column order.",
        "- The rescorer accepted all three groups with zero rejected cells and zero incomplete groups.",
        "- Independent summary arithmetic matches the saved fold means and population standard deviations within 1e-12.",
        "",
        "## Superseded probe",
        "",
        "The earlier one-fold 729-column RF probe was stopped after the feature-selection correction and is excluded from every result above.",
    ]
    report_path = report_dir / "V1_SELECTED_FEATURES_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    record = {
        "passed": True,
        "checks": {
            "input_shape_and_hashes": True,
            "run_complete_30_cells": True,
            "prediction_hashes_and_ranges": True,
            "per_model_test_partition_353_exactly_once": True,
            "rescore_zero_rejected_zero_incomplete": True,
            "summary_arithmetic_atol_1e-12": True,
        },
        "cell_checks": cell_checks,
        "outputs": [comparison_path.relative_to(ROOT).as_posix(), report_path.relative_to(ROOT).as_posix()],
    }
    (report_dir / "verification.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "results/nazli_selected_v1_felipe_2026-10-05")
    args = parser.parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    print(json.dumps(run(base), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
