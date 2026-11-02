#!/usr/bin/env python3
"""Verify the Nazli inferred-827 V1 prediction cells and score summaries."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/nazli_inferred_827_v1_2026-10-08"
MODELS = ("rf", "ipcarf", "rflda")
FEATURE_KEY = "nazli_inferred_827"


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    require(path.is_file(), f"missing JSON: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_inputs(base: Path):
    manifest_path = base / "inputs/manifest.csv"
    manifest_frame = pd.read_csv(manifest_path)
    require(len(manifest_frame) == 1, "manifest must contain one row")
    manifest = manifest_frame.iloc[0]
    require(manifest["feature_key"] == FEATURE_KEY, "manifest feature key changed")
    require(int(manifest["n_samples"]) == 355, "manifest sample count changed")
    require(int(manifest["n_features"]) == 827, "manifest feature count changed")
    require(int(manifest["n_labels"]) == 44, "manifest label count changed")

    x_path = resolve_path(str(manifest["x_path"]))
    y_path = resolve_path(str(manifest["y_path"]))
    split_path = resolve_path(str(manifest["split_path"]))
    x = pd.read_csv(x_path, index_col=0)
    y = pd.read_csv(y_path, index_col=0)
    splits = pd.read_csv(split_path)
    x.index = x.index.astype(str)
    y.index = y.index.astype(str)
    require(x.shape == (355, 827), "X shape changed")
    require(y.shape == (355, 44), "Y shape changed")
    require(x.index.tolist() == y.index.tolist(), "X/Y ID order changed")
    require(bool(np.isfinite(x.to_numpy(float)).all()), "X contains non-finite values")
    require(bool(np.isin(y.to_numpy(), [0, 1]).all()), "Y is not binary")
    hashes = {
        "manifest": sha256(manifest_path),
        "x": sha256(x_path),
        "y": sha256(y_path),
        "split": sha256(split_path),
    }
    require(hashes["x"] == manifest["x_sha256"], "X hash differs from manifest")
    require(hashes["y"] == manifest["y_sha256"], "Y hash differs from manifest")
    require(hashes["split"] == manifest["split_sha256"], "split hash differs from manifest")
    return manifest, x, y, splits, hashes


def verify_cells(base: Path) -> None:
    manifest, x, y, splits, hashes = load_inputs(base)
    config = read_json(base / "run/run_config.json")
    require(config.get("completed") is True, "run config is incomplete")
    require(config.get("models") == list(MODELS), "run model order changed")
    require(config.get("model_profile") == "paper", "run profile is not paper")
    cells = config.get("cells")
    require(isinstance(cells, list) and len(cells) == 30, "run config does not have 30 cells")
    require(Counter(str(cell.get("model")) for cell in cells) == Counter({m: 10 for m in MODELS}),
            "run config model/fold counts changed")
    require(all(cell.get("outcome") in {"fitted", "reused"} for cell in cells),
            "run contains an unfinished cell")

    run_root = base / f"run/within/v1__all44__{FEATURE_KEY}__paper"
    all_ids = x.index.tolist()
    labels = y.columns.tolist()
    for model in MODELS:
        covered: list[str] = []
        for fold in range(1, 11):
            test_positions = splits[f"test_fold{fold}"].dropna().astype(int).tolist()
            expected_ids = [all_ids[position] for position in test_positions]
            stem = f"fold{fold:02d}"
            directory = run_root / model
            prediction_path = directory / f"{stem}.predictions.csv"
            contract_path = directory / f"{stem}.contract.json"
            done_path = directory / f"{stem}.done"
            require(prediction_path.is_file() and contract_path.is_file() and done_path.is_file(),
                    f"missing cell artifact: {model}/{stem}")
            contract = read_json(contract_path)
            prediction_hash = sha256(prediction_path)
            require(contract.get("prediction_file_sha256") == prediction_hash,
                    f"prediction hash mismatch: {model}/{stem}")
            require(done_path.read_text().strip() == prediction_hash,
                    f"done marker mismatch: {model}/{stem}")
            input_hashes = contract.get("input_hashes")
            require(isinstance(input_hashes, dict), f"missing input hashes: {model}/{stem}")
            require(input_hashes.get("x") == hashes["x"], f"wrong X used: {model}/{stem}")
            require(contract.get("feature_key") == FEATURE_KEY, f"wrong feature key: {model}/{stem}")
            prediction = pd.read_csv(prediction_path)
            require(prediction.columns[0] == "ID" and prediction.columns[1:].tolist() == labels,
                    f"prediction schema changed: {model}/{stem}")
            require(prediction["ID"].astype(str).tolist() == expected_ids,
                    f"prediction IDs do not match fold: {model}/{stem}")
            values = prediction.iloc[:, 1:].to_numpy(float)
            require(bool(np.isfinite(values).all() and (values >= 0).all() and (values <= 1).all()),
                    f"invalid prediction values: {model}/{stem}")
            covered.extend(expected_ids)
        require(Counter(covered) == Counter(all_ids), f"{model} does not cover all IDs once")
    print("inferred 827 cell verification passed (30 cells)")


def _csv_data_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def verify_scores(base: Path) -> None:
    _manifest, _x, _y, _splits, _hashes = load_inputs(base)
    scores = base / "scores"
    summary = pd.read_csv(scores / "summary__paper_youden.csv")
    folds = pd.read_csv(scores / "fold_metrics__paper_youden.csv")
    require(len(summary) == 3, "score summary does not have three model groups")
    require(set(summary["model"]) == set(MODELS), "score summary model set changed")
    require(bool(summary["complete"].all()), "score summary contains an incomplete group")
    require(bool((summary["n_folds_evaluated"] == 10).all()), "summary does not contain ten folds per model")
    require(len(folds) == 30, "fold score table does not contain 30 rows")
    require(_csv_data_rows(scores / "rejected_cells.csv") == 0, "rescorer rejected cells")
    require(_csv_data_rows(scores / "incomplete_groups.csv") == 0, "rescorer found incomplete groups")
    for row in summary.itertuples(index=False):
        selected = folds[folds["model"] == row.model]
        require(len(selected) == 10, f"wrong fold count for {row.model}")
        for metric in ("hamming", "label_ranking", "micro_roc", "micro_auprc", "precision", "recall", "fscore", "accuracy"):
            measured = float(selected[metric].mean())
            stored = float(getattr(row, f"{metric}_mean"))
            require(abs(measured - stored) <= 1e-12, f"summary arithmetic mismatch: {row.model}/{metric}")
    print("inferred 827 score verification passed (3 groups, 30 folds)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--check", choices=("cells", "scores"), required=True)
    args = parser.parse_args(argv)
    base = args.base if args.base.is_absolute() else ROOT / args.base
    try:
        verify_cells(base) if args.check == "cells" else verify_scores(base)
    except (VerificationError, FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
