"""Pivot final_comparison.csv into a generalization matrix.

Rows = (feature_set, model, metric); columns = the four train→test regimes
(V1->V1, V2->V2, V1->V2, V2->V1). This is the comparison-column view requested in
Supervisor item 7.

Caveat: within-version regimes use each version's own filtered label space (V1: 45,
V2: 124 diseases), while the cross-version regimes use the shared 40-disease space.
The columns are therefore directionally informative, not a strict like-for-like cell
comparison. The label counts are emitted alongside for transparency.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REGIME = {  # (evaluation, train, test) -> column label
    ("within_version_cv", "v1", "v1"): "V1->V1",
    ("within_version_cv", "v2", "v2"): "V2->V2",
    ("cross_version_transfer", "v1", "v2"): "V1->V2",
    ("cross_version_transfer", "v2", "v1"): "V2->V1",
}
METRICS = ["micro_roc", "micro_auprc", "fscore", "hamming", "label_ranking", "accuracy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final-comparison", default="results/final_comparison.csv")
    p.add_argument("--outdir", default="results")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = PROJECT_ROOT / args.final_comparison
    outdir = PROJECT_ROOT / args.outdir
    df = pd.read_csv(src)

    df["regime"] = [
        REGIME.get((e, s, t))
        for e, s, t in zip(df["evaluation"], df["train_dataset"], df["test_dataset"])
    ]
    df = df[df["regime"].notna()].copy()

    rows = []
    for (feat, model), g in df.groupby(["feature_set", "model"]):
        for metric in METRICS:
            row = {"feature_set": feat, "model": model, "metric": metric}
            for regime in ["V1->V1", "V2->V2", "V1->V2", "V2->V1"]:
                sub = g[g["regime"] == regime]
                row[regime] = round(float(sub[f"{metric}_mean"].iloc[0]), 4) if len(sub) else ""
            rows.append(row)

    matrix = pd.DataFrame(rows).sort_values(["metric", "feature_set", "model"], kind="stable")
    out = outdir / "generalization_matrix.csv"
    matrix.to_csv(out, index=False)

    # label-count provenance
    prov = (
        df.groupby("regime")["n_labels"].agg(["min", "max"]).reset_index()
        if "n_labels" in df.columns else pd.DataFrame()
    )
    prov_path = outdir / "generalization_matrix_label_counts.csv"
    prov.to_csv(prov_path, index=False)

    print(f"Saved {out} ({len(matrix)} rows)")
    print(f"Saved {prov_path}")
    print(matrix[matrix["metric"] == "micro_auprc"].to_string(index=False))


if __name__ == "__main__":
    main()
