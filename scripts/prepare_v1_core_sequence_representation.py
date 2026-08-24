"""Create the 583-dimensional V1 core sequence representation for paired probes."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prepare_inductive_feature_representations import assert_no_target_contamination


ROOT = Path(__file__).resolve().parents[1]
CORE_COMPONENTS = [
    "kmer_matrix_k1",
    "kmer_matrix_k2",
    "kmer_matrix_k3",
    "kmer_matrix_k4",
    "monoDiKGap_matrix",
    "monoMonoKGap_matrix",
    "rc_kmer_matrix_k1",
    "rc_kmer_matrix_k2",
    "rc_kmer_matrix_k3",
    "rc_kmer_matrix_k4",
    "zcurve_matrix",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        required=True,
        help="Explicit path to the audited V1 source manifest.",
    )
    parser.add_argument(
        "--outdir",
        default="inductive_inputs/corrected_10fold_2026-08-17/v1_core_sequence",
    )
    args = parser.parse_args()

    source_manifest_path = ROOT / args.source_manifest
    source = pd.read_csv(source_manifest_path).iloc[0].copy()
    x = pd.read_csv(source["x_path"])
    y = pd.read_csv(source["y_path"])
    prefixes = tuple(f"{component}__" for component in CORE_COMPONENTS)
    selected_columns = ["sample_id"] + [
        column for column in x.columns if column.startswith(prefixes)
    ]
    core = x[selected_columns].copy()
    if core.shape[1] - 1 != 583:
        raise ValueError(
            f"Expected 583 core sequence features, got {core.shape[1] - 1}."
        )
    contamination = assert_no_target_contamination(
        core,
        y,
        context="v1 core sequence representation",
    )

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    x_path = outdir / "v1_core_sequence_X.csv"
    manifest_path = outdir / "feature_representation_manifest.csv"
    core.to_csv(x_path, index=False)

    source["feature_set"] = "+".join(CORE_COMPONENTS)
    source["feature_key"] = "core_sequence_583"
    source["domains"] = "+".join(["rna"] * len(CORE_COMPONENTS))
    source["components"] = "+".join(CORE_COMPONENTS)
    source["x_path"] = x_path.as_posix()
    source["n_features"] = 583
    source["binary_feature_columns"] = contamination["binary_feature_columns"]
    source["exact_label_copy_count"] = contamination["exact_label_copy_count"]
    source["target_contamination_flagged"] = contamination["flagged"]
    pd.DataFrame([source]).to_csv(manifest_path, index=False)
    print(f"Saved {x_path}")
    print(f"Saved {manifest_path}")


if __name__ == "__main__":
    main()
