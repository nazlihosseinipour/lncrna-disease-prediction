from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print grouped RFLDA/IPCARF commands from a prepared feature manifest."
    )
    parser.add_argument(
        "--manifest",
        default="inductive_inputs/feature_representations/feature_representation_manifest.csv",
        help="Manifest created by prepare_inductive_feature_representations.py",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["rflda", "ipcarf"],
        default=["rflda", "ipcarf"],
        help="Which model commands to print.",
    )
    parser.add_argument(
        "--versions",
        nargs="+",
        choices=["v1", "v2"],
        default=None,
        help="Optional version filter.",
    )
    parser.add_argument(
        "--contains",
        default=None,
        help="Optional substring filter over feature_set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(PROJECT_ROOT / args.manifest)

    if args.versions:
        manifest = manifest[manifest["version"].isin(args.versions)]
    if args.contains:
        manifest = manifest[manifest["feature_set"].str.contains(args.contains, case=False, na=False)]

    model_dir = {"rflda": "RFLDA", "ipcarf": "IPCARF"}
    model_col = {"rflda": "rflda_command", "ipcarf": "ipcarf_command"}

    for model in args.models:
        print(f"\n### {model.upper()}")
        for _, row in manifest.iterrows():
            print(f"# {row['version']} | {row['feature_set']}")
            print(f"cd lncRNA_CIBCB2025-main/{model_dir[model]}")
            print(row[model_col[model]])
            print("cd ../..")
            print()


if __name__ == "__main__":
    main()
