"""
Run all NN feature methods over one or more sequence CSVs.

Inputs: CSVs with id,seq or ID/seqs.
If none are provided or a provided path is missing, the script will try to
locate/generate sequences_for_oop.csv from existing outputs:
  - Data/output_data/sequences_for_oop.csv
  - Data/output_data/sequences_v2.csv
  - Data/output_data/website_sequences_for_oop.csv
  - Data/output_data/website_sequences.csv (ID, seqs -> id,seq)
  - Data/output_data/website_full_matrix.csv (ID, seqs columns)

Outputs: <outdir>/<version_name>/nn/
"""

from pathlib import Path
import argparse
import pandas as pd

from mainfolder.core.feature_extractor import FeatureExtractor
from mainfolder.features.nn_features import NNFeatures
from mainfolder.utils.loader import preprocess_sequences
from mainfolder.utils.utils import ALPHABET


def normalize_id_seq(csv_path: Path):
    df = pd.read_csv(csv_path, dtype=str)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if "seqs" in df.columns and "seq" not in df.columns:
        df = df.rename(columns={"seqs": "seq"})
    if "id" not in df.columns or "seq" not in df.columns:
        raise ValueError(f"{csv_path} must have id/seq or ID/seqs columns")
    df = df[["id", "seq"]].dropna(subset=["id", "seq"])
    return df["id"].tolist(), df["seq"].tolist()


def find_or_build_sequences_csv() -> Path:
    candidates = [
        Path("Data/output_data/sequences_for_oop.csv"),
        Path("Data/output_data/sequences_v2.csv"),
        Path("Data/output_data/website_sequences_for_oop.csv"),
        Path("Data/output_data/website_sequences.csv"),
        Path("Data/output_data/website_full_matrix.csv"),
    ]
    for c in candidates:
        if c.exists():
            # If it's website_sequences.csv or website_full_matrix.csv, normalize to id,seq and save sequences_for_oop.csv
            if c.name in ("website_sequences.csv", "website_full_matrix.csv"):
                df = pd.read_csv(c, dtype=str)
                cols = [col.lower() for col in df.columns]
                df.columns = cols
                if "seqs" not in cols or "id" not in cols:
                    continue
                seq_df = df[["id", "seqs"]].rename(columns={"seqs": "seq"}).dropna(subset=["id", "seq"])
                out = Path("Data/output_data/sequences_for_oop.csv")
                out.parent.mkdir(parents=True, exist_ok=True)
                seq_df.to_csv(out, index=False)
                return out
            return c
    raise FileNotFoundError(
        "No sequences CSV found. Expected one of sequences_for_oop.csv, sequences_v2.csv, "
        "website_sequences_for_oop.csv, website_sequences.csv, or website_full_matrix.csv under Data/output_data/."
    )


def main():
    p = argparse.ArgumentParser(description="Run all NN feature methods over one or more CSVs.")
    p.add_argument(
        "--seqs_csv",
        nargs="+",
        help="One or more CSVs with id/seq or ID/seqs (e.g. sequences_for_oop.csv, website_full_matrix.csv)",
    )
    p.add_argument("--outdir", required=True, help="Base output directory")
    p.add_argument("--version_name", required=True, help="Data version label (e.g., v2, v3)")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size for NN models")
    p.add_argument("--layer", type=int, help="Layer for *_tokens methods")
    p.add_argument("--window", type=int, default=1024, help="Window for chunked methods")
    p.add_argument("--stride", type=int, default=512, help="Stride for chunked methods")
    p.add_argument("--agg", choices=["mean", "max"], default="mean", help="Aggregation for chunked methods")
    args = p.parse_args()

    if args.seqs_csv:
        seqs_list = [Path(p) for p in args.seqs_csv]
    else:
        seqs_list = [find_or_build_sequences_csv()]

    out_base = Path(args.outdir) / args.version_name / "nn"
    out_base.mkdir(parents=True, exist_ok=True)

    method_ids = sorted(NNFeatures.METHOD_MAP.keys())

    for seqs_path in seqs_list:
        if not seqs_path.exists():
            raise FileNotFoundError(f"File not found: {seqs_path}")
        ids, seqs = normalize_id_seq(seqs_path)
        ids2, seqs2 = preprocess_sequences(ids, seqs, valid_alphabet=set(ALPHABET), strict=False)
        stem = seqs_path.stem

        for mid in method_ids:
            name = NNFeatures.METHOD_MAP[mid]
            print(f"[run] {args.version_name}/{stem} -> method {mid} ({name})")
            kwargs = {
                "return_format": "dataframe",
                "sample_ids": ids2,
                "batch_size": args.batch_size,
            }
            if mid in (101, 104) and args.layer is not None:
                kwargs["layer"] = args.layer
            if mid in (130, 131):
                kwargs.update({"window": args.window, "stride": args.stride, "agg": args.agg})

            cols, df = FeatureExtractor.run("nn", mid, seqs2, **kwargs)
            out_path = out_base / f"{stem}_{name}.csv"
            if isinstance(df, pd.DataFrame):
                df.to_csv(out_path, index=False)
            else:
                pd.DataFrame(df, columns=cols).to_csv(out_path, index=False)
            print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
