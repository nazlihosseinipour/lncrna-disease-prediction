"""
Run all NN feature methods over one or more sequence CSVs.

Inputs: CSVs with id,seq or ID/seqs.
If none are provided or a provided path is missing, the script will try to
locate a usable sequences CSV from existing outputs:
  - Data/output_data/sequences_for_oop.csv
  - Data/output_data/sequences_v2.csv
  - Data/output_data/website_sequences_for_oop.csv
  - Data/output_data/website_sequences.csv (ID, seqs)
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


def find_default_sequences_csv() -> Path:
    candidates = [
        Path("Data/output_data/sequences_for_oop.csv"),
        Path("Data/output_data/sequences_v2.csv"),
        Path("Data/output_data/website_sequences_for_oop.csv"),
        Path("Data/output_data/website_sequences.csv"),
        Path("Data/output_data/website_full_matrix.csv"),
    ]
    for c in candidates:
        if c.exists():
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
    p.add_argument("--version_name", required=True, help="Data version label (e.g., v1, v2)")
    p.add_argument("--batch_size", type=int, default=2, help="Batch size for NN models")
    p.add_argument("--layer", type=int, help="Layer for *_tokens methods")
    p.add_argument("--window", type=int, default=1024, help="Window for chunked methods")
    p.add_argument("--stride", type=int, default=512, help="Stride for chunked methods")
    p.add_argument("--agg", choices=["mean", "max"], default="mean", help="Aggregation for chunked methods")
    p.add_argument(
        "--methods",
        nargs="+",
        type=int,
        help="Optional subset of NN method IDs to run (e.g. 100 101 130 for MP-RNA only)",
    )
    args = p.parse_args()

    if args.seqs_csv:
        seqs_list = [Path(p) for p in args.seqs_csv]
    else:
        selected = find_default_sequences_csv()
        print(f"[info] no --seqs_csv provided, using {selected}")
        seqs_list = [selected]

    out_base = Path(args.outdir) / args.version_name / "nn"
    out_base.mkdir(parents=True, exist_ok=True)

    method_ids = sorted(NNFeatures.METHOD_MAP.keys())
    if args.methods:
        unknown = sorted(set(args.methods) - set(method_ids))
        if unknown:
            raise ValueError(
                f"Unknown NN method IDs requested: {unknown}. Available: {method_ids}"
            )
        method_ids = [mid for mid in method_ids if mid in set(args.methods)]

    for seqs_path in seqs_list:
        if not seqs_path.exists():
            raise FileNotFoundError(f"File not found: {seqs_path}")
        ids, seqs = normalize_id_seq(seqs_path)
        ids2, seqs2 = preprocess_sequences(ids, seqs, valid_alphabet=set(ALPHABET), strict=False)
        if not seqs2:
            raise ValueError(
                f"{seqs_path} produced 0 valid sequences after preprocessing. "
                "Check seq/seqs content and allowed RNA alphabet."
            )
        stem = seqs_path.stem
        print(
            f"[info] {seqs_path}: kept {len(seqs2)}/{len(seqs)} sequences after preprocessing"
        )
        failures = []
        success_count = 0

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

            try:
                cols, df = FeatureExtractor.run("nn", mid, seqs2, **kwargs)
            except Exception as exc:
                failures.append(
                    {
                        "source_csv": str(seqs_path),
                        "method_id": mid,
                        "method_name": name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"[warn] failed method {mid} ({name}): {type(exc).__name__}: {exc}")
                continue

            out_path = out_base / f"{stem}_{name}.csv"
            if isinstance(df, pd.DataFrame):
                df.to_csv(out_path, index=False)
            else:
                pd.DataFrame(df, columns=cols).to_csv(out_path, index=False)
            success_count += 1
            print(f"[saved] {out_path}")

        if failures:
            fail_path = out_base / f"{stem}_nn_failures.csv"
            pd.DataFrame(failures).to_csv(fail_path, index=False)
            print(f"[saved] {fail_path}")
        if success_count == 0:
            raise RuntimeError(
                f"All NN methods failed for {seqs_path}. "
                f"See {out_base / f'{stem}_nn_failures.csv'} for details."
            )


if __name__ == "__main__":
    main()
