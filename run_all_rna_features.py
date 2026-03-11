"""
Run every RNA feature method over one or more sequence CSVs.

Notes:
- Methods 3–6 need dinucleotide properties; we require --props_csv so nothing is skipped.
- Accepts multiple inputs; if a file has columns ID/seqs, it is normalized to id/seq.

Usage example:
  python run_all_rna_features.py \
    --seqs_csv Data/output_data/sequences_for_oop.csv Data/output_data/website_full_matrix.csv \
    --outdir final-output \
    --props_csv path/to/dinuc_props.csv
"""

from pathlib import Path
import argparse
import pandas as pd

from mainfolder.utils.loader import load_sequences_csv, preprocess_sequences
from mainfolder.core.feature_extractor import FeatureExtractor
from mainfolder.features.rna_features import RnaFeatures
from mainfolder.utils.utils import ALPHABET


def load_props_csv(path: str):
    """Load dinucleotide properties from CSV with columns: dinuc,p1,p2,..."""
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if "dinuc" not in cols:
        raise ValueError("props_csv must include a 'dinuc' column")
    df["dinuc"] = df["dinuc"].str.upper().str.replace("T", "U")
    props = {}
    for _, row in df.iterrows():
        dinuc = row["dinuc"]
        vals = [float(x) for x in row.drop(labels="dinuc").tolist()]
        props[dinuc] = vals
    if not props:
        raise ValueError("props_csv contained no rows")
    return props


def normalize_id_seq(csv_path: Path):
    """
    Load a CSV that may be id/seq or ID/seqs and return ids, seqs lists.
    Extra columns (disease labels, etc.) are ignored.
    """
    df = pd.read_csv(csv_path, dtype=str)
    cols_lower = [c.lower() for c in df.columns]
    df.columns = cols_lower
    if "seqs" in df.columns and "seq" not in df.columns:
        df = df.rename(columns={"seqs": "seq"})
    if "id" not in df.columns:
        raise ValueError(f"{csv_path} must have an id/ID column")
    if "seq" not in df.columns:
        raise ValueError(f"{csv_path} must have a seq/seqs column")
    df = df[["id", "seq"]].dropna(subset=["id", "seq"])
    return df["id"].tolist(), df["seq"].tolist()


def main():
    p = argparse.ArgumentParser(description="Run all RNA feature methods over one or more CSVs.")
    p.add_argument(
        "--seqs_csv",
        nargs="+",
        required=True,
        help="One or more CSVs with id/seq or ID/seqs (e.g. sequences_for_oop.csv, website_full_matrix.csv)",
    )
    p.add_argument("--outdir", required=True, help="Base folder to save outputs")
    p.add_argument("--version_name", required=True, help="Data version label (e.g., v2, v3)")
    p.add_argument("--k", type=int, default=3, help="k for k-mer based methods (1,2,8,9)")
    p.add_argument("--lam", type=int, default=2, help="lam for PseDNC (method 3)")
    p.add_argument("--weight", type=float, default=0.5, help="w for PseDNC (method 3)")
    p.add_argument("--L", type=int, default=3, help="lag for DAC/DCC/DACC (methods 4,5,6)")
    p.add_argument("--k_gap", type=int, default=1, help="gap for monoMonoKGap/monoDiKGap (methods 11,12)")
    p.add_argument(
        "--props_csv",
        required=True,
        help="CSV with dinucleotide properties (dinuc + feature columns) so methods 3–6 can run",
    )
    args = p.parse_args()

    # place results under <outdir>/<version_name>/rna/
    base_out = Path(args.outdir) / args.version_name / "rna"
    base_out.mkdir(parents=True, exist_ok=True)

    props = load_props_csv(args.props_csv)

    # method_id -> kwargs (props always provided so nothing is skipped)
    method_params = {
        1: {"k": args.k},
        2: {"k": args.k},
        3: {"lam": args.lam, "w": args.weight, "props": props},
        4: {"L": args.L, "props": props},
        5: {"L": args.L, "props": props},
        6: {"L": args.L, "props": props},
        7: {},
        8: {},   # di_composition uses k=2 internally
        9: {},   # tri_composition uses k=3 internally
        10: {},
        11: {"k_gap": args.k_gap},
        12: {"k_gap": args.k_gap},
    }

    for seqs_csv in args.seqs_csv:
        seqs_path = Path(seqs_csv)
        ids, seqs = normalize_id_seq(seqs_path)
        # Drop sequences with invalid chars (including N) to avoid downstream errors
        ids2, seqs2 = preprocess_sequences(ids, seqs, valid_alphabet=set(ALPHABET), strict=False)

        stem = seqs_path.stem
        for mid, name in sorted(RnaFeatures.METHOD_MAP.items()):
            kwargs = method_params.get(mid, {}).copy()
            kwargs.update({"return_format": "dataframe", "sample_ids": ids2})
            # Only pass normalize to methods that accept it
            if mid in (1, 2, 7, 8, 9, 10, 11, 12):
                kwargs["normalize"] = True
            print(f"[run] {args.version_name}/{stem} -> method {mid} ({name})")
            cols, df = FeatureExtractor.run("rna", mid, seqs2, **kwargs)
            out_path = base_out / f"{stem}_{name}.csv"
            if isinstance(df, pd.DataFrame):
                df.to_csv(out_path, index=False)
            else:
                pd.DataFrame(df, columns=cols).to_csv(out_path, index=False)
            print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
