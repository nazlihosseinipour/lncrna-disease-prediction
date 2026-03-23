"""
Run all Cross feature methods (GIP, SVD) over one or more matrices.

Inputs: CSV matrices (rows=lncRNAs, cols=diseases). If the first column is an ID,
it is used as the row index; otherwise all columns are treated as numeric.

Outputs: <outdir>/<version_name>/cross/
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from mainfolder.core.feature_extractor import FeatureExtractor
from mainfolder.features.cross_features import CrossFeatures


def load_matrix(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    cols = list(df.columns)
    if cols and cols[0].lower() in ("id", "lnc_id", "ncRNA Symbol".lower()):
        df = df.set_index(cols[0])
    # keep only numeric columns (drop seqs or other text columns)
    for c in df.columns:
        try:
            df[c] = df[c].astype(float)
        except Exception:
            df = df.drop(columns=[c])
    if df.shape[1] == 0:
        raise ValueError(
            f"{csv_path} has no numeric disease columns after filtering. "
            "Expected an lncRNA x disease numeric matrix (with optional ID first column)."
        )
    if df.shape[0] == 0:
        raise ValueError(f"{csv_path} has zero rows after loading; cannot run cross features.")
    return df


def main():
    p = argparse.ArgumentParser(description="Run all cross feature methods over one or more matrices.")
    p.add_argument(
        "--matrix_csv",
        nargs="+",
        required=True,
        help="One or more CSVs with lncRNA x disease matrices (e.g., website_disease_matrix.csv)",
    )
    p.add_argument("--outdir", required=True, help="Base output directory")
    p.add_argument("--version_name", required=True, help="Data version label (e.g., v1, v2)")
    p.add_argument("--k", type=int, default=64, help="k for SVD features (method 17)")
    args = p.parse_args()

    out_base = Path(args.outdir) / args.version_name / "cross"
    out_base.mkdir(parents=True, exist_ok=True)

    # method_ids from CrossFeatures.METHOD_MAP
    method_ids = sorted(CrossFeatures.METHOD_MAP.keys())

    for m_csv in args.matrix_csv:
        m_path = Path(m_csv)
        if not m_path.exists():
            raise FileNotFoundError(f"Matrix file not found: {m_path}")
        M = load_matrix(m_path)
        stem = m_path.stem

        for mid in method_ids:
            name = CrossFeatures.METHOD_MAP[mid]
            print(f"[run] {args.version_name}/{stem} -> method {mid} ({name})")
            if mid == 16:
                gip_lnc, gip_dis = FeatureExtractor.run("cross", mid, matrix=M)
                gip_lnc.to_csv(out_base / f"{stem}_gip_lncRNA.csv")
                gip_dis.to_csv(out_base / f"{stem}_gip_disease.csv")
                print(f"[saved] {out_base / f'{stem}_gip_lncRNA.csv'}")
                print(f"[saved] {out_base / f'{stem}_gip_disease.csv'}")
            elif mid == 17:
                lnc_feat, dis_feat = FeatureExtractor.run("cross", mid, matrix=M, k=args.k)
                pd.DataFrame(lnc_feat).to_csv(out_base / f"{stem}_svd_lncRNA.csv", index=False)
                pd.DataFrame(dis_feat).to_csv(out_base / f"{stem}_svd_disease.csv", index=False)
                print(f"[saved] {out_base / f'{stem}_svd_lncRNA.csv'}")
                print(f"[saved] {out_base / f'{stem}_svd_disease.csv'}")


if __name__ == "__main__":
    main()
