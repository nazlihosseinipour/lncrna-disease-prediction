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
from mainfolder.utils.output_summary import add_shape_record, shape_from_result, write_shape_summary


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
    shape_records = []

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
                lnc_path = out_base / f"{stem}_gip_lncRNA.csv"
                dis_path = out_base / f"{stem}_gip_disease.csv"
                gip_lnc.to_csv(lnc_path)
                gip_dis.to_csv(dis_path)
                lnc_rows, lnc_cols = shape_from_result(gip_lnc)
                dis_rows, dis_cols = shape_from_result(gip_dis)
                add_shape_record(
                    shape_records,
                    feature_group="cross",
                    feature_name="gip_lncRNA",
                    method_id=mid,
                    source_name=stem,
                    output_path=lnc_path,
                    rows=lnc_rows,
                    cols=lnc_cols,
                )
                add_shape_record(
                    shape_records,
                    feature_group="cross",
                    feature_name="gip_disease",
                    method_id=mid,
                    source_name=stem,
                    output_path=dis_path,
                    rows=dis_rows,
                    cols=dis_cols,
                )
                print(f"[saved] {lnc_path}")
                print(f"[saved] {dis_path}")
            elif mid == 17:
                lnc_feat, dis_feat = FeatureExtractor.run("cross", mid, matrix=M, k=args.k)
                lnc_df = pd.DataFrame(lnc_feat)
                dis_df = pd.DataFrame(dis_feat)
                lnc_path = out_base / f"{stem}_svd_lncRNA.csv"
                dis_path = out_base / f"{stem}_svd_disease.csv"
                lnc_df.to_csv(lnc_path, index=False)
                dis_df.to_csv(dis_path, index=False)
                lnc_rows, lnc_cols = shape_from_result(lnc_df)
                dis_rows, dis_cols = shape_from_result(dis_df)
                add_shape_record(
                    shape_records,
                    feature_group="cross",
                    feature_name="svd_lncRNA",
                    method_id=mid,
                    source_name=stem,
                    output_path=lnc_path,
                    rows=lnc_rows,
                    cols=lnc_cols,
                )
                add_shape_record(
                    shape_records,
                    feature_group="cross",
                    feature_name="svd_disease",
                    method_id=mid,
                    source_name=stem,
                    output_path=dis_path,
                    rows=dis_rows,
                    cols=dis_cols,
                )
                print(f"[saved] {lnc_path}")
                print(f"[saved] {dis_path}")

    summary_path = write_shape_summary(shape_records, out_base / "cross_feature_shapes.csv")
    print(f"[saved] {summary_path}")


if __name__ == "__main__":
    main()
