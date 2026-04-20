"""
Saved v1 vs v2 comparison reports for dataset matrices and generated feature outputs.

This module compares:
1. The source label matrices (default: Data/raw/sequences.csv vs Data/output_data/website_full_matrix.csv)
2. The generated outputs under final_output/v1 and final_output/v2
3. The per-run output-summary CSVs (*_feature_shapes.csv) when present

Outputs are written to a report directory as CSV files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_V1_MATRIX = Path("Data/raw/sequences.csv")
DEFAULT_V2_MATRIX = Path("Data/output_data/website_full_matrix.csv")
DEFAULT_FEATURE_BASE = Path("final_output")
DEFAULT_REPORT_DIR = Path("Data/output_data/comparison_reports")

ID_CANDIDATES = ("ID", "id", "lnc_id", "ncRNA Symbol")
SEQ_CANDIDATES = ("seq", "seqs")
DOMAIN_DIR_CANDIDATES = {
    "rna": ("rna",),
    "disease": ("disease", "Dis"),
    "cross": ("cross",),
    "nn": ("nn",),
}
DOMAIN_SUMMARY_FILES = {
    "rna": "rna_feature_shapes.csv",
    "disease": "disease_feature_shapes.csv",
    "cross": "cross_feature_shapes.csv",
    "nn": "nn_feature_shapes.csv",
}
DOMAINS = tuple(DOMAIN_DIR_CANDIDATES.keys())


def csv_shape(path: Path) -> str:
    try:
        df = pd.read_csv(path)
        return f"{len(df)}x{len(df.columns)}"
    except Exception as exc:
        return f"error: {exc}"


def _normalized_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in columns}


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    cols_lookup = _normalized_lookup(df.columns)

    unnamed_cols = [col for col in df.columns if str(col).strip().lower().startswith("unnamed:")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        cols_lookup = _normalized_lookup(df.columns)

    id_col = None
    for candidate in ID_CANDIDATES:
        id_col = cols_lookup.get(candidate.lower())
        if id_col is not None:
            break
    if id_col is None:
        raise ValueError(f"{path} missing an ID column. Tried: {', '.join(ID_CANDIDATES)}")

    df = df.rename(columns={id_col: "ID"})
    cols_lookup = _normalized_lookup(df.columns)

    seq_cols = []
    for candidate in SEQ_CANDIDATES:
        seq_col = cols_lookup.get(candidate.lower())
        if seq_col is not None and seq_col != "ID":
            seq_cols.append(seq_col)
    if seq_cols:
        df = df.drop(columns=sorted(set(seq_cols)))

    label_cols = [col for col in df.columns if col != "ID"]
    if not label_cols:
        raise ValueError(f"{path} has no disease/label columns after dropping ID/sequence columns")

    numeric_block = (
        df[label_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    out = pd.concat(
        [
            df["ID"].astype(str).str.strip().rename("ID"),
            numeric_block,
        ],
        axis=1,
    )
    out = out.dropna(subset=["ID"])
    out = out[out["ID"].astype(str).str.strip().ne("")]
    return out


def dataset_stats(df: pd.DataFrame) -> dict[str, float | int]:
    label_cols = [c for c in df.columns if c != "ID"]
    labels = df[label_cols]
    n_seq = len(df)
    n_dis = len(label_cols)
    total_ones = int(labels.to_numpy(dtype=int, copy=False).sum())
    avg_pos_per_row_sequence = float(labels.sum(axis=1).mean()) if n_seq else 0.0
    avg_pos_per_col_disease = float(labels.sum(axis=0).mean()) if n_dis else 0.0
    density = total_ones / (n_seq * n_dis) if n_seq and n_dis else 0.0
    return {
        "num_sequences": int(n_seq),
        "num_diseases": int(n_dis),
        "total_ones": total_ones,
        "avg_diseases_per_seq": avg_pos_per_row_sequence,  # backward-compatible alias
        "avg_pos_per_row_sequence": avg_pos_per_row_sequence,
        "avg_pos_per_col_disease": avg_pos_per_col_disease,
        "label_density": density,
    }


def matrix_version_summary(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    stats1 = dataset_stats(df1)
    stats2 = dataset_stats(df2)
    delta = {
        key: (
            round(stats2[key] - stats1[key], 6)
            if isinstance(stats1[key], float) or isinstance(stats2[key], float)
            else int(stats2[key] - stats1[key])
        )
        for key in stats1
    }
    return pd.DataFrame(
        [
            {"version": "v1", **stats1},
            {"version": "v2", **stats2},
            {"version": "delta_v2_minus_v1", **delta},
        ]
    )


def compare_matrix_changes(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids1 = set(df1["ID"].astype(str))
    ids2 = set(df2["ID"].astype(str))
    diseases1 = [c for c in df1.columns if c != "ID"]
    diseases2 = [c for c in df2.columns if c != "ID"]
    disease_set1 = set(diseases1)
    disease_set2 = set(diseases2)

    common_ids = sorted(ids1 & ids2)
    new_ids = sorted(ids2 - ids1)
    dropped_ids = sorted(ids1 - ids2)
    common_diseases = sorted(disease_set1 & disease_set2)
    new_diseases = sorted(disease_set2 - disease_set1)
    dropped_diseases = sorted(disease_set1 - disease_set2)

    sub1 = df1.set_index("ID").loc[common_ids, common_diseases].sort_index() if common_ids and common_diseases else pd.DataFrame(index=common_ids, columns=common_diseases)
    sub2 = df2.set_index("ID").loc[common_ids, common_diseases].sort_index() if common_ids and common_diseases else pd.DataFrame(index=common_ids, columns=common_diseases)
    sub1 = sub1.fillna(0).astype(int)
    sub2 = sub2.fillna(0).astype(int)

    zero_to_one = ((sub2 == 1) & (sub1 == 0))
    one_to_zero = ((sub1 == 1) & (sub2 == 0))

    summary = pd.DataFrame(
        [
            {
                "v1_sequences": len(ids1),
                "v2_sequences": len(ids2),
                "sequence_delta_v2_minus_v1": len(ids2) - len(ids1),
                "common_sequences": len(common_ids),
                "new_sequences_in_v2": len(new_ids),
                "dropped_sequences_from_v1": len(dropped_ids),
                "v1_diseases": len(disease_set1),
                "v2_diseases": len(disease_set2),
                "disease_delta_v2_minus_v1": len(disease_set2) - len(disease_set1),
                "common_diseases": len(common_diseases),
                "new_diseases_in_v2": len(new_diseases),
                "dropped_diseases_from_v1": len(dropped_diseases),
                "added_interactions_0_to_1": int(zero_to_one.to_numpy(dtype=int, copy=False).sum()),
                "removed_interactions_1_to_0": int(one_to_zero.to_numpy(dtype=int, copy=False).sum()),
            }
        ]
    )

    disease_rank = pd.DataFrame(
        {
            "disease": common_diseases,
            "v1_positive_count_common_sequences": sub1.sum(axis=0).astype(int).tolist(),
            "v2_positive_count_common_sequences": sub2.sum(axis=0).astype(int).tolist(),
            "positive_count_delta_v2_minus_v1": (sub2.sum(axis=0) - sub1.sum(axis=0)).astype(int).tolist(),
            "zero_to_one_count": zero_to_one.sum(axis=0).astype(int).tolist(),
            "one_to_zero_count": one_to_zero.sum(axis=0).astype(int).tolist(),
        }
    )
    if not disease_rank.empty:
        disease_rank["total_changed"] = disease_rank["zero_to_one_count"] + disease_rank["one_to_zero_count"]
        disease_rank["abs_positive_count_delta"] = disease_rank["positive_count_delta_v2_minus_v1"].abs()
        disease_rank = disease_rank.sort_values(
            by=["total_changed", "abs_positive_count_delta", "disease"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        disease_rank.insert(0, "change_rank", range(1, len(disease_rank) + 1))

    new_diseases_df = pd.DataFrame({"disease": new_diseases})
    dropped_diseases_df = pd.DataFrame({"disease": dropped_diseases})
    return summary, disease_rank, new_diseases_df, dropped_diseases_df


def _resolve_domain_dir(base: Path, version: str, domain: str) -> Path | None:
    version_root = base / version
    for candidate in DOMAIN_DIR_CANDIDATES[domain]:
        p = version_root / candidate
        if p.exists():
            return p
    return None


def compare_feature_files(base: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for domain in DOMAINS:
        dir_v1 = _resolve_domain_dir(base, "v1", domain)
        dir_v2 = _resolve_domain_dir(base, "v2", domain)
        if dir_v1 is None or dir_v2 is None:
            continue

        files_v1 = {p.name: p for p in dir_v1.glob("*") if p.is_file()}
        files_v2 = {p.name: p for p in dir_v2.glob("*") if p.is_file()}
        common = sorted(set(files_v1) & set(files_v2))
        for name in common:
            rows.append(
                {
                    "domain": domain,
                    "file": name,
                    "v1_shape": csv_shape(files_v1[name]),
                    "v2_shape": csv_shape(files_v2[name]),
                }
            )
    return pd.DataFrame(rows, columns=["domain", "file", "v1_shape", "v2_shape"])


def compare_shape_summaries(base: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    empty_cols = [
        "domain",
        "feature_group",
        "feature_name",
        "method_id",
        "source_name",
        "rows_v1",
        "cols_v1",
        "output_path_v1",
        "rows_v2",
        "cols_v2",
        "output_path_v2",
        "row_delta_v2_minus_v1",
        "col_delta_v2_minus_v1",
    ]
    for domain in DOMAINS:
        dir_v1 = _resolve_domain_dir(base, "v1", domain)
        dir_v2 = _resolve_domain_dir(base, "v2", domain)
        if dir_v1 is None or dir_v2 is None:
            continue

        summary_name = DOMAIN_SUMMARY_FILES[domain]
        path_v1 = dir_v1 / summary_name
        path_v2 = dir_v2 / summary_name
        if not path_v1.exists() or not path_v2.exists():
            continue

        df1 = pd.read_csv(path_v1).fillna("")
        df2 = pd.read_csv(path_v2).fillna("")
        merge_cols = ["feature_group", "feature_name", "method_id", "source_name"]
        for frame in (df1, df2):
            for col in merge_cols:
                if col not in frame.columns:
                    frame[col] = ""
        keep_cols = merge_cols + ["rows", "cols", "output_path"]
        merged = df1[keep_cols].merge(
            df2[keep_cols],
            how="outer",
            on=merge_cols,
            suffixes=("_v1", "_v2"),
        )
        merged.insert(0, "domain", domain)
        for col in ("rows_v1", "cols_v1", "rows_v2", "cols_v2"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged["row_delta_v2_minus_v1"] = merged["rows_v2"] - merged["rows_v1"]
        merged["col_delta_v2_minus_v1"] = merged["cols_v2"] - merged["cols_v1"]
        rows.append(merged)

    if not rows:
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(rows, ignore_index=True)


def write_reports(
    *,
    v1_matrix_path: Path,
    v2_matrix_path: Path,
    feature_base: Path,
    report_dir: Path,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)

    df1 = load_matrix(v1_matrix_path)
    df2 = load_matrix(v2_matrix_path)

    version_summary = matrix_version_summary(df1, df2)
    matrix_change_summary, disease_rank, new_diseases_df, dropped_diseases_df = compare_matrix_changes(df1, df2)
    feature_file_comparison = compare_feature_files(feature_base)
    shape_summary_comparison = compare_shape_summaries(feature_base)

    outputs = {
        "matrix_version_summary": report_dir / "matrix_version_summary.csv",
        "matrix_change_summary": report_dir / "matrix_change_summary.csv",
        "disease_change_ranking": report_dir / "disease_change_ranking.csv",
        "new_diseases_in_v2": report_dir / "new_diseases_in_v2.csv",
        "dropped_diseases_from_v1": report_dir / "dropped_diseases_from_v1.csv",
        "feature_file_shape_comparison": report_dir / "feature_file_shape_comparison.csv",
        "feature_shape_summary_comparison": report_dir / "feature_shape_summary_comparison.csv",
    }

    version_summary.to_csv(outputs["matrix_version_summary"], index=False)
    matrix_change_summary.to_csv(outputs["matrix_change_summary"], index=False)
    disease_rank.to_csv(outputs["disease_change_ranking"], index=False)
    new_diseases_df.to_csv(outputs["new_diseases_in_v2"], index=False)
    dropped_diseases_df.to_csv(outputs["dropped_diseases_from_v1"], index=False)
    feature_file_comparison.to_csv(outputs["feature_file_shape_comparison"], index=False)
    shape_summary_comparison.to_csv(outputs["feature_shape_summary_comparison"], index=False)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Save v1 vs v2 dataset and output comparison reports.")
    parser.add_argument("--v1-matrix", default=str(DEFAULT_V1_MATRIX), help="Matrix CSV representing v1")
    parser.add_argument("--v2-matrix", default=str(DEFAULT_V2_MATRIX), help="Matrix CSV representing v2")
    parser.add_argument("--feature-base", default=str(DEFAULT_FEATURE_BASE), help="Base directory containing v1/ and v2/ outputs")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Directory to save comparison CSVs")
    parser.add_argument("--topk", type=int, default=15, help="How many changed diseases to print in the console summary")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    v1_matrix_path = Path(args.v1_matrix)
    v2_matrix_path = Path(args.v2_matrix)
    feature_base = Path(args.feature_base)
    report_dir = Path(args.report_dir)

    outputs = write_reports(
        v1_matrix_path=v1_matrix_path,
        v2_matrix_path=v2_matrix_path,
        feature_base=feature_base,
        report_dir=report_dir,
    )

    version_summary = pd.read_csv(outputs["matrix_version_summary"])
    matrix_change_summary = pd.read_csv(outputs["matrix_change_summary"])
    disease_rank = pd.read_csv(outputs["disease_change_ranking"])
    summary_row = matrix_change_summary.iloc[0]

    print("\n=== Matrix Version Summary ===")
    print(version_summary.to_string(index=False))

    print("\n=== Matrix Change Summary ===")
    print(matrix_change_summary.to_string(index=False))

    print(f"\nTop changed diseases (top {args.topk})")
    if disease_rank.empty:
        print("No common diseases to rank.")
    else:
        print(
            disease_rank[
                [
                    "change_rank",
                    "disease",
                    "v1_positive_count_common_sequences",
                    "v2_positive_count_common_sequences",
                    "positive_count_delta_v2_minus_v1",
                    "zero_to_one_count",
                    "one_to_zero_count",
                    "total_changed",
                ]
            ]
            .head(args.topk)
            .to_string(index=False)
        )

    print("\nSaved reports")
    for key, path in outputs.items():
        print(f"{key}: {path}")

    print(
        "\nQuick summary: "
        f"v2 has {int(summary_row['new_diseases_in_v2'])} new diseases, "
        f"{int(summary_row['added_interactions_0_to_1'])} 0->1 changes, and "
        f"{int(summary_row['removed_interactions_1_to_0'])} 1->0 changes "
        f"on {int(summary_row['common_sequences'])} common sequences."
    )


if __name__ == "__main__":
    main()
