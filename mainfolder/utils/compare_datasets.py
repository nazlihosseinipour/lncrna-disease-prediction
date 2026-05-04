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


DEFAULT_V1_MATRIX = Path("Data/output_data/sequences_for_oop.csv")
DEFAULT_V2_MATRIX = Path("Data/output_data/website_full_matrix.csv")
DEFAULT_FEATURE_BASE = Path("final_output")
DEFAULT_REPORT_DIR = Path("Data/output_data/comparison_reports")

VERSION_DIR_CANDIDATES = {
    "v1": ("v1", "V1"),
    "v2": ("v2", "V2"),
}
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


def raw_csv_shape(path: Path) -> tuple[int, int]:
    df = pd.read_csv(path)
    return int(len(df)), int(len(df.columns))


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


def original_input_summary(v1_matrix_path: Path, v2_matrix_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for version, path in (("v1", v1_matrix_path), ("v2", v2_matrix_path)):
        raw_rows, raw_cols = raw_csv_shape(path)
        normalized = load_matrix(path)
        stats = dataset_stats(normalized)
        rows.append(
            {
                "version": version,
                "matrix_path": str(path),
                "raw_csv_rows": raw_rows,
                "raw_csv_cols": raw_cols,
                "normalized_rows": int(stats["num_sequences"]),
                "normalized_disease_cols": int(stats["num_diseases"]),
            }
        )
    return pd.DataFrame(rows)


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


def _resolve_version_dir(base: Path, version: str) -> Path | None:
    for candidate in VERSION_DIR_CANDIDATES.get(version, (version,)):
        p = base / candidate
        if p.exists():
            return p
    return None


def _resolve_domain_dir(base: Path, version: str, domain: str) -> Path | None:
    version_root = _resolve_version_dir(base, version)
    if version_root is None:
        return None
    for candidate in DOMAIN_DIR_CANDIDATES[domain]:
        p = version_root / candidate
        if p.exists():
            return p
    return None


def _resolve_output_file(
    *,
    base: Path,
    version: str,
    domain: str,
    output_path: str,
) -> Path | None:
    candidate = Path(output_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    domain_dir = _resolve_domain_dir(base, version, domain)
    if domain_dir is not None:
        basename_candidate = domain_dir / candidate.name
        if basename_candidate.exists():
            return basename_candidate

    if not candidate.is_absolute():
        repo_candidate = Path.cwd() / candidate
        if repo_candidate.exists():
            return repo_candidate

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


def _expected_feature_shape(
    *,
    domain: str,
    feature_name: str,
    method_id: int | None,
    n_sequences: int,
    n_diseases: int,
) -> tuple[int | None, int | None, str]:
    feature_key = str(feature_name).strip().lower()

    if domain == "rna":
        return n_sequences, None, "RNA features should keep one row per sequence; feature width is method-specific."

    if domain == "nn":
        if int(method_id or -1) == 101 or feature_key == "mp_tokens":
            return None, None, "Token-level output; compare unique sample_id count to input sequence count."
        return n_sequences, None, "Sequence-level NN embeddings should keep one row per sequence."

    if domain == "cross":
        if "gip_lnc" in feature_key:
            return n_sequences, n_sequences, "lncRNA GIP kernel should be square over sequences."
        if "gip_disease" in feature_key:
            return n_diseases, n_diseases, "Disease GIP kernel should be square over diseases."
        if "svd_lnc" in feature_key:
            return n_sequences, None, "SVD lncRNA embedding should keep one row per sequence; embedding width is method-specific."
        if "svd_disease" in feature_key:
            return n_diseases, None, "SVD disease embedding should keep one row per disease; embedding width is method-specific."

    if domain == "disease":
        if "term_similarity_wang" in feature_key or "disease_similarity_bma" in feature_key:
            return n_diseases, n_diseases, "Disease similarity matrices should be square over diseases."
        if "lfs_from_y" in feature_key:
            return n_sequences, n_sequences, "LFS should be square over sequences."

    return None, None, "No sanity rule defined for this output."


def _safe_bool_match(observed: int | None, expected: int | None) -> str:
    if expected is None:
        return "not_checked"
    if observed is None:
        return "missing"
    return "ok" if int(observed) == int(expected) else "mismatch"


def _token_sample_count(
    *,
    base: Path,
    version: str,
    domain: str,
    output_path: str,
) -> int | None:
    resolved = _resolve_output_file(base=base, version=version, domain=domain, output_path=output_path)
    if resolved is None or not resolved.exists():
        return None
    df = pd.read_csv(resolved, usecols=["sample_id"])
    return int(df["sample_id"].astype(str).nunique())


def build_feature_sanity_report(
    *,
    v1_matrix_path: Path,
    v2_matrix_path: Path,
    feature_base: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    version_meta = {
        "v1": load_matrix(v1_matrix_path),
        "v2": load_matrix(v2_matrix_path),
    }

    for version, matrix_df in version_meta.items():
        n_sequences = int(len(matrix_df))
        n_diseases = int(len([c for c in matrix_df.columns if c != "ID"]))

        for domain in DOMAINS:
            domain_dir = _resolve_domain_dir(feature_base, version, domain)
            if domain_dir is None:
                continue
            summary_path = domain_dir / DOMAIN_SUMMARY_FILES[domain]
            if not summary_path.exists():
                continue

            summary_df = pd.read_csv(summary_path).fillna("")
            if summary_df.empty:
                continue

            rows: list[dict[str, object]] = []
            for record in summary_df.to_dict("records"):
                method_id_raw = record.get("method_id", "")
                method_id = int(method_id_raw) if str(method_id_raw).strip() else None
                expected_rows, expected_cols, note = _expected_feature_shape(
                    domain=domain,
                    feature_name=str(record.get("feature_name", "")),
                    method_id=method_id,
                    n_sequences=n_sequences,
                    n_diseases=n_diseases,
                )
                output_rows = int(record.get("rows", 0) or 0)
                output_cols = int(record.get("cols", 0) or 0)

                sample_count = None
                sample_status = "not_checked"
                if domain == "nn" and method_id == 101:
                    sample_count = _token_sample_count(
                        base=feature_base,
                        version=version,
                        domain=domain,
                        output_path=str(record.get("output_path", "")),
                    )
                    sample_status = _safe_bool_match(sample_count, n_sequences)

                rows.append(
                    {
                        "version": version,
                        "domain": domain,
                        "feature_group": str(record.get("feature_group", "")),
                        "feature_name": str(record.get("feature_name", "")),
                        "method_id": method_id,
                        "source_name": str(record.get("source_name", "")),
                        "original_sequence_rows": n_sequences,
                        "original_disease_cols": n_diseases,
                        "output_rows": output_rows,
                        "output_cols": output_cols,
                        "expected_rows": expected_rows,
                        "expected_cols": expected_cols,
                        "row_status": _safe_bool_match(output_rows, expected_rows),
                        "col_status": _safe_bool_match(output_cols, expected_cols),
                        "observed_unique_samples": sample_count,
                        "sample_status": sample_status,
                        "output_path": str(record.get("output_path", "")),
                        "sanity_note": note,
                    }
                )

            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(
            columns=[
                "version",
                "domain",
                "feature_group",
                "feature_name",
                "method_id",
                "source_name",
                "original_sequence_rows",
                "original_disease_cols",
                "output_rows",
                "output_cols",
                "expected_rows",
                "expected_cols",
                "row_status",
                "col_status",
                "observed_unique_samples",
                "sample_status",
                "output_path",
                "sanity_note",
            ]
        )
    return pd.concat(frames, ignore_index=True)


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
    original_summary = original_input_summary(v1_matrix_path, v2_matrix_path)
    feature_file_comparison = compare_feature_files(feature_base)
    shape_summary_comparison = compare_shape_summaries(feature_base)
    feature_sanity_report = build_feature_sanity_report(
        v1_matrix_path=v1_matrix_path,
        v2_matrix_path=v2_matrix_path,
        feature_base=feature_base,
    )

    outputs = {
        "original_input_shape_summary": report_dir / "original_input_shape_summary.csv",
        "matrix_version_summary": report_dir / "matrix_version_summary.csv",
        "matrix_change_summary": report_dir / "matrix_change_summary.csv",
        "disease_change_ranking": report_dir / "disease_change_ranking.csv",
        "new_diseases_in_v2": report_dir / "new_diseases_in_v2.csv",
        "dropped_diseases_from_v1": report_dir / "dropped_diseases_from_v1.csv",
        "feature_file_shape_comparison": report_dir / "feature_file_shape_comparison.csv",
        "feature_shape_summary_comparison": report_dir / "feature_shape_summary_comparison.csv",
        "feature_sanity_check": report_dir / "feature_sanity_check.csv",
    }

    original_summary.to_csv(outputs["original_input_shape_summary"], index=False)
    version_summary.to_csv(outputs["matrix_version_summary"], index=False)
    matrix_change_summary.to_csv(outputs["matrix_change_summary"], index=False)
    disease_rank.to_csv(outputs["disease_change_ranking"], index=False)
    new_diseases_df.to_csv(outputs["new_diseases_in_v2"], index=False)
    dropped_diseases_df.to_csv(outputs["dropped_diseases_from_v1"], index=False)
    feature_file_comparison.to_csv(outputs["feature_file_shape_comparison"], index=False)
    shape_summary_comparison.to_csv(outputs["feature_shape_summary_comparison"], index=False)
    feature_sanity_report.to_csv(outputs["feature_sanity_check"], index=False)
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
    feature_sanity = pd.read_csv(outputs["feature_sanity_check"])
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

    if not feature_sanity.empty:
        mismatches = feature_sanity[
            feature_sanity["row_status"].eq("mismatch")
            | feature_sanity["col_status"].eq("mismatch")
            | feature_sanity["sample_status"].eq("mismatch")
        ]
        print(
            "\nFeature sanity summary: "
            f"{len(feature_sanity)} outputs checked, "
            f"{len(mismatches)} with at least one mismatch."
        )

    print(
        "\nQuick summary: "
        f"v2 has {int(summary_row['new_diseases_in_v2'])} new diseases, "
        f"{int(summary_row['added_interactions_0_to_1'])} 0->1 changes, and "
        f"{int(summary_row['removed_interactions_1_to_0'])} 1->0 changes "
        f"on {int(summary_row['common_sequences'])} common sequences."
    )


if __name__ == "__main__":
    main()
