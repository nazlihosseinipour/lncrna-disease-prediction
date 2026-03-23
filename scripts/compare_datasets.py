"""
Compare v1 vs v2 outputs across all feature domains (rna, disease, cross, nn),
and also provide the legacy label-stats comparison for two matrices.

Usage:
  python scripts/compare_datasets.py
"""

from pathlib import Path
import pandas as pd

# Legacy matrix comparison 
# Legacy matrix comparison (label stats). Adjust these to real files on disk.
# v1: raw sequences matrix with diseases
# v2: website full matrix with diseases
LEGACY_V1 = Path("Data/raw/sequences.csv")
LEGACY_V2 = Path("Data/output_data/website_full_matrix.csv")
ID_COL = "ID"    # expected ID column
SEQ_COL = "seqs" # drop this if present

BASE = Path("final-output")
DOMAINS = ["rna", "disease", "cross", "nn"]


def csv_shape(path: Path) -> str:
    try:
        df = pd.read_csv(path, nrows=None)
        return f"{len(df)}x{len(df.columns)}"
    except Exception as e:
        return f"error: {e}"


def compare_domain(domain: str):
    dir_v1 = BASE / "v1" / domain
    dir_v2 = BASE / "v2" / domain
    if not dir_v1.exists() or not dir_v2.exists():
        return pd.DataFrame(columns=["domain", "file", "v1_shape", "v2_shape"])

    files_v1 = {p.name: p for p in dir_v1.glob("*") if p.is_file()}
    files_v2 = {p.name: p for p in dir_v2.glob("*") if p.is_file()}
    common = sorted(set(files_v1) & set(files_v2))

    rows = []
    for name in common:
        p1, p2 = files_v1[name], files_v2[name]
        rows.append(
            {
                "domain": domain,
                "file": name,
                "v1_shape": csv_shape(p1),
                "v2_shape": csv_shape(p2)
            }
        )
    return pd.DataFrame(rows)



def load_matrix(path: Path, id_col: str, seq_col: str | None):
    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}
    id_key = id_col if id_col in df.columns else cols_lower.get(id_col.lower())
    if id_key is None:
        raise ValueError(f"{path} missing ID column '{id_col}'")
    df = df.rename(columns={id_key: "ID"})
    for seq_candidate in filter(None, {seq_col, "seq", "seqs"}):
        seq_key = seq_candidate if seq_candidate in df.columns else cols_lower.get(str(seq_candidate).lower())
        if seq_key and seq_key in df.columns:
            df = df.drop(columns=[seq_key])
    return df


def dataset_stats(df: pd.DataFrame):
    if "ID" not in df.columns:
        raise ValueError("DataFrame must have an 'ID' column")
    label_cols = [c for c in df.columns if c != "ID"]
    labels = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    n_seq = len(df)
    n_dis = len(label_cols)
    total_ones = labels.to_numpy(dtype=int, copy=False).sum()
    # Row axis = sequences (one row per ID), column axis = diseases.
    avg_pos_per_row_sequence = float(labels.sum(axis=1).mean()) if n_seq else 0.0
    avg_pos_per_col_disease = float(labels.sum(axis=0).mean()) if n_dis else 0.0
    density = total_ones / (n_seq * n_dis) if n_seq and n_dis else 0.0
    return {
        "num_sequences": n_seq,
        "num_diseases": n_dis,
        "avg_diseases_per_seq": avg_pos_per_row_sequence,  # backward-compatible alias
        "avg_pos_per_row_sequence": avg_pos_per_row_sequence,
        "avg_pos_per_col_disease": avg_pos_per_col_disease,
        "label_density": density,
        "total_ones": total_ones,
    }


def compare_interactions(df1: pd.DataFrame, df2: pd.DataFrame):
    common_ids = sorted(set(df1["ID"]) & set(df2["ID"]))
    label_cols1 = [c for c in df1.columns if c != "ID"]
    label_cols2 = [c for c in df2.columns if c != "ID"]
    common_labels = sorted(set(label_cols1) & set(label_cols2))
    sub1 = df1.set_index("ID").loc[common_ids, common_labels].sort_index()
    sub2 = df2.set_index("ID").loc[common_ids, common_labels].sort_index()
    mat1 = sub1.to_numpy(dtype=int, copy=False)
    mat2 = sub2.to_numpy(dtype=int, copy=False)
    added = ((mat2 == 1) & (mat1 == 0)).sum()
    removed = ((mat1 == 1) & (mat2 == 0)).sum()
    return {
        "common_sequences": len(common_ids),
        "common_diseases": len(common_labels),
        "added_interactions": int(added),
        "removed_interactions": int(removed),
    }


def legacy_compare():
    rows = []
    try:
        df1 = load_matrix(LEGACY_V1, ID_COL, SEQ_COL)
        df2 = load_matrix(LEGACY_V2, ID_COL, SEQ_COL)
        stats1 = dataset_stats(df1)
        stats2 = dataset_stats(df2)
        cmp_stats = compare_interactions(df1, df2)
        summary = pd.DataFrame(
            {
                "version": ["v1", "v2"],
                "num_sequences": [stats1["num_sequences"], stats2["num_sequences"]],
                "num_diseases": [stats1["num_diseases"], stats2["num_diseases"]],
                # Backward-compatible legacy field name.
                "avg_diseases_per_seq": [stats1["avg_diseases_per_seq"], stats2["avg_diseases_per_seq"]],
                "avg_pos_per_row_sequence": [stats1["avg_pos_per_row_sequence"], stats2["avg_pos_per_row_sequence"]],
                "avg_pos_per_col_disease": [stats1["avg_pos_per_col_disease"], stats2["avg_pos_per_col_disease"]],
                "label_density": [stats1["label_density"], stats2["label_density"]],
            }
        )
        delta = pd.DataFrame(
            {
                "common_sequences": [cmp_stats["common_sequences"]],
                "common_diseases": [cmp_stats["common_diseases"]],
                "added_interactions": [cmp_stats["added_interactions"]],
                "removed_interactions": [cmp_stats["removed_interactions"]],
            }
        )
        rows.append(("summary", summary))
        rows.append(("delta", delta))
    except Exception as e:
        print(f"[legacy] skipped: {e}")
        rows = []
    return rows


def main():
    # Legacy comparison (label stats)
    legacy = legacy_compare()
    for title, df in legacy:
        print(f"\n=== Legacy {title} ===")
        print(df.to_string(index=False))

    # Domain file shape comparison
    frames = [compare_domain(d) for d in DOMAINS]
    summary = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if summary.empty:
        print("\nNo comparable files found under final-output/v1 and final-output/v2.")
        return
    for domain in DOMAINS:
        sub = summary[summary["domain"] == domain]
        if sub.empty:
            continue
        print(f"\n=== {domain.upper()} (common files: {len(sub)}) ===")
        print(sub[["file", "v1_shape", "v2_shape"]].to_string(index=False))


if __name__ == "__main__":
    main()
