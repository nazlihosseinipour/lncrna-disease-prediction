
# from pathlib import Path
# import pandas as pd
# import argparse

# def read_any(path):
#     return pd.read_csv(path, sep="\t" if str(path).lower().endswith(".tsv") else ",")

# def build_sequences_csv(raw_sequences_csv: str, id_col: str, seq_col: str, out_csv: str) -> str:
#     df = read_any(raw_sequences_csv)
#     # Drop obvious junk cols like 'Unnamed: 0'
#     df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
#     seqs = df[[id_col, seq_col]].dropna().copy()
#     seqs.columns = ["id", "seq"]
#     Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
#     seqs.to_csv(out_csv, index=False)
#     return out_csv

# def build_Y_from_sequences(raw_sequences_csv: str, id_col: str, seq_col: str, out_csv: str) -> str:
#     df = read_any(raw_sequences_csv)
#     df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]
#     # All columns except id & seq are disease labels
#     disease_cols = [c for c in df.columns if c not in (id_col, seq_col)]
#     Y = df.set_index(id_col)[disease_cols]
#     Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
#     Y.to_csv(out_csv)
#     return out_csv

# def build_Y_matrix_from_pairs(pairs_csv: str, lnc_col: str, dis_col: str, out_csv: str) -> str:
#     df = read_any(pairs_csv)[[lnc_col, dis_col]].dropna()
#     mat = (df.assign(val=1)
#              .pivot_table(index=lnc_col, columns=dis_col, values="val",
#                           aggfunc="max", fill_value=0))
#     Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
#     mat.to_csv(out_csv)
#     return out_csv

# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("--raw_sequences_csv", required=True)
#     p.add_argument("--raw_pairs_csv", required=False, help="Optional; if omitted, Y is derived from sequences.csv")
#     p.add_argument("--out_dir", default="Data/processed")
#     p.add_argument("--id_col", default="ID")
#     p.add_argument("--seq_col", default="seqs")
#     p.add_argument("--lnc_col", default="ID")
#     p.add_argument("--dis_col", default="Disease Name")
#     a = p.parse_args()

#     out = Path(a.out_dir)
#     out.mkdir(parents=True, exist_ok=True)

#     build_sequences_csv(a.raw_sequences_csv, a.id_col, a.seq_col, str(out / "sequences.csv"))

#     if a.raw_pairs_csv:
#         # only use if identifiers match your ID system
#         build_Y_matrix_from_pairs(a.raw_pairs_csv, a.lnc_col, a.dis_col, str(out / "Y.csv"))
#     else:
#         build_Y_from_sequences(a.raw_sequences_csv, a.id_col, a.seq_col, str(out / "Y.csv"))

#     print("[ok] wrote:", out / "sequences.csv", "and", out / "Y.csv")

from pathlib import Path
import pandas as pd
import argparse


def read_any(path):
    """Read CSV/TSV based on file extension."""
    return pd.read_csv(path, sep="\t" if str(path).lower().endswith(".tsv") else ",")


def build_sequences_csv(raw_sequences_csv: str, id_col: str, seq_col: str, out_csv: str) -> str:
    """Create a clean sequences.csv with only [id, seq]."""
    df = read_any(raw_sequences_csv)
    # Drop obvious junk cols like 'Unnamed: 0'
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

    seqs = df[[id_col, seq_col]].dropna().copy()
    seqs.columns = ["id", "seq"]

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    seqs.to_csv(out_csv, index=False)
    return out_csv


def build_Y_from_sequences(raw_sequences_csv: str, id_col: str, seq_col: str) -> pd.DataFrame:
    """
    Build Y from the wide label format in raw_sequences_csv.
    All columns except id_col and seq_col are interpreted as disease labels.
    """
    df = read_any(raw_sequences_csv)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

    # All columns except id & seq are disease labels
    disease_cols = [c for c in df.columns if c not in (id_col, seq_col)]
    Y = df.set_index(id_col)[disease_cols]
    return Y


def build_Y_matrix_from_pairs(
    pairs_csv: str,
    lnc_col: str,
    dis_col: str,
    valid_ids=None,
) -> pd.DataFrame:
    """
    Build a lnc × disease matrix from a long-format pairs file (like website_alldata.csv).

    If valid_ids is given, only keep pairs whose lnc_col matches an ID from valid_ids
    (case-insensitive), so you can restrict to lncRNAs that you have sequences for.
    """
    df = read_any(pairs_csv)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False)]

    df = df[[lnc_col, dis_col]].dropna()

    if valid_ids is not None:
        valid = {str(v).upper() for v in valid_ids}
        df = df[df[lnc_col].astype(str).str.upper().isin(valid)]

    # binary matrix (0/1), one row per lnc, one col per disease
    mat = (
        df.assign(val=1)
          .drop_duplicates(subset=[lnc_col, dis_col])
          .pivot_table(
              index=lnc_col,
              columns=dis_col,
              values="val",
              aggfunc="max",
              fill_value=0,
          )
    )
    return mat


def merge_Y_matrices(Y_seq: pd.DataFrame | None, Y_pairs: pd.DataFrame | None) -> pd.DataFrame:
    """
    Merge two Y matrices by taking the logical OR of their labels.

    Rows = lnc IDs (union of both), columns = diseases (union of both).
    """
    if Y_seq is None and Y_pairs is None:
        raise ValueError("At least one of Y_seq or Y_pairs must be provided.")

    if Y_seq is None:
        return (Y_pairs > 0).astype(int)
    if Y_pairs is None:
        return (Y_seq > 0).astype(int)

    Y_seq = (Y_seq > 0).astype(int)
    Y_pairs = (Y_pairs > 0).astype(int)

    all_index = sorted(set(Y_seq.index) | set(Y_pairs.index))
    all_cols = sorted(set(Y_seq.columns) | set(Y_pairs.columns))

    Y_seq = Y_seq.reindex(index=all_index, columns=all_cols, fill_value=0)
    Y_pairs = Y_pairs.reindex(index=all_index, columns=all_cols, fill_value=0)

    Y = ((Y_seq + Y_pairs) > 0).astype(int)
    return Y


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_sequences_csv", required=True)
    p.add_argument("--raw_pairs_csv", required=False,
                   help="Optional; website_alldata-style file to build/merge Y from.")
    p.add_argument("--out_dir", default="Data/processed")

    p.add_argument("--id_col", default="ID")
    p.add_argument("--seq_col", default="seqs")

    # for pairs / website_alldata
    p.add_argument("--lnc_col", default="ID",
                   help="Column in raw_pairs_csv with the lncRNA identifier (e.g. 'ncRNA Symbol').")
    p.add_argument("--dis_col", default="Disease Name",
                   help="Column in raw_pairs_csv with the disease name.")
    p.add_argument(
        "--merge_pairs",
        action="store_true",
        help=(
            "If set, merge labels from raw_sequences_csv and raw_pairs_csv "
            "instead of only using raw_pairs_csv."
        ),
    )

    a = p.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1)write clean sequences.csv
    seq_out = out / "sequences.csv"
    build_sequences_csv(a.raw_sequences_csv, a.id_col, a.seq_col, str(seq_out))

    # 2) build Y
    Y_seq = None
    Y_pairs = None

    # labels coming from sequences.csv
    Y_seq = build_Y_from_sequences(a.raw_sequences_csv, a.id_col, a.seq_col)

    if a.raw_pairs_csv:
        # build from website_alldata or other pair file, restricted to IDs we have sequences for
        Y_pairs = build_Y_matrix_from_pairs(
            a.raw_pairs_csv,
            a.lnc_col,
            a.dis_col,
            valid_ids=Y_seq.index,  # only keep pairs with known sequences
        )

        if a.merge_pairs:
            Y = merge_Y_matrices(Y_seq, Y_pairs)
        else:
            # original behavior: only use pairs file if provided
            Y = (Y_pairs > 0).astype(int)
    else:
        # no pairs file -> fall back to labels in sequences.csv
        Y = (Y_seq > 0).astype(int)

    Y_out = out / "Y.csv"
    Y.to_csv(Y_out)

    print("[ok] wrote:", seq_out, "and", Y_out)
