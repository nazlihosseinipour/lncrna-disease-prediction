import pandas as pd, numpy as np, sys
from typing import List, Optional, Tuple
from utils import ALPHABET 

#  LOADERS

def load_sequences(path: str) -> List[str]:
    """Load RNA sequences from a TXT file (one sequence per line)."""
    with open(path) as f:
        seqs = [ln.strip() for ln in f if ln.strip()]
    if not seqs:
        raise ValueError("No sequences found in file.")
    return seqs


def load_txt_list(path: Optional[str]) -> Optional[List[str]]:
    """Load simple list of IDs or terms from a TXT file."""
    if not path:
        return None
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_csv_df(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path, index_col=0)


def load_edges_child_parent(path: str) -> List[Tuple[str, str]]:
    """Load ontology edges from CSV (columns: child,parent)."""
    df = pd.read_csv(path)
    cols = [c.lower() for c in df.columns]
    if not {"child", "parent"}.issubset(cols):
        raise ValueError("edges CSV must have columns: child,parent")
    df.columns = cols
    return list(df[["child", "parent"]].itertuples(index=False, name=None))


def load_sequences_csv(path: str) -> Tuple[List[str], List[str]]:
    """
    Load sequences from a CSV file with columns: id,seq.
    Returns (ids, seqs).
    """
    df = pd.read_csv(path)
    if not {"id", "seq"}.issubset(df.columns):
        raise ValueError(f"{path} must have columns: id,seq")
    return df["id"].astype(str).tolist(), df["seq"].astype(str).tolist()


#  PREPROCESSING

def preprocess_sequences(
    ids: Optional[List[str]],
    seqs: List[str],
    *,
    to_upper: bool = True,
    replace_t_with_u: bool = True,
    valid_alphabet: set = set(ALPHABET) | {"N"},
    strict: bool = False,
) -> Tuple[Optional[List[str]], List[str]]:
    """
    Preprocess and validate sequences before feature extraction.

    - Uppercases sequences (default)
    - Converts T → U (default)
    - Removes or errors on invalid characters
    - Returns (ids2, seqs2)
    """
    ids2 = [] if ids is not None else None
    seqs2, dropped = [], []

    for i, s in enumerate(seqs):
        x = s.strip()
        if to_upper:
            x = x.upper()
        if replace_t_with_u:
            x = x.replace("T", "U")

        if set(x) - valid_alphabet:
            dropped.append((ids[i] if ids else i, s))
            if strict:
                raise ValueError(f"Invalid characters in sequence {ids[i] if ids else i}: {s}")
            continue

        if ids2 is not None:
            ids2.append(ids[i])
        seqs2.append(x)

    if dropped:
        print(f"[warn] Dropped {len(dropped)} invalid sequence(s)", file=sys.stderr)

    return ids2, seqs2


# OUTPUT 

def save_output(obj, out: Optional[str]):
    """
    Save or print output automatically based on type.
    - DataFrame → CSV
    - Tuple → NPZ
    - Other → text
    """
    if out is None:
        if isinstance(obj, pd.DataFrame):
            print(obj.to_csv())
        else:
            print(obj)
        return

    if isinstance(obj, pd.DataFrame):
        obj.to_csv(out)
    elif isinstance(obj, tuple):
        if not out.endswith(".npz"):
            out += ".npz"
        np.savez(out, *obj)
    else:
        with open(out, "w") as f:
            f.write(str(obj))

    print(f"[saved] {out}", file=sys.stderr)
