import numpy as np 
import pandas as pd
import Levenshtein
# kmer_api.py
from collections import Counter
from typing import Iterable, List, Sequence, Tuple, Literal, Optional
import itertools

try:
    import pandas as pd
except ImportError:
    pd = None  



class RnaFeatures: 


   #Y=UΣV^T
        #Where:
        #Y: your original matrix (e.g. lncRNA–disease matrix)
        #U: matrix of left singular vectors (represents lncRNA features) 
        #Σ: diagonal matrix with singular values (importance of each feature)
        #V^T: transpose of right singular vectors (represents disease features)

    @staticmethod
    def extract_svd_features(matrix, k) : 

        # Perform SVD decomposition
        #SVD tells us :"Hey, I found these patterns that explain the most important ways lncRNAs and diseases are related."

        matrix = np.array(matrix, dtype=float)
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)

        #i needed to do this to be able to do the multiplication 
        if k is not None: 
         u_k = u[:, :k]       # keep first k columns of U
         s_k = np.diag(s[:k]) # keep first k singular values and make them diagonal
         vt_k = vt[:k, :]     # keep first k rows of VT

        # Convert Σ Create diagonal matrix from singular values
        s = np.diag(np.sqrt(s_k))  # use sqrt(s) to match paper logic (bcs usually it's used for both in the original formula so now it's one item we also do want half of it)


        # Feature matrices
        lncRNA_features = u_k @ s       # left side = lncRNA 
        disease_features = (s @ vt_k).T     # right side = disease

        return lncRNA_features, disease_features
  


    @staticmethod
    def edit_distance (seqA , seqB ): 
        # matrix = np.zeros(size=(size.length(), seqB.legnth()))
        # #create lables for rows (incRNAs) and columns (diseases)    #A.shape→(rows, columns)
        # seqALables = [f"L{i+1}" for i in range(matrix.shape[0])]      #A.shape[0] = number of rows
        # seqBLables= [f"D{j+1}" for j in range(matrix.shape[1])]          #A.shape[1] = number of cloumn
        # matrixReadable = pd.DataFrame(matrix, index=seqALables, columns=seqBLables)
        return Levenshtein.distance(seqA, seqB)
    
    




# rna_features.py
from __future__ import annotations
from collections import Counter
from typing import Iterable, List, Dict, Tuple, Optional, Literal
import itertools
import math

try:
    import pandas as pd
except ImportError:
    pd = None


# ======== common helpers ========

ALPHABET = ("A", "C", "G", "U")
COMP_TRANS = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C"})

def _clean(seq: str) -> str:
    """Uppercase and T->U so DNA-ish inputs are tolerated for RNA mode."""
    return seq.upper().replace("T", "U")

def revcomp(s: str) -> str:
    return s.translate(COMP_TRANS)[::-1]

def make_columns(k: int) -> List[str]:
    """All RNA k-mers over (A,C,G,U) in lexicographic order."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]

def make_canonical_columns(k: int) -> List[str]:
    """Unique canonical labels (min(kmer, revcomp(kmer))) in lexicographic order."""
    if k < 1:
        raise ValueError("k must be >= 1")
    reps = {min(s, revcomp(s)) for s in ("".join(p) for p in itertools.product(ALPHABET, repeat=k))}
    return sorted(reps)




def _kmer_row(seq: str, columns: List[str], *, normalize: bool = True) -> List[float]:
    """One sequence -> one row aligned to `columns` (non-canonical)."""
    k = len(columns[0])
    s = _clean(seq)
    n = len(s)
    W = max(n - k + 1, 0)
    if W == 0:
        return [0.0] * len(columns)
    cnt = Counter(s[i:i+k] for i in range(W))
    row = [cnt.get(col, 0) for col in columns]
    if not normalize:
        return row
    return [v / W for v in row]

def _canonical_kmer_row(seq: str, columns: List[str], *, normalize: bool = True) -> List[float]:
    """One sequence -> one row aligned to canonical `columns`."""
    k = len(columns[0])
    s = _clean(seq)
    n = len(s)
    W = max(n - k + 1, 0)
    if W == 0:
        return [0.0] * len(columns)
    cnt = Counter(min(s[i:i+k], revcomp(s[i:i+k])) for i in range(W))
    row = [cnt.get(col, 0) for col in columns]
    if not normalize:
        return row
    return [v / W for v in row]


# ======== public feature functions (by README numbering) ========

# (1) Kmer
# Take an RNA string (letters A/C/G/U), cut it into overlapping chunks of length k, count how often each possible chunk appears, and return those numbers in a fixed column order.
#(here it will use function make_columns() and kmer_row() and it will end up putting them together to create single row of with the frequency of the existing options based on the all the options and it will give that as an return 
def kmer_matrix(
    seqs: Iterable[str], k: int, *, normalize: bool = True,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    cols = make_columns(k)
    rows = [_kmer_row(seq, cols, normalize=normalize) for seq in seqs]
    if return_format == "matrix":
        return cols, rows
    if pd is None:
        raise RuntimeError("pandas not installed; set return_format='matrix' or install pandas.")
    df = pd.DataFrame(rows, columns=cols)
    if sample_ids is not None:
        df.insert(0, "sample_id", list(sample_ids))
    return cols, df



# (2) Reverse Compliment (Canonical) Kmer
def rc_kmer_matrix(
    seqs: Iterable[str], k: int, *, normalize: bool = True,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    cols = make_canonical_columns(k)
    rows = [_canonical_kmer_row(seq, cols, normalize=normalize) for seq in seqs]
    if return_format == "matrix":
        return cols, rows
    if pd is None:
        raise RuntimeError("pandas not installed; set return_format='matrix' or install pandas.")
    df = pd.DataFrame(rows, columns=cols)
    if sample_ids is not None:
        df.insert(0, "sample_id", list(sample_ids))
    return cols, df





# (3) Pseudo dinucleotide composition (PseDNC)  —— needs properties
def psednc_matrix(
    seqs: Iterable[str],
    *,
    props: Dict[str, List[float]],
    lam: int,
    w: float = 0.5,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    """
    Generic PseDNC placeholder.
    - props: dict mapping 16 dinucleotides (AA..UU) -> list of M properties (normalized)
    - lam (λ): number of correlation factors (lags)
    - w: weight between composition and correlation
    This is method-specific and requires a curated property table.
    """
    raise NotImplementedError("PseDNC requires a dinucleotide property table (props), λ, and w. Provide them, then implement per your chosen formula.")




# (4) Dinucleotide-based auto covariance —— needs properties
def di_auto_cov_matrix(
    seqs: Iterable[str],
    *,
    props: Dict[str, List[float]],
    L: int,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    """
    Placeholder: requires per-dinucleotide properties and max lag L.
    """
    raise NotImplementedError("Auto covariance requires dinucleotide property vectors and a lag L.")





# (5) Dinucleotide-based cross covariance —— needs properties
def di_cross_cov_matrix(
    seqs: Iterable[str],
    *,
    props: Dict[str, List[float]],
    L: int,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    raise NotImplementedError("Cross covariance requires multiple properties per dinucleotide and lag L.")




# (6) Dinucleotide-based auto-cross covariance —— needs properties
def di_acc_matrix(
    seqs: Iterable[str],
    *,
    props: Dict[str, List[float]],
    L: int,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    raise NotImplementedError("Auto-cross covariance requires property table and lag L.")





# (7) Nucleic acid composition (mono composition)
def mono_composition_matrix(
    seqs: Iterable[str], *, normalize: bool = True,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    cols = list(ALPHABET)  # ["A","C","G","U"]
    rows = []
    for seq in seqs:
        s = _clean(seq)
        cnt = Counter(s)
        row = [cnt.get(b, 0) for b in cols]
        if normalize:
            total = len(s) or 1
            row = [v / total for v in row]
        rows.append(row)
    if return_format == "matrix":
        return cols, rows
    if pd is None:
        raise RuntimeError("pandas not installed.")
    df = pd.DataFrame(rows, columns=cols)
    if sample_ids is not None:
        df.insert(0, "sample_id", list(sample_ids))
    return cols, df





# (8) Di-nucleotide composition (k=2 composition)
def di_composition_matrix(
    seqs: Iterable[str], *, normalize: bool = True,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    return kmer_matrix(seqs, 2, normalize=normalize, return_format=return_format, sample_ids=sample_ids)





# (9) Tri-nucleotide composition (k=3 composition)
def tri_composition_matrix(
    seqs: Iterable[str], *, normalize: bool = True,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    return kmer_matrix(seqs, 3, normalize=normalize, return_format=return_format, sample_ids=sample_ids)





# (10) z-curve (global, 3 features)
def zcurve_matrix(
    seqs: Iterable[str], *,
    normalize: bool = True,  # normalize by length (recommended)
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    cols = ["ZC_x", "ZC_y", "ZC_z"]
    rows = []
    for seq in seqs:
        s = _clean(seq)
        c = Counter(s)
        A, Cn, G, U = c.get("A", 0), c.get("C", 0), c.get("G", 0), c.get("U", 0)
        x = (A + G) - (Cn + U)
        y = (A + Cn) - (G + U)
        z = (A + U) - (Cn + G)
        if normalize:
            n = len(s) or 1
            rows.append([x / n, y / n, z / n])
        else:
            rows.append([x, y, z])
    if return_format == "matrix":
        return cols, rows
    if pd is None:
        raise RuntimeError("pandas not installed.")
    df = pd.DataFrame(rows, columns=cols)
    if sample_ids is not None:
        df.insert(0, "sample_id", list(sample_ids))
    return cols, df







# (11) monoMonoKGap
def monoMonoKGap_matrix(
    seqs: Iterable[str], k_gap: int, *, normalize: bool = True,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    labels = [f"{a}_{b}_gap{k_gap}" for a in ALPHABET for b in ALPHABET]
    rows = []
    for seq in seqs:
        s = _clean(seq)
        n = len(s)
        W = max(n - k_gap - 1, 0)  # number of valid pairs
        cnt = Counter((s[i], s[i + k_gap + 1]) for i in range(W))
        row = [cnt.get((a, b), 0) for a in ALPHABET for b in ALPHABET]
        if normalize and W > 0:
            row = [v / W for v in row]
        rows.append(row)
    if return_format == "matrix":
        return labels, rows
    if pd is None:
        raise RuntimeError("pandas not installed.")
    df = pd.DataFrame(rows, columns=labels)
    if sample_ids is not None:
        df.insert(0, "sample_id", list(sample_ids))
    return labels, df





# (12) monoDiKGap
def monoDiKGap_matrix(
    seqs: Iterable[str], k_gap: int, *, normalize: bool = True,
    return_format: Literal["matrix", "dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    dinucs = ["".join(p) for p in itertools.product(ALPHABET, repeat=2)]
    labels = [f"{a}_{d}_gap{k_gap}" for a in ALPHABET for d in dinucs]
    rows = []
    for seq in seqs:
        s = _clean(seq)
        n = len(s)
        W = max(n - k_gap - 2, 0)
        cnt = Counter((s[i], s[i + k_gap + 1 : i + k_gap + 3]) for i in range(W))
        row = [cnt.get((a, d), 0) for a in ALPHABET for d in dinucs]
        if normalize and W > 0:
            row = [v / W for v in row]
        rows.append(row)
    if return_format == "matrix":
        return labels, rows
    if pd is None:
        raise RuntimeError("pandas not installed.")
    df = pd.DataFrame(rows, columns=labels)
    if sample_ids is not None:
        df.insert(0, "sample_id", list(sample_ids))
    return labels, df





# ======== dispatcher (matches README numbers 1..12) ========

def extract_rna_features(
    method_id: int,
    seqs: Iterable[str],
    *,
    k: Optional[int] = None,          # for (1),(2) k-mer size
    normalize: bool = True,
    k_gap: Optional[int] = None,      # for (11),(12)
    props: Optional[Dict[str, List[float]]] = None,  # for (3)-(6)
    lam: Optional[int] = None,        # for (3)
    w: float = 0.5,                   # for (3)
    L: Optional[int] = None,          # for (4)-(6)
    return_format: Literal["matrix","dataframe"] = "matrix",
    sample_ids: Optional[Iterable[str]] = None,
):
    """
    Unified entry point mirroring the README numbering.
    Returns (columns, matrix/df).
    """
    if method_id == 1:
        if k is None: raise ValueError("k is required for k-mer.")
        return kmer_matrix(seqs, k, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    if method_id == 2:
        if k is None: raise ValueError("k is required for reverse-complement k-mer.")
        return rc_kmer_matrix(seqs, k, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    if method_id == 3:
        if props is None or lam is None:
            raise ValueError("PseDNC needs props and lam (λ).")
        return psednc_matrix(seqs, props=props, lam=lam, w=w, return_format=return_format, sample_ids=sample_ids)

    if method_id == 4:
        if props is None or L is None:
            raise ValueError("Auto covariance needs props and L.")
        return di_auto_cov_matrix(seqs, props=props, L=L, return_format=return_format, sample_ids=sample_ids)

    if method_id == 5:
        if props is None or L is None:
            raise ValueError("Cross covariance needs props and L.")
        return di_cross_cov_matrix(seqs, props=props, L=L, return_format=return_format, sample_ids=sample_ids)

    if method_id == 6:
        if props is None or L is None:
            raise ValueError("Auto-cross covariance needs props and L.")
        return di_acc_matrix(seqs, props=props, L=L, return_format=return_format, sample_ids=sample_ids)

    if method_id == 7:
        return mono_composition_matrix(seqs, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    if method_id == 8:
        return di_composition_matrix(seqs, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    if method_id == 9:
        return tri_composition_matrix(seqs, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    if method_id == 10:
        return zcurve_matrix(seqs, normalize=True, return_format=return_format, sample_ids=sample_ids)

    if method_id == 11:
        if k_gap is None: raise ValueError("k_gap is required for monoMonoKGap.")
        return monoMonoKGap_matrix(seqs, k_gap, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    if method_id == 12:
        if k_gap is None: raise ValueError("k_gap is required for monoDiKGap.")
        return monoDiKGap_matrix(seqs, k_gap, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    raise ValueError(f"Unknown RNA method id: {method_id}")
