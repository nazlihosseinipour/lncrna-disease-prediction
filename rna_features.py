from collections import Counter
from typing import Iterable, List, Dict, Tuple, Optional, Literal
import itertools

try:
    import pandas as pd
except ImportError:
    pd = None

class RnaFeatures: 
    # -------------------- helpers --------------------

    ALPHABET = ("A", "C", "G", "U")
    COMP_TRANS = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C"})

    def _clean(seq: str) -> str:
        """Uppercase and map T->U so DNA-style inputs still work in RNA mode."""
        return seq.upper().replace("T", "U")

    def revcomp(s: str) -> str:
        return s.translate(COMP_TRANS)[::-1]


    # -------------------- columns generators --------------------

    def make_columns(k: int) -> List[str]:
        """All RNA k-mers over (A,C,G,U) in lexicographic order."""
        if k < 1:
            raise ValueError("k must be >= 1")
        return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]

    def make_canonical_columns(k: int) -> List[str]:
        """Unique reverse-complement canonical labels: min(kmer, revcomp(kmer))."""
        if k < 1:
            raise ValueError("k must be >= 1")
        reps = {min(s, revcomp(s)) for s in ("".join(p) for p in itertools.product(ALPHABET, repeat=k))}
        return sorted(reps)


    # -------------------- row builders --------------------

    def _kmer_row(seq: str, columns: List[str], *, normalize: bool = True) -> List[float]:
        """One sequence -> one row aligned to `columns` (non-canonical)."""
        if not columns:
            raise ValueError("columns must come from make_columns(k)")
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
        if not columns:
            raise ValueError("columns must come from make_canonical_columns(k)")
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


    # -------------------- public API (by README numbering) --------------------

    # (1) K-mer
    def kmer_matrix(
        seqs: Iterable[str], k: int, *, normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        cols = make_columns(k)
        rows = [_kmer_row(seq, cols, normalize=normalize) for seq in seqs]
        if return_format == "matrix" or pd is None:
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df

    # (2) Reverse-complement (canonical) K-mer
    def rc_kmer_matrix(
        seqs: Iterable[str], k: int, *, normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        cols = make_canonical_columns(k)
        rows = [_canonical_kmer_row(seq, cols, normalize=normalize) for seq in seqs]
        if return_format == "matrix" or pd is None:
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df

    # (7) Nucleic acid composition (mono)
    def mono_composition_matrix(
        seqs: Iterable[str], *, normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        cols = list(ALPHABET)
        rows = []
        for seq in seqs:
            s = _clean(seq)
            cnt = Counter(s)
            row = [cnt.get(b, 0) for b in cols]
            if normalize:
                total = len(s) or 1
                row = [v / total for v in row]
            rows.append(row)
        if return_format == "matrix" or pd is None:
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df

    # (8) Di-nucleotide composition (k=2)
    def di_composition_matrix(
        seqs: Iterable[str], *, normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        return kmer_matrix(seqs, 2, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    # (9) Tri-nucleotide composition (k=3)
    def tri_composition_matrix(
        seqs: Iterable[str], *, normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        return kmer_matrix(seqs, 3, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    # (10) z-curve (3 features)
    def zcurve_matrix(
        seqs: Iterable[str], *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        cols = ["ZC_x", "ZC_y", "ZC_z"]
        rows = []
        for seq in seqs:
            s = _clean(seq)
            c = Counter(s)
            A = c.get("A", 0)
            Cn = c.get("C", 0)
            G = c.get("G", 0)
            U = c.get("U", 0)
            x = (A + G) - (Cn + U)
            y = (A + Cn) - (G + U)
            z = (A + U) - (Cn + G)
            if normalize:
                n = len(s) or 1
                rows.append([x / n, y / n, z / n])
            else:
                rows.append([x, y, z])
        if return_format == "matrix" or pd is None:
            return cols, rows
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
            W = max(n - k_gap - 1, 0)
            cnt = Counter((s[i], s[i + k_gap + 1]) for i in range(W))
            row = [cnt.get((a, b), 0) for a in ALPHABET for b in ALPHABET]
            if normalize and W > 0:
                row = [v / W for v in row]
            rows.append(row)
        if return_format == "matrix" or pd is None:
            return labels, rows
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
        if return_format == "matrix" or pd is None:
            return labels, rows
        df = pd.DataFrame(rows, columns=labels)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return labels, df


    # -------------------- dispatcher --------------------

    def extract_rna_features(
        method_id: int,
        seqs: Iterable[str],
        *,
        k: Optional[int] = None,
        normalize: bool = True,
        k_gap: Optional[int] = None,
        props: Optional[Dict[str, List[float]]] = None,  # placeholders for 3-6
        lam: Optional[int] = None,
        w: float = 0.5,
        L: Optional[int] = None,
        return_format: Literal["matrix","dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        """
        Unified entry point mirroring the README numbering (1..12).
        Implemented here: 1,2,7,8,9,10,11,12.
        """
        if method_id == 1:
            if k is None: raise ValueError("k is required for k-mer.")
            return kmer_matrix(seqs, k, normalize=normalize, return_format=return_format, sample_ids=sample_ids)
        if method_id == 2:
            if k is None: raise ValueError("k is required for reverse-complement k-mer.")
            return rc_kmer_matrix(seqs, k, normalize=normalize, return_format=return_format, sample_ids=sample_ids)
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

        raise ValueError(f"RNA method id {method_id} not implemented in this module.")
