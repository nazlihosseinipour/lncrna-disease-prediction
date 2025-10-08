from collections import Counter
from typing import Iterable, List, Dict, Optional, Literal, Tuple
import itertools
import util

try:
    import pandas as pd
except ImportError:
    pd = None

# gotta create a contructor for this calss too 

class RnaFeatures:
# fix asap : 
# wait i've never talked about what are and what is the cls actually?? 
# gotta import the functions that i've already imported bcs rn i'm not sure how items like : make_columns gets used while they are in another functions bcs we do sth like cls.make_columns() which i'm not sure how it happens ngl 


#also where is these functions , i'm sure i created them did i delted them by mistake?? :  
# Pseudo dinucleotide composition	3
# Dinucleotide-based auto covariance	4
# Dinucleotide-based cross covariance	5
# Dinucleotide-based auto-cross covariance	6


# make all of these features look better this one looks really bad rn you gotta put and add like exact explanation of what each item does you dont have to explain the how just what it tdoe and like what is @given and then @returns what and what we will use that for? 
    
    # (1) K-mer
    @classmethod
    def kmer_matrix(
        cls,
        seqs: Iterable[str],
        k: int,
        *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[List[float]]]:
        cols = cls.make_columns(k)
        rows = [cls._kmer_row(seq, cols, normalize=normalize) for seq in seqs]
        if return_format == "matrix" or pd is None:
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df  # type: ignore[return-value]

    # (2) Reverse-complement (canonical) K-mer
    @classmethod
    def rc_kmer_matrix(
        cls,
        seqs: Iterable[str],
        k: int,
        *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[List[float]]]:
        cols = cls.make_canonical_columns(k)
        rows = [cls._canonical_kmer_row(seq, cols, normalize=normalize) for seq in seqs]
        if return_format == "matrix" or pd is None:
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df  # type: ignore[return-value]

    #(3)
    #(4) 
    #(5)  
    #(6) 
    # (7) Nucleic acid composition (mono)
    @classmethod
    def mono_composition_matrix(
        cls,
        seqs: Iterable[str],
        *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[List[float]]]:
        cols = list(cls.ALPHABET)
        rows: List[List[float]] = []
        for seq in seqs:
            s = cls._clean(seq)
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
        return cols, df  # type: ignore[return-value]

    # (8) Di-nucleotide composition (k=2)
    @classmethod
    def di_composition_matrix(
        cls,
        seqs: Iterable[str],
        *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        return cls.kmer_matrix(seqs, 2, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    # (9) Tri-nucleotide composition (k=3)
    @classmethod
    def tri_composition_matrix(
        cls,
        seqs: Iterable[str],
        *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        return cls.kmer_matrix(seqs, 3, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

    # (10) z-curve (3 features)
    @classmethod
    def zcurve_matrix(
        cls,
        seqs: Iterable[str],
        *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[List[float]]]:
        cols = ["ZC_x", "ZC_y", "ZC_z"]
        rows: List[List[float]] = []
        for seq in seqs:
            s = cls._clean(seq)
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
        return cols, df  # type: ignore[return-value]

    # (11) monoMonoKGap
    @classmethod
    def monoMonoKGap_matrix(
        cls,
        seqs: Iterable[str],
        k_gap: int,
        *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[List[float]]]:
        labels = [f"{a}_{b}_gap{k_gap}" for a in cls.ALPHABET for b in cls.ALPHABET]
        rows: List[List[float]] = []
        for seq in seqs:
            s = cls._clean(seq)
            n = len(s)
            W = max(n - k_gap - 1, 0)
            cnt = Counter((s[i], s[i + k_gap + 1]) for i in range(W))
            row = [cnt.get((a, b), 0) for a in cls.ALPHABET for b in cls.ALPHABET]
            if normalize and W > 0:
                row = [v / W for v in row]
            rows.append(row)
        if return_format == "matrix" or pd is None:
            return labels, rows
        df = pd.DataFrame(rows, columns=labels)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return labels, df  # type: ignore[return-value]

    # (12) monoDiKGap
    @classmethod
    def monoDiKGap_matrix(
        cls,
        seqs: Iterable[str],
        k_gap: int,
        *,
        normalize: bool = True,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[List[str], List[List[float]]]:
        dinucs = ["".join(p) for p in itertools.product(cls.ALPHABET, repeat=2)]
        labels = [f"{a}_{d}_gap{k_gap}" for a in cls.ALPHABET for d in dinucs]
        rows: List[List[float]] = []
        for seq in seqs:
            s = cls._clean(seq)
            n = len(s)
            W = max(n - k_gap - 2, 0)
            cnt = Counter((s[i], s[i + k_gap + 1: i + k_gap + 3]) for i in range(W))
            row = [cnt.get((a, d), 0) for a in cls.ALPHABET for d in dinucs]
            if normalize and W > 0:
                row = [v / W for v in row]
            rows.append(row)
        if return_format == "matrix" or pd is None:
            return labels, rows
        df = pd.DataFrame(rows, columns=labels)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return labels, df  # type: ignore[return-value]

    # -------------------- dispatcher --------------------

    @classmethod
    def extract_rna_features(
        cls,
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
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        """
        Unified entry point mirroring monoDiKGap
        Implemented here: 1,2,7,8,9,10,11,12.
        """
        if method_id == 1:
            if k is None:
                raise ValueError("k is required for k-mer.")
            return cls.kmer_matrix(seqs, k, normalize=normalize, return_format=return_format, sample_ids=sample_ids)
        if method_id == 2:
            if k is None:
                raise ValueError("k is required for reverse-complement k-mer.")
            return cls.rc_kmer_matrix(seqs, k, normalize=normalize, return_format=return_format, sample_ids=sample_ids)
        if method_id == 7:
            return cls.mono_composition_matrix(seqs, normalize=normalize, return_format=return_format, sample_ids=sample_ids)
        if method_id == 8:
            return cls.di_composition_matrix(seqs, normalize=normalize, return_format=return_format, sample_ids=sample_ids)
        if method_id == 9:
            return cls.tri_composition_matrix(seqs, normalize=normalize, return_format=return_format, sample_ids=sample_ids)
        if method_id == 10:
            return cls.zcurve_matrix(seqs, normalize=True, return_format=return_format, sample_ids=sample_ids)
        if method_id == 11:
            if k_gap is None:
                raise ValueError("k_gap is required for monoMonoKGap.")
            return cls.monoMonoKGap_matrix(seqs, k_gap, normalize=normalize, return_format=return_format, sample_ids=sample_ids)
        if method_id == 12:
            if k_gap is None:
                raise ValueError("k_gap is required for monoDiKGap.")
            return cls.monoDiKGap_matrix(seqs, k_gap, normalize=normalize, return_format=return_format, sample_ids=sample_ids)

        raise ValueError(f"RNA method id {method_id} not implemented in this module.")
