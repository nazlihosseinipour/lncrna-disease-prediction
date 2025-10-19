from typing import Iterable, List, Dict, Optional, Literal, Tuple
from feature_module import FeatureModule
import pandas as pd,  numpy as np , itertools
from collections import Counter
from utils import (
    ALPHABET, DINUCS, _clean, 
    make_columns, make_canonical_columns,
    _kmer_row, _canonical_kmer_row, _dinuc_properties)
from validators import (
    require_seqs, require_k, require_return_format, 
    require_sample_ids_len, require_lam , 
    require_weight , require_L , require_k_gap )

class RnaFeatures(FeatureModule):


    def __init__ (self): 
        super().__init__()

    METHOD_MAP = {
        1: "kmer_matrix", 2: "rc_kmer_matrix",
        3: "psednc_matrix", 4: "di_auto_cov_matrix",
        5: "di_cross_cov_matrix", 6: "di_acc_matrix",
        7: "mono_composition_matrix", 8: "di_composition_matrix",
        9: "tri_composition_matrix", 10: "zcurve_matrix",
        11: "monoMonoKGap_matrix", 12: "monoDiKGap_matrix",
    }

    @classmethod
    def extract(cls, method_id, *args, **kwargs):
        return super().extract(method_id, *args, **kwargs)

    @staticmethod
    def _clean_and_check(x: str) -> str:
        from utils import _clean  # local import avoids cycles
        if not isinstance(x, str):
            raise TypeError("each sequence must be a string.")
        s = _clean(x)
        bad = [ch for ch in s if ch not in "ACGU"]
        if bad:
            raise ValueError(f"sequence contains invalid characters after cleaning: {sorted(set(bad))}")
        return s


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
        
        require_return_format(return_format)
        require_seqs(seqs)
        require_k(k)
        # materialize once (safe for generators), then check content
        seqs = [cls._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        require_sample_ids_len(sample_ids, len(seqs))

        cols = make_columns(k)
        rows = [_kmer_row(seq, cols, normalize=normalize) for seq in seqs]
        if return_format == "matrix":
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df  


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
        
        require_return_format(return_format)
        require_seqs(seqs)
        require_k(k)
        seqs = [cls._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        require_sample_ids_len(sample_ids, len(seqs))
    
        cols = make_canonical_columns(k)
        rows = [_canonical_kmer_row(seq, cols, normalize=normalize) for seq in seqs]
        if return_format == "matrix":
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df  
   
    # (3) Pseudo dinucleotide composition (PseDNC)
    @staticmethod
    def psednc_matrix(
        seqs: Iterable[str],
        *,
        props: Dict[str, List[float]],
        lam: int,
        w: float = 0.5,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        require_return_format(return_format)
        require_seqs(seqs)
        require_lam(lam)
        require_weight(w)
        # props checks (RNA-specific): non-empty, uniform lengths, valid keys
        if not props:
            raise ValueError("props cannot be empty for PseDNC.")
        lens = {len(v) for v in props.values()}
        if len(lens) != 1:
            raise ValueError("all property vectors in props must have the same length.")
        bad = [k for k in props.keys() if len(k) != 2 or any(ch not in "ACGU" for ch in k)]
        if bad:
            raise ValueError(f"props has invalid dinucleotide keys: {bad[:5]}...")
        # seqs
        seqs = [RnaFeatures._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        if lam <= 0: raise ValueError("lam must be >= 1")

        cols = DINUCS + [f"theta_{i+1}" for i in range(lam)]
        rows: List[List[float]] = []

        for seq in seqs:
            s = _clean(seq)
            n = len(s)
            if n < 2:
                rows.append([0.0] * len(cols))
                continue

            # composition (normalized)
            dinucs = [s[i:i+2] for i in range(n - 1)]
            comp = Counter(dinucs)
            comp_vec = [comp.get(d, 0) / (n - 1) for d in DINUCS]

            # property vectors per dinucleotide (once, outside lag loop)
            vals = _dinuc_properties(seq, props)  # list[list[float]] of length n-1
            theta: List[float] = []
            for lag in range(1, lam + 1):
                if len(vals) <= lag:
                    theta.append(0.0)
                else:
                    corr = float(np.mean([
                        float(np.dot(vals[i], vals[i + lag]))
                        for i in range(len(vals) - lag)
                    ]))
                    theta.append(corr)

            # combine with weighting
            denom = 1.0 + w * sum(theta)
            pse = [(c / denom) for c in comp_vec] + [(w * t / denom) for t in theta]
            rows.append(pse)

        if return_format == "matrix":
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df

    # (4) Dinucleotide auto covariance (DAC)
    @staticmethod
    def di_auto_cov_matrix(
        seqs: Iterable[str],
        *,
        props: Dict[str, List[float]],
        L: int,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        require_return_format(return_format)
        require_seqs(seqs)
        require_L(L)
        if not props:
            raise ValueError("props cannot be empty for DAC.")
        lens = {len(v) for v in props.values()}
        if len(lens) != 1:
            raise ValueError("all property vectors in props must have the same length.")
        bad = [k for k in props.keys() if len(k) != 2 or any(ch not in "ACGU" for ch in k)]
        if bad:
            raise ValueError(f"props has invalid dinucleotide keys: {bad[:5]}...")
        seqs = [RnaFeatures._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        
        M = len(next(iter(props.values())))  # number of properties per dinuc
        cols = [f"AUTO_p{m}_lag{l}" for m in range(M) for l in range(1, L + 1)]
        rows: List[List[float]] = []

        for seq in seqs:
            vals = _dinuc_properties(seq, props)  # list[list[float]]
            n = len(vals)
            if n == 0:
                rows.append([0.0] * len(cols))
                continue

            features: List[float] = []
            for m in range(M):
                mean_m = float(np.mean([v[m] for v in vals]))
                for lag in range(1, L + 1):
                    if n <= lag:
                        features.append(0.0)
                    else:
                        ac = float(np.mean([
                            (vals[i][m] - mean_m) * (vals[i + lag][m] - mean_m)
                            for i in range(n - lag)
                        ]))
                        features.append(ac)
            rows.append(features)

        if return_format == "matrix":
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df


    
    # (5) Dinucleotide cross covariance (DCC)
    @staticmethod
    def di_cross_cov_matrix(
        seqs: Iterable[str],
        *,
        props: Dict[str, List[float]],
        L: int,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        require_return_format(return_format)
        require_seqs(seqs)
        require_L(L)
        if not props:
            raise ValueError("props cannot be empty for DCC.")
        lens = {len(v) for v in props.values()}
        if len(lens) != 1:
            raise ValueError("all property vectors in props must have the same length.")
        bad = [k for k in props.keys() if len(k) != 2 or any(ch not in "ACGU" for ch in k)]
        if bad:
            raise ValueError(f"props has invalid dinucleotide keys: {bad[:5]}...")
        seqs = [RnaFeatures._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        if not props:
            raise ValueError("props cannot be empty for DCC.")
        
        M = len(next(iter(props.values())))
        cols = [
            f"CROSS_p{m1}_p{m2}_lag{l}"
            for m1 in range(M) for m2 in range(M) if m1 != m2
            for l in range(1, L + 1)
        ]
        rows: List[List[float]] = []

        for seq in seqs:
            vals = _dinuc_properties(seq, props)
            n = len(vals)
            if n == 0:
                rows.append([0.0] * len(cols))
                continue

            features: List[float] = []
            means = [float(np.mean([v[m] for v in vals])) for m in range(M)]
            for m1 in range(M):
                mean1 = means[m1]
                for m2 in range(M):
                    if m1 == m2:
                        continue
                    mean2 = means[m2]
                    for lag in range(1, L + 1):
                        if n <= lag:
                            features.append(0.0)
                        else:
                            cc = float(np.mean([
                                (vals[i][m1] - mean1) * (vals[i + lag][m2] - mean2)
                                for i in range(n - lag)
                            ]))
                            features.append(cc)
            rows.append(features)

        if return_format == "matrix":
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df

    # (6) Dinucleotide auto-cross covariance (DACC = DAC + DCC)
    @staticmethod
    def di_acc_matrix(
        seqs: Iterable[str],
        *,
        props: Dict[str, List[float]],
        L: int,
        return_format: Literal["matrix", "dataframe"] = "matrix",
        sample_ids: Optional[Iterable[str]] = None,
    ):
        
        require_return_format(return_format)
        require_seqs(seqs)
        require_L(L)
        if not props:
            raise ValueError("props cannot be empty for DACC.")
        lens = {len(v) for v in props.values()}
        if len(lens) != 1:
            raise ValueError("all property vectors in props must have the same length.")
        bad = [k for k in props.keys() if len(k) != 2 or any(ch not in "ACGU" for ch in k)]
        if bad:
            raise ValueError(f"props has invalid dinucleotide keys: {bad[:5]}...")
        seqs = [RnaFeatures._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")        
        

        # call siblings explicitly via the class, and request matrix shape
        cols_auto, X_auto = RnaFeatures.di_auto_cov_matrix(
            seqs, props=props, L=L, return_format="matrix"
        )
        cols_cross, X_cross = RnaFeatures.di_cross_cov_matrix(
            seqs, props=props, L=L, return_format="matrix"
        )

        cols = cols_auto + cols_cross
        rows = [a + b for a, b in zip(X_auto, X_cross)]

        if return_format == "matrix":
            return cols, rows
        df = pd.DataFrame(rows, columns=cols)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return cols, df


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
        
        require_return_format(return_format)
        require_seqs(seqs)
        seqs = [cls._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        require_sample_ids_len(sample_ids, len(seqs))

        cols = list(ALPHABET)
        rows: List[List[float]] = []
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
        require_return_format(return_format)
        require_seqs(seqs)
        seqs = [cls._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        require_sample_ids_len(sample_ids, len(seqs))

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
        require_return_format(return_format)
        require_seqs(seqs)
        seqs = [cls._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        require_sample_ids_len(sample_ids, len(seqs))    

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
        
        require_return_format(return_format)
        require_seqs(seqs)
        seqs = [cls._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        require_sample_ids_len(sample_ids, len(seqs))

        cols = ["ZC_x", "ZC_y", "ZC_z"]
        rows: List[List[float]] = []
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
        if return_format == "matrix":
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
        
        require_return_format(return_format)
        require_seqs(seqs)
        require_k_gap(k_gap)
        seqs = [cls._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        require_sample_ids_len(sample_ids, len(seqs))

        labels = [f"{a}_{b}_gap{k_gap}" for a in ALPHABET for b in ALPHABET]
        rows: List[List[float]] = []
        for seq in seqs:
            s = _clean(seq)
            n = len(s)
            W = max(n - k_gap - 1, 0)
            cnt = Counter((s[i], s[i + k_gap + 1]) for i in range(W))
            row = [cnt.get((a, b), 0) for a in ALPHABET for b in ALPHABET]
            if normalize and W > 0:
                row = [v / W for v in row]
            rows.append(row)
        if return_format == "matrix":
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
        
        require_return_format(return_format)
        require_seqs(seqs)
        require_k_gap(k_gap)
        seqs = [cls._clean_and_check(x) for x in seqs]
        if not seqs:
            raise ValueError("seqs is empty.")
        require_sample_ids_len(sample_ids, len(seqs))

        dinucs = ["".join(p) for p in itertools.product(ALPHABET, repeat=2)]
        labels = [f"{a}_{d}_gap{k_gap}" for a in ALPHABET for d in dinucs]
        rows: List[List[float]] = []
        for seq in seqs:
            s = _clean(seq)
            n = len(s)
            W = max(n - k_gap - 2, 0)
            cnt = Counter((s[i], s[i + k_gap + 1: i + k_gap + 3]) for i in range(W))
            row = [cnt.get((a, d), 0) for a in ALPHABET for d in dinucs]
            if normalize and W > 0:
                row = [v / W for v in row]
            rows.append(row)
        if return_format == "matrix":
            return labels, rows
        df = pd.DataFrame(rows, columns=labels)
        if sample_ids is not None:
            df.insert(0, "sample_id", list(sample_ids))
        return labels, df  # type: ignore[return-value]
