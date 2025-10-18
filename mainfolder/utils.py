from typing import Iterable, Dict, List, Type
from collections import Counter
import itertools

ALPHABET = ("A", "C", "G", "U")
COMP_TRANS = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C"})
DINUCS = ["".join(p) for p in itertools.product(ALPHABET, repeat=2)]

def _clean(seq: str) -> str:
    """Uppercase and map T->U so DNA-style inputs still work in RNA mode."""
    return seq.upper().replace("T", "U")

""" columns generators """
def make_columns(k: int) -> List[str]:
        """All RNA k-mers over (A,C,G,U) in lexicographic order."""
        if k < 1:
            raise ValueError("k must be >= 1")
        return ["".join(p) for p in itertools.product(ALPHABET, repeat=k)]


def make_canonical_columns(k: int) -> List[str]:
    """Unique reverse-complement canonical labels: min(kmer, revcomp(kmer))."""
    if k < 1:
            raise ValueError("k must be >= 1")
    reps = {
        min(s, revcomp(s))
        for s in ("".join(p) for p in itertools.product(ALPHABET, repeat=k))
    }
    return sorted(reps)


""" row builders """
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
    cnt = Counter(s[i:i + k] for i in range(W))
    row = [float(cnt.get(col, 0)) for col in columns]
    return row if not normalize else [v / W for v in row]


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
    cnt = Counter(min(s[i:i + k], revcomp(s[i:i + k])) for i in range(W))
    row = [cnt.get(col, 0) for col in columns]
    return row if not normalize else [v / W for v in row]
    

def _dinuc_properties(seq: str, props: Dict[str, List[float]]) -> List[List[float]]:
    """
    Turn a sequence into a list of property vectors per dinucleotide.

    @param seq: RNA sequence string
    @param props: dictionary mapping dinucleotide -> property vector
    @return: list of property vectors, one per dinucleotide in the sequence
    """
    s = _clean(seq)  # assumes _clean is defined globally in util.py
    n = len(s)
    if n < 2:
        return []

    dinucs = [s[i:i+2] for i in range(n-1)]
    return [props[d] for d in dinucs if d in props]


def describe_methods(cls: Type) -> str:
    method_map: Dict[int, str] = getattr(cls, "METHOD_MAP", {})
    if not method_map:
        return f"{cls.__name__}: no methods registered."
    pairs = [f"{mid} -> {name}" for mid, name in sorted(method_map.items())]
    return f"{cls.__name__} methods: " + ", ".join(pairs)

def ensure_methods(cls: Type, required: Iterable[int]) -> None:
    method_map: Dict[int, str] = getattr(cls, "METHOD_MAP", {})
    missing = [mid for mid in required if mid not in method_map]
    if missing:
        have: List[int] = sorted(method_map.keys())
        raise AssertionError(
            f"{cls.__name__} is missing method ids: {missing}. "
            f"Currently has: {have}"
        )
