from collections import Counter
from typing import List, Dict
import itertools

class util: 

    ALPHABET = "ACGU"
    DINUCS = [a+b for a in ALPHABET for b in ALPHABET]

    def _clean(seq: str) -> str:
        """Uppercase RNA sequence and replace T with U."""
        return seq.upper().replace("T", "U")

    def make_columns(k: int, alphabet: str = ALPHABET):
        return ["".join(p) for p in itertools.product(alphabet, repeat=k)]

    def reverse_complement(kmer: str) -> str:
        comp = {"A":"U","U":"A","C":"G","G":"C"}
        return "".join(comp[b] for b in reversed(kmer))

    def make_canonical_columns(k: int):
        kmers = make_columns(k)
        canon = {min(k, reverse_complement(k)) for k in kmers}
        return sorted(canon)

    def _kmer_row(seq: str, cols, normalize=True):
        seq = _clean(seq)
        n = len(seq)
        k = len(cols[0])
        W = n - k + 1
        cnt = Counter(seq[i:i+k] for i in range(W) if "N" not in seq[i:i+k])
        row = [cnt.get(c, 0) for c in cols]
        if normalize and W > 0:
            row = [v/W for v in row]
        return row

    def _canonical_kmer_row(seq: str, cols, normalize=True):
        seq = _clean(seq)
        n = len(seq)
        k = len(cols[0])
        W = n - k + 1
        cnt = Counter(min(seq[i:i+k], reverse_complement(seq[i:i+k]))
                    for i in range(W) if "N" not in seq[i:i+k])
        row = [cnt.get(c, 0) for c in cols]
        if normalize and W > 0:
            row = [v/W for v in row]
        return row

    def _dinuc_properties(seq, props):
        """Return list of property vectors for each dinucleotide in seq."""
        seq = _clean(seq)
        return [props[d] for d in (seq[i:i+2] for i in range(len(seq)-1)) if d in props]
    



    
"before editing version is below as backup  "
#         ALPHABET = ("A", "C", "G", "U")
#     COMP_TRANS = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C"})
#     DINUCS = ["".join(p) for p in itertools.product(ALPHABET, repeat=2)]


#     def _clean(seq: str) -> str:
#         """Uppercase and map T->U so DNA-style inputs still work in RNA mode."""
#         return seq.upper().replace("T", "U")
    
#     @classmethod
#     def revcomp(cls, s: str) -> str:
#         return s.translate(cls.COMP_TRANS)[::-1]
    


#     """ columns generators """

#     @classmethod
#     def make_columns(cls, k: int) -> List[str]:
#         """All RNA k-mers over (A,C,G,U) in lexicographic order."""
#         if k < 1:
#             raise ValueError("k must be >= 1")
#         return ["".join(p) for p in itertools.product(cls.ALPHABET, repeat=k)]

#     @classmethod
#     def make_canonical_columns(cls, k: int) -> List[str]:
#         """Unique reverse-complement canonical labels: min(kmer, revcomp(kmer))."""
#         if k < 1:
#             raise ValueError("k must be >= 1")
#         reps = {
#             min(s, cls.revcomp(s))
#             for s in ("".join(p) for p in itertools.product(cls.ALPHABET, repeat=k))
#         }
#         return sorted(reps)

#     """ row builders """
#     @classmethod
#     def _kmer_row(cls, seq: str, columns: List[str], *, normalize: bool = True) -> List[float]:
#         """One sequence -> one row aligned to `columns` (non-canonical)."""
#         if not columns:
#             raise ValueError("columns must come from make_columns(k)")
#         k = len(columns[0])
#         s = cls._clean(seq)
#         n = len(s)
#         W = max(n - k + 1, 0)
#         if W == 0:
#             return [0.0] * len(columns)
#         cnt = Counter(s[i:i + k] for i in range(W))
#         row = [cnt.get(col, 0) for col in columns]
#         return row if not normalize else [v / W for v in row]

#     @classmethod
#     def _canonical_kmer_row(cls, seq: str, columns: List[str], *, normalize: bool = True) -> List[float]:
#         """One sequence -> one row aligned to canonical `columns`."""
#         if not columns:
#             raise ValueError("columns must come from make_canonical_columns(k)")
#         k = len(columns[0])
#         s = cls._clean(seq)
#         n = len(s)
#         W = max(n - k + 1, 0)
#         if W == 0:
#             return [0.0] * len(columns)
#         cnt = Counter(min(s[i:i + k], cls.revcomp(s[i:i + k])) for i in range(W))
#         row = [cnt.get(col, 0) for col in columns]
#         return row if not normalize else [v / W for v in row]
    

#     """Dinucleotide feature helpers"""""
    
# #??????? idk why i'm getting yellow line under _clean what is wrong with it?????????????? why is it no known ???????????###################################################################################################
#     def _dinuc_properties(seq: str, props: Dict[str, List[float]]):
#         """Turn a sequence into a list of property vectors per dinucleotide."""
#         s = _clean(seq)
#         n = len(s)
#         dinucs = [s[i:i+2] for i in range(n-1)]
#         return [props[d] for d in dinucs if d in props]
    