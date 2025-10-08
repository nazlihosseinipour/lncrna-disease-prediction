#i wanna have here sth like util sort function sort of vibe which i can put the helper functions 

class util: 

    ALPHABET = ("A", "C", "G", "U")
    COMP_TRANS = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C"})

    @staticmethod
    def _clean(seq: str) -> str:
        """Uppercase and map T->U so DNA-style inputs still work in RNA mode."""
        return seq.upper().replace("T", "U")

    @classmethod
    def revcomp(cls, s: str) -> str:
        return s.translate(cls.COMP_TRANS)[::-1]
    


    """ columns generators """

    @classmethod
    def make_columns(cls, k: int) -> List[str]:
        """All RNA k-mers over (A,C,G,U) in lexicographic order."""
        if k < 1:
            raise ValueError("k must be >= 1")
        return ["".join(p) for p in itertools.product(cls.ALPHABET, repeat=k)]

    @classmethod
    def make_canonical_columns(cls, k: int) -> List[str]:
        """Unique reverse-complement canonical labels: min(kmer, revcomp(kmer))."""
        if k < 1:
            raise ValueError("k must be >= 1")
        reps = {
            min(s, cls.revcomp(s))
            for s in ("".join(p) for p in itertools.product(cls.ALPHABET, repeat=k))
        }
        return sorted(reps)

    """ row builders """
    @classmethod
    def _kmer_row(cls, seq: str, columns: List[str], *, normalize: bool = True) -> List[float]:
        """One sequence -> one row aligned to `columns` (non-canonical)."""
        if not columns:
            raise ValueError("columns must come from make_columns(k)")
        k = len(columns[0])
        s = cls._clean(seq)
        n = len(s)
        W = max(n - k + 1, 0)
        if W == 0:
            return [0.0] * len(columns)
        cnt = Counter(s[i:i + k] for i in range(W))
        row = [cnt.get(col, 0) for col in columns]
        return row if not normalize else [v / W for v in row]

    @classmethod
    def _canonical_kmer_row(cls, seq: str, columns: List[str], *, normalize: bool = True) -> List[float]:
        """One sequence -> one row aligned to canonical `columns`."""
        if not columns:
            raise ValueError("columns must come from make_canonical_columns(k)")
        k = len(columns[0])
        s = cls._clean(seq)
        n = len(s)
        W = max(n - k + 1, 0)
        if W == 0:
            return [0.0] * len(columns)
        cnt = Counter(min(s[i:i + k], cls.revcomp(s[i:i + k])) for i in range(W))
        row = [cnt.get(col, 0) for col in columns]
        return row if not normalize else [v / W for v in row]