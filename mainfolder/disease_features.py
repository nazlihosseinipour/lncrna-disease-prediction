from __future__ import annotations
from typing import Dict, Iterable, List, Set, Tuple, Union, Optional
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from feature_module import FeatureModule
from validators import require_edges, require_edge_weight, require_term , require_edges, require_edge_weight, require_disease_to_terms, require_threshold, require_matrix_like

MatrixLike = Union[np.ndarray, pd.DataFrame]


class DiseaseFeatures(FeatureModule):
    """
    Minimal utilities for:
      13) Wang semantic similarity between two terms (scalar)
      14) Disease × Disease similarity (BMA over term sets) -> DataFrame
      15) LFS (lncRNA Functional Similarity) from Y and disease similarity -> DataFrame

    Inputs you provide to __init__:
      - edges_child_parent: iterable of (child_term, parent_term) pairs defining a DAG
      - edge_weight: propagation decay along edges (0 < w <= 1)
    """

    METHOD_MAP = {
        13: "wang_term_similarity",
        14: "disease_similarity_bma",
        15: "lfs_from_Y",
    }

    @classmethod
    def extract(cls, method_id, *args, **kwargs):
        return super().extract(method_id, *args, **kwargs)


    def __init__(self, edges_child_parent: Iterable[Tuple[str, str]], edge_weight: float = 0.8):
        super().__init__()
        # validate input using central validators
        self.edge_weight = require_edge_weight(edge_weight)
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        for child, parent in require_edges(edges_child_parent):
            self.parents[child].add(parent)
        # cache for S-values
        self._s_cache: Dict[str, Dict[str, float]] = {}

    #13.Wang semantics 

    def _S_values(self, term: str) -> Dict[str, float]:
        """S(term)=1; propagate up parents multiplying by edge_weight; keep max path contribution."""
        if term in self._s_cache:
            return self._s_cache[term]
        S = {term: 1.0}
        q = deque([(term, 1.0)])
        w = self.edge_weight
        while q:
            node, contrib = q.popleft()
            for p in self.parents.get(node, ()):
                cand = contrib * w
                if cand > S.get(p, 0.0):
                    S[p] = cand
                    q.append((p, cand))
        self._s_cache[term] = S
        return S

    def wang_term_similarity(self, t1: str, t2: str) -> float:
        """
        Sim_Wang(t1,t2) = sum_{a∈A∩B}(S1(a)+S2(a)) / [sum_{a∈A}S1(a) + sum_{b∈B}S2(b)].
        Returns a scalar in [0,1].
        """
        t1 = require_term(t1, "t1")
        t2 = require_term(t2, "t2")

        if not t1 or not t2:
            raise ValueError("t1 and t2 must be non-empty strings.")
        S1 = self._S_values(t1)
        S2 = self._S_values(t2)
        denom = sum(S1.values()) + sum(S2.values())
        if denom == 0.0:
            return 0.0
        inter = set(S1) & set(S2)
        num = sum(S1[a] + S2[a] for a in inter)
        return float(num / denom)

    # 14.Disease × Disease (BMA)

    def disease_similarity_bma(
        self,
        disease_to_terms: Dict[str, Iterable[str]],
        diseases_order: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Best-Match Average between two diseases D1,D2:
          BMA = ( avg_{t∈T1} max_{u∈T2} sim(t,u) + avg_{u∈T2} max_{t∈T1} sim(t,u) ) / 2

        Returns a symmetric DataFrame (disease × disease) with ones on the diagonal.
        """
        disease_to_terms = require_disease_to_terms(disease_to_terms)

        # dedupe term lists, preserve order
        termsets = {d: list(dict.fromkeys(T or [])) for d, T in disease_to_terms.items()}
        diseases = diseases_order or sorted(termsets.keys())

        # assert diseases_order doesn't include unknown keys
        unknown = [d for d in diseases if d not in termsets]
        if unknown:
            raise ValueError(f"diseases_order includes unknown disease ids: {unknown[:8]}...")

        n = len(diseases)
        if n == 0:
            raise ValueError("No diseases provided after filtering.")

        K = np.zeros((n, n), dtype=float)

        # small cache for term-term sims
        cache: Dict[Tuple[str, str], float] = {}

        def sim_tt(a: str, b: str) -> float:
            key = (a, b) if a <= b else (b, a)
            if key not in cache:
                cache[key] = self.wang_term_similarity(*key)
            return cache[key]

        def bma(T1: List[str], T2: List[str]) -> float:
            if not T1 or not T2:
                return 0.0
            a = float(np.mean([max(sim_tt(t, u) for u in T2) for t in T1]))
            b = float(np.mean([max(sim_tt(u, t) for t in T1) for u in T2]))
            return (a + b) / 2.0

        for i in range(n):
            K[i, i] = 1.0
            Ti = termsets[diseases[i]]
            for j in range(i + 1, n):
                Tj = termsets[diseases[j]]
                K[i, j] = K[j, i] = bma(Ti, Tj)

        return pd.DataFrame(K, index=diseases, columns=diseases)

    #15.LFS from Y + disease similarity 

    def lfs_from_Y(
        self,
        Y: MatrixLike,
        disease_sim: pd.DataFrame,
        *,
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Build LFS (lncRNA × lncRNA) by BMA over associated disease sets, using disease_sim.

        Steps:
          - For each lncRNA i, collect D_i = { d | Y[i,d] > threshold }.
          - LFS(i,j) = BMA over D_i and D_j using disease_sim[d1,d2].
        """
        Y = require_matrix_like(Y)
        threshold = require_threshold(threshold)
        
        # to array + labels
        if isinstance(Y, pd.DataFrame):
            A = Y.to_numpy(dtype=float, copy=False)
            lnc_ids = list(map(str, Y.index))
            dis_ids = list(map(str, Y.columns))
        else:
            A = np.asarray(Y, dtype=float)
            if A.ndim != 2:
                raise ValueError("Y must be a 2D matrix or a DataFrame.")
            lnc_ids = [f"L{i+1}" for i in range(A.shape[0])]
            dis_ids = [f"D{j+1}" for j in range(A.shape[1])]

        if not isinstance(disease_sim, pd.DataFrame):
            raise TypeError("disease_sim must be a pandas DataFrame.")

        # verify disease_sim covers all diseases in Y (order is aligned below)
        missing = [d for d in dis_ids if d not in disease_sim.index or d not in disease_sim.columns]
        if missing:
            raise ValueError(
                "disease_sim must contain all diseases from Y.columns as both index and columns. "
                f"Missing: {missing[:10]}..."
            )

        # align disease_sim to Y’s column order
        S_dis = disease_sim.loc[dis_ids, dis_ids].to_numpy(dtype=float)

        # indicess of associated diseases per lncRNA
        assoc_idx = [np.where(A[i] > float(threshold))[0] for i in range(A.shape[0])]

        def bma_idx(I: np.ndarray, J: np.ndarray) -> float:
            if I.size == 0 or J.size == 0:
                return 0.0
            a = float(np.mean(np.max(S_dis[np.ix_(I, J)], axis=1)))
            b = float(np.mean(np.max(S_dis[np.ix_(J, I)], axis=1)))
            return (a + b) / 2.0

        n = A.shape[0]
        LFS = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                val = bma_idx(assoc_idx[i], assoc_idx[j])
                LFS[i, j] = LFS[j, i] = val

        return pd.DataFrame(LFS, index=lnc_ids, columns=lnc_ids)
