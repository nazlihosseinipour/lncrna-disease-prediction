from __future__ import annotations
from typing import Dict, Iterable, List, Set, Tuple, Union, Optional
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from feature_extractor import FeatureExtractor

MatrixLike = Union[np.ndarray, pd.DataFrame]

# gotta create an extractor here too 

class DiseaseFeatures(FeatureExtractor):
    """
    Minimal utilities for:
      1) Wang semantic similarity on a disease DAG (MeSH-like)
      2) LFS (lncRNA Functional Similarity) from a lncRNA×disease association matrix

    Inputs you provide:
      - edges: iterable of (child_term, parent_term) pairs defining a DAG
      - disease_to_terms: dict {disease_id: iterable of MeSH terms}
      - Y: lncRNA×disease association matrix (rows=lncRNAs, cols=diseases)
    """

    def __init__(self, edges_child_parent: Iterable[Tuple[str, str]], edge_weight: float = 0.8):
        # Build parent adjacency for Wang propagation
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        for child, parent in edges_child_parent:
            self.parents[child].add(parent)
        self.edge_weight = float(edge_weight)
        self._s_cache: Dict[str, Dict[str, float]] = {}  # term -> {ancestor: S-value}



    #  13) Wang’s method on terms -
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
        """Sim_Wang(t1,t2) = sum_{a∈A∩B}(S1(a)+S2(a)) / [sum_{a∈A}S1(a) + sum_{b∈B}S2(b)]."""
        S1 = self._S_values(t1)
        S2 = self._S_values(t2)
        denom = sum(S1.values()) + sum(S2.values())
        if denom == 0.0:
            return 0.0
        inter = set(S1) & set(S2)
        num = sum(S1[a] + S2[a] for a in inter)
        return num / denom



    #  14a) Disease × Disease similarity via BMA over term sets 
    def disease_similarity_bma(
        self,
        disease_to_terms: Dict[str, Iterable[str]],
        diseases_order: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Best-Match Average between two diseases D1,D2:
          BMA = ( avg_{t∈T1} max_{u∈T2} sim(t,u) + avg_{u∈T2} max_{t∈T1} sim(t,u) ) / 2
        """
        diseases = diseases_order or sorted(disease_to_terms.keys())
        n = len(diseases)
        K = np.zeros((n, n), dtype=float)

        # small cache for term-term sims
        cache: Dict[Tuple[str, str], float] = {}
        def sim_tt(a: str, b: str) -> float:
            key = (a, b) if a <= b else (b, a)
            if key not in cache:
                cache[key] = self.wang_term_similarity(*key)
            return cache[key]

        def bma(T1: List[str], T2: List[str]) -> float:
            if not T1 or not T2: return 0.0
            a = np.mean([max(sim_tt(t, u) for u in T2) for t in T1])
            b = np.mean([max(sim_tt(u, t) for t in T1) for u in T2])
            return float((a + b) / 2.0)

        # build matrix
        termsets = {d: list(dict.fromkeys(disease_to_terms.get(d, ()))) for d in diseases}
        for i in range(n):
            K[i, i] = 1.0
            Ti = termsets[diseases[i]]
            for j in range(i + 1, n):
                Tj = termsets[diseases[j]]
                K[i, j] = K[j, i] = bma(Ti, Tj)

        return pd.DataFrame(K, index=diseases, columns=diseases)

    #  15) LFS (lncRNA Functional Similarity) from Y and disease similarity
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
        # to array + labels
        if isinstance(Y, pd.DataFrame):
            A = Y.to_numpy(dtype=float, copy=False)
            lnc_ids = list(map(str, Y.index))
            dis_ids = list(map(str, Y.columns))
        else:
            A = np.asarray(Y, dtype=float)
            lnc_ids = [f"L{i+1}" for i in range(A.shape[0])]
            dis_ids = [f"D{j+1}" for j in range(A.shape[1])]

        # align disease_sim to Y’s column order
        S_dis = disease_sim.loc[dis_ids, dis_ids].to_numpy(dtype=float)

        # indices of associated diseases per lncRNA
        assoc_idx = [np.where(A[i] > float(threshold))[0] for i in range(A.shape[0])]

        def bma_idx(I: np.ndarray, J: np.ndarray) -> float:
            if I.size == 0 or J.size == 0:
                return 0.0
            # best match each way using the precomputed disease similarity matrix
            a = np.mean(np.max(S_dis[np.ix_(I, J)], axis=1))
            b = np.mean(np.max(S_dis[np.ix_(J, I)], axis=1))
            return float((a + b) / 2.0)

        n = A.shape[0]
        LFS = np.eye(n, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                val = bma_idx(assoc_idx[i], assoc_idx[j])
                LFS[i, j] = LFS[j, i] = val

        return pd.DataFrame(LFS, index=lnc_ids, columns=lnc_ids)
