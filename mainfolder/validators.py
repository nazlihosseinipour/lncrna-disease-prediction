# validators.py
from typing import Iterable, Optional, Dict
import numpy as np, pandas as pd



"""validators for rna_features """
def require_seqs(seqs: Iterable[str]) -> None:
    """Lightweight: only checks type/None. Do NOT consume generators here."""
    if seqs is None:
        raise ValueError("seqs must not be None.")
    # duck-typing: we just require it's iterable; content/type checks happen after materialization
    try:
        iter(seqs)
    except TypeError:
        raise TypeError("seqs must be an iterable of strings.")

def require_k(k: Optional[int]) -> None:
    if k is None or not isinstance(k, int) or k < 1:
        raise ValueError("k must be an integer >= 1.")

def require_k_gap(k_gap: Optional[int]) -> None:
    if k_gap is None or not isinstance(k_gap, int) or k_gap < 0:
        raise ValueError("k_gap must be an integer >= 0.")

def require_L(L: Optional[int]) -> None:
    if L is None or not isinstance(L, int) or L < 1:
        raise ValueError("L must be an integer >= 1.")

def require_lam(lam: Optional[int]) -> None:
    if lam is None or not isinstance(lam, int) or lam < 1:
        raise ValueError("lam must be an integer >= 1.")

def require_weight(w: float) -> None:
    if not (isinstance(w, (int, float)) and 0 <= w <= 1):
        raise ValueError("w must be a number in [0, 1].")

def require_return_format(fmt: str) -> None:
    if fmt not in ("matrix", "dataframe"):
        raise ValueError("return_format must be 'matrix' or 'dataframe'.")

def require_sample_ids_len(sample_ids: Optional[Iterable[str]], expected_len: int) -> None:
    if sample_ids is None:
        return
    try:
        n_ids = sum(1 for _ in sample_ids)
    except TypeError:
        raise TypeError("sample_ids must be an iterable of strings.")
    if n_ids != expected_len:
        raise ValueError(f"sample_ids length ({n_ids}) must match number of sequences ({expected_len}).")
    

"""validators for disease_features """
def require_edges(edges):
    edges = list(edges)
    if not edges:
        raise ValueError("edges_child_parent cannot be empty.")
    for child, parent in edges:
        if not child or not parent:
            raise ValueError("Edge terms must be non-empty strings.")
    return edges

def require_edge_weight(w):
    if not isinstance(w, (int, float)) or not (0.0 < w <= 1.0):
        raise ValueError("edge_weight must be a number in (0, 1].")
    return float(w)

def require_term(term: str, name="term"):
    if not term or not isinstance(term, str):
        raise ValueError(f"{name} must be a non-empty string.")
    return term


def require_disease_to_terms(disease_to_terms: Dict[str, Iterable[str]]):
    if not isinstance(disease_to_terms, dict) or not disease_to_terms:
        raise ValueError("disease_to_terms must be a non-empty dict.")
    for k, v in disease_to_terms.items():
        if not isinstance(k, str):
            raise TypeError("disease_to_terms keys must be strings.")
        if not isinstance(v, Iterable):
            raise TypeError(f"Values for disease {k} must be iterable of terms.")
    return disease_to_terms

def require_threshold(threshold: float) -> float:
    if not isinstance(threshold, (int, float)):
        raise TypeError("threshold must be a number.")
    return float(threshold)

def require_matrix_like(Y):
    if not isinstance(Y, (pd.DataFrame, np.ndarray, list, tuple)):
        raise TypeError("Y must be a DataFrame, ndarray, or nested list/tuple.")
    return Y
