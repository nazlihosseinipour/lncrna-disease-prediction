from feature_module import FeatureModule
import numpy as np
import pandas as pd
from typing import Tuple, Union


# here i gotta rewrite the gip in a way that acts like oop so that i can kinda reuse it or myabe create and entire claas for it and here call the class but for now this is the reminder , what i have is good but make it public API worthy 

MatrixLike = Union[np.ndarray, pd.DataFrame]

class CrossFeatures(FeatureModule):
    """
    Utilities for:
      - GIP kernels for lncRNAs (rows) and diseases (columns)
      - SVD-based feature split (left/right)
    """

    def __init__(self):
        super().__init__()


    METHOD_MAP = {
        16: "GIP kernels for lncRNAs (rows) and diseases (columns) " , 
        17: "SVD-based ",
    }

    @classmethod
    def extract(cls, method_id, *args, **kwargs):
        return super().extract(method_id, *args, **kwargs)


    @staticmethod
    def _as_array_and_labels(M: MatrixLike):
        """
        Accepts a NumPy array or a pandas DataFrame.
        Returns (A, row_labels, col_labels) where A is a 2D float64 ndarray.
        If labels are not available, generate 'L1..Ln' (rows) and 'D1..Dm' (cols).
        """
        if isinstance(M, pd.DataFrame):
            A = M.to_numpy(dtype=float, copy=False)
            row_labels = [str(i) for i in M.index] if M.index is not None else [f"L{i+1}" for i in range(A.shape[0])]
            col_labels = [str(c) for c in M.columns] if M.columns is not None else [f"D{j+1}" for j in range(A.shape[1])]
        else:
            A = np.asarray(M, dtype=float)
            row_labels = [f"L{i+1}" for i in range(A.shape[0])]
            col_labels = [f"D{j+1}" for j in range(A.shape[1])]
        if A.ndim != 2:
            raise ValueError("Input matrix must be 2D.")
        return A, row_labels, col_labels

    @staticmethod
    def _safe_gamma(norms_squared: np.ndarray) -> float:
        """
        gamma = 1 / mean(||x||^2), with guards for zeros/NaNs.
        """
        mean_val = float(np.mean(norms_squared))
        if not np.isfinite(mean_val) or mean_val <= 0.0:
            # Fallback if rows/cols are all zeros or degenerate
            return 1.0
        return 1.0 / mean_val

   
    def calculate_gip_lncRNA_and_dis(self, matrix: MatrixLike) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (GIP_lncRNA, GIP_disease):
          - GIP_lncRNA: kernel over rows (lncRNAs)
          - GIP_disease: kernel over columns (diseases)
        """
        A, row_labels, col_labels = self._as_array_and_labels(matrix)

        # --- lncRNAs (rows) ---
        profiles_r = A                      # shape (n, m)
        norms_sq_r = np.sum(profiles_r * profiles_r, axis=1)      # (n,)
        G_r = profiles_r @ profiles_r.T                            # (n, n)
        d2_r = norms_sq_r[:, None] + norms_sq_r[None, :] - 2.0 * G_r
        d2_r = np.maximum(d2_r, 0.0)
        gamma_r = self._safe_gamma(norms_sq_r)
        K_r = np.exp(-gamma_r * d2_r)
        df_gip_lnc = pd.DataFrame(K_r, index=row_labels, columns=row_labels)

        # --- diseases (cols) ---
        profiles_c = A.T                    # shape (m, n)
        norms_sq_c = np.sum(profiles_c * profiles_c, axis=1)      # (m,)
        G_c = profiles_c @ profiles_c.T                            # (m, m)
        d2_c = norms_sq_c[:, None] + norms_sq_c[None, :] - 2.0 * G_c
        d2_c = np.maximum(d2_c, 0.0)
        gamma_c = self._safe_gamma(norms_sq_c)
        K_c = np.exp(-gamma_c * d2_c)
        df_gip_dis = pd.DataFrame(K_c, index=col_labels, columns=col_labels)

        return df_gip_lnc, df_gip_dis

    # Y = U Σ V^T
    # Where:
    # Y: your original matrix (e.g. lncRNA–disease matrix)
    # U: matrix of left singular vectors (represents lncRNA features)
    # Σ: diagonal matrix with singular values (importance of each feature)
    # V^T: transpose of right singular vectors (represents disease features)
    @staticmethod
    def extract_svd_features(matrix: MatrixLike, k: int):
        # Perform SVD decomposition
        # SVD tells us: "Hey, I found these patterns that explain the most important
        # ways lncRNAs and diseases are related."
        M = matrix.to_numpy(dtype=float, copy=False) if isinstance(matrix, pd.DataFrame) else np.asarray(matrix, dtype=float)
        U, s_vals, VT = np.linalg.svd(M, full_matrices=False)  # M ≈ U diag(s) VT

        # Select the first k components (clamped to available rank)
        if k is None or k > len(s_vals):
            k = len(s_vals)
        U_k  = U[:, :k]            # (n × k)
        s_k  = s_vals[:k]          # (k,)
        VT_k = VT[:k, :]           # (k × m)

        # use sqrt(s) to match paper logic (because usually it's used for both
        # sides in the original formula so now it's one item we also do want half of it)
        S_sqrt = np.diag(np.sqrt(s_k))  # (k × k)

        # Feature matrices
        lncRNA_features  = U_k @ S_sqrt        # left side = lncRNA (n × k)
        disease_features = (S_sqrt @ VT_k).T   # right side = disease (m × k)

        return lncRNA_features, disease_features
    

