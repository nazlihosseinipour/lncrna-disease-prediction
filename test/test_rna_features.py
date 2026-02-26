import math
import sys
from pathlib import Path

import numpy as np

# Allow running tests directly by adding the project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mainfolder.features.rna_features import RnaFeatures



def test_kmer_and_rc_kmer_normalize():
    seqs = ["AUGC", "AAAAAA"]
    cols, X = RnaFeatures.kmer_matrix(seqs, k=2, normalize=True, return_format="matrix")
    assert len(cols) == 16
    assert len(X) == 2
    assert all(math.isclose(sum(row), 1.0, rel_tol=1e-9, abs_tol=1e-9) for row in X)

    cols_rc, X_rc = RnaFeatures.rc_kmer_matrix(seqs, k=2, normalize=True, return_format="matrix")
    assert 0 < len(cols_rc) < 16  # canonical collapse
    assert len(X_rc) == 2
    assert all(math.isclose(sum(row), 1.0, rel_tol=1e-9, abs_tol=1e-9) for row in X_rc)


def test_composition_matches_kmer_special_cases():
    seqs = ["AUGCUUAGC"]
    _, mono = RnaFeatures.mono_composition_matrix(seqs, normalize=True, return_format="matrix")
    _, k1 = RnaFeatures.kmer_matrix(seqs, k=1, normalize=True, return_format="matrix")
    assert np.allclose(mono[0], k1[0])

    _, di = RnaFeatures.di_composition_matrix(seqs, normalize=True, return_format="matrix")
    _, k2 = RnaFeatures.kmer_matrix(seqs, k=2, normalize=True, return_format="matrix")
    assert np.allclose(di[0], k2[0])

    _, tri = RnaFeatures.tri_composition_matrix(seqs, normalize=True, return_format="matrix")
    _, k3 = RnaFeatures.kmer_matrix(seqs, k=3, normalize=True, return_format="matrix")
    assert np.allclose(tri[0], k3[0])


def test_zcurve_and_gap_features():
    seqs = ["AUGCUUAGC"]

    cols, X = RnaFeatures.zcurve_matrix(seqs, normalize=True, return_format="matrix")
    assert cols == ["ZC_x", "ZC_y", "ZC_z"]
    assert len(X) == 1 and len(X[0]) == 3
    assert all(-1.0 <= v <= 1.0 for v in X[0])

    labels_mm, X_mm = RnaFeatures.monoMonoKGap_matrix(seqs, k_gap=1, normalize=True, return_format="matrix")
    assert len(labels_mm) == 16
    assert math.isclose(sum(X_mm[0]), 1.0, rel_tol=1e-9, abs_tol=1e-9)

    labels_md, X_md = RnaFeatures.monoDiKGap_matrix(seqs, k_gap=0, normalize=True, return_format="matrix")
    assert len(labels_md) == 64
    assert math.isclose(sum(X_md[0]), 1.0, rel_tol=1e-9, abs_tol=1e-9)
