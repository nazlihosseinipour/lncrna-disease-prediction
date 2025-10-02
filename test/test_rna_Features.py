# test/test_rna_features.py
import sys, os, math, numpy as np

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try both class names so this works regardless of your naming
try:
    from rna_features import RNAFeatures as RF
except (ImportError, AttributeError):
    from rna_features import rna_features as RF

def test_kmer_and_rc_kmer():
    rna = RF()
    seqs = ["AUGC", "AAAAAA"]

    cols, X = rna.kmer_matrix(seqs, k=2, normalize=True, return_format="matrix")
    assert len(cols) == 16
    assert len(X) == 2
    for row in X:
        assert math.isclose(sum(row), 1.0, rel_tol=1e-9, abs_tol=1e-9)

    cols_rc, X_rc = rna.rc_kmer_matrix(seqs, k=2, normalize=True, return_format="matrix")
    assert len(cols_rc) < 16
    assert len(X_rc) == 2
    for row in X_rc:
        assert math.isclose(sum(row), 1.0, rel_tol=1e-9, abs_tol=1e-9)

def test_make_and_canonical_columns():
    rna = RF()
    cols1 = rna.make_columns(1)
    cols2 = rna.make_columns(2)
    cols3 = rna.make_columns(3)
    assert len(cols1) == 4
    assert len(cols2) == 16
    assert len(cols3) == 64

    canon2 = rna.make_canonical_columns(2)
    assert 0 < len(canon2) < 16

def test_compositions_match_kmer_special_cases():
    rna = RF()
    seqs = ["AUGCUUAGC"]

    _, mono = rna.mono_composition_matrix(seqs, normalize=True, return_format="matrix")
    _, k1   = rna.kmer_matrix(seqs, k=1, normalize=True, return_format="matrix")
    assert np.allclose(mono[0], k1[0])

    _, di  = rna.di_composition_matrix(seqs, normalize=True, return_format="matrix")
    _, k2  = rna.kmer_matrix(seqs, k=2, normalize=True, return_format="matrix")
    assert np.allclose(di[0], k2[0])

    _, tri = rna.tri_composition_matrix(seqs, normalize=True, return_format="matrix")
    _, k3  = rna.kmer_matrix(seqs, k=3, normalize=True, return_format="matrix")
    assert np.allclose(tri[0], k3[0])

def test_zcurve_and_gap_features():
    rna = RF()
    seqs = ["AUGCUUAGC"]

    cols, X = rna.zcurve_matrix(seqs, normalize=True, return_format="matrix")
    assert cols == ["ZC_x", "ZC_y", "ZC_z"]
    assert len(X) == 1 and len(X[0]) == 3
    for v in X[0]:
        assert -1.0 <= v <= 1.0

    labels_mm, X_mm = rna.monoMonoKGap_matrix(seqs, k_gap=1, normalize=True, return_format="matrix")
    assert len(labels_mm) == 16
    assert math.isclose(sum(X_mm[0]), 1.0, rel_tol=1e-9, abs_tol=1e-9)

    labels_md, X_md = rna.monoDiKGap_matrix(seqs, k_gap=0, normalize=True, return_format="matrix")
    assert len(labels_md) == 64
    assert math.isclose(sum(X_md[0]), 1.0, rel_tol=1e-9, abs_tol=1e-9)
