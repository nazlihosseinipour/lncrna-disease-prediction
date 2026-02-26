import os
import sys
import numpy as np
import pandas as pd

# Ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mainfolder.features.disease_features import DiseaseFeatures
from mainfolder.features.cross_features import CrossFeatures
import toy_matrix


def toy_dag():
    return [
        ("A", "ROOT"),
        ("B", "ROOT"),
        ("C", "B"),
    ]


def toy_disease_terms():
    return {
        "D1": ["A"],
        "D2": ["B"],
        "D3": ["C"],
    }


def test_disease_wang_and_bma_symmetry():
    df = DiseaseFeatures(toy_dag(), edge_weight=0.8)
    sim_bc = df.wang_term_similarity("B", "C")
    sim_ab = df.wang_term_similarity("A", "B")
    assert 0.0 <= sim_bc <= 1.0
    assert 0.0 <= sim_ab <= 1.0

    K = df.disease_similarity_bma(toy_disease_terms())
    assert isinstance(K, pd.DataFrame)
    assert K.shape == (3, 3)
    assert np.allclose(K.values, K.values.T)
    assert np.allclose(np.diag(K), 1.0)


def test_disease_lfs_from_Y_basic():
    df = DiseaseFeatures(toy_dag(), edge_weight=0.8)
    K_dis = df.disease_similarity_bma(toy_disease_terms())
    Y = pd.DataFrame(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
        index=["L1", "L2", "L3"],
        columns=["D1", "D2", "D3"]
    )
    LFS = df.lfs_from_Y(Y, disease_sim=K_dis)
    assert isinstance(LFS, pd.DataFrame)
    assert LFS.shape == (3, 3)
    assert np.allclose(LFS.values, LFS.values.T)
    assert np.allclose(np.diag(LFS), 1.0)


def test_cross_gip_outputs_shape_and_symmetry():
    mat = toy_matrix.ToyMatrix().getData()  # lncRNA x disease matrix
    gip_lnc, gip_dis = CrossFeatures.calculate_gip_lncRNA_and_dis(mat)
    assert gip_lnc.shape[0] == gip_lnc.shape[1] == mat.shape[0]
    assert gip_dis.shape[0] == gip_dis.shape[1] == mat.shape[1]
    assert np.allclose(gip_lnc.values, gip_lnc.values.T)
    assert np.allclose(gip_dis.values, gip_dis.values.T)
