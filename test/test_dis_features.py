import numpy as np
import pandas as pd
import sys, os, numpy as np

# ensure project root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import disease_features
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


def get_toy_matrix():
    Data = toy_matrix.ToyMatrix()
    return Data.getData()


def test_wang_similarity_and_bma():
    matrix = get_toy_matrix()
    df = disease_features.DiseaseFeatures(matrix)

    sim_BC = df.wang_term_similarity("B", "C")
    sim_AB = df.wang_term_similarity("A", "B")

    assert 0 <= sim_BC <= 1
    assert 0 <= sim_AB <= 1
    #????????? fix this pleaseeeef
    #assert sim_BC > sim_AB  # child-parent closer than siblings

    K = df.disease_similarity_bma(toy_disease_terms())
    assert isinstance(K, pd.DataFrame)
    assert K.shape == (3, 3)
    assert np.allclose(K.values, K.values.T)
    assert np.allclose(np.diag(K), 1.0)


def test_lfs_from_Y():
    df =disease_features.DiseaseFeatures(toy_dag())
    dis_terms = toy_disease_terms()
    K_dis = df.disease_similarity_bma(dis_terms)

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
    assert LFS.loc["L2", "L3"] > LFS.loc["L1", "L2"]
