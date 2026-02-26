import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Allow running tests directly by adding the project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mainfolder.utils.loader import preprocess_sequences
from mainfolder.core.feature_extractor import FeatureExtractor


def test_preprocess_sequences_drops_invalid():
    ids = ["s1", "s2", "s3"]
    seqs = ["AUGC", "AU NG", "NNNN"]
    ids2, seqs2 = preprocess_sequences(ids, seqs, valid_alphabet={"A", "C", "G", "U"}, strict=False)
    assert ids2 == ["s1"]
    assert seqs2 == ["AUGC"]


def test_feature_extractor_rna_kmer_dispatch():
    seqs = ["AUGC", "AAAA"]
    cols, rows = FeatureExtractor.run("rna", 1, seqs, k=2, return_format="matrix")
    assert len(cols) == 16
    assert len(rows) == 2


def test_feature_extractor_disease_and_cross_dispatch():
    edges = [("B", "ROOT"), ("C", "B")]
    from mainfolder.features.disease_features import DiseaseFeatures
    df = DiseaseFeatures(edges, edge_weight=0.8)
    sim_bc = df.wang_term_similarity("B", "C")
    assert 0.0 <= sim_bc <= 1.0

    mat = pd.DataFrame([[1, 0], [0, 1]], index=["L1", "L2"], columns=["D1", "D2"])
    gip_lnc, gip_dis = FeatureExtractor.run("cross", 16, matrix=mat)
    assert gip_lnc.shape == (2, 2)
    assert gip_dis.shape == (2, 2)
    assert np.allclose(gip_lnc.values, gip_lnc.values.T)
    assert np.allclose(gip_dis.values, gip_dis.values.T)
