import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "lncRNA_CIBCB2025-main"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from IPCARF.ipcarf import IPCARF
from RFLDA.rflda import RFLDA


def test_ipcarf_scales_features_before_pca():
    x = pd.DataFrame(
        {
            "small": [0.0, 1.0, 2.0, 3.0],
            "huge": [0.0, 1.0e12, 2.0e12, 3.0e12],
            "other": [1.0, 0.0, 1.0, 0.0],
        }
    )
    y = pd.DataFrame({"label": [0, 0, 1, 1]})
    model = IPCARF(n_components=2)
    pipe = model._fit_pipeline_with_pca_fallback(
        x,
        y,
        RandomForestRegressor(n_estimators=1, random_state=0),
    )

    assert list(pipe.named_steps) == ["scale", "pca", "rf"]
    scaled = pipe.named_steps["scale"].transform(x)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-12)


def test_ipcarf_paper_profile_can_disable_unpublished_scaling():
    model = IPCARF(n_components=2, standardize=False, allow_pca_fallback=False)
    pipe = model._build_pipeline(
        IncrementalPCA(n_components=2),
        RandomForestRegressor(n_estimators=1, random_state=0),
    )

    assert list(pipe.named_steps) == ["pca", "rf"]


def test_rflda_feature_intervals_always_include_the_full_matrix():
    model = RFLDA()

    assert model._find_intervals(pd.DataFrame(np.zeros((2, 49)))) == [49]
    assert model._find_intervals(pd.DataFrame(np.zeros((2, 123)))) == [50, 100, 123]


def test_rflda_paper_profile_uses_five_folds_and_subset_accuracy():
    model = RFLDA(n_folds=5, selection_metric="subset_accuracy")

    assert model.splitter.n_splits == 5
    assert model.selection_metric == "subset_accuracy"
