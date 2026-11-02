from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lncRNA_CIBCB2025-main"))

from scripts.felipe_paper_models import (  # noqa: E402
    MODELS,
    build_paper_model,
    effective_params,
    fitted_selection,
)


def test_paper_profile_parameters_are_recorded_from_instances():
    rf = effective_params("rf", build_paper_model("rf"))
    ipcarf = effective_params("ipcarf", build_paper_model("ipcarf"))
    rflda = effective_params("rflda", build_paper_model("rflda"))
    for params in (rf, ipcarf, rflda):
        assert params["n_estimators"] == 150 and params["random_state"] == 0
        assert params["binary_mode"] is False
    assert rf["max_features"] == "sklearn_default" and rf["oob_score"] is False
    assert ipcarf["standardize"] is False and ipcarf["allow_pca_fallback"] is False
    assert ipcarf["inner_cv_folds"] == 5
    assert ipcarf["n_components_grid"] == [2, 4, 8, 16, 32, 64, 128]
    assert rflda["inner_cv_folds"] == 5 and rflda["selection_metric"] == "subset_accuracy"


def test_n_jobs_does_not_change_effective_params():
    for key in MODELS:
        assert effective_params(key, build_paper_model(key, n_jobs=2)) == effective_params(
            key, build_paper_model(key)
        )
        assert build_paper_model(key, n_jobs=2).n_jobs == 2


def test_fitted_selection_after_tiny_fit():
    rng = np.random.RandomState(0)
    x = pd.DataFrame(rng.rand(60, 12))
    y = pd.DataFrame(rng.randint(0, 2, size=(60, 3)))
    for key in MODELS:
        model = build_paper_model(key, n_jobs=1)
        model.n_estimators = 5  # speed only; this test checks bookkeeping
        model.fit(x, y)
        chosen = fitted_selection(key, model)
        assert set(chosen) == {"rf": set(), "ipcarf": {"n_components"}, "rflda": {"best_n_features"}}[key]
        assert np.asarray(model.predict_proba(x.iloc[:5])).shape == (5, 3)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        build_paper_model("svm")
