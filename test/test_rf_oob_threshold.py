from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "lncRNA_CIBCB2025-main") not in sys.path:
    sys.path.insert(0, str(ROOT / "lncRNA_CIBCB2025-main"))

from RF.rf import RF
from parse_results.evaluate import Evaluator
from scripts.run_inductive_within_cv import select_global_f1_threshold


def test_select_global_f1_threshold_finds_training_optimum():
    y = np.array([[0, 0, 1, 1]])
    scores = np.array([[0.05, 0.20, 0.35, 0.90]])

    threshold = select_global_f1_threshold(y, scores)

    assert threshold == 0.35


def test_select_global_f1_threshold_falls_back_for_one_class():
    assert select_global_f1_threshold(np.zeros((2, 3)), np.full((2, 3), 0.1)) == 0.5


def test_evaluator_uses_fixed_threshold_by_default():
    evaluator = Evaluator()

    assert evaluator.threshold_mode == "fixed"


def test_evaluator_rejects_runner_only_oob_mode():
    with pytest.raises(ValueError, match="Unsupported threshold mode"):
        Evaluator(threshold_mode="train_oob_f1")


def test_ranking_loss_uses_continuous_scores():
    y = pd.DataFrame({"d1": [1, 0], "d2": [0, 1]})
    scores = pd.DataFrame({"d1": [0.4, 0.3], "d2": [0.2, 0.45]})

    metrics = Evaluator().evaluate(y, scores)

    assert metrics[1] == 0.0


def test_rf_exposes_multilabel_oob_predictions():
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(80, 5)))
    y = pd.DataFrame(
        {
            "d1": (x[0] + 0.2 * x[1] > 0).astype(int),
            "d2": (x[2] - 0.3 * x[3] > 0).astype(int),
        }
    )
    model = RF(binary_mode=False, oob_score=True)
    model.n_estimators = 40
    model.n_jobs = 1

    model.fit(x, y)
    oob = model.get_oob_predictions()

    assert oob.shape == y.shape
    assert np.isfinite(oob).all()
