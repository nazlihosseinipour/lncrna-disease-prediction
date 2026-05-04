from pathlib import Path

import numpy as np
import pandas as pd

from mainfolder.utils.inductive_models import (
    build_rflda_param_grid,
    choose_youden_threshold_binary,
    choose_youden_thresholds_multilabel,
    load_feature_matrices,
    make_multilabel_dataset,
    run_nested_cv,
)


def test_build_rflda_param_grid_includes_final_feature_count() -> None:
    assert build_rflda_param_grid(120, step=50) == [50, 100, 120]


def test_choose_youden_thresholds_multilabel_supports_global_and_per_label() -> None:
    y_true = np.array([[1, 0], [1, 1], [0, 0], [0, 1]])
    y_score = np.array([[0.9, 0.2], [0.8, 0.8], [0.1, 0.3], [0.2, 0.7]])

    t_bin, j_bin = choose_youden_threshold_binary(y_true[:, 0], y_score[:, 0])
    assert 0.01 <= t_bin <= 0.99
    assert np.isfinite(j_bin)

    t_per, j_per = choose_youden_thresholds_multilabel(y_true, y_score, mode="per-label")
    assert t_per.shape == (2,)
    assert j_per.shape == (2,)

    t_global, j_global = choose_youden_thresholds_multilabel(y_true, y_score, mode="global")
    assert np.allclose(t_global[0], t_global[1])
    assert np.allclose(j_global[0], j_global[1], equal_nan=True)


def test_make_multilabel_dataset_merges_feature_files_by_id(tmp_path: Path) -> None:
    x1 = tmp_path / "x1.csv"
    x2 = tmp_path / "x2.csv"
    y = tmp_path / "y.csv"

    pd.DataFrame({"sample_id": ["a", "b"], "f0": [1.0, 2.0]}).to_csv(x1, index=False)
    pd.DataFrame({"ID": ["a", "b"], "g0": [3.0, 4.0]}).to_csv(x2, index=False)
    pd.DataFrame({"ID": ["a", "b"], "seqs": ["ACGU", "UGCA"], "D1": [1, 0], "D2": [0, 1]}).to_csv(y, index=False)

    bundle = make_multilabel_dataset([x1, x2], y)

    assert bundle.X.shape == (2, 2)
    assert bundle.Y.shape == (2, 2)
    assert bundle.ids == ["a", "b"]


def test_run_nested_cv_multilabel_returns_expected_reports(tmp_path: Path) -> None:
    ids = [f"s{i}" for i in range(12)]
    x = tmp_path / "x.csv"
    y = tmp_path / "y.csv"

    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "sample_id": ids,
            "f0": rng.normal(size=12),
            "f1": rng.normal(size=12),
            "f2": rng.normal(size=12),
            "f3": rng.normal(size=12),
        }
    )
    Y = pd.DataFrame(
        {
            "ID": ids,
            "D1": [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
            "D2": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    X.to_csv(x, index=False)
    Y.to_csv(y, index=False)

    bundle = make_multilabel_dataset([x], y)
    reports = run_nested_cv(
        dataset_name="toy",
        task="multilabel",
        model_name="rflda",
        bundle=bundle,
        param_grid=[2, 4],
        outer_splits=3,
        inner_splits=2,
        n_estimators=10,
        random_state=0,
        threshold_mode="per-label",
    )

    assert not reports["summary"].empty
    assert not reports["fold_metrics"].empty
    assert not reports["thresholds"].empty
    assert set(reports["summary"]["metric"]) == {
        "hamming_loss",
        "ranking_error",
        "micro_auroc",
        "micro_auprc",
        "precision",
        "recall",
        "f1",
        "accuracy",
    }
