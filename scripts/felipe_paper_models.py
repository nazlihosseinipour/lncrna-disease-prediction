"""Single construction point for the paper-profile RF / IPCARF / RFLDA models.

Both Felipe runners (within-version CV and strict transfer) build models here so
the paper profile and the recorded ``effective_params`` cannot drift apart.
Callers must put ``lncRNA_CIBCB2025-main`` on ``sys.path`` before calling
``build_paper_model``.
"""
from __future__ import annotations

MODELS = ("rf", "ipcarf", "rflda")
MODEL_SOURCES = {
    "rf": "lncRNA_CIBCB2025-main/RF/rf.py",
    "ipcarf": "lncRNA_CIBCB2025-main/IPCARF/ipcarf.py",
    "rflda": "lncRNA_CIBCB2025-main/RFLDA/rflda.py",
}


def build_paper_model(model_key: str, *, n_jobs: int | None = None):
    """Return an unfitted paper-profile model; ``n_jobs`` only changes parallelism."""
    if model_key == "rf":
        from RF.rf import RF

        model = RF(binary_mode=False)
    elif model_key == "ipcarf":
        from IPCARF.ipcarf import IPCARF

        model = IPCARF(binary_mode=False, standardize=False, allow_pca_fallback=False)
    elif model_key == "rflda":
        from RFLDA.rflda import RFLDA

        model = RFLDA(binary_mode=False, n_folds=5, selection_metric="subset_accuracy")
    else:
        raise ValueError(f"unknown model {model_key!r}; expected one of {MODELS}")
    if n_jobs is not None:
        model.n_jobs = n_jobs
    return model


def effective_params(model_key: str, model) -> dict[str, object]:
    """Read the parameters that determine predictions from the constructed model.

    Values are read from the instance, not restated, so a changed default is
    recorded. ``n_jobs`` is excluded: it changes speed, not the fitted forest.
    """
    params: dict[str, object] = {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "n_estimators": int(model.n_estimators),
        "random_state": int(model.random_state),
        "binary_mode": bool(model.converter.binary_mode),
    }
    if model_key == "rf":
        params.update(
            {
                "estimator": "sklearn.ensemble.RandomForestRegressor",
                "max_features": "sklearn_default",
                "oob_score": bool(model.oob_score),
            }
        )
    elif model_key == "ipcarf":
        params.update(
            {
                "standardize": bool(model.standardize),
                "allow_pca_fallback": bool(model.allow_pca_fallback),
                "inner_cv_folds": int(model.n_folds),
                "n_components_grid": [int(n) for n in model.n_components_optimize],
                "inner_cv_scoring": "estimator_default_r2",
            }
        )
    elif model_key == "rflda":
        params.update(
            {
                "inner_cv_folds": int(model.n_folds),
                "selection_metric": str(model.selection_metric),
                "inner_cv_shuffle": bool(model.shuffle),
                "feature_step": 50,
            }
        )
    else:
        raise ValueError(f"unknown model {model_key!r}")
    return params


def fitted_selection(model_key: str, model) -> dict[str, object]:
    """Hyperparameters chosen during fit (recorded, never used as an acceptance key)."""
    if model_key == "ipcarf":
        return {"n_components": int(model.n_components)}
    if model_key == "rflda":
        return {"best_n_features": int(model.best_n_features)}
    return {}
