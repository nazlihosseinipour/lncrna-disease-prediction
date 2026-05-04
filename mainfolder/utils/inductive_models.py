from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    label_ranking_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

ID_CANDIDATES = ("sample_id", "ID", "id", "lnc_id", "ncRNA Symbol")
SEQ_CANDIDATES = ("seq", "seqs")
DEFAULT_THRESHOLDS = np.linspace(0.01, 0.99, 99)


@dataclass
class DatasetBundle:
    X: np.ndarray
    Y: np.ndarray
    ids: list[str]
    feature_names: list[str]
    label_names: list[str]

    @property
    def positive_rate(self) -> float:
        return float(self.Y.sum() / self.Y.size) if self.Y.size else 0.0


def _normalized_lookup(columns: list[str]) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in columns}


def _extract_id_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    cols_lookup = _normalized_lookup(df.columns.tolist())
    for candidate in ID_CANDIDATES:
        col = cols_lookup.get(candidate.lower())
        if col is not None:
            out = df.rename(columns={col: "ID"})
            return out, "ID"

    out = df.copy()
    out.insert(0, "ID", [str(i) for i in range(len(out))])
    return out, "ID"


def load_feature_matrix(path: str | Path, *, prefix_columns: bool = True) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    unnamed = [col for col in df.columns if str(col).strip().lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df, _ = _extract_id_column(df)
    if df["ID"].duplicated().any():
        dup_count = int(df["ID"].duplicated().sum())
        raise ValueError(
            f"{csv_path} has {dup_count} duplicated IDs. "
            "This loader expects one feature row per sample. Token-level outputs like mp_tokens are not supported."
        )

    feature_cols = [col for col in df.columns if col != "ID"]
    if not feature_cols:
        raise ValueError(f"{csv_path} has no feature columns after loading.")

    feature_block = df[feature_cols].apply(pd.to_numeric, errors="raise")
    if prefix_columns:
        stem = csv_path.stem
        feature_block = feature_block.rename(columns={col: f"{stem}__{col}" for col in feature_block.columns})

    out = pd.concat([df[["ID"]].copy(), feature_block], axis=1)
    out["ID"] = out["ID"].astype(str).str.strip()
    return out


def load_feature_matrices(paths: list[str | Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("At least one feature matrix path is required.")

    merged = load_feature_matrix(paths[0], prefix_columns=len(paths) > 1)
    for extra_path in paths[1:]:
        extra = load_feature_matrix(extra_path, prefix_columns=True)
        merged = merged.merge(extra, on="ID", how="inner")
    if merged.empty:
        raise ValueError("No common IDs remained after merging feature matrices.")
    return merged


def load_multilabel_matrix(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    unnamed = [col for col in df.columns if str(col).strip().lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df, _ = _extract_id_column(df)
    cols_lookup = _normalized_lookup(df.columns.tolist())

    seq_cols: list[str] = []
    for candidate in SEQ_CANDIDATES:
        col = cols_lookup.get(candidate.lower())
        if col is not None and col != "ID":
            seq_cols.append(col)
    if seq_cols:
        df = df.drop(columns=sorted(set(seq_cols)))

    label_cols = [col for col in df.columns if col != "ID"]
    if not label_cols:
        raise ValueError(f"{csv_path} has no label columns after dropping ID and sequence columns.")

    block = df[label_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    out = pd.concat([df[["ID"]].copy(), block], axis=1)
    out["ID"] = out["ID"].astype(str).str.strip()
    if out["ID"].duplicated().any():
        raise ValueError(f"{csv_path} has duplicated IDs in the label matrix.")
    return out


def load_binary_labels(path: str | Path, *, label_col: str | None = None) -> pd.DataFrame:
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    unnamed = [col for col in df.columns if str(col).strip().lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    df, _ = _extract_id_column(df)
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"{csv_path} missing requested label column: {label_col}")
        value_col = label_col
    else:
        candidate_cols = [col for col in df.columns if col != "ID"]
        if len(candidate_cols) != 1:
            raise ValueError(
                f"{csv_path} must have exactly one non-ID label column for binary mode. "
                f"Found: {candidate_cols}"
            )
        value_col = candidate_cols[0]

    out = df[["ID", value_col]].copy()
    out = out.rename(columns={value_col: "label"})
    out["ID"] = out["ID"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    if out["ID"].duplicated().any():
        raise ValueError(f"{csv_path} has duplicated IDs in binary label mode.")
    return out


def make_multilabel_dataset(x_paths: list[str | Path], y_path: str | Path) -> DatasetBundle:
    X_df = load_feature_matrices(x_paths)
    Y_df = load_multilabel_matrix(y_path)
    merged = X_df.merge(Y_df, on="ID", how="inner")
    if merged.empty:
        raise ValueError("No common IDs remained after aligning X and Y.")

    feature_names = [col for col in X_df.columns if col != "ID"]
    label_names = [col for col in Y_df.columns if col != "ID"]
    X = merged[feature_names].to_numpy(dtype=float)
    Y = merged[label_names].to_numpy(dtype=int)
    ids = merged["ID"].astype(str).tolist()
    return DatasetBundle(X=X, Y=Y, ids=ids, feature_names=feature_names, label_names=label_names)


def make_binary_dataset(
    x_paths: list[str | Path],
    y_path: str | Path,
    *,
    label_col: str | None = None,
) -> DatasetBundle:
    X_df = load_feature_matrices(x_paths)
    y_df = load_binary_labels(y_path, label_col=label_col)
    merged = X_df.merge(y_df, on="ID", how="inner")
    if merged.empty:
        raise ValueError("No common IDs remained after aligning X and y.")

    feature_names = [col for col in X_df.columns if col != "ID"]
    X = merged[feature_names].to_numpy(dtype=float)
    y = merged["label"].to_numpy(dtype=int)
    ids = merged["ID"].astype(str).tolist()
    return DatasetBundle(X=X, Y=y, ids=ids, feature_names=feature_names, label_names=["label"])


def build_rflda_param_grid(n_features_total: int, *, step: int = 50) -> list[int]:
    if n_features_total < 1:
        raise ValueError("n_features_total must be >= 1")
    values = list(range(step, n_features_total + 1, step))
    if not values or values[-1] != n_features_total:
        values.append(n_features_total)
    return sorted(set(max(1, min(n_features_total, int(v))) for v in values))


def build_rflda_estimator(
    *,
    task: str,
    n_features: int,
    n_estimators: int,
    random_state: int,
    class_weight: str | None = "balanced",
) -> Any:
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=n_features,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=random_state,
    )
    if task == "binary":
        return rf
    return MultiOutputClassifier(rf, n_jobs=1)


def build_ipcarf_estimator(
    *,
    task: str,
    n_components: int,
    n_estimators: int,
    random_state: int,
    class_weight: str | None = "balanced",
) -> Any:
    def make_pipeline() -> Pipeline:
        return Pipeline(
            [
                ("var", VarianceThreshold()),
                ("ipca", IncrementalPCA(n_components=n_components)),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        max_features="sqrt",
                        class_weight=class_weight,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    estimator = make_pipeline()
    if task == "binary":
        return estimator
    return MultiOutputClassifier(estimator, n_jobs=1)


def _extract_classes(estimator: Any) -> np.ndarray:
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_)
    if hasattr(estimator, "named_steps"):
        for step in reversed(list(estimator.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    raise AttributeError("Could not extract classes_ from estimator.")


def _positive_probability(prob: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if prob.ndim == 1:
        return prob.astype(float)
    if prob.shape[1] == 1:
        return np.ones(prob.shape[0], dtype=float) if int(classes[0]) == 1 else np.zeros(prob.shape[0], dtype=float)
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
        return prob[:, pos_idx].astype(float)
    return prob[:, -1].astype(float)


def predict_positive_proba(estimator: Any, X: np.ndarray, *, task: str) -> np.ndarray:
    raw = estimator.predict_proba(X)
    if task == "binary":
        classes = _extract_classes(estimator)
        return _positive_probability(np.asarray(raw), classes)

    if isinstance(raw, list):
        cols = []
        for probs, sub_estimator in zip(raw, estimator.estimators_):
            classes = _extract_classes(sub_estimator)
            cols.append(_positive_probability(np.asarray(probs), classes))
        return np.column_stack(cols)

    if isinstance(raw, np.ndarray) and raw.ndim == 3:
        cols = []
        for probs, sub_estimator in zip(raw.transpose(1, 0, 2), estimator.estimators_):
            classes = _extract_classes(sub_estimator)
            cols.append(_positive_probability(np.asarray(probs), classes))
        return np.column_stack(cols)

    raise TypeError(f"Unsupported predict_proba return type for multi-label task: {type(raw)}")


def safe_micro_auprc(y_true: np.ndarray, y_score: np.ndarray, *, task: str) -> float:
    try:
        if task == "binary":
            return float(average_precision_score(y_true, y_score))
        return float(average_precision_score(y_true.ravel(), y_score.ravel()))
    except Exception:
        return float("nan")


def safe_micro_auroc(y_true: np.ndarray, y_score: np.ndarray, *, task: str) -> float:
    try:
        if task == "binary":
            return float(roc_auc_score(y_true, y_score))
        return float(roc_auc_score(y_true.ravel(), y_score.ravel()))
    except Exception:
        return float("nan")


def _sensitivity_specificity_binary(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    tn = int(np.logical_and(y_true == 0, y_pred == 0).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens), float(spec)


def choose_youden_threshold_binary(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    best_t = 0.5
    best_j = -np.inf
    for t in DEFAULT_THRESHOLDS:
        y_pred = (y_score >= t).astype(int)
        sens, spec = _sensitivity_specificity_binary(y_true, y_pred)
        youden = sens + spec - 1.0
        if youden > best_j:
            best_j = youden
            best_t = float(t)
    return best_t, float(best_j)


def choose_youden_thresholds_multilabel(
    Y_true: np.ndarray,
    Y_score: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    Y_true = np.asarray(Y_true).astype(int)
    Y_score = np.asarray(Y_score).astype(float)
    n_labels = Y_true.shape[1]

    if mode == "global":
        t, j = choose_youden_threshold_binary(Y_true.ravel(), Y_score.ravel())
        return np.full(n_labels, t, dtype=float), np.full(n_labels, j, dtype=float)

    thresholds = np.full(n_labels, 0.5, dtype=float)
    youden_scores = np.full(n_labels, np.nan, dtype=float)
    for col in range(n_labels):
        thresholds[col], youden_scores[col] = choose_youden_threshold_binary(Y_true[:, col], Y_score[:, col])
    return thresholds, youden_scores


def apply_thresholds(y_score: np.ndarray, thresholds: np.ndarray | float, *, task: str) -> np.ndarray:
    if task == "binary":
        return (np.asarray(y_score) >= float(thresholds)).astype(int)
    y_score = np.asarray(y_score)
    thresholds_arr = np.asarray(thresholds).reshape(1, -1)
    return (y_score >= thresholds_arr).astype(int)


def evaluate_binary(y_true: np.ndarray, y_score: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "ranking_error": float("nan"),
        "micro_auroc": safe_micro_auroc(y_true, y_score, task="binary"),
        "micro_auprc": safe_micro_auprc(y_true, y_score, task="binary"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    return out


def evaluate_multilabel(Y_true: np.ndarray, Y_score: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    out = {
        "hamming_loss": float(hamming_loss(Y_true, Y_pred)),
        "ranking_error": float(label_ranking_loss(Y_true, Y_score)),
        "micro_auroc": safe_micro_auroc(Y_true, Y_score, task="multilabel"),
        "micro_auprc": safe_micro_auprc(Y_true, Y_score, task="multilabel"),
        "precision": float(precision_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "recall": float(recall_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "f1": float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)),
        "accuracy": float(accuracy_score(Y_true, Y_pred)),
    }
    return out


def _build_estimator(
    *,
    model_name: str,
    task: str,
    param_value: int,
    n_estimators: int,
    random_state: int,
    class_weight: str | None,
) -> Any:
    if model_name == "rflda":
        return build_rflda_estimator(
            task=task,
            n_features=param_value,
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
        )
    if model_name == "ipcarf":
        return build_ipcarf_estimator(
            task=task,
            n_components=param_value,
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def _iter_splits(X: np.ndarray, Y: np.ndarray, *, task: str, n_splits: int, random_state: int):
    if task == "binary":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return splitter.split(X, Y)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return splitter.split(X)


def _inner_score(
    *,
    model_name: str,
    task: str,
    param_value: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_estimators: int,
    inner_splits: int,
    random_state: int,
    class_weight: str | None,
) -> float:
    scores: list[float] = []
    for split_idx, (train_idx, valid_idx) in enumerate(
        _iter_splits(X, Y, task=task, n_splits=inner_splits, random_state=random_state),
        start=1,
    ):
        estimator = _build_estimator(
            model_name=model_name,
            task=task,
            param_value=param_value,
            n_estimators=n_estimators,
            random_state=random_state + split_idx,
            class_weight=class_weight,
        )
        estimator.fit(X[train_idx], Y[train_idx])
        score = predict_positive_proba(estimator, X[valid_idx], task=task)
        scores.append(safe_micro_auprc(Y[valid_idx], score, task=task))

    valid_scores = [s for s in scores if not np.isnan(s)]
    return float(np.mean(valid_scores)) if valid_scores else float("-inf")


def select_best_param(
    *,
    model_name: str,
    task: str,
    param_grid: list[int],
    X: np.ndarray,
    Y: np.ndarray,
    n_estimators: int,
    inner_splits: int,
    random_state: int,
    class_weight: str | None,
) -> tuple[int, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_param = param_grid[0]
    best_score = float("-inf")

    for param_value in param_grid:
        mean_score = _inner_score(
            model_name=model_name,
            task=task,
            param_value=param_value,
            X=X,
            Y=Y,
            n_estimators=n_estimators,
            inner_splits=inner_splits,
            random_state=random_state,
            class_weight=class_weight,
        )
        rows.append({"param_value": param_value, "mean_inner_micro_auprc": mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_param = param_value

    return best_param, pd.DataFrame(rows)


def fit_oof_probabilities(
    *,
    model_name: str,
    task: str,
    param_value: int,
    X: np.ndarray,
    Y: np.ndarray,
    n_estimators: int,
    inner_splits: int,
    random_state: int,
    class_weight: str | None,
) -> np.ndarray:
    if task == "binary":
        oof = np.zeros(len(Y), dtype=float)
    else:
        oof = np.zeros((Y.shape[0], Y.shape[1]), dtype=float)

    for split_idx, (train_idx, valid_idx) in enumerate(
        _iter_splits(X, Y, task=task, n_splits=inner_splits, random_state=random_state),
        start=1,
    ):
        estimator = _build_estimator(
            model_name=model_name,
            task=task,
            param_value=param_value,
            n_estimators=n_estimators,
            random_state=random_state + 1000 + split_idx,
            class_weight=class_weight,
        )
        estimator.fit(X[train_idx], Y[train_idx])
        oof[valid_idx] = predict_positive_proba(estimator, X[valid_idx], task=task)
    return oof


def summarize_metrics(fold_df: pd.DataFrame, *, dataset_name: str, model_name: str, positive_rate: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "hamming_loss",
        "ranking_error",
        "micro_auroc",
        "micro_auprc",
        "precision",
        "recall",
        "f1",
        "accuracy",
    ]
    for metric in metric_cols:
        vals = pd.to_numeric(fold_df[metric], errors="coerce")
        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "metric": metric,
                "mean": float(vals.mean(skipna=True)),
                "std": float(vals.std(ddof=1, skipna=True)),
                "n_folds": int(vals.notna().sum()),
                "positive_rate": float(positive_rate),
            }
        )
    return pd.DataFrame(rows)


def run_nested_cv(
    *,
    dataset_name: str,
    task: str,
    model_name: str,
    bundle: DatasetBundle,
    param_grid: list[int],
    outer_splits: int = 10,
    inner_splits: int = 5,
    n_estimators: int = 150,
    random_state: int = 0,
    class_weight: str | None = "balanced",
    threshold_mode: str = "per-label",
) -> dict[str, pd.DataFrame]:
    X = np.asarray(bundle.X, dtype=float)
    Y = np.asarray(bundle.Y)

    fold_rows: list[dict[str, Any]] = []
    inner_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        _iter_splits(X, Y, task=task, n_splits=outer_splits, random_state=random_state),
        start=1,
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        best_param, inner_df = select_best_param(
            model_name=model_name,
            task=task,
            param_grid=param_grid,
            X=X_train,
            Y=Y_train,
            n_estimators=n_estimators,
            inner_splits=inner_splits,
            random_state=random_state + fold_idx,
            class_weight=class_weight,
        )
        inner_df.insert(0, "outer_fold", fold_idx)
        inner_df.insert(0, "model", model_name)
        inner_df.insert(0, "dataset", dataset_name)
        inner_rows.append(inner_df)

        oof_train_score = fit_oof_probabilities(
            model_name=model_name,
            task=task,
            param_value=best_param,
            X=X_train,
            Y=Y_train,
            n_estimators=n_estimators,
            inner_splits=inner_splits,
            random_state=random_state + fold_idx,
            class_weight=class_weight,
        )

        if task == "binary":
            threshold, youden = choose_youden_threshold_binary(Y_train, oof_train_score)
            thresholds = float(threshold)
            threshold_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "outer_fold": fold_idx,
                    "label": "label",
                    "threshold": threshold,
                    "youden_j": youden,
                }
            )
        else:
            thresholds, youdens = choose_youden_thresholds_multilabel(
                Y_train,
                oof_train_score,
                mode=threshold_mode,
            )
            for label_name, threshold, youden in zip(bundle.label_names, thresholds, youdens):
                threshold_rows.append(
                    {
                        "dataset": dataset_name,
                        "model": model_name,
                        "outer_fold": fold_idx,
                        "label": label_name,
                        "threshold": float(threshold),
                        "youden_j": float(youden),
                    }
                )

        estimator = _build_estimator(
            model_name=model_name,
            task=task,
            param_value=best_param,
            n_estimators=n_estimators,
            random_state=random_state + 5000 + fold_idx,
            class_weight=class_weight,
        )
        estimator.fit(X_train, Y_train)
        test_score = predict_positive_proba(estimator, X_test, task=task)
        test_pred = apply_thresholds(test_score, thresholds, task=task)

        metrics = (
            evaluate_binary(Y_test, test_score, test_pred)
            if task == "binary"
            else evaluate_multilabel(Y_test, test_score, test_pred)
        )
        fold_rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "outer_fold": fold_idx,
                "best_param": int(best_param),
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                **metrics,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary_df = summarize_metrics(
        fold_df,
        dataset_name=dataset_name,
        model_name=model_name,
        positive_rate=bundle.positive_rate,
    )
    inner_df = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame()
    threshold_df = pd.DataFrame(threshold_rows)

    return {
        "summary": summary_df,
        "fold_metrics": fold_df,
        "inner_search": inner_df,
        "thresholds": threshold_df,
        "dataset_info": pd.DataFrame(
            [
                {
                    "dataset": dataset_name,
                    "task": task,
                    "num_samples": int(len(bundle.ids)),
                    "num_features": int(bundle.X.shape[1]),
                    "num_labels": int(bundle.Y.shape[1] if task != "binary" else 1),
                    "positive_rate": float(bundle.positive_rate),
                }
            ]
        ),
    }
