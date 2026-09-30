"""Leakage-safe binary OOF probabilities and fixed equal-weight blending."""

from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from automl.engine.ensemble.blender import blend_predictions
from automl.engine.training.sklearn_trainer import _build_pipeline


@dataclass
class OOFResult:
    oof: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    blended_oof: np.ndarray
    blended_test: np.ndarray
    fold_ids: np.ndarray
    target: np.ndarray
    classes: list[Any]
    scores: dict[str, float]
    fold_scores: dict[str, list[float]]
    backends: dict[str, str]
    parameters: dict[str, dict]
    training_seconds: float


def evaluate_oof(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    factories: dict[str, Callable[[], Any]],
    folds: int = 5,
    seed: int = 42,
    max_seconds: float = 300.0,
    check_allowed: Callable[[], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> OOFResult:
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if not factories or len(factories) > 2:
        raise ValueError("OOF currently accepts one or two individual models")
    if X.empty or X_test.empty or len(X) != len(y) or y.isna().any():
        raise ValueError("OOF requires nonempty features/test data and a complete aligned target")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns must match")

    encoder = LabelEncoder()
    target = encoder.fit_transform(y)
    if len(encoder.classes_) != 2:
        raise ValueError("OOF currently supports binary classification only")
    if np.bincount(target).min() < folds:
        raise ValueError("Each target class must have at least folds rows")

    started = monotonic()
    oof = {name: np.full(len(X), np.nan) for name in factories}
    test_folds = {name: [] for name in factories}
    fold_scores = {name: [] for name in factories}
    fold_scores["oof_blend"] = []
    backends, parameters = {}, {}
    fold_ids = np.full(len(X), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    def check_budget() -> None:
        if check_allowed:
            check_allowed()
        if monotonic() - started >= max_seconds:
            raise TimeoutError("OOF time budget exceeded; no result promoted")

    def probabilities(pipeline, data):
        values = np.asarray(pipeline.predict_proba(data), dtype=float)
        if values.shape != (len(data), 2) or not np.isfinite(values).all():
            raise ValueError("Model must produce finite binary class probabilities")
        if np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1):
            raise ValueError("Model probabilities must be bounded and sum to one")
        classes = np.asarray(pipeline.classes_)
        if not np.array_equal(np.sort(classes), [0, 1]):
            raise ValueError("Model probability columns must correspond to encoded classes 0 and 1")
        return values[:, int(np.flatnonzero(classes == 1)[0])]

    completed = 0
    if progress:
        progress(0, folds * len(factories), "Evaluating OOF folds")
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, target)):
        fold_ids[valid_idx] = fold
        validation_predictions = []
        for name, factory in factories.items():
            check_budget()
            model = factory()
            if not hasattr(model, "predict_proba"):
                raise ValueError(f"Model '{name}' must support predict_proba")
            params = model.get_params(deep=False)
            if "random_state" in params and params["random_state"] is None:
                model.set_params(random_state=seed)
            backends[name] = f"{type(model).__module__}.{type(model).__name__}"
            parameters[name] = model.get_params(deep=False)
            # All imputers/scalers/encoders are fitted on training rows of this fold.
            pipeline = _build_pipeline(X.iloc[train_idx], model)
            pipeline.fit(X.iloc[train_idx], target[train_idx])
            check_budget()
            valid_probs = probabilities(pipeline, X.iloc[valid_idx])
            oof[name][valid_idx] = valid_probs
            test_folds[name].append(probabilities(pipeline, X_test))
            validation_predictions.append(valid_probs)
            fold_scores[name].append(float(roc_auc_score(target[valid_idx], valid_probs)))
            completed += 1
            if progress:
                progress(completed, folds * len(factories), f"Fold {fold + 1}/{folds}: {name}")
        fold_scores["oof_blend"].append(float(roc_auc_score(
            target[valid_idx], blend_predictions(validation_predictions),
        )))

    check_budget()
    test = {name: blend_predictions(values) for name, values in test_folds.items()}
    blended_oof = blend_predictions(list(oof.values()))
    blended_test = blend_predictions(list(test.values()))
    scores = {name: float(roc_auc_score(target, values)) for name, values in oof.items()}
    scores["oof_blend"] = float(roc_auc_score(target, blended_oof))
    return OOFResult(oof, test, blended_oof, blended_test, fold_ids, target,
                     encoder.classes_.tolist(), scores, fold_scores, backends,
                     parameters, monotonic() - started)
