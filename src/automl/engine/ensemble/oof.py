"""Leakage-safe binary OOF probabilities, multi-model blending, and Level-2 stacking."""

from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from automl.engine.ensemble.blender import (
    blend_predictions,
    optimize_ensemble_weights,
    stack_predictions,
)
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
    weights: dict[str, float] | None = None
    method: str = "average"


def evaluate_oof(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    factories: dict[str, Callable[[], Any]],
    folds: int = 5,
    seed: int = 42,
    method: str = "average",
    meta_model: str = "ridge",
    max_seconds: float = 300.0,
    check_allowed: Callable[[], None] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> OOFResult:
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer >= 2")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if not factories:
        raise ValueError("OOF requires at least one model factory (one or two or more)")
    if X.empty or X_test.empty or len(X) != len(y) or y.isna().any():
        raise ValueError("OOF requires nonempty features/test data and a complete aligned target")
    if list(X.columns) != list(X_test.columns):
        raise ValueError("Train and test feature columns must match")

    method_aliases = {
        "mean": "average",
        "avg": "average",
        "stacking": "stacked",
        "stack": "stacked",
        "nelder_mead": "simplex",
    }
    canonical_method = method_aliases.get(method.lower(), method.lower())
    valid_methods = {"average", "rank", "simplex", "stacked"}
    if canonical_method not in valid_methods:
        raise ValueError(
            f"Unknown OOF blending method '{method}'. Valid options are: {sorted(valid_methods)}"
        )

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
            fold_scores[name].append(float(roc_auc_score(target[valid_idx], valid_probs)))
            completed += 1
            if progress:
                progress(completed, folds * len(factories), f"Fold {fold + 1}/{folds}: {name}")

    check_budget()
    test = {name: blend_predictions(values) for name, values in test_folds.items()}
    oof_list = list(oof.values())
    test_list = list(test.values())
    model_names = list(factories.keys())
    weights: dict[str, float] | None = None

    if canonical_method == "average":
        blended_oof = blend_predictions(oof_list, method="average")
        blended_test = blend_predictions(test_list, method="average")
        equal_w = 1.0 / len(model_names)
        weights = {name: equal_w for name in model_names}
    elif canonical_method == "rank":
        blended_oof = blend_predictions(oof_list, method="rank")
        blended_test = blend_predictions(test_list, method="rank")
        equal_w = 1.0 / len(model_names)
        weights = {name: equal_w for name in model_names}
    elif canonical_method == "simplex":
        opt_weights, _ = optimize_ensemble_weights(
            oof_list, target, metric="roc_auc", task_type="binary_classification", method="average"
        )
        blended_oof = blend_predictions(oof_list, weights=opt_weights, method="average")
        blended_test = blend_predictions(test_list, weights=opt_weights, method="average")
        weights = {name: float(w) for name, w in zip(model_names, opt_weights)}
    elif canonical_method == "stacked":
        stacked_oof, stacked_test, fitted_meta = stack_predictions(
            oof_list,
            target,
            test_predictions=test_list,
            meta_model=meta_model,
            task_type="binary_classification",
            random_state=seed,
        )
        blended_oof = stacked_oof
        blended_test = stacked_test if stacked_test is not None else blend_predictions(test_list)
        if hasattr(fitted_meta, "coef_"):
            raw_coefs = np.asarray(fitted_meta.coef_).ravel()
            if len(raw_coefs) == len(model_names):
                weights = {name: float(c) for name, c in zip(model_names, raw_coefs)}

    # Compute fold scores for the blend consistently
    for fold in range(folds):
        fold_mask = fold_ids == fold
        fold_scores["oof_blend"].append(float(roc_auc_score(target[fold_mask], blended_oof[fold_mask])))

    scores = {name: float(roc_auc_score(target, values)) for name, values in oof.items()}
    scores["oof_blend"] = float(roc_auc_score(target, blended_oof))
    return OOFResult(
        oof=oof,
        test=test,
        blended_oof=blended_oof,
        blended_test=blended_test,
        fold_ids=fold_ids,
        target=target,
        classes=encoder.classes_.tolist(),
        scores=scores,
        fold_scores=fold_scores,
        backends=backends,
        parameters=parameters,
        training_seconds=monotonic() - started,
        weights=weights,
        method=canonical_method,
    )

