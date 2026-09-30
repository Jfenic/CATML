from __future__ import annotations

from typing import Any
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def normalize_weights(weights: list[float] | tuple[float, ...] | np.ndarray | None, n_items: int) -> np.ndarray:
    """Validates and normalizes weights to sum to 1.0.

    Args:
        weights: Optional collection of non-negative weights.
        n_items: Expected number of items.

    Returns:
        1D numpy array of normalized weights of length n_items.
    """
    if weights is not None:
        if len(weights) != n_items:
            raise ValueError(
                f"Weights length ({len(weights)}) does not match expected length ({n_items})."
            )
        weights_arr = np.asarray(weights, dtype=float)
        if np.any(weights_arr < 0):
            raise ValueError("Weights must be non-negative.")
        total_weight = float(np.sum(weights_arr))
        if total_weight <= 0:
            raise ValueError("Sum of weights must be strictly positive.")
        return weights_arr / total_weight
    return np.full(n_items, 1.0 / n_items, dtype=float)


def blend_predictions(
    predictions: list[np.ndarray | list[float] | list[list[float]]],
    weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    task_type: str = "binary_classification",
    method: str = "average",
) -> np.ndarray:
    """Blends multiple prediction arrays using weighted averaging or rank averaging.

    Args:
        predictions: List of prediction arrays (1D for regression/probabilities, 2D for multiclass/probabilities).
        weights: Optional non-negative weights for each prediction array.
        task_type: Task type ('binary_classification', 'multiclass_classification', or 'regression').
        method: Blending method ('average' or 'rank').

    Returns:
        Blended numpy array with the same shape as each individual prediction array.
    """
    if not predictions:
        raise ValueError("Cannot blend empty predictions list.")

    arrays = [np.asarray(p, dtype=float) for p in predictions]
    n_models = len(arrays)

    first_shape = arrays[0].shape
    for i, arr in enumerate(arrays):
        if arr.shape != first_shape:
            raise ValueError(
                f"Prediction array shape mismatch at index {i}: expected {first_shape}, got {arr.shape}"
            )

    normalized_weights = normalize_weights(weights, n_models)

    # Rank normalization
    if method == "rank":
        processed_arrays: list[np.ndarray] = []
        for arr in arrays:
            if arr.ndim == 1:
                n = len(arr)
                if n > 1:
                    ranks = (rankdata(arr) - 1.0) / (n - 1.0)
                else:
                    ranks = arr.copy()
                processed_arrays.append(ranks)
            elif arr.ndim == 2 and arr.shape[1] == 2:
                # For binary probabilities, rank positive class and complement
                n = len(arr)
                p1 = arr[:, 1]
                if n > 1:
                    r1 = (rankdata(p1) - 1.0) / (n - 1.0)
                else:
                    r1 = p1.copy()
                processed_arrays.append(np.column_stack([1.0 - r1, r1]))
            else:
                processed_arrays.append(arr)
        arrays = processed_arrays

    # Weighted accumulation in-place to minimize peak memory
    blended = np.zeros_like(arrays[0], dtype=float)
    for w, arr in zip(normalized_weights, arrays):
        blended += w * arr

    # If classification probabilities in 2D, re-normalize rows to guarantee sum == 1.0
    is_classification = "classification" in task_type or task_type in {
        "binary_classification",
        "multiclass_classification",
    }
    if is_classification and blended.ndim == 2:
        row_sums = blended.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        blended = blended / row_sums

    return blended


def rank_average_predictions(
    predictions: list[np.ndarray | list[float] | list[list[float]]],
    weights: list[float] | tuple[float, ...] | np.ndarray | None = None,
    task_type: str = "binary_classification",
) -> np.ndarray:
    """Convenience function blending predictions using rank averaging (fractional ranks)."""
    return blend_predictions(predictions, weights=weights, task_type=task_type, method="rank")


def optimize_ensemble_weights(
    predictions: list[np.ndarray | list[float] | list[list[float]]],
    y_true: np.ndarray | list[Any],
    metric: str = "roc_auc",
    task_type: str = "binary_classification",
    method: str = "average",
    max_iter: int = 300,
) -> tuple[np.ndarray, float]:
    """Optimizes ensemble mixing weights to maximize validation metric on OOF predictions.

    Uses bounded Nelder-Mead on softmax parameterization to guarantee non-negative weights
    that sum strictly to 1.0 while maximizing the target evaluation metric.

    Args:
        predictions: List of model prediction arrays (e.g. out-of-fold probabilities).
        y_true: True ground truth labels or targets.
        metric: Metric to optimize ('roc_auc', 'log_loss', 'accuracy', 'f1', 'rmse', 'r2').
        task_type: Problem task type.
        method: Blending method ('average' or 'rank').
        max_iter: Maximum optimization iterations.

    Returns:
        tuple of (optimal_weights_array, best_score)
    """
    if not predictions:
        raise ValueError("Cannot optimize weights for empty predictions list.")

    arrays = [np.asarray(p, dtype=float) for p in predictions]
    n_models = len(arrays)
    y_arr = np.asarray(y_true)

    # Encode string targets if binary classification
    if "binary" in task_type and not np.issubdtype(y_arr.dtype, np.number):
        unique_labels = np.unique(y_arr)
        if len(unique_labels) == 2:
            y_arr = np.where(y_arr == unique_labels[1], 1, 0)

    equal_weights = np.full(n_models, 1.0 / n_models, dtype=float)
    metric_lower = metric.lower()

    def _eval_score(preds: np.ndarray) -> float:
        try:
            if metric_lower in ("roc_auc", "auc"):
                p = preds[:, 1] if (preds.ndim == 2 and preds.shape[1] == 2) else preds
                return float(roc_auc_score(y_arr, p))
            elif metric_lower in ("log_loss", "cross_entropy"):
                return -float(log_loss(y_arr, preds))
            elif metric_lower == "accuracy":
                if preds.ndim == 2:
                    y_pred = np.argmax(preds, axis=1)
                else:
                    y_pred = (preds >= 0.5).astype(int)
                return float(accuracy_score(y_arr, y_pred))
            elif metric_lower in ("f1", "f1_macro"):
                if preds.ndim == 2:
                    y_pred = np.argmax(preds, axis=1)
                else:
                    y_pred = (preds >= 0.5).astype(int)
                return float(f1_score(y_arr, y_pred, average="macro"))
            elif metric_lower in ("rmse", "root_mean_squared_error"):
                return -float(np.sqrt(mean_squared_error(y_arr, preds)))
            elif metric_lower in ("r2", "r2_score"):
                return float(r2_score(y_arr, preds))
            else:
                if "classification" in task_type:
                    p = preds[:, 1] if (preds.ndim == 2 and preds.shape[1] == 2) else preds
                    return float(roc_auc_score(y_arr, p))
                else:
                    return -float(np.sqrt(mean_squared_error(y_arr, preds)))
        except Exception:
            return -1e9

    baseline_blend = blend_predictions(arrays, weights=equal_weights, task_type=task_type, method=method)
    baseline_score = _eval_score(baseline_blend)

    if n_models == 1:
        raw_score = baseline_score
        if metric_lower in ("log_loss", "cross_entropy", "rmse", "root_mean_squared_error"):
            raw_score = abs(raw_score)
        return equal_weights, raw_score

    def _objective(theta: np.ndarray) -> float:
        w = softmax(theta)
        blended = blend_predictions(arrays, weights=w, task_type=task_type, method=method)
        raw_score = _eval_score(blended)
        if "classification" in task_type:
            try:
                if blended.ndim == 2:
                    p = blended[:, 1] if blended.shape[1] == 2 else blended
                    if p.ndim == 2:
                        y_one_hot = np.zeros_like(p)
                        for c_idx in range(p.shape[1]):
                            y_one_hot[y_arr == c_idx, c_idx] = 1.0
                        brier = float(np.mean((p - y_one_hot) ** 2))
                    else:
                        brier = float(np.mean((p - y_arr) ** 2))
                else:
                    brier = float(np.mean((blended - y_arr) ** 2))
                return -(raw_score - 0.05 * brier)
            except Exception:
                pass
        return -raw_score

    theta_0 = np.zeros(n_models, dtype=float)
    try:
        res = minimize(
            _objective,
            theta_0,
            method="Nelder-Mead",
            options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4},
        )
        opt_weights = softmax(res.x)
        opt_blend = blend_predictions(arrays, weights=opt_weights, task_type=task_type, method=method)
        opt_score = _eval_score(opt_blend)

        if opt_score >= baseline_score:
            final_score = opt_score
            final_weights = np.round(opt_weights, 5)
            final_weights = final_weights / final_weights.sum()
        else:
            final_score = baseline_score
            final_weights = equal_weights
    except Exception:
        final_score = baseline_score
        final_weights = equal_weights

    if metric_lower in ("log_loss", "cross_entropy", "rmse", "root_mean_squared_error"):
        final_score = abs(final_score)

    return final_weights, float(final_score)
