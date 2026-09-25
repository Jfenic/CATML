from __future__ import annotations

from typing import Any
import numpy as np


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
) -> np.ndarray:
    """Blends multiple prediction arrays using weighted averaging.

    Args:
        predictions: List of prediction arrays (1D for regression/probabilities, 2D for multiclass/probabilities).
        weights: Optional non-negative weights for each prediction array.
        task_type: Task type ('binary_classification', 'multiclass_classification', or 'regression').

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
