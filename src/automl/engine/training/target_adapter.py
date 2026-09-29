from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class TargetAdapter:
    """
    Adapts target series for machine learning models and metric evaluations.

    For classification tasks with non-numeric labels (e.g. string labels 'No'/'Yes')
    or arbitrary categories, encodes labels deterministically to integers [0, 1, ... K-1],
    preserving classes_ to decode predictions back to original classes or map
    calibrated probabilities accurately.

    For regression tasks or already standard continuous targets, acts as a passthrough.
    """

    def __init__(self) -> None:
        self._encoder: LabelEncoder | None = None
        self.is_encoded: bool = False
        self.classes_: np.ndarray | None = None

    def fit_transform(
        self,
        y: pd.Series | np.ndarray | list[Any],
        task_type: str = "binary_classification",
    ) -> np.ndarray:
        series = pd.Series(y) if not isinstance(y, pd.Series) else y
        is_classification = task_type in {"binary_classification", "multiclass_classification"}

        if not is_classification:
            # Regression or clustering: pass through as numeric float/int
            self.is_encoded = False
            self.classes_ = None
            return series.to_numpy()

        dtype_str = str(series.dtype).lower()
        is_discrete_non_int = (
            series.dtype.kind in {"O", "b", "S", "U"}
            or "str" in dtype_str
            or "object" in dtype_str
            or "bool" in dtype_str
            or isinstance(series.dtype, pd.CategoricalDtype)
        )

        # Check if already 0-indexed integer
        if not is_discrete_non_int and series.dtype.kind in {"i", "u"}:
            unique_vals = set(series.dropna().unique())
            expected_vals = set(range(len(unique_vals)))
            if unique_vals == expected_vals:
                self.is_encoded = False
                self.classes_ = np.array(sorted(unique_vals))
                return series.to_numpy()

        # Needs label encoding
        self._encoder = LabelEncoder()
        encoded = self._encoder.fit_transform(series)
        self.classes_ = self._encoder.classes_
        self.is_encoded = True
        return encoded

    def transform(self, y: pd.Series | np.ndarray | list[Any]) -> np.ndarray:
        if not self.is_encoded or self._encoder is None:
            return np.asarray(y)
        series = pd.Series(y) if not isinstance(y, pd.Series) else y
        return self._encoder.transform(series)

    def inverse_transform(self, predictions: np.ndarray | list[Any]) -> np.ndarray:
        if not self.is_encoded or self._encoder is None:
            return np.asarray(predictions)
        preds_array = np.asarray(predictions)
        # Only inverse transform integer-like class predictions, not probabilities
        if np.issubdtype(preds_array.dtype, np.integer):
            return self._encoder.inverse_transform(preds_array)
        return preds_array
