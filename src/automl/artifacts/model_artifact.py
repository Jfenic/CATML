from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelArtifact:
    """
    Self-contained, serializable machine learning model artifact.

    Encapsulates the fully fitted scikit-learn Pipeline (preprocessing, encoders,
    scalers, imputers, and estimator), feature metadata, target adapter, and metrics.
    Can be serialized/deserialized independently without database or workspace dependencies.
    """

    pipeline: Any
    model_id: str
    task_type: str
    feature_names: list[str]
    target_name: str | None = None
    target_adapter: Any = None
    metric: str = "roc_auc"
    score: float = 0.0
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Run inference using the self-contained pipeline.

        Accepts either a pandas DataFrame (with required column names) or a numpy
        ndarray (with matching number of feature columns). Automatically inverts
        target label encoding if a TargetAdapter was fitted.
        """
        X_proc = self._prepare_features(X)
        raw_preds = self.pipeline.predict(X_proc)

        if self.target_adapter is not None and getattr(self.target_adapter, "is_encoded", False):
            try:
                preds = self.target_adapter.inverse_transform(raw_preds)
                return np.asarray(preds)
            except Exception:
                pass
        return np.asarray(raw_preds)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Compute predicted class probabilities if supported by the underlying model.
        """
        if not hasattr(self.pipeline, "predict_proba"):
            raise AttributeError(
                f"Underlying model '{self.model_id}' does not support predict_proba."
            )
        X_proc = self._prepare_features(X)
        probs = self.pipeline.predict_proba(X_proc)
        return np.asarray(probs)

    def _prepare_features(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame is missing required features: {missing}")
            return X[self.feature_names]
        elif isinstance(X, np.ndarray):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Feature count mismatch: expected {len(self.feature_names)} features, got {arr.shape[1]}."
                )
            return pd.DataFrame(arr, columns=self.feature_names)
        else:
            try:
                df = pd.DataFrame(X)
                if df.shape[1] == len(self.feature_names):
                    df.columns = self.feature_names
                    return df
            except Exception:
                pass
            raise TypeError(f"Unsupported input type for inference: {type(X)}")

    def save(self, path: str | Path) -> Path:
        """
        Serialize this artifact to disk as a standalone portable file.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifact:
        """
        Load a standalone ModelArtifact from disk.
        """
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Model artifact not found at '{target}'")
        loaded = joblib.load(target)
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Loaded object is not a ModelArtifact instance (got {type(loaded)})"
            )
        return loaded
