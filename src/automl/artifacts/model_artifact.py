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
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provenance and "provenance" in self.metadata:
            self.provenance = dict(self.metadata["provenance"])
        elif not self.provenance:
            self.provenance = self._collect_default_provenance()

    def _collect_default_provenance(self) -> dict[str, Any]:
        import sys
        from datetime import datetime, timezone
        from importlib.metadata import version, PackageNotFoundError

        dep_versions: dict[str, str] = {}
        for name in ("numpy", "pandas", "scikit-learn", "joblib", "lightgbm", "xgboost", "catboost", "optuna"):
            try:
                dep_versions[name] = version(name)
            except PackageNotFoundError:
                pass

        estimator_step = self.pipeline[-1] if hasattr(self.pipeline, "__getitem__") and hasattr(self.pipeline, "steps") else self.pipeline

        return {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "estimator_class": type(estimator_step).__name__,
            "estimator_module": type(estimator_step).__module__,
            "dependencies": dep_versions,
        }

    def describe(self) -> dict[str, Any]:
        """
        Return a structured summary of the artifact's metadata and provenance.
        """
        return {
            "model_id": self.model_id,
            "task_type": self.task_type,
            "target_name": self.target_name,
            "metric": self.metric,
            "score": self.score,
            "feature_count": len(self.feature_names),
            "features": list(self.feature_names),
            "parameters": dict(self.parameters),
            "metadata": dict(self.metadata),
            "provenance": dict(self.provenance),
        }

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
