from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from automl.domain.optimization.search_space import ParameterSpec, SearchSpace
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModelPluginPort
from automl.domain.tasks.task_type import TaskType


def blend_predictions(
    predictions: list[np.ndarray | list[float] | list[list[float]]],
    weights: list[float] | None = None,
    task_type: str = "binary_classification",
) -> np.ndarray:
    """Blends multiple model predictions using soft voting or weighted averaging.

    Args:
        predictions: List of prediction arrays (1D or 2D).
        weights: Optional list of non-negative weights.
        task_type: Task type ('binary_classification', 'multiclass_classification', or 'regression').

    Returns:
        Averaged/blended numpy array of predictions.
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

    if weights is not None:
        if len(weights) != n_models:
            raise ValueError(
                f"Weights length ({len(weights)}) does not match number of predictions ({n_models})."
            )
        weights_arr = np.asarray(weights, dtype=float)
        if np.any(weights_arr < 0):
            raise ValueError("Weights must be non-negative.")
        total_weight = float(np.sum(weights_arr))
        if total_weight <= 0:
            raise ValueError("Sum of weights must be strictly positive.")
        normalized_weights = weights_arr / total_weight
    else:
        normalized_weights = np.full(n_models, 1.0 / n_models, dtype=float)

    # Weighted sum across models
    blended = np.zeros_like(arrays[0], dtype=float)
    for w, arr in zip(normalized_weights, arrays):
        blended += w * arr

    # If classification probabilities in 2D, re-normalize each row to sum to 1.0
    if task_type in {
        TaskType.BINARY_CLASSIFICATION.value,
        TaskType.MULTICLASS_CLASSIFICATION.value,
        "binary_classification",
        "multiclass_classification",
    } and blended.ndim == 2:
        row_sums = blended.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums[row_sums == 0] = 1.0
        blended = blended / row_sums

    return blended


class VotingEnsembleEstimator(BaseEstimator):
    """Scikit-Learn compatible estimator for soft-voting ensembles and blending."""

    def __init__(
        self,
        estimators: list[Any] | list[tuple[str, Any]] | None = None,
        weights: list[float] | None = None,
        task_type: str = "binary_classification",
        voting: str = "soft",
        refit: bool = True,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.task_type = task_type
        self.voting = voting
        self.refit = refit

        # Fitted attributes
        self.fitted_estimators_: list[Any] = []
        self.classes_: np.ndarray | None = None
        self.is_fitted_: bool = False

    @property
    def is_regression(self) -> bool:
        return self.task_type in {TaskType.REGRESSION.value, "regression"}

    def _normalize_estimators(self) -> list[Any]:
        if self.estimators is None or len(self.estimators) == 0:
            if self.is_regression:
                return [
                    Ridge(alpha=1.0),
                    RandomForestRegressor(n_estimators=50, random_state=42),
                ]
            else:
                return [
                    LogisticRegression(max_iter=1000, random_state=42),
                    RandomForestClassifier(n_estimators=50, random_state=42),
                ]

        normalized: list[Any] = []
        for item in self.estimators:
            if isinstance(item, tuple) and len(item) == 2:
                normalized.append(item[1])
            else:
                normalized.append(item)
        return normalized

    def _get_normalized_weights(self, n_estimators: int) -> np.ndarray:
        if self.weights is not None:
            if len(self.weights) != n_estimators:
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match number of estimators ({n_estimators})."
                )
            w = np.asarray(self.weights, dtype=float)
            if np.any(w < 0):
                raise ValueError("Weights must be non-negative.")
            s = float(np.sum(w))
            if s <= 0:
                raise ValueError("Sum of weights must be positive.")
            return w / s
        return np.full(n_estimators, 1.0 / n_estimators, dtype=float)

    def fit(self, X: Any, y: Any) -> VotingEnsembleEstimator:
        raw_estimators = self._normalize_estimators()
        n_estimators = len(raw_estimators)
        if n_estimators == 0:
            raise ValueError("No estimators available to fit.")

        self._get_normalized_weights(n_estimators)

        if not self.is_regression:
            y_arr = np.asarray(y)
            self.classes_ = np.unique(y_arr)

        self.fitted_estimators_ = []
        for estimator in raw_estimators:
            # Check if estimator is already fitted and refit is False
            already_fitted = (
                hasattr(estimator, "classes_")
                or hasattr(estimator, "n_features_in_")
                or hasattr(estimator, "estimator_")
            )
            if already_fitted and not self.refit:
                self.fitted_estimators_.append(estimator)
            else:
                from sklearn.base import clone

                fitted = clone(estimator)
                fitted.fit(X, y)
                self.fitted_estimators_.append(fitted)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("This VotingEnsembleEstimator is not fitted yet.")
        if self.is_regression:
            raise AttributeError("predict_proba is not available for regression tasks.")

        probs_list: list[np.ndarray] = []
        for estimator in self.fitted_estimators_:
            if hasattr(estimator, "predict_proba"):
                prob = estimator.predict_proba(X)
                # Map estimator classes to self.classes_ if necessary
                if hasattr(estimator, "classes_") and not np.array_equal(estimator.classes_, self.classes_):
                    prob_aligned = np.zeros((prob.shape[0], len(self.classes_)), dtype=float)
                    for idx, c in enumerate(estimator.classes_):
                        target_idx = np.where(self.classes_ == c)[0]
                        if len(target_idx) > 0:
                            prob_aligned[:, target_idx[0]] = prob[:, idx]
                    prob = prob_aligned
            elif hasattr(estimator, "decision_function"):
                df = estimator.decision_function(X)
                if df.ndim == 1:
                    # binary sigmoid
                    pos_prob = 1.0 / (1.0 + np.exp(-df))
                    prob = np.column_stack([1.0 - pos_prob, pos_prob])
                else:
                    # softmax
                    exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                    prob = exp_df / np.sum(exp_df, axis=1, keepdims=True)
            else:
                # One-hot encode hard predict
                preds = estimator.predict(X)
                prob = np.zeros((len(preds), len(self.classes_)), dtype=float)
                for idx, c in enumerate(self.classes_):
                    prob[preds == c, idx] = 1.0

            probs_list.append(prob)

        return blend_predictions(
            predictions=probs_list,
            weights=self.weights,
            task_type=self.task_type,
        )

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted_:
            raise ValueError("This VotingEnsembleEstimator is not fitted yet.")

        if self.is_regression:
            preds_list = [estimator.predict(X) for estimator in self.fitted_estimators_]
            return blend_predictions(
                predictions=preds_list,
                weights=self.weights,
                task_type=self.task_type,
            )

        if self.voting == "soft":
            proba = self.predict_proba(X)
            class_indices = np.argmax(proba, axis=1)
            return self.classes_[class_indices]
        else:
            # Hard majority voting
            preds_list = [estimator.predict(X) for estimator in self.fitted_estimators_]
            n_samples = len(preds_list[0])
            weights = self._get_normalized_weights(len(self.fitted_estimators_))

            final_preds = []
            for sample_idx in range(n_samples):
                score_per_class: dict[Any, float] = {}
                for m_idx, preds in enumerate(preds_list):
                    val = preds[sample_idx]
                    score_per_class[val] = score_per_class.get(val, 0.0) + weights[m_idx]
                best_class = max(score_per_class.items(), key=lambda x: x[1])[0]
                final_preds.append(best_class)
            return np.asarray(final_preds)


class VotingEnsemblePlugin(ModelPluginPort):
    """CATML Model Plugin for Voting Ensemble and Blending."""

    def __init__(self) -> None:
        self.plugin_id = "voting_ensemble"
        self.name = "Voting Ensemble & Blending"
        self.version = "1.0.0"
        self.plugin_type = PluginType.MODEL
        self.capabilities = PluginCapability(
            supported_tasks=[
                TaskType.BINARY_CLASSIFICATION.value,
                TaskType.MULTICLASS_CLASSIFICATION.value,
                TaskType.REGRESSION.value,
            ],
            supported_modalities=["tabular"],
            requires_gpu=False,
            supports_proba=True,
        )

    def build_estimator(
        self,
        parameters: dict[str, Any] | None = None,
        task_type: str = "binary_classification",
        **kwargs: Any,
    ) -> VotingEnsembleEstimator:
        params = dict(parameters or {})
        params.update(kwargs)

        estimators = params.pop("estimators", None)
        weights = params.pop("weights", None)
        voting = params.pop("voting", "soft")
        refit = params.pop("refit", True)
        actual_task_type = params.pop("task_type", task_type)

        return VotingEnsembleEstimator(
            estimators=estimators,
            weights=weights,
            task_type=actual_task_type,
            voting=voting,
            refit=refit,
            **params,
        )

    def get_search_space(self, task_type: str) -> SearchSpace:
        space = SearchSpace()
        if task_type != TaskType.REGRESSION.value:
            space.add(ParameterSpec.categorical("voting", ["soft", "hard"], default="soft"))
        return space

    def blend(
        self,
        predictions: list[np.ndarray | list[float] | list[list[float]]],
        weights: list[float] | None = None,
        task_type: str = "binary_classification",
    ) -> np.ndarray:
        return blend_predictions(predictions, weights=weights, task_type=task_type)

    def from_models(
        self,
        models: list[Any],
        weights: list[float] | None = None,
        task_type: str = "binary_classification",
        refit: bool = False,
    ) -> VotingEnsembleEstimator:
        estimator = VotingEnsembleEstimator(
            estimators=models,
            weights=weights,
            task_type=task_type,
            voting="soft",
            refit=refit,
        )
        if not refit:
            # If not refitting, consider it pre-fitted
            estimator.fitted_estimators_ = list(models)
            estimator.is_fitted_ = True
            for m in models:
                if hasattr(m, "classes_"):
                    estimator.classes_ = np.asarray(m.classes_)
                    break
        return estimator
