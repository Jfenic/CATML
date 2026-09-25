from __future__ import annotations

from collections.abc import Callable
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import NotFittedError

from automl.engine.ensemble.blender import blend_predictions, normalize_weights


def _check_estimator_fitted(estimator: Any) -> bool:
    """Checks whether an estimator is fitted."""
    # First try sklearn's check_is_fitted if available
    try:
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(estimator)
        return True
    except (NotFittedError, ValueError, TypeError):
        pass
    except Exception:
        pass

    # Heuristic fallback for non-sklearn standard fitted attributes
    return any(
        hasattr(estimator, attr)
        for attr in ("classes_", "n_features_in_", "estimator_", "estimators_", "coef_")
    )


class VotingEnsembleEstimator(BaseEstimator):
    """Scikit-Learn compatible estimator for soft/hard voting ensembles and blending.

    Delegates blending mathematics to `blend_predictions` and model creation to an optional factory.
    """

    def __init__(
        self,
        estimators: list[Any] | list[tuple[str, Any]] | None = None,
        weights: list[float] | None = None,
        task_type: str = "binary_classification",
        voting: str = "soft",
        refit: bool = True,
        default_estimator_factory: Callable[[str], list[Any] | list[tuple[str, Any]]] | None = None,
    ) -> None:
        self.estimators = estimators
        self.weights = weights
        self.task_type = task_type
        self.voting = voting
        self.refit = refit
        self.default_estimator_factory = default_estimator_factory

        # Fitted attributes
        self.fitted_estimators_: list[Any] = []
        self.classes_: np.ndarray | None = None
        self.is_fitted_: bool = False

    @property
    def is_regression(self) -> bool:
        return "regression" in self.task_type

    def _resolve_estimators(self) -> list[Any]:
        """Extracts and unwraps estimators, delegating to default factory if none provided."""
        raw = self.estimators
        if raw is None or len(raw) == 0:
            if self.default_estimator_factory is not None:
                raw = self.default_estimator_factory(self.task_type)
            else:
                raise ValueError(
                    "No estimators provided to VotingEnsembleEstimator and no default_estimator_factory is configured."
                )

        normalized: list[Any] = []
        for item in raw:
            if isinstance(item, tuple) and len(item) == 2:
                normalized.append(item[1])
            else:
                normalized.append(item)

        if not normalized:
            raise ValueError("Resolved estimators list is empty.")
        return normalized

    def _validate_pre_fitted_estimators(self, estimators: list[Any], y: Any | None = None) -> None:
        """Thoroughly validates pre-fitted estimators when refit=False."""
        observed_classes: list[np.ndarray] = []

        for idx, est in enumerate(estimators):
            est_name = getattr(est, "__class__", type(est)).__name__
            if not _check_estimator_fitted(est):
                raise ValueError(
                    f"Estimator at index {idx} ({est_name}) is not fitted. "
                    "When refit=False, all base estimators must be fitted before passing to VotingEnsembleEstimator."
                )

            if not hasattr(est, "predict") or not callable(getattr(est, "predict")):
                raise TypeError(
                    f"Estimator at index {idx} ({est_name}) does not implement a callable 'predict' method."
                )

            if not self.is_regression and hasattr(est, "classes_"):
                est_classes = np.asarray(est.classes_)
                observed_classes.append(est_classes)

        # Validate class consistency across models for classification
        if not self.is_regression and observed_classes:
            ref_classes = observed_classes[0]
            for idx, c in enumerate(observed_classes[1:], start=1):
                if not set(c).issubset(set(ref_classes)) and not set(ref_classes).issubset(set(c)):
                    raise ValueError(
                        f"Class mismatch between base estimator 0 ({ref_classes}) "
                        f"and estimator {idx} ({c}). Base models must predict compatible target classes."
                    )
            # Use the most comprehensive set of classes
            all_unique = np.unique(np.concatenate(observed_classes))
            self.classes_ = all_unique

    def fit(self, X: Any, y: Any) -> VotingEnsembleEstimator:
        raw_estimators = self._resolve_estimators()
        n_estimators = len(raw_estimators)

        # Validate weights matching estimator count
        normalize_weights(self.weights, n_estimators)

        if not self.is_regression:
            y_arr = np.asarray(y)
            self.classes_ = np.unique(y_arr)

        self.fitted_estimators_ = []

        if not self.refit:
            self._validate_pre_fitted_estimators(raw_estimators, y=y)
            self.fitted_estimators_ = list(raw_estimators)
        else:
            for estimator in raw_estimators:
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
                # Align probabilities if estimator classes differ in order or size
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
                    pos_prob = 1.0 / (1.0 + np.exp(-df))
                    prob = np.column_stack([1.0 - pos_prob, pos_prob])
                else:
                    exp_df = np.exp(df - np.max(df, axis=1, keepdims=True))
                    prob = exp_df / np.sum(exp_df, axis=1, keepdims=True)
            else:
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
            weights = normalize_weights(self.weights, len(self.fitted_estimators_))

            final_preds = []
            for sample_idx in range(n_samples):
                score_per_class: dict[Any, float] = {}
                for m_idx, preds in enumerate(preds_list):
                    val = preds[sample_idx]
                    score_per_class[val] = score_per_class.get(val, 0.0) + weights[m_idx]
                best_class = max(score_per_class.items(), key=lambda x: x[1])[0]
                final_preds.append(best_class)
            return np.asarray(final_preds)
