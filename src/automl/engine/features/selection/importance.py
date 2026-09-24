from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

from automl.domain.features.selection_strategy import FeatureRank, FeatureSetCandidate
from automl.domain.ports import FeatureAnalysisContext, FeatureSelectorPort
from automl.engine.features.common import prepare_feature_matrix
from automl.engine.profiling.dataset_profiler import load_dataframe


class TreeImportanceSelector(FeatureSelectorPort):
    method_id: str = "importance"
    selector_type: str = "embedded"

    def __init__(self, n_estimators: int = 50) -> None:
        self.n_estimators = n_estimators
        self._ranks: list[FeatureRank] = []

    def fit(self, context: FeatureAnalysisContext) -> None:
        df = load_dataframe(context.dataset_path)
        X, y = prepare_feature_matrix(
            df,
            feature_names=context.active_feature_names,
            target_column=context.target_column,
            task_type=context.task_type,
        )
        if y is None:
            raise ValueError("Tree importance selector requires a valid target column.")

        if context.task_type == "regression":
            model = ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                random_state=context.random_seed,
                n_jobs=1,
            )
        else:
            model = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                random_state=context.random_seed,
                n_jobs=1,
            )

        model.fit(X, y)
        importances = model.feature_importances_

        ranked_indices = np.argsort(-importances)
        self._ranks = [
            FeatureRank(
                feature_name=X.columns[idx],
                score=float(importances[idx]),
                rank=i + 1,
                method=self.method_id,
                metadata={"tree_gini_importance": float(importances[idx])},
            )
            for i, idx in enumerate(ranked_indices)
        ]

    def rank_features(self) -> list[FeatureRank]:
        return list(self._ranks)

    def select(self, k: int) -> FeatureSetCandidate:
        if not self._ranks:
            raise ValueError("Selector has not been fitted yet.")
        selected = [r.feature_name for r in self._ranks[:k]]
        return FeatureSetCandidate.create(
            name=f"{self.method_id}_top_{k}",
            feature_names=selected,
            method=self.method_id,
            k=len(selected),
            metadata={"source": self.method_id},
        )


class L1Selector(FeatureSelectorPort):
    method_id: str = "l1"
    selector_type: str = "embedded"

    def __init__(self, alpha: float = 0.01) -> None:
        self.alpha = alpha
        self._ranks: list[FeatureRank] = []

    def fit(self, context: FeatureAnalysisContext) -> None:
        df = load_dataframe(context.dataset_path)
        X, y = prepare_feature_matrix(
            df,
            feature_names=context.active_feature_names,
            target_column=context.target_column,
            task_type=context.task_type,
        )
        if y is None:
            raise ValueError("L1 selector requires a valid target column.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if context.task_type == "regression":
            model = Lasso(alpha=self.alpha, random_state=context.random_seed, max_iter=2000)
            model.fit(X_scaled, y)
            coefs = np.abs(model.coef_)
        else:
            # L1 penalty via liblinear
            model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=1.0 / max(self.alpha, 1e-5),
                l1_ratio=1.0,
                random_state=context.random_seed,
                max_iter=2000,
            )
            model.fit(X_scaled, y)
            if model.coef_.ndim > 1:
                coefs = np.mean(np.abs(model.coef_), axis=0)
            else:
                coefs = np.abs(model.coef_)

        ranked_indices = np.argsort(-coefs)
        self._ranks = [
            FeatureRank(
                feature_name=X.columns[idx],
                score=float(coefs[idx]),
                rank=i + 1,
                method=self.method_id,
                metadata={"l1_abs_coefficient": float(coefs[idx])},
            )
            for i, idx in enumerate(ranked_indices)
        ]

    def rank_features(self) -> list[FeatureRank]:
        return list(self._ranks)

    def select(self, k: int) -> FeatureSetCandidate:
        if not self._ranks:
            raise ValueError("Selector has not been fitted yet.")
        selected = [r.feature_name for r in self._ranks[:k]]
        return FeatureSetCandidate.create(
            name=f"{self.method_id}_top_{k}",
            feature_names=selected,
            method=self.method_id,
            k=len(selected),
            metadata={"source": self.method_id},
        )
