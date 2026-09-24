from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from automl.domain.features.selection_strategy import FeatureRank, FeatureSetCandidate
from automl.domain.ports import FeatureAnalysisContext, FeatureSelectorPort
from automl.engine.features.common import prepare_feature_matrix
from automl.engine.profiling.dataset_profiler import load_dataframe


class CorrelationSelector(FeatureSelectorPort):
    method_id: str = "correlation"
    selector_type: str = "filter"

    def __init__(self) -> None:
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
            raise ValueError("Correlation selector requires a valid target column.")

        scores: list[float] = []
        for col in X.columns:
            vals = X[col].to_numpy()
            if np.all(vals == vals[0]) or np.all(y == y[0]):
                scores.append(0.0)
            else:
                corr, _ = spearmanr(vals, y)
                scores.append(0.0 if np.isnan(corr) else abs(float(corr)))

        ranked_indices = np.argsort(-np.array(scores))
        self._ranks = [
            FeatureRank(
                feature_name=X.columns[idx],
                score=float(scores[idx]),
                rank=i + 1,
                method=self.method_id,
                metadata={"abs_spearman_corr": float(scores[idx])},
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
