from __future__ import annotations

from typing import Any
import numpy as np

from automl.domain.features.selection_strategy import FeatureRank, FeatureSetCandidate
from automl.domain.ports import FeatureAnalysisContext, FeatureSelectorPort


class EnsembleRankSelector(FeatureSelectorPort):
    method_id: str = "ensemble"
    selector_type: str = "ensemble"

    def __init__(
        self,
        selectors: list[FeatureSelectorPort] | None = None,
        combine_method: str = "weighted_rank",
        weights: dict[str, float] | None = None,
    ) -> None:
        self.selectors = selectors or []
        self.combine_method = combine_method
        self.weights = weights or {}
        self._ranks: list[FeatureRank] = []

    def fit(self, context: FeatureAnalysisContext) -> None:
        if not self.selectors:
            raise ValueError("EnsembleRankSelector requires at least one selector.")

        method_ranks: dict[str, list[FeatureRank]] = {}
        for selector in self.selectors:
            selector.fit(context)
            method_ranks[selector.method_id] = selector.rank_features()

        self._ranks = self.combine_rankings(
            method_ranks,
            combine_method=self.combine_method,
            weights=self.weights,
        )

    def rank_features(self) -> list[FeatureRank]:
        return list(self._ranks)

    def select(self, k: int) -> FeatureSetCandidate:
        if not self._ranks:
            raise ValueError("Selector has not been fitted yet.")
        selected = [r.feature_name for r in self._ranks[:k]]
        return FeatureSetCandidate.create(
            name=f"{self.method_id}_{self.combine_method}_top_{k}",
            feature_names=selected,
            method=f"{self.method_id}_{self.combine_method}",
            k=len(selected),
            metadata={"combine_method": self.combine_method},
        )

    @staticmethod
    def combine_rankings(
        method_ranks: dict[str, list[FeatureRank]],
        combine_method: str = "weighted_rank",
        weights: dict[str, float] | None = None,
    ) -> list[FeatureRank]:
        if not method_ranks:
            return []

        weights = weights or {}
        # Collect all unique feature names
        all_features: set[str] = set()
        for ranks in method_ranks.values():
            for r in ranks:
                all_features.add(r.feature_name)

        n_features = len(all_features)
        combined_scores: dict[str, float] = {f: 0.0 for f in all_features}
        feature_breakdown: dict[str, dict[str, Any]] = {f: {} for f in all_features}

        if combine_method == "borda":
            for method, ranks in method_ranks.items():
                w = weights.get(method, 1.0)
                for r in ranks:
                    points = (n_features - r.rank + 1) * w
                    combined_scores[r.feature_name] += points
                    feature_breakdown[r.feature_name][method] = {
                        "rank": r.rank,
                        "borda_points": points,
                    }
        else:  # default: weighted_rank
            total_weight = sum(weights.get(m, 1.0) for m in method_ranks.keys())
            for method, ranks in method_ranks.items():
                w = weights.get(method, 1.0) / (total_weight if total_weight > 0 else 1.0)
                for r in ranks:
                    # Normalized rank score: 1.0 for rank 1, linearly decreasing
                    norm_score = ((n_features - r.rank + 1) / max(n_features, 1)) * w
                    combined_scores[r.feature_name] += norm_score
                    feature_breakdown[r.feature_name][method] = {
                        "rank": r.rank,
                        "score": r.score,
                        "weighted_score": norm_score,
                    }

        sorted_features = sorted(
            combined_scores.keys(),
            key=lambda f: combined_scores[f],
            reverse=True,
        )

        return [
            FeatureRank(
                feature_name=fname,
                score=float(combined_scores[fname]),
                rank=i + 1,
                method=f"ensemble_{combine_method}",
                metadata={"breakdown": feature_breakdown[fname]},
            )
            for i, fname in enumerate(sorted_features)
        ]
