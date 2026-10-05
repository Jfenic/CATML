"""
Domain contracts and entities for Meta-Learning and Dataset Fingerprinting.
Pure Python standard library implementation adhering to Hexagonal Architecture.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


@dataclass
class DatasetFingerprint:
    """Statistical and structural meta-features characterising a tabular dataset."""
    dataset_id: str
    dataset_name: str
    task_type: str
    row_count: int
    feature_count: int
    numerical_feature_count: int
    categorical_feature_count: int
    missing_cells_ratio: float
    target_entropy: float = 0.0

    @property
    def numerical_ratio(self) -> float:
        if self.feature_count <= 0:
            return 0.0
        return round(self.numerical_feature_count / self.feature_count, 3)

    @property
    def categorical_ratio(self) -> float:
        if self.feature_count <= 0:
            return 0.0
        return round(self.categorical_feature_count / self.feature_count, 3)

    @property
    def feature_to_row_ratio(self) -> float:
        if self.row_count <= 0:
            return 0.0
        return round(self.feature_count / self.row_count, 4)

    def to_vector(self) -> list[float]:
        """Normalized 6-dimensional meta-feature vector in [0.0, 1.0]."""
        # Logarithmic scaling for dimension cardinality
        log_rows = math.log10(max(1, self.row_count)) / 7.0  # 10^7 upper bound
        log_feats = math.log10(max(1, self.feature_count)) / 4.0  # 10^4 upper bound
        return [
            min(1.0, max(0.0, log_rows)),
            min(1.0, max(0.0, log_feats)),
            min(1.0, max(0.0, self.numerical_ratio)),
            min(1.0, max(0.0, self.categorical_ratio)),
            min(1.0, max(0.0, self.missing_cells_ratio)),
            min(1.0, max(0.0, self.target_entropy)),
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "rows": self.row_count,
            "features": self.feature_count,
            "numerical_ratio": self.numerical_ratio,
            "categorical_ratio": self.categorical_ratio,
            "missing_ratio": self.missing_cells_ratio,
            "target_entropy": round(self.target_entropy, 3),
            "vector": [round(v, 4) for v in self.to_vector()],
        }


@dataclass
class SimilarDatasetMatch:
    """Similarity match against a canonical benchmark or historical dataset."""
    name: str
    similarity: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "similarity": round(self.similarity, 2),
            "reasons": list(self.reasons),
        }


@dataclass
class HistoricalModelRanking:
    """Empirical performance ranking for a model family across similar datasets."""
    model: str
    experiments: int
    mean_rank: float
    win_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "experiments": self.experiments,
            "mean_rank": round(self.mean_rank, 1),
            "win_rate": round(self.win_rate, 2),
        }


@dataclass
class WarmStartRecommendation:
    """Warm-start configuration derived from meta-learning knowledge."""
    recommended_model: str
    params: dict[str, Any]
    expected_search_reduction: str = "~35%"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommended_model": self.recommended_model,
            "params": dict(self.params),
            "expected_search_reduction": self.expected_search_reduction,
            "reason": self.reason,
        }


@dataclass
class MetaLearningKnowledge:
    """Complete meta-learning synthesis for a dataset."""
    version: str
    dataset_name: str
    current_fingerprint: dict[str, Any]
    similar_datasets: list[SimilarDatasetMatch] = field(default_factory=list)
    historical_rankings: list[HistoricalModelRanking] = field(default_factory=list)
    warm_start: WarmStartRecommendation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "fingerprint": dict(self.current_fingerprint),
            "current_fingerprint": dict(self.current_fingerprint),
            "similar_datasets": [s.to_dict() for s in self.similar_datasets],
            "historical_rankings": [r.to_dict() for r in self.historical_rankings],
            "warm_start": self.warm_start.to_dict() if self.warm_start else {},
        }
