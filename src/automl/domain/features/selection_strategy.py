from __future__ import annotations

from dataclasses import dataclass, field
import uuid
from typing import Any


@dataclass
class FeatureRank:
    feature_name: str
    score: float
    rank: int
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "score": self.score,
            "rank": self.rank,
            "method": self.method,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureRank:
        return cls(
            feature_name=data["feature_name"],
            score=float(data["score"]),
            rank=int(data["rank"]),
            method=data["method"],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class FeatureSetCandidate:
    id: str
    name: str
    feature_names: list[str]
    method: str
    k: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        name: str,
        feature_names: list[str],
        method: str,
        k: int,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureSetCandidate:
        return cls(
            id=str(uuid.uuid4())[:8],
            name=name,
            feature_names=list(feature_names),
            method=method,
            k=k,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "feature_names": list(self.feature_names),
            "method": self.method,
            "k": self.k,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSetCandidate:
        return cls(
            id=data["id"],
            name=data["name"],
            feature_names=list(data["feature_names"]),
            method=data["method"],
            k=int(data["k"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class FeatureSelectionStrategy:
    methods: list[str] = field(default_factory=lambda: ["mutual_information", "importance"])
    top_k: list[int] = field(default_factory=lambda: [3, 5])
    combine_method: str = "weighted_rank"  # "weighted_rank" | "borda" | "intersection"
    include_reduction: bool = False
    reduction_variances: list[float] = field(default_factory=lambda: [0.95])

    def to_dict(self) -> dict[str, Any]:
        return {
            "methods": list(self.methods),
            "top_k": list(self.top_k),
            "combine_method": self.combine_method,
            "include_reduction": self.include_reduction,
            "reduction_variances": list(self.reduction_variances),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureSelectionStrategy:
        return cls(
            methods=list(data.get("methods", ["mutual_information", "importance"])),
            top_k=[int(k) for k in data.get("top_k", [3, 5])],
            combine_method=data.get("combine_method", "weighted_rank"),
            include_reduction=bool(data.get("include_reduction", False)),
            reduction_variances=[float(v) for v in data.get("reduction_variances", [0.95])],
        )
