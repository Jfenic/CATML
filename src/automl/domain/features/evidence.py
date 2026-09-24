from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FeatureEvidence:
    feature_id: str
    mutual_information: float | None = None
    shap_importance: float | None = None
    permutation_importance: float | None = None
    linear_coefficient: float | None = None
    ablation_impact: float | None = None
    experiment_count: int = 0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "mutual_information": self.mutual_information,
            "shap_importance": self.shap_importance,
            "permutation_importance": self.permutation_importance,
            "linear_coefficient": self.linear_coefficient,
            "ablation_impact": self.ablation_impact,
            "experiment_count": self.experiment_count,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureEvidence:
        return cls(
            feature_id=data["feature_id"],
            mutual_information=data.get("mutual_information"),
            shap_importance=data.get("shap_importance"),
            permutation_importance=data.get("permutation_importance"),
            linear_coefficient=data.get("linear_coefficient"),
            ablation_impact=data.get("ablation_impact"),
            experiment_count=int(data.get("experiment_count", 0)),
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class FeatureInteractionEvidence:
    features: tuple[str, ...]
    interaction_score: float = 0.0
    experimental_gain: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "interaction_score": self.interaction_score,
            "experimental_gain": self.experimental_gain,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureInteractionEvidence:
        return cls(
            features=tuple(data["features"]),
            interaction_score=float(data.get("interaction_score", 0.0)),
            experimental_gain=float(data.get("experimental_gain", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
        )
