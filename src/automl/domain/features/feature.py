from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FeatureStatus(str, Enum):
    ACTIVE = "ACTIVE"
    EXCLUDED = "EXCLUDED"
    PRIORITY = "PRIORITY"
    EXPERIMENTAL = "EXPERIMENTAL"
    GENERATED = "GENERATED"


class Modality(str, Enum):
    TABULAR = "TABULAR"


@dataclass
class Feature:
    id: str
    dataset_id: str
    name: str
    modality: Modality = Modality.TABULAR
    semantic_type: str = "unknown"
    physical_dtype: str = "unknown"
    status: FeatureStatus = FeatureStatus.ACTIVE
    user_priority: float = 0.0

    def exclude(self) -> None:
        self.status = FeatureStatus.EXCLUDED

    def prioritize(self, score: float = 1.0) -> None:
        self.status = FeatureStatus.PRIORITY
        self.user_priority = score
