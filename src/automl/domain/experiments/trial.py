from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExperimentStatus(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class TrialStatus(str, Enum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Experiment:
    id: str
    run_id: str
    name: str
    hypothesis: str
    feature_names: list[str]
    model_ids: list[str]
    metric: str
    validation_strategy: str = "holdout"
    status: ExperimentStatus = ExperimentStatus.CREATED
    created_by: str = "user"
    priority: str = "normal"
    feature_set_id: str | None = None


@dataclass
class Trial:
    id: str
    experiment_id: str
    model_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int = 42
    status: TrialStatus = TrialStatus.CREATED


@dataclass
class TrialResult:
    trial_id: str
    experiment_id: str
    model_id: str
    primary_metric: str
    primary_score: float
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    training_time_seconds: float = 0.0
    failure_reason: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.failure_reason is None
