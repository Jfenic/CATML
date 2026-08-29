from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from automl.domain.runs.states import RunPhase, RunStatus


@dataclass
class RunConfig:
    task_type: str
    target: str
    metric: str = "accuracy"
    validation_strategy: str = "holdout"
    test_size: float = 0.2
    random_seed: int = 42
    cv_folds: int = 5
    models_include: list[str] = field(default_factory=list)
    models_exclude: list[str] = field(default_factory=list)
    features_excluded: list[str] = field(default_factory=list)
    features_priority: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "target": self.target,
            "metric": self.metric,
            "validation_strategy": self.validation_strategy,
            "test_size": self.test_size,
            "random_seed": self.random_seed,
            "cv_folds": self.cv_folds,
            "models_include": list(self.models_include),
            "models_exclude": list(self.models_exclude),
            "features_excluded": list(self.features_excluded),
            "features_priority": list(self.features_priority),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        return cls(
            task_type=data["task_type"],
            target=data["target"],
            metric=data.get("metric", "accuracy"),
            validation_strategy=data.get("validation_strategy", "holdout"),
            test_size=float(data.get("test_size", 0.2)),
            random_seed=int(data.get("random_seed", 42)),
            cv_folds=int(data.get("cv_folds", 5)),
            models_include=list(data.get("models_include", [])),
            models_exclude=list(data.get("models_exclude", [])),
            features_excluded=list(data.get("features_excluded", [])),
            features_priority=list(data.get("features_priority", [])),
            extra=dict(data.get("extra", {})),
        )


@dataclass
class AutoMLRun:
    id: str
    workspace_id: str
    dataset_id: str
    config: RunConfig
    status: RunStatus = RunStatus.CREATED
    current_phase: RunPhase = RunPhase.DATASET_REGISTRATION

    def transition_to(self, status: RunStatus, phase: RunPhase | None = None) -> None:
        self.status = status
        if phase is not None:
            self.current_phase = phase
