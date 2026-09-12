from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from automl.domain.experiments.priority import Priority
from automl.domain.experiments.trial import Experiment, ExperimentStatus


@dataclass
class ExperimentCandidate:
    id: str
    run_id: str
    name: str
    hypothesis: str
    feature_names: list[str]
    model_ids: list[str]
    metric: str
    validation_strategy: str = "holdout"
    priority: Priority | None = None
    feature_set_id: str | None = None
    tags: list[str] = field(default_factory=list)
    created_by: str = "planner"

    @classmethod
    def create(
        cls,
        run_id: str,
        name: str,
        hypothesis: str,
        feature_names: list[str],
        model_ids: list[str],
        metric: str,
        validation_strategy: str = "holdout",
        priority: Priority | None = None,
        feature_set_id: str | None = None,
        tags: list[str] | None = None,
        created_by: str = "planner",
    ) -> ExperimentCandidate:
        candidate_id = f"cand_{uuid.uuid4().hex[:8]}"
        return cls(
            id=candidate_id,
            run_id=run_id,
            name=name,
            hypothesis=hypothesis,
            feature_names=list(feature_names),
            model_ids=list(model_ids),
            metric=metric,
            validation_strategy=validation_strategy,
            priority=priority,
            feature_set_id=feature_set_id,
            tags=list(tags or []),
            created_by=created_by,
        )

    def to_experiment(self, experiment_id: str | None = None) -> Experiment:
        exp_id = experiment_id or f"exp_{uuid.uuid4().hex[:8]}"
        priority_str = self.priority.level.value if self.priority else "normal"
        return Experiment(
            id=exp_id,
            run_id=self.run_id,
            name=self.name,
            hypothesis=self.hypothesis,
            feature_names=list(self.feature_names),
            model_ids=list(self.model_ids),
            metric=self.metric,
            validation_strategy=self.validation_strategy,
            status=ExperimentStatus.CREATED,
            created_by=self.created_by,
            priority=priority_str,
            feature_set_id=self.feature_set_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "name": self.name,
            "hypothesis": self.hypothesis,
            "feature_names": self.feature_names,
            "model_ids": self.model_ids,
            "metric": self.metric,
            "validation_strategy": self.validation_strategy,
            "priority": self.priority.to_dict() if self.priority else None,
            "feature_set_id": self.feature_set_id,
            "tags": self.tags,
            "created_by": self.created_by,
        }
