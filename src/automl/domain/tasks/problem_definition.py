from __future__ import annotations

from dataclasses import dataclass, field

from automl.domain.tasks.task_type import TaskType, metrics_for_task, models_for_task


@dataclass
class ProblemDefinition:
    dataset_id: str
    task_type: TaskType
    target_column: str | None
    default_metric: str
    available_metrics: list[str] = field(default_factory=list)
    recommended_models: list[str] = field(default_factory=list)
    inferred: bool = True
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "task_type": self.task_type.value,
            "task_label": self.task_type.name,
            "target_column": self.target_column,
            "default_metric": self.default_metric,
            "available_metrics": list(self.available_metrics),
            "recommended_models": list(self.recommended_models),
            "inferred": self.inferred,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_task(
        cls,
        dataset_id: str,
        task_type: TaskType,
        target_column: str | None,
        *,
        inferred: bool = True,
        reasoning: str = "",
    ) -> ProblemDefinition:
        from automl.domain.tasks.task_type import default_metric_for

        return cls(
            dataset_id=dataset_id,
            task_type=task_type,
            target_column=target_column,
            default_metric=default_metric_for(task_type),
            available_metrics=metrics_for_task(task_type),
            recommended_models=models_for_task(task_type),
            inferred=inferred,
            reasoning=reasoning,
        )
