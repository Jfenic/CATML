from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelSpec:
    id: str
    name: str
    task_types: list[str] = field(default_factory=lambda: ["binary_classification"])
    description: str = ""


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ModelSpec] = {}

    def register(self, model: ModelSpec) -> None:
        self._models[model.id] = model

    def unregister(self, model_id: str) -> None:
        self._models.pop(model_id, None)

    def get(self, model_id: str) -> ModelSpec | None:
        return self._models.get(model_id)

    def list(self, task_type: str | None = None) -> list[ModelSpec]:
        models = list(self._models.values())
        if task_type is None:
            return models
        return [m for m in models if task_type in m.task_types]

    def resolve_active(
        self,
        include: list[str],
        exclude: list[str],
        task_type: str | None = None,
    ) -> list[str]:
        compatible = {m.id for m in self.list(task_type)} if task_type else set(self._models.keys())
        if include:
            candidates = [m for m in include if m in self._models and m in compatible]
        else:
            candidates = [m for m in compatible if m in self._models]
        return [m for m in candidates if m not in exclude]

    def validate_for_task(self, model_ids: list[str], task_type: str) -> list[str]:
        compatible = {m.id for m in self.list(task_type)}
        invalid = [m for m in model_ids if m not in compatible]
        if invalid:
            raise ValueError(
                f"Models not compatible with task '{task_type}': {', '.join(invalid)}. "
                f"Available: {', '.join(sorted(compatible))}"
            )
        return model_ids
