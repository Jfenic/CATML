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

    def resolve_active(self, include: list[str], exclude: list[str]) -> list[str]:
        if include:
            candidates = [m for m in include if m in self._models]
        else:
            candidates = list(self._models.keys())
        return [m for m in candidates if m not in exclude]
