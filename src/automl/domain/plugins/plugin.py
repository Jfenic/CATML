from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PluginType(str, Enum):
    MODEL = "model"
    METRIC = "metric"
    OPTIMIZER = "optimizer"
    PREPROCESSOR = "preprocessor"


@dataclass(frozen=True)
class PluginCapability:
    supported_tasks: list[str] = field(default_factory=lambda: ["binary_classification"])
    supported_modalities: list[str] = field(default_factory=lambda: ["tabular"])
    requires_gpu: bool = False
    supports_proba: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def is_compatible_with_task(self, task_type: str) -> bool:
        if "*" in self.supported_tasks:
            return True
        return task_type in self.supported_tasks

    def is_compatible_with_modality(self, modality: str) -> bool:
        if "*" in self.supported_modalities:
            return True
        return modality in self.supported_modalities

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_tasks": list(self.supported_tasks),
            "supported_modalities": list(self.supported_modalities),
            "requires_gpu": self.requires_gpu,
            "supports_proba": self.supports_proba,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginCapability:
        return cls(
            supported_tasks=list(data.get("supported_tasks", ["binary_classification"])),
            supported_modalities=list(data.get("supported_modalities", ["tabular"])),
            requires_gpu=bool(data.get("requires_gpu", False)),
            supports_proba=bool(data.get("supports_proba", False)),
            extra=dict(data.get("extra", {})),
        )
