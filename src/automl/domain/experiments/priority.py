from __future__ import annotations

from enum import Enum


class ExperimentPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    PINNED = "pinned"

    @classmethod
    def parse(cls, value: str) -> ExperimentPriority:
        normalized = value.strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(f"Invalid priority: {value}. Use: low, normal, high, pinned")
