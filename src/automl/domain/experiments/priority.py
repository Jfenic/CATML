from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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


@dataclass
class PriorityScoreBreakdown:
    gain_estimate: float = 0.5
    uncertainty: float = 0.5
    cost_penalty: float = 0.0
    user_boost: float = 0.0
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain_estimate": round(self.gain_estimate, 4),
            "uncertainty": round(self.uncertainty, 4),
            "cost_penalty": round(self.cost_penalty, 4),
            "user_boost": round(self.user_boost, 4),
            "explanation": self.explanation,
        }


@dataclass
class Priority:
    system_score: float
    user_score: float = 0.0
    effective_score: float = 0.5
    level: ExperimentPriority = ExperimentPriority.NORMAL
    is_pinned: bool = False
    breakdown: PriorityScoreBreakdown = field(default_factory=PriorityScoreBreakdown)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_score": round(self.system_score, 4),
            "user_score": round(self.user_score, 4),
            "effective_score": round(self.effective_score, 4),
            "level": self.level.value,
            "is_pinned": self.is_pinned,
            "breakdown": self.breakdown.to_dict(),
        }

