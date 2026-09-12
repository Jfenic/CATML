from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OptimizationBudget:
    max_trials: int = 20
    timeout_seconds: float | None = None
    patience: int = 5
    min_delta: float = 0.0001

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_trials": self.max_trials,
            "timeout_seconds": self.timeout_seconds,
            "patience": self.patience,
            "min_delta": self.min_delta,
        }
