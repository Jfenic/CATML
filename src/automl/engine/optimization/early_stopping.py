from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingPolicy:
    patience: int = 5
    min_delta: float = 0.0001
    mode: str = "max"
    best_score: float = -float("inf")
    wait_count: int = 0

    def __post_init__(self) -> None:
        if self.mode == "min" and self.best_score == -float("inf"):
            self.best_score = float("inf")

    def update(self, score: float) -> bool:
        if self.mode == "max":
            improved = (score - self.best_score) > self.min_delta
        else:
            improved = (self.best_score - score) > self.min_delta

        if improved:
            self.best_score = score
            self.wait_count = 0
            return True
        else:
            self.wait_count += 1
            return False

    def should_stop(self) -> bool:
        return self.wait_count >= self.patience

    def step(self, score: float) -> bool:
        self.update(score)
        return self.should_stop()
