from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from automl.domain.experiments.candidate import ExperimentCandidate


@dataclass
class BudgetPolicy:
    max_experiments: int | None = None
    max_trials: int | None = None
    max_estimated_time_seconds: float | None = None
    min_effective_score: float = 0.0
    allowed_models: list[str] | None = None
    excluded_models: list[str] = field(default_factory=list)

    def is_candidate_eligible(
        self,
        candidate: ExperimentCandidate,
        executed_experiments: int = 0,
        executed_trials: int = 0,
    ) -> bool:
        # Pinned candidates always bypass soft filters (score thresholds, etc.)
        is_pinned = candidate.priority is not None and candidate.priority.is_pinned

        if self.max_experiments is not None and (executed_experiments >= self.max_experiments):
            if not is_pinned:
                return False

        if self.max_trials is not None:
            projected_trials = executed_trials + len(candidate.model_ids)
            if projected_trials > self.max_trials and not is_pinned:
                return False

        if candidate.priority is not None:
            if candidate.priority.effective_score < self.min_effective_score and not is_pinned:
                return False

        if self.allowed_models is not None:
            allowed_set = set(self.allowed_models)
            if not any(m in allowed_set for m in candidate.model_ids):
                return False

        if self.excluded_models:
            excluded_set = set(self.excluded_models)
            # If all candidate models are excluded, reject candidate
            if all(m in excluded_set for m in candidate.model_ids):
                return False

        return True

    def filter_candidates(
        self,
        candidates: list[ExperimentCandidate],
        executed_experiments: int = 0,
        executed_trials: int = 0,
    ) -> list[ExperimentCandidate]:
        eligible: list[ExperimentCandidate] = []
        current_exp = executed_experiments
        current_tri = executed_trials

        for cand in candidates:
            if self.is_candidate_eligible(cand, current_exp, current_tri):
                eligible.append(cand)
                current_exp += 1
                current_tri += len(cand.model_ids)

        return eligible

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_experiments": self.max_experiments,
            "max_trials": self.max_trials,
            "max_estimated_time_seconds": self.max_estimated_time_seconds,
            "min_effective_score": self.min_effective_score,
            "allowed_models": self.allowed_models,
            "excluded_models": self.excluded_models,
        }
