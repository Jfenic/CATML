from __future__ import annotations

from automl.domain.datasets.profile import DatasetProfile
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import (
    ExperimentPriority,
    Priority,
    PriorityScoreBreakdown,
)
from automl.domain.ports import PriorityScorerPort
from automl.domain.runs.run import AutoMLRun


class RuleBasedPriorityScorer(PriorityScorerPort):
    """
    Computes explainable priority scores combining expected gain,
    uncertainty/exploration value, computational cost penalty, and user overrides.
    """

    def score(
        self,
        candidate: ExperimentCandidate,
        run: AutoMLRun,
        profile: DatasetProfile,
        user_priority: ExperimentPriority | str | None = None,
    ) -> Priority:
        parsed_user_priority: ExperimentPriority
        if user_priority is None:
            parsed_user_priority = ExperimentPriority.NORMAL
        elif isinstance(user_priority, ExperimentPriority):
            parsed_user_priority = user_priority
        else:
            parsed_user_priority = ExperimentPriority.parse(str(user_priority))

        # 1. Gain and novelty estimation
        tags = set(candidate.tags)
        gain_estimate = 0.70
        uncertainty = 0.10
        cost_penalty = 0.0
        reasons: list[str] = []

        if "baseline" in tags:
            gain_estimate = 0.85
            uncertainty = 0.05
            reasons.append("Baseline establishes primary reference performance.")
        elif "prioritized" in tags:
            gain_estimate = 0.80
            uncertainty = 0.15
            reasons.append("Evaluates user-selected high-value feature subset.")
        elif "model_exploration" in tags:
            gain_estimate = 0.75
            uncertainty = 0.20
            reasons.append("Explores non-linear model capacity.")

        # 2. Computational cost penalty based on models and data scale
        expensive_models = {"svc", "svr", "dbscan"}
        moderate_models = {"random_forest"}
        if any(m in expensive_models for m in candidate.model_ids):
            cost_penalty += 0.20
            reasons.append("Contains quadratic/expensive training algorithms (-0.20 cost penalty).")
        elif any(m in moderate_models for m in candidate.model_ids):
            cost_penalty += 0.05
            reasons.append("Contains ensemble algorithms with moderate cost (-0.05 cost penalty).")

        # 3. System score calculation
        raw_system_score = gain_estimate + uncertainty - cost_penalty
        system_score = max(0.05, min(0.95, round(raw_system_score, 4)))

        # 4. User priority boost and effective score
        user_boost = 0.0
        user_score = 0.0
        is_pinned = False

        if parsed_user_priority == ExperimentPriority.PINNED:
            is_pinned = True
            user_boost = 100.0
            user_score = 10.0
            effective_score = round(100.0 + system_score, 4)
            reasons.append("PINNED by user: immediate scheduling execution.")
        elif parsed_user_priority == ExperimentPriority.HIGH:
            user_boost = 0.25
            user_score = 0.25
            effective_score = round(min(1.0, system_score + user_boost), 4)
            reasons.append("HIGH user priority boost (+0.25).")
        elif parsed_user_priority == ExperimentPriority.LOW:
            user_boost = -0.25
            user_score = -0.25
            effective_score = round(max(0.01, system_score + user_boost), 4)
            reasons.append("LOW user priority penalty (-0.25).")
        else:
            effective_score = system_score
            reasons.append("Standard normal priority.")

        breakdown = PriorityScoreBreakdown(
            gain_estimate=gain_estimate,
            uncertainty=uncertainty,
            cost_penalty=cost_penalty,
            user_boost=user_boost,
            explanation=" ".join(reasons),
        )

        return Priority(
            system_score=system_score,
            user_score=user_score,
            effective_score=effective_score,
            level=parsed_user_priority,
            is_pinned=is_pinned,
            breakdown=breakdown,
        )
