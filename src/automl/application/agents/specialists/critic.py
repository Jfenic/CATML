"""Critic specialist responsible for evaluating experiment outcomes against evidence and baselines."""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from automl.application.agents.contracts import ContextPayload, EvaluationFeedback
from automl.application.agents.ports import LLMProviderPort, SpecialistPort


class Critic:
    """Specialist that interprets existing metrics and evaluates empirical improvements."""

    def __init__(self, llm_provider: LLMProviderPort | None = None) -> None:
        self._llm_provider = llm_provider

    @property
    def name(self) -> str:
        return "critic"

    def analyze(
        self, context: ContextPayload, latest_result: dict[str, Any] | None = None, **kwargs: Any
    ) -> EvaluationFeedback:
        """Evaluate recent experimental results against baselines and statistical evidence."""
        if self._llm_provider is not None:
            try:
                return self._analyze_with_llm(context, latest_result)
            except Exception:
                pass

        return self._analyze_deterministically(context, latest_result)

    def _analyze_deterministically(
        self, context: ContextPayload, latest_result: dict[str, Any] | None = None
    ) -> EvaluationFeedback:
        """Deterministic empirical evaluation conforming to Propose != Accept."""
        leaderboard = context.leaderboard
        direction = context.metric_direction.lower()
        metric_name = context.target_metric

        # Case 1: No previous experiments, nothing to evaluate yet
        if not leaderboard and not latest_result:
            return EvaluationFeedback(
                feedback_id=f"feed-crit-{uuid4().hex[:8]}",
                run_id=context.run_id,
                specialist_name=self.name,
                is_improvement=False,
                metric_name=metric_name,
                recommendation="explore_alternative",
                variance_observation="No experimental evidence available yet.",
                evidence_summary={},
            )

        # Determine current score & baseline score
        if latest_result:
            current_score = float(latest_result.get("score", 0.0))
            experiment_id = latest_result.get("experiment_id")
            # Baseline is best on leaderboard
            baseline_score = float(leaderboard[0]["score"]) if leaderboard else None
        elif len(leaderboard) >= 2:
            current_score = float(leaderboard[0]["score"])
            experiment_id = leaderboard[0].get("experiment_id")
            baseline_score = float(leaderboard[1]["score"])
        elif len(leaderboard) == 1:
            current_score = float(leaderboard[0]["score"])
            experiment_id = leaderboard[0].get("experiment_id")
            baseline_score = None
        else:
            current_score = 0.0
            experiment_id = None
            baseline_score = None

        # Case 2: First baseline experiment
        if baseline_score is None:
            return EvaluationFeedback(
                feedback_id=f"feed-crit-{uuid4().hex[:8]}",
                run_id=context.run_id,
                specialist_name=self.name,
                is_improvement=True,
                metric_name=metric_name,
                experiment_id=experiment_id,
                baseline_score=None,
                current_score=current_score,
                variance_observation="Initial baseline established. No prior reference point for variance comparison.",
                recommendation="accept",
                evidence_summary={"baseline_score": current_score},
            )

        # Case 3: Comparison with baseline
        delta = current_score - baseline_score
        is_improvement = (delta > 0) if direction == "maximize" else (delta < 0)

        # Variance & overfitting diagnosis (only when evidence is present)
        variance_obs = "Variance within expected bounds across evaluation folds."
        fold_scores = latest_result.get("fold_scores") if latest_result else None
        if fold_scores and len(fold_scores) >= 2:
            spread = max(fold_scores) - min(fold_scores)
            if spread > 0.15:
                variance_obs = f"Warning: High cross-validation spread detected (fold spread: {spread:.4f}). Potential instability."
                if is_improvement:
                    # Downgrade recommendation if variance is volatile
                    return EvaluationFeedback(
                        feedback_id=f"feed-crit-{uuid4().hex[:8]}",
                        run_id=context.run_id,
                        specialist_name=self.name,
                        is_improvement=is_improvement,
                        metric_name=metric_name,
                        experiment_id=experiment_id,
                        baseline_score=baseline_score,
                        current_score=current_score,
                        variance_observation=variance_obs,
                        recommendation="explore_alternative",
                        evidence_summary={"fold_spread": spread, "delta": delta},
                    )

        if is_improvement:
            recommendation = "accept"
        else:
            recommendation = "reject"

        return EvaluationFeedback(
            feedback_id=f"feed-crit-{uuid4().hex[:8]}",
            run_id=context.run_id,
            specialist_name=self.name,
            is_improvement=is_improvement,
            metric_name=metric_name,
            experiment_id=experiment_id,
            baseline_score=baseline_score,
            current_score=current_score,
            variance_observation=variance_obs,
            recommendation=recommendation,
            evidence_summary={
                "delta": delta,
                "direction": direction,
                "current_score": current_score,
                "baseline_score": baseline_score,
            },
        )

    def _analyze_with_llm(
        self, context: ContextPayload, latest_result: dict[str, Any] | None = None
    ) -> EvaluationFeedback:
        """Query LLM provider using structured prompt."""
        prompt = (
            f"Evaluate the experiment outcomes against baseline:\n"
            f"Target Metric: {context.target_metric} ({context.metric_direction})\n"
            f"Leaderboard: {json.dumps(context.leaderboard)}\n"
            f"Latest Result: {json.dumps(latest_result or {})}\n"
        )
        system_prompt = (
            "You are the Critic specialist in CATML. Objectively evaluate performance improvements. "
            "Report overfitting or high variance only when justified by empirical evidence. Output valid JSON."
        )
        schema = {
            "type": "object",
            "properties": {
                "is_improvement": {"type": "boolean"},
                "recommendation": {"type": "string"},
                "variance_observation": {"type": "string"},
            },
            "required": ["is_improvement", "recommendation", "variance_observation"],
        }
        assert self._llm_provider is not None
        response = self._llm_provider.generate(
            prompt=prompt, system_prompt=system_prompt, response_schema=schema
        )
        data = response.parsed or json.loads(response.content)
        return EvaluationFeedback(
            feedback_id=f"feed-crit-{uuid4().hex[:8]}",
            run_id=context.run_id,
            specialist_name=self.name,
            is_improvement=bool(data["is_improvement"]),
            metric_name=context.target_metric,
            variance_observation=data["variance_observation"],
            recommendation=data["recommendation"],
            evidence_summary={},
        )
