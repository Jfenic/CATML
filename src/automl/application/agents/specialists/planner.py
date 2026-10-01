"""Planner specialist responsible for formulating scientific experiment proposals."""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from automl.application.agents.contracts import CandidateProposal, ContextPayload
from automl.application.agents.ports import LLMProviderPort, SpecialistPort


class Planner:
    """Specialist that proposes experiment and hyperparameter configurations."""

    def __init__(self, llm_provider: LLMProviderPort | None = None) -> None:
        self._llm_provider = llm_provider

    @property
    def name(self) -> str:
        return "planner"

    def analyze(
        self, context: ContextPayload, **kwargs: Any
    ) -> CandidateProposal:
        """Analyze context payload and propose the next experiment or trial."""
        if self._llm_provider is not None:
            try:
                return self._analyze_with_llm(context)
            except Exception:
                # Deterministic fallback if LLM provider fails or raises
                pass

        return self._analyze_deterministically(context)

    def _analyze_deterministically(self, context: ContextPayload) -> CandidateProposal:
        """Deterministic rule-based planning conforming to Propose != Accept."""
        tested_models = {
            entry.get("model_id")
            for entry in context.leaderboard
            if entry.get("model_id")
        }

        # 1. Baseline proposal
        if not tested_models:
            model_id = (
                "logistic_regression"
                if context.task_type != "regression"
                else "linear_regression"
            )
            return CandidateProposal(
                proposal_id=f"prop-plan-{uuid4().hex[:8]}",
                specialist_name=self.name,
                run_id=context.run_id,
                hypothesis=f"Establish initial baseline score using default {model_id}.",
                action_type="create_experiment",
                action_payload={"model_id": model_id, "hyperparameters": {}},
                verification_plan={
                    "metric": context.target_metric,
                    "direction": context.metric_direction,
                    "target": "establish_baseline",
                },
                estimated_cost={"trials": 1, "duration_seconds": 15.0},
                confidence=0.95,
            )

        # 2. Next family proposal (decision trees / random forest)
        if "random_forest" not in tested_models:
            return CandidateProposal(
                proposal_id=f"prop-plan-{uuid4().hex[:8]}",
                specialist_name=self.name,
                run_id=context.run_id,
                hypothesis="Evaluate non-linear interactions using an ensemble of decision trees (RandomForest).",
                action_type="create_experiment",
                action_payload={
                    "model_id": "random_forest",
                    "hyperparameters": {"n_estimators": 100},
                },
                verification_plan={
                    "metric": context.target_metric,
                    "direction": context.metric_direction,
                    "compare_to_baseline": True,
                },
                estimated_cost={"trials": 1, "duration_seconds": 30.0},
                confidence=0.90,
            )

        # 3. Gradient boosting family (LightGBM)
        if "lightgbm" not in tested_models:
            return CandidateProposal(
                proposal_id=f"prop-plan-{uuid4().hex[:8]}",
                specialist_name=self.name,
                run_id=context.run_id,
                hypothesis="Test histogram-based gradient boosting (LightGBM) to capture complex patterns with fast convergence.",
                action_type="create_experiment",
                action_payload={
                    "model_id": "lightgbm",
                    "hyperparameters": {"learning_rate": 0.05, "n_estimators": 100},
                },
                verification_plan={
                    "metric": context.target_metric,
                    "direction": context.metric_direction,
                    "compare_to_baseline": True,
                },
                estimated_cost={"trials": 1, "duration_seconds": 25.0},
                confidence=0.88,
            )

        # 4. Hyperparameter tuning proposal on current best model
        best_entry = context.leaderboard[0]
        best_model = best_entry.get("model_id", "random_forest")
        return CandidateProposal(
            proposal_id=f"prop-plan-{uuid4().hex[:8]}",
            specialist_name=self.name,
            run_id=context.run_id,
            hypothesis=f"Tune hyperparameters of current best model ({best_model}) via Bayesian TPE search to maximize {context.target_metric}.",
            action_type="tune_hyperparameters",
            action_payload={"model_id": best_model, "n_trials": 10},
            verification_plan={
                "metric": context.target_metric,
                "direction": context.metric_direction,
                "baseline_score": best_entry.get("score"),
            },
            estimated_cost={"trials": 10, "duration_seconds": 120.0},
            confidence=0.85,
        )

    def _analyze_with_llm(self, context: ContextPayload) -> CandidateProposal:
        """Query LLM provider using structured prompt and response schema."""
        prompt = (
            f"Analyze the following AutoML experiment state and propose the next scientific experiment:\n"
            f"Run ID: {context.run_id}\n"
            f"Task: {context.task_type}\n"
            f"Target Metric: {context.target_metric} ({context.metric_direction})\n"
            f"Leaderboard: {json.dumps(context.leaderboard)}\n"
            f"Profile: {json.dumps(context.dataset_profile)}\n"
        )
        system_prompt = (
            "You are the Planner specialist in CATML. Propose a concrete next experiment candidate. "
            "Never execute mutating actions. Output valid JSON matching the candidate proposal schema."
        )
        schema = {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "action_type": {"type": "string"},
                "model_id": {"type": "string"},
                "hyperparameters": {"type": "object"},
                "estimated_trials": {"type": "integer"},
            },
            "required": ["hypothesis", "action_type", "model_id"],
        }
        assert self._llm_provider is not None
        response = self._llm_provider.generate(
            prompt=prompt, system_prompt=system_prompt, response_schema=schema
        )
        data = response.parsed or json.loads(response.content)
        return CandidateProposal(
            proposal_id=f"prop-plan-{uuid4().hex[:8]}",
            specialist_name=self.name,
            run_id=context.run_id,
            hypothesis=data["hypothesis"],
            action_type=data["action_type"],
            action_payload={
                "model_id": data["model_id"],
                "hyperparameters": data.get("hyperparameters", {}),
            },
            verification_plan={
                "metric": context.target_metric,
                "direction": context.metric_direction,
            },
            estimated_cost={"trials": data.get("estimated_trials", 1)},
            confidence=0.85,
        )
