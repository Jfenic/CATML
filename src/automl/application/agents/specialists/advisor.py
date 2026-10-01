"""Feature Advisor specialist responsible for proposing feature engineering candidates and detecting leakage."""
from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from automl.application.agents.contracts import CandidateProposal, ContextPayload
from automl.application.agents.ports import LLMProviderPort, SpecialistPort


class FeatureAdvisor:
    """Specialist that analyzes dataset profiles to propose feature candidates and flag data leakage."""

    def __init__(self, llm_provider: LLMProviderPort | None = None) -> None:
        self._llm_provider = llm_provider

    @property
    def name(self) -> str:
        return "feature_advisor"

    def analyze(
        self, context: ContextPayload, **kwargs: Any
    ) -> CandidateProposal:
        """Analyze dataset profile and feature evidence to propose feature sets or flag leakage."""
        if self._llm_provider is not None:
            try:
                return self._analyze_with_llm(context)
            except Exception:
                pass

        return self._analyze_deterministically(context)

    def _analyze_deterministically(self, context: ContextPayload) -> CandidateProposal:
        """Deterministic feature proposal conforming to Propose != Accept."""
        profile = context.dataset_profile
        numeric_cols = profile.get("numeric_columns", [])
        categorical_cols = profile.get("categorical_columns", [])
        target_col = profile.get("target_column", "").lower()

        # 1. Leakage detection heuristics
        suspicious_leakage_cols = [
            col for col in numeric_cols + categorical_cols
            if col.lower() != target_col and (
                col.lower() in ("target", "label", "outcome", "ground_truth", "is_churned")
                or target_col and target_col in col.lower() and len(col) > len(target_col)
            )
        ]
        if suspicious_leakage_cols:
            return CandidateProposal(
                proposal_id=f"prop-adv-{uuid4().hex[:8]}",
                specialist_name=self.name,
                run_id=context.run_id,
                hypothesis=(
                    f"Warning: Potential target leakage detected in feature(s): {suspicious_leakage_cols}. "
                    "Propose excluding these features from training to prevent optimistic evaluation bias."
                ),
                action_type="exclude_features",
                action_payload={"excluded_features": suspicious_leakage_cols},
                verification_plan={
                    "metric": context.target_metric,
                    "direction": context.metric_direction,
                    "target": "verify_leakage_removal",
                },
                estimated_cost={"trials": 1, "duration_seconds": 10.0},
                confidence=0.92,
            )

        # 2. Existing feature sets inspection
        existing_set_names = {fset.get("name") for fset in context.feature_evidence if fset.get("name")}

        # 3. Numeric interactions proposal (differences and ratios)
        if len(numeric_cols) >= 2 and "interactions_differences" not in existing_set_names:
            pairs = []
            for i in range(min(len(numeric_cols), 4)):
                for j in range(i + 1, min(len(numeric_cols), 4)):
                    pairs.append(f"inter_diff_{numeric_cols[i]}_minus_{numeric_cols[j]}")

            return CandidateProposal(
                proposal_id=f"prop-adv-{uuid4().hex[:8]}",
                specialist_name=self.name,
                run_id=context.run_id,
                hypothesis=(
                    "Propose pairwise numerical differences to capture relative spread and contrast "
                    f"between top numeric features: {numeric_cols[:4]}."
                ),
                action_type="propose_feature_set",
                action_payload={
                    "feature_set_name": "interactions_differences",
                    "transformation": "difference",
                    "candidate_features": pairs,
                    "source_columns": numeric_cols[:4],
                },
                verification_plan={
                    "metric": context.target_metric,
                    "direction": context.metric_direction,
                    "ablation_comparison": "baseline",
                },
                estimated_cost={"trials": 1, "duration_seconds": 20.0},
                confidence=0.86,
            )

        # 4. Categorical frequency / target encoding proposal
        if categorical_cols and "categorical_frequency_encoding" not in existing_set_names:
            return CandidateProposal(
                proposal_id=f"prop-adv-{uuid4().hex[:8]}",
                specialist_name=self.name,
                run_id=context.run_id,
                hypothesis=(
                    f"Propose frequency encoding for categorical columns ({categorical_cols[:3]}) "
                    "to preserve cardinality distribution without high dimensionality expansion."
                ),
                action_type="propose_feature_set",
                action_payload={
                    "feature_set_name": "categorical_frequency_encoding",
                    "transformation": "frequency_encoding",
                    "source_columns": categorical_cols[:3],
                },
                verification_plan={
                    "metric": context.target_metric,
                    "direction": context.metric_direction,
                    "ablation_comparison": "baseline",
                },
                estimated_cost={"trials": 1, "duration_seconds": 20.0},
                confidence=0.82,
            )

        # 5. Default feature scaling/normalization proposal
        return CandidateProposal(
            proposal_id=f"prop-adv-{uuid4().hex[:8]}",
            specialist_name=self.name,
            run_id=context.run_id,
            hypothesis="Evaluate robust standardization for numeric variables to mitigate outlier sensitivity.",
            action_type="propose_feature_set",
            action_payload={
                "feature_set_name": "robust_scaled_features",
                "transformation": "robust_scaler",
                "source_columns": numeric_cols,
            },
            verification_plan={
                "metric": context.target_metric,
                "direction": context.metric_direction,
                "ablation_comparison": "baseline",
            },
            estimated_cost={"trials": 1, "duration_seconds": 15.0},
            confidence=0.80,
        )

    def _analyze_with_llm(self, context: ContextPayload) -> CandidateProposal:
        """Query LLM provider using structured prompt."""
        prompt = (
            f"Analyze dataset profile and suggest feature engineering:\n"
            f"Profile: {json.dumps(context.dataset_profile)}\n"
            f"Feature Evidence: {json.dumps(context.feature_evidence)}\n"
        )
        system_prompt = (
            "You are the Feature Advisor in CATML. Propose candidate features or warn of leakage. "
            "Never execute mutating actions. Output valid JSON matching the schema."
        )
        schema = {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "action_type": {"type": "string"},
                "feature_set_name": {"type": "string"},
                "source_columns": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["hypothesis", "action_type", "feature_set_name"],
        }
        assert self._llm_provider is not None
        response = self._llm_provider.generate(
            prompt=prompt, system_prompt=system_prompt, response_schema=schema
        )
        data = response.parsed or json.loads(response.content)
        return CandidateProposal(
            proposal_id=f"prop-adv-{uuid4().hex[:8]}",
            specialist_name=self.name,
            run_id=context.run_id,
            hypothesis=data["hypothesis"],
            action_type=data["action_type"],
            action_payload={
                "feature_set_name": data["feature_set_name"],
                "source_columns": data.get("source_columns", []),
            },
            verification_plan={
                "metric": context.target_metric,
                "direction": context.metric_direction,
            },
            estimated_cost={"trials": 1, "duration_seconds": 20.0},
            confidence=0.85,
        )
