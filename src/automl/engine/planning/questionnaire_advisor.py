"""Questionnaire advisor service that infers problem context via heuristics or LLM semantic reasoning."""
from __future__ import annotations

import json
import re
from typing import Any

from automl.domain.tasks.questionnaire import (
    DatasetQuestionnaire,
    ErrorCostPriority,
    ExplainabilityLevel,
    LatencyConstraint,
    TemporalStructure,
)


class QuestionnaireAdvisor:
    """Infers problem context and framing recommendations from dataset metadata and profiles."""

    def __init__(self, llm_provider: Any = None) -> None:
        self._llm_provider = llm_provider

    def infer(
        self,
        dataset_id: str,
        dataset_name: str,
        column_names: list[str],
        target_column: str,
        profile_summary: dict[str, Any] | None = None,
    ) -> DatasetQuestionnaire:
        """Synthesize a framing questionnaire using LLM if available, falling back to heuristics."""
        if self._llm_provider is not None:
            try:
                return self._infer_with_llm(
                    dataset_id=dataset_id,
                    dataset_name=dataset_name,
                    column_names=column_names,
                    target_column=target_column,
                    profile_summary=profile_summary,
                )
            except Exception:
                # Deterministic fallback if LLM provider fails
                pass

        return self._infer_heuristic(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            column_names=column_names,
            target_column=target_column,
            profile_summary=profile_summary,
        )

    def _infer_heuristic(
        self,
        dataset_id: str,
        dataset_name: str,
        column_names: list[str],
        target_column: str,
        profile_summary: dict[str, Any] | None = None,
    ) -> DatasetQuestionnaire:
        """Deterministic heuristic inference based on semantic tokens in dataset and column names."""
        name_lower = dataset_name.lower()
        cols_lower = [c.lower() for c in column_names]
        target_lower = target_column.lower()

        # Defaults
        domain_hint = "general_tabular"
        error_cost = ErrorCostPriority.BALANCED
        temporal = TemporalStructure.CROSS_SECTIONAL
        latency = LatencyConstraint.STANDARD_INTERACTIVE
        explainability = ExplainabilityLevel.MODERATE
        imbalance = "auto"
        primary_metric: str | None = None
        split_strategy = "stratified_kfold"
        reasons: list[str] = []

        # 1. Churn / Customer Retention
        if any(term in name_lower or any(term in c for c in cols_lower) for term in ("churn", "attrition", "tenure", "contract", "retention")):
            domain_hint = "customer_churn"
            error_cost = ErrorCostPriority.AVOID_FALSE_NEGATIVES
            primary_metric = "pr_auc"
            reasons.append("Subscription churn detected: Missed churners incur severe acquisition loss. PR-AUC prioritized.")

        # 2. Fraud / Anomaly Detection
        elif any(term in name_lower or any(term in c for c in cols_lower) for term in ("fraud", "chargeback", "transaction", "breach", "anomaly")):
            domain_hint = "fraud_detection"
            error_cost = ErrorCostPriority.AVOID_FALSE_NEGATIVES
            primary_metric = "pr_auc"
            imbalance = "balanced_weights"
            latency = LatencyConstraint.ULTRA_LOW_REALTIME
            reasons.append("Transaction fraud detected: High cost of missed fraudulent events with sub-second real-time latency.")

        # 3. Medical / Healthcare
        elif any(term in name_lower or any(term in c for c in cols_lower) for term in ("patient", "diagnosis", "disease", "cancer", "clinical", "hospital")):
            domain_hint = "healthcare_diagnosis"
            error_cost = ErrorCostPriority.AVOID_FALSE_NEGATIVES
            explainability = ExplainabilityLevel.HIGHLY_REGULATED
            primary_metric = "recall"
            reasons.append("Medical diagnosis context: Minimizing false negatives is critical for patient safety with auditable boundaries.")

        # 4. Credit Scoring / Financial Risk
        elif any(term in name_lower or any(term in c for c in cols_lower) for term in ("credit", "loan", "default", "borrower", "debt")):
            domain_hint = "credit_risk"
            explainability = ExplainabilityLevel.HIGHLY_REGULATED
            error_cost = ErrorCostPriority.AVOID_FALSE_NEGATIVES
            primary_metric = "roc_auc"
            reasons.append("Credit lending context: Monotonic, explainable risk models required for regulatory compliance.")

        # 5. Temporal / Sequential Structure Detection
        temporal_cols = [c for c in cols_lower if re.search(r"(date|time|timestamp|year|month|day|hour|sensor|step)", c)]
        if temporal_cols or any(term in name_lower for term in ("time", "temporal", "sensor", "rainfall", "weather")):
            temporal = TemporalStructure.SEQUENTIAL_TIME_SERIES
            split_strategy = "time_series_split"
            reasons.append(f"Sequential ordering detected in features ({temporal_cols[:2]}). TimeSeriesSplit recommended to avoid look-ahead leakage.")

        # 6. Grouped Cohort Detection
        group_cols = [c for c in cols_lower if re.search(r"(customer_id|client_id|patient_id|user_id|session_id|group_id)", c)]
        if group_cols and temporal != TemporalStructure.SEQUENTIAL_TIME_SERIES:
            temporal = TemporalStructure.GROUPED_COHORTS
            split_strategy = "group_kfold"
            reasons.append(f"Grouped subject IDs detected ({group_cols[0]}). GroupKFold recommended to evaluate out-of-subject generalization.")

        # 7. Imbalance check from profile
        if profile_summary:
            target_rate = profile_summary.get("target_rate")
            if target_rate is not None and (target_rate < 0.10 or target_rate > 0.90):
                imbalance = "balanced_weights"
                if primary_metric is None:
                    primary_metric = "pr_auc"
                reasons.append(f"Target distribution is imbalanced ({target_rate:.1%}). Class weighting applied.")

        if not reasons:
            reasons.append("Standard tabular classification/regression with balanced default trade-offs.")

        return DatasetQuestionnaire(
            dataset_id=dataset_id,
            domain_hint=domain_hint,
            error_cost=error_cost,
            temporal_structure=temporal,
            latency_constraint=latency,
            explainability=explainability,
            imbalance_strategy=imbalance,
            primary_metric_override=primary_metric,
            split_strategy_recommendation=split_strategy,
            auto_generated=True,
            confidence=0.90,
            reasoning=" ".join(reasons),
        )

    def _infer_with_llm(
        self,
        dataset_id: str,
        dataset_name: str,
        column_names: list[str],
        target_column: str,
        profile_summary: dict[str, Any] | None = None,
    ) -> DatasetQuestionnaire:
        """Synthesize questionnaire using LLM semantic understanding."""
        prompt = (
            f"Analyze the dataset metadata and provide the optimal problem context questionnaire:\n"
            f"Dataset Name: {dataset_name}\n"
            f"Target Column: {target_column}\n"
            f"Columns: {', '.join(column_names[:30])}\n"
            f"Profile: {json.dumps(profile_summary or {})}\n"
        )
        system_prompt = (
            "You are the Dataset Framing Specialist in CATML. Classify the problem domain, error cost priority, "
            "temporal structure, latency constraints, explainability level, and recommended validation metric. "
            "Output strictly valid JSON matching the schema."
        )
        schema = {
            "type": "object",
            "properties": {
                "domain_hint": {"type": "string"},
                "error_cost": {
                    "type": "string",
                    "enum": ["balanced", "avoid_false_negatives", "avoid_false_positives", "custom_cost_matrix"],
                },
                "temporal_structure": {
                    "type": "string",
                    "enum": ["cross_sectional", "sequential_time_series", "grouped_cohorts", "unknown"],
                },
                "latency_constraint": {
                    "type": "string",
                    "enum": ["ultra_low_realtime", "standard_interactive", "batch_offline"],
                },
                "explainability": {
                    "type": "string",
                    "enum": ["highly_regulated", "moderate", "performance_first"],
                },
                "imbalance_strategy": {"type": "string"},
                "primary_metric_override": {"type": ["string", "null"]},
                "split_strategy_recommendation": {"type": "string"},
                "confidence": {"type": "number"},
                "reasoning": {"type": "string"},
            },
            "required": ["domain_hint", "error_cost", "temporal_structure", "split_strategy_recommendation"],
        }

        resp = self._llm_provider.generate(
            prompt=prompt, system_prompt=system_prompt, response_schema=schema
        )
        data = resp.parsed or json.loads(resp.content)
        data["dataset_id"] = dataset_id
        return DatasetQuestionnaire.from_dict(data)
