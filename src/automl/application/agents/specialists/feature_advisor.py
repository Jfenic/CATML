"""Feature Advisor specialist allowing LLM agents and heuristics to propose and validate derived features."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from automl.domain.features.derived_feature import (
    DerivedFeatureDefinition,
    DerivedFeatureType,
    FeatureEvaluationResult,
)
from automl.engine.features.generation.derived_feature_engine import DerivedFeatureEngine


class FeatureAdvisor:
    """Specialist that proposes derived feature engineering hypotheses and validates them against data."""

    def __init__(self, llm_provider: Any = None) -> None:
        self.llm_provider = llm_provider
        self.engine = DerivedFeatureEngine()

    def propose_features(
        self,
        column_names: list[str],
        numeric_columns: list[str],
        target_column: str,
        domain_hint: str = "general_tabular",
        skewed_columns: list[str] | None = None,
    ) -> list[DerivedFeatureDefinition]:
        """Proposes derived feature candidates using LLM if available, falling back to heuristics."""
        if self.llm_provider is not None:
            try:
                return self._propose_with_llm(
                    column_names=column_names,
                    numeric_columns=numeric_columns,
                    target_column=target_column,
                    domain_hint=domain_hint,
                )
            except Exception:
                # Fallback to deterministic heuristics if LLM fails
                pass

        return self._propose_heuristic(
            numeric_columns=numeric_columns,
            target_column=target_column,
            domain_hint=domain_hint,
            skewed_columns=skewed_columns or [],
        )

    def _propose_heuristic(
        self,
        numeric_columns: list[str],
        target_column: str,
        domain_hint: str = "general_tabular",
        skewed_columns: list[str] | None = None,
    ) -> list[DerivedFeatureDefinition]:
        """Heuristic generation of derived feature formulas."""
        candidates: list[DerivedFeatureDefinition] = []
        valid_nums = [c for c in numeric_columns if c != target_column]

        # 1. Log transformations for highly skewed columns
        for col in (skewed_columns or []):
            if col in valid_nums:
                candidates.append(
                    DerivedFeatureDefinition(
                        name=f"log1p_{col}",
                        expression_type=DerivedFeatureType.FORMULA,
                        expression=f"log1p(clip({col}, 0, 1e9))",
                        description=f"Log1p transformation of skewed variable {col}",
                        source_columns=[col],
                        created_by="feature_advisor_heuristic",
                    )
                )

        # 2. Domain-informed heuristics
        hint_lower = domain_hint.lower()
        if "churn" in hint_lower:
            tenure_cols = [c for c in valid_nums if "tenure" in c.lower() or "month" in c.lower()]
            charge_cols = [c for c in valid_nums if "charge" in c.lower() or "amount" in c.lower() or "fee" in c.lower()]
            if tenure_cols and charge_cols:
                t_col, c_col = tenure_cols[0], charge_cols[0]
                candidates.append(
                    DerivedFeatureDefinition(
                        name=f"clv_est_{c_col}_x_{t_col}",
                        expression_type=DerivedFeatureType.FORMULA,
                        expression=f"{c_col} * {t_col}",
                        description="Estimated Customer Lifetime Value proxy",
                        source_columns=[c_col, t_col],
                        created_by="feature_advisor_heuristic",
                    )
                )
                candidates.append(
                    DerivedFeatureDefinition(
                        name=f"charge_per_tenure_{c_col}_div_{t_col}",
                        expression_type=DerivedFeatureType.FORMULA,
                        expression=f"{c_col} / ({t_col} + 1.0)",
                        description="Average charge rate per unit tenure",
                        source_columns=[c_col, t_col],
                        created_by="feature_advisor_heuristic",
                    )
                )

        elif "credit" in hint_lower or "fraud" in hint_lower:
            income_cols = [c for c in valid_nums if "income" in c.lower() or "salary" in c.lower()]
            debt_cols = [c for c in valid_nums if "debt" in c.lower() or "loan" in c.lower() or "amount" in c.lower()]
            if income_cols and debt_cols:
                inc, dbt = income_cols[0], debt_cols[0]
                candidates.append(
                    DerivedFeatureDefinition(
                        name=f"dti_ratio_{dbt}_div_{inc}",
                        expression_type=DerivedFeatureType.FORMULA,
                        expression=f"{dbt} / ({inc} + 1.0)",
                        description="Debt-to-income leverage ratio",
                        source_columns=[dbt, inc],
                        created_by="feature_advisor_heuristic",
                    )
                )

        # 3. Standard top numeric interaction if at least 2 numeric features
        if len(valid_nums) >= 2 and len(candidates) < 3:
            col_a, col_b = valid_nums[0], valid_nums[1]
            candidates.append(
                DerivedFeatureDefinition(
                    name=f"ratio_{col_a}_div_{col_b}",
                    expression_type=DerivedFeatureType.FORMULA,
                    expression=f"{col_a} / ({col_b} + 1e-6)",
                    description=f"Safe ratio interaction of {col_a} and {col_b}",
                    source_columns=[col_a, col_b],
                    created_by="feature_advisor_heuristic",
                )
            )

        return candidates

    def _propose_with_llm(
        self,
        column_names: list[str],
        numeric_columns: list[str],
        target_column: str,
        domain_hint: str,
    ) -> list[DerivedFeatureDefinition]:
        """Queries LLM provider with structured schema for high-value feature hypotheses."""
        prompt = (
            f"Propose 2 to 4 high-value derived features for tabular dataset:\n"
            f"Domain: {domain_hint}\n"
            f"Target Column: {target_column}\n"
            f"Numeric Columns: {', '.join(numeric_columns[:25])}\n"
            f"All Columns: {', '.join(column_names[:30])}\n"
            "Use either math formulas (e.g. `col_a / (col_b + 1e-5)` or `log1p(col)`) "
            "or concise Python functions with signature `def compute_feature(df):`."
        )
        system_prompt = (
            "You are the Feature Engineering Specialist in CATML. Output valid JSON array of feature proposals."
        )
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "expression_type": {"type": "string", "enum": ["formula", "python_code"]},
                    "expression": {"type": "string"},
                    "description": {"type": "string"},
                    "source_columns": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "expression_type", "expression", "description"],
            },
        }

        resp = self.llm_provider.generate(
            prompt=prompt, system_prompt=system_prompt, response_schema=schema
        )
        raw_items = resp.parsed or json.loads(resp.content)
        proposals: list[DerivedFeatureDefinition] = []
        for item in raw_items:
            item["created_by"] = "llm_agent"
            proposals.append(DerivedFeatureDefinition.from_dict(item))
        return proposals

    def validate_and_filter(
        self,
        df: pd.DataFrame,
        proposals: list[DerivedFeatureDefinition],
    ) -> tuple[list[DerivedFeatureDefinition], list[FeatureEvaluationResult]]:
        """
        Validates all candidate proposals against dataset.
        Discards invalid proposals, zero-variance (constant) columns, or columns with > 50% NaNs.
        """
        valid_definitions: list[DerivedFeatureDefinition] = []
        results: list[FeatureEvaluationResult] = []

        for prop in proposals:
            res, _ = self.engine.evaluate(df, prop)
            results.append(res)
            # Accept only robust features
            if res.is_valid and not res.is_constant and res.null_percentage <= 0.50:
                valid_definitions.append(prop)

        return valid_definitions, results
