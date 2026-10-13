"""Scope and plan validation (pure domain)."""
from __future__ import annotations

from automl.domain.preparation.models import (
    LEARNING_OPERATIONS,
    PreparationOperation,
    PreparationPlan,
    PreparationScope,
    PreparationStep,
)


class PreparationValidationError(ValueError):
    """Illegal preparation plan or step."""


def assert_step_scope_legal(step: PreparationStep) -> None:
    """Reject learning ops on dataset scope (anti-leakage)."""
    if step.operation in LEARNING_OPERATIONS and step.scope != PreparationScope.TRAINING_PIPELINE:
        raise PreparationValidationError(
            f"Operation '{step.operation.value}' learns from data and must use "
            f"scope=training_pipeline (got scope={step.scope.value})."
        )
    if step.operation == PreparationOperation.DEDUPE:
        policy = (step.params or {}).get("policy")
        if policy not in {"keep_first", "drop", "flag_only"}:
            raise PreparationValidationError(
                "dedupe requires params.policy in {'keep_first', 'drop', 'flag_only'}"
            )


def validate_plan(plan: PreparationPlan) -> list[str]:
    """
    Validate all steps. Returns warnings (non-fatal).
    Raises PreparationValidationError on illegal scopes/ops.
    """
    if not plan.steps:
        raise PreparationValidationError("PreparationPlan must contain at least one step.")
    warnings: list[str] = []
    for step in plan.steps:
        assert_step_scope_legal(step)
        if step.scope == PreparationScope.DATASET and step.operation in LEARNING_OPERATIONS:
            # unreachable if assert_step_scope_legal holds; kept for clarity
            raise PreparationValidationError("Illegal learning op on dataset scope")
        if step.operation == PreparationOperation.DROP_COLUMNS and not step.columns:
            raise PreparationValidationError("drop_columns requires columns")
        if step.operation == PreparationOperation.KEEP_COLUMNS and not step.columns:
            raise PreparationValidationError("keep_columns requires columns")
        if (
            step.scope == PreparationScope.TRAINING_PIPELINE
            and step.operation
            in {
                PreparationOperation.DROP_COLUMNS,
                PreparationOperation.KEEP_COLUMNS,
                PreparationOperation.DEDUPE,
            }
        ):
            warnings.append(
                f"Step {step.step_id}: {step.operation.value} on training_pipeline "
                "is unusual; prefer dataset scope for structural edits."
            )
    return warnings
