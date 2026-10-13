"""CQRS commands for Data Preparation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AnalyzeDataQualityCommand:
    dataset_id: str


@dataclass(frozen=True)
class SuggestPreparationPlanCommand:
    dataset_id: str
    name: str | None = None
    include_pipeline_marks: bool = True


@dataclass(frozen=True)
class PreviewPreparationPlanCommand:
    plan_id: str


@dataclass(frozen=True)
class ValidatePreparationPlanCommand:
    plan_id: str


@dataclass(frozen=True)
class ApplyPreparationPlanCommand:
    """Apply dataset-scope steps to a new version. Pipeline steps recorded only."""

    plan_id: str
    approved: bool = False
    notes: str | None = None


@dataclass(frozen=True)
class CreatePreparationPlanCommand:
    """Create an explicit plan from step dicts (CLI/tests)."""

    dataset_id: str
    steps: tuple[dict[str, Any], ...] = ()
    name: str | None = None
    notes: str | None = None
