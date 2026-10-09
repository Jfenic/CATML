"""CQRS commands for CATML Explore studies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CreateStudyCommand:
    """Command to configure and register an exploratory study."""
    dataset_id: str
    name: str = ""
    workspace_id: str = "default"
    target_column: str | None = None
    analysis_types: tuple[str, ...] = ("descriptive", "association")
    parameters: dict[str, Any] = field(default_factory=dict)
    time_budget_seconds: float | None = None


@dataclass(frozen=True)
class RunAnalysisCommand:
    """Command to execute analysis algorithms on a registered study."""
    study_id: str
    analysis_types: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ArchiveStudyCommand:
    """Command to archive an exploratory study."""
    study_id: str


@dataclass(frozen=True)
class CreateHypothesisCommand:
    """Command to propose an actionable ML hypothesis from a finding."""
    study_id: str
    finding_id: str
    description: str
    proposed_action: str
    experiment_delta: dict[str, Any] = field(default_factory=dict)
