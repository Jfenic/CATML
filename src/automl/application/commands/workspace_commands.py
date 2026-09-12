from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from automl.domain.policies.budget import BudgetPolicy


@dataclass(frozen=True)
class AddModelCommand:
    run_id: str
    model_id: str


@dataclass(frozen=True)
class ExcludeModelCommand:
    run_id: str
    model_id: str


@dataclass(frozen=True)
class ExcludeFeatureCommand:
    dataset_id: str
    feature_name: str
    run_id: str | None = None


@dataclass(frozen=True)
class PrioritizeFeatureCommand:
    dataset_id: str
    feature_name: str
    score: float = 1.0
    run_id: str | None = None


@dataclass(frozen=True)
class CreateFeatureSetCommand:
    dataset_id: str
    name: str
    feature_names: list[str]
    lineage: str = ""


@dataclass(frozen=True)
class CreateExperimentCommand:
    run_id: str
    name: str
    feature_names: list[str] | None = None
    feature_set_id: str | None = None
    model_ids: list[str] | None = None
    hypothesis: str = ""
    priority: str = "normal"


@dataclass(frozen=True)
class RunExperimentCommand:
    run_id: str
    experiment_id: str


@dataclass(frozen=True)
class PauseRunCommand:
    run_id: str


@dataclass(frozen=True)
class ResumeRunCommand:
    run_id: str


@dataclass(frozen=True)
class CancelRunCommand:
    run_id: str


@dataclass(frozen=True)
class CloneRunCommand:
    run_id: str
    new_name: str | None = None


@dataclass(frozen=True)
class PlanExperimentsCommand:
    run_id: str
    auto_enqueue: bool = True
    user_priorities: dict[str, str] | None = None


@dataclass(frozen=True)
class PrioritizeCandidateCommand:
    run_id: str
    candidate_id: str
    priority: str


@dataclass(frozen=True)
class ExecuteNextExperimentCommand:
    run_id: str
    budget: BudgetPolicy | None = None


@dataclass(frozen=True)
class RunScheduledExperimentsCommand:
    run_id: str
    max_experiments: int | None = None
    max_trials: int | None = None
    budget: BudgetPolicy | None = None

