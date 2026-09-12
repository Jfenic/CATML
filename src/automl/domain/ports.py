from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from automl.domain.datasets.profile import DatasetProfile
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import ExperimentPriority, Priority
from automl.domain.experiments.trial import Experiment, Trial, TrialResult
from automl.domain.features.registry import FeatureRegistry
from automl.domain.models.registry import ModelRegistry
from automl.domain.optimization.search_space import SearchSpace
from automl.domain.runs.run import AutoMLRun



@dataclass
class TrialExecution:
    trial: Trial
    experiment: Experiment
    run: AutoMLRun
    feature_names: list[str]
    dataset_path: str
    target_column: str
    task_type: str
    metric: str
    validation_strategy: str
    test_size: float
    cv_folds: int
    random_seed: int


class TrainerPort(Protocol):
    def run(self, execution: TrialExecution) -> TrialResult:
        ...


class EvaluatorPort(Protocol):
    def evaluate(self, execution: TrialExecution, predictions: Any, y_true: Any) -> dict[str, float]:
        ...


class ExperimentRepositoryPort(Protocol):
    def save_run(self, run: AutoMLRun) -> None:
        ...

    def get_run(self, run_id: str) -> AutoMLRun | None:
        ...

    def save_dataset_profile(self, profile: DatasetProfile) -> None:
        ...

    def get_dataset_profile(self, dataset_id: str) -> DatasetProfile | None:
        ...

    def save_experiment(self, experiment: Experiment) -> None:
        ...

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        ...

    def list_experiments(self, run_id: str) -> list[Experiment]:
        ...

    def save_trial(self, trial: Trial) -> None:
        ...

    def get_trial(self, trial_id: str) -> Trial | None:
        ...

    def save_trial_result(self, result: TrialResult) -> None:

        ...

    def list_trial_results(self, experiment_id: str) -> list[TrialResult]:
        ...

    def get_leaderboard(self, run_id: str) -> list[TrialResult]:
        ...


class ExperimentPlannerPort(Protocol):
    def propose(
        self,
        run: AutoMLRun,
        profile: DatasetProfile,
        feature_registry: FeatureRegistry,
        model_registry: ModelRegistry,
        history: list[TrialResult] | None = None,
    ) -> list[ExperimentCandidate]:
        ...


class PriorityScorerPort(Protocol):
    def score(
        self,
        candidate: ExperimentCandidate,
        run: AutoMLRun,
        profile: DatasetProfile,
        user_priority: ExperimentPriority | str | None = None,
    ) -> Priority:
        ...


class OptimizerPort(Protocol):
    def suggest(self, trial_number: int, search_space: SearchSpace) -> dict[str, Any]:
        ...

    def observe(
        self,
        trial_number: int,
        parameters: dict[str, Any],
        score: float,
        succeeded: bool = True,
    ) -> None:
        ...

    def should_stop(self) -> bool:
        ...

    def best_parameters(self) -> dict[str, Any]:
        ...

    def best_score(self) -> float:
        ...

