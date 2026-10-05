from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from automl import __version__
from automl.domain.datasets.profile import Dataset
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import ExperimentPriority, Priority
from automl.domain.experiments.trial import Experiment, ExperimentStatus, Trial, TrialResult, TrialStatus
from automl.domain.features.evidence import FeatureEvidence, FeatureInteractionEvidence
from automl.domain.features.feature import Feature
from automl.domain.features.feature_set import FeatureSet
from automl.domain.features.registry import FeatureRegistry
from automl.domain.features.selection_strategy import (
    FeatureRank,
    FeatureSelectionStrategy,
    FeatureSetCandidate,
)
from automl.domain.models.registry import ModelRegistry
from automl.domain.optimization.budget import OptimizationBudget
from automl.domain.optimization.search_space import SearchSpace
from automl.domain.policies.budget import BudgetPolicy
from automl.domain.ports import (
    ExperimentPlannerPort,
    FeatureAnalysisContext,
    FeatureSelectorPort,
    OptimizerPort,
    PriorityScorerPort,
    TrialExecution,
)
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus
from automl.domain.tasks.problem_definition import ProblemDefinition
from automl.domain.tasks.task_type import TASK_CATALOG, TaskType, default_metric_for
from automl.engine.features.reduction.pca import PCAReducer
from automl.engine.features.selection.correlation import CorrelationSelector
from automl.engine.features.selection.ensemble import EnsembleRankSelector
from automl.engine.features.selection.importance import TreeImportanceSelector
from automl.engine.features.selection.mutual_information import MutualInformationSelector
from automl.engine.features.selection.variance import VarianceSelector
from automl.engine.optimization.early_stopping import EarlyStoppingPolicy
from automl.engine.optimization.random_search import RandomSearchOptimizer
from automl.engine.optimization.search_space_builder import SearchSpaceBuilder
from automl.engine.optimization.trial_factory import TrialFactory
from automl.engine.planning.ablation_planner import AblationPlanner
from automl.engine.planning.experiment_planner import RuleBasedExperimentPlanner
from automl.engine.planning.task_planner import plan_from_dataframe
from automl.engine.priority.scheduler import ExperimentQueue, Scheduler
from automl.application.plugins.registry import PluginRegistry
from automl.engine.priority.scorer import RuleBasedPriorityScorer
from automl.engine.profiling.dataset_profiler import infer_task_type, load_dataframe, profile_dataset
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository
from automl.domain.modalities.modality import Modality
from automl.domain.pipelines.graph import NodeType, PipelineGraph, PipelineNode
from automl.engine.pipeline.fusion import FeatureFusionNode
from automl.engine.pipeline.graph_validator import GraphValidator
from automl.engine.vision.image_encoder import ImageEncoderNode
from automl.plugins.metrics.business_metrics import CostSensitiveMetricPlugin, WeightedF1MetricPlugin
from automl.plugins.modalities.image_plugin import ImageModalityPlugin
from automl.plugins.models.catboost_plugin import CatBoostPlugin
from automl.plugins.models.ensemble import VotingEnsemblePlugin
from automl.plugins.models.gradient_boosting import LightGBMPlugin, XGBoostPlugin
from automl.plugins.models.sklearn_models import ExtraTreesPlugin, MLPPlugin, default_model_specs
from automl.plugins.models.sklearn_plugin import create_default_sklearn_plugins
from automl.plugins.models.vision_plugin import TimmVisionPlugin
from automl.plugins.optimizers.optuna_optimizer import OptunaOptimizer

PLATFORM_VERSION = __version__


def _init_default_plugins(workspace: AutoMLWorkspace) -> None:
    for p in create_default_sklearn_plugins():
        workspace.plugin_registry.register(p)
    workspace.plugin_registry.register(LightGBMPlugin())
    workspace.plugin_registry.register(XGBoostPlugin())
    workspace.plugin_registry.register(CatBoostPlugin())
    workspace.plugin_registry.register(ExtraTreesPlugin())
    workspace.plugin_registry.register(MLPPlugin())
    workspace.plugin_registry.register(VotingEnsemblePlugin())
    workspace.plugin_registry.register(CostSensitiveMetricPlugin())
    workspace.plugin_registry.register(WeightedF1MetricPlugin())
    workspace.plugin_registry.register(ImageModalityPlugin())
    workspace.plugin_registry.register(TimmVisionPlugin())


@dataclass
class AutoMLWorkspace:
    id: str
    root_dir: Path
    repository: SQLiteExperimentRepository
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)
    planner: ExperimentPlannerPort = field(default_factory=RuleBasedExperimentPlanner)
    scorer: PriorityScorerPort = field(default_factory=RuleBasedPriorityScorer)
    scheduler: Scheduler = field(default_factory=Scheduler)
    plugin_registry: PluginRegistry = field(default_factory=PluginRegistry)
    execution_check: Callable[[], None] | None = field(default=None, repr=False)
    execution_progress: Callable[[int, int, str], None] | None = field(default=None, repr=False)
    _runs: dict[str, AutoMLRun] = field(default_factory=dict)
    _datasets: dict[str, Dataset] = field(default_factory=dict)
    _feature_registries: dict[str, FeatureRegistry] = field(default_factory=dict)
    _queues: dict[str, ExperimentQueue] = field(default_factory=dict)
    _candidate_feature_sets: dict[str, list[FeatureSetCandidate]] = field(default_factory=dict)
    _feature_ranks: dict[str, list[FeatureRank]] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, root_dir: str | Path | None = None) -> AutoMLWorkspace:
        base = Path(root_dir or Path.cwd() / ".automl" / name)
        base.mkdir(parents=True, exist_ok=True)
        workspace_id = f"ws_{uuid.uuid4().hex[:8]}"
        repo = SQLiteExperimentRepository(base / "automl.db")
        workspace = cls(id=workspace_id, root_dir=base, repository=repo)
        for spec in default_model_specs():
            workspace.model_registry.register(spec)
        _init_default_plugins(workspace)
        return workspace

    @classmethod
    def load(cls, root_dir: str | Path) -> AutoMLWorkspace:
        base = Path(root_dir)
        repo = SQLiteExperimentRepository(base / "automl.db")
        datasets = repo.list_datasets()
        runs = repo.list_runs()
        workspace_id = (
            datasets[0].workspace_id
            if datasets
            else (runs[0].workspace_id if runs else f"ws_{uuid.uuid4().hex[:8]}")
        )
        workspace = cls(id=workspace_id, root_dir=base, repository=repo)
        for spec in default_model_specs():
            workspace.model_registry.register(spec)
        _init_default_plugins(workspace)
        for dataset in repo.list_datasets(workspace_id):
            workspace._datasets[dataset.id] = dataset
            workspace._hydrate_feature_registry(dataset.id)
        for run in repo.list_runs(workspace_id):
            workspace._runs[run.id] = run
        return workspace

    @classmethod
    def load_or_create(cls, name: str, root_dir: str | Path | None = None) -> AutoMLWorkspace:
        base = Path(root_dir or Path.cwd() / ".automl" / name)
        db_path = base / "automl.db"
        if db_path.exists():
            return cls.load(base)
        return cls.create(name, root_dir=base)

    def _hydrate_feature_registry(self, dataset_id: str) -> None:
        features = self.repository.list_features(dataset_id)
        profile = self.repository.get_dataset_profile(dataset_id)
        if features:
            if profile:
                id_names = set(profile.identifier_column_names)
                for f in features:
                    if f.semantic_type == "unknown" and f.name in id_names:
                        f.semantic_type = "identifier"
                        self.repository.save_feature(f)
            registry = FeatureRegistry(features)
        else:
            profile = self.repository.get_dataset_profile(dataset_id)
            if profile is None:
                return
            registry = FeatureRegistry()
            for column in profile.columns:
                if column.name == profile.target_column:
                    continue
                registry.register(
                    Feature(
                        id=f"feat_{column.name}",
                        dataset_id=dataset_id,
                        name=column.name,
                        physical_dtype=column.dtype,
                        semantic_type="identifier" if column.is_identifier else "unknown",
                    )
                )
        self._feature_registries[dataset_id] = registry

    def _emit(self, event_type: str, payload: dict, run_id: str | None = None) -> None:
        self.repository.append_event(event_type, payload, run_id=run_id)

    def _get_run(self, run_id: str) -> AutoMLRun:
        run = self._runs.get(run_id) or self.repository.get_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        self._runs[run_id] = run
        return run

    def _get_dataset(self, dataset_id: str) -> Dataset:
        dataset = self._datasets.get(dataset_id) or self.repository.get_dataset(dataset_id)
        if dataset is None:
            raise KeyError(f"Dataset not found: {dataset_id}")
        self._datasets[dataset_id] = dataset
        return dataset

    def register_dataset(
        self,
        name: str,
        path: str | Path,
        target: str,
        task_type: str | None = None,
    ) -> Dataset:
        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        resolved = str(Path(path).resolve())
        df = load_dataframe(resolved)
        if task_type:
            resolved_task = TaskType.parse(task_type)
        else:
            resolved_task = TaskType.parse(infer_task_type(df[target]))

        dataset = Dataset(
            id=dataset_id,
            workspace_id=self.id,
            name=name,
            path=resolved,
            target_column=target,
            task_type=resolved_task.value,
        )
        profile = profile_dataset(dataset)
        self._datasets[dataset_id] = dataset
        self.repository.save_dataset(dataset)
        self.repository.save_dataset_profile(profile)

        problem = plan_from_dataframe(dataset, df, task_type=resolved_task if task_type else None)
        self.repository.save_problem_definition(problem)

        registry = FeatureRegistry()
        for column in profile.columns:
            if column.name == target:
                continue
            feature = Feature(
                id=f"feat_{column.name}",
                dataset_id=dataset_id,
                name=column.name,
                physical_dtype=column.dtype,
                semantic_type="identifier" if column.is_identifier else "unknown",
            )
            registry.register(feature)
            self.repository.save_feature(feature)
        self._feature_registries[dataset_id] = registry
        self._emit("DatasetRegistered", {"dataset_id": dataset_id, "name": name})
        return dataset

    def create_run(
        self,
        dataset: Dataset,
        metric: str | None = None,
        validation_strategy: str | None = None,
        group_column: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> AutoMLRun:
        problem = self.repository.get_problem_definition(dataset.id)
        if problem is None:
            df = load_dataframe(dataset.path)
            problem = plan_from_dataframe(dataset, df)
            self.repository.save_problem_definition(problem)

        run_metric = metric or problem.default_metric
        strat = validation_strategy or ("group_kfold" if group_column else "holdout")
        config = RunConfig(
            task_type=problem.task_type.value,
            target=dataset.target_column,
            metric=run_metric,
            validation_strategy=strat,
            group_column=group_column,
            extra=dict(extra or {}),
        )
        run = AutoMLRun(
            id=f"run_{uuid.uuid4().hex[:8]}",
            workspace_id=self.id,
            dataset_id=dataset.id,
            config=config,
            status=RunStatus.PROFILING,
            current_phase=RunPhase.DATASET_PROFILING,
        )
        self._runs[run.id] = run
        self.repository.save_run(run)
        self._emit("RunCreated", {"run_id": run.id, "dataset_id": dataset.id}, run_id=run.id)
        return run

    def get_feature_registry(self, dataset_id: str) -> FeatureRegistry:
        if dataset_id not in self._feature_registries:
            self._hydrate_feature_registry(dataset_id)
        registry = self._feature_registries.get(dataset_id)
        if registry is None:
            raise KeyError(f"No feature registry for dataset {dataset_id}")
        return registry

    def create_feature_set(
        self,
        dataset_id: str,
        name: str,
        feature_names: list[str],
        lineage: str = "",
    ) -> FeatureSet:
        feature_set = FeatureSet(
            id=f"fset_{uuid.uuid4().hex[:8]}",
            dataset_id=dataset_id,
            name=name,
            feature_names=feature_names,
            lineage=lineage,
        )
        self.repository.save_feature_set(feature_set)
        self._emit("FeatureSetCreated", feature_set.to_dict())
        return feature_set

    def create_experiment(
        self,
        run: AutoMLRun,
        name: str,
        feature_names: list[str] | None = None,
        feature_set_id: str | None = None,
        model_ids: list[str] | None = None,
        hypothesis: str = "",
        priority: str = "normal",
        validation_strategy: str | None = None,
        group_column: str | None = None,
    ) -> Experiment:
        dataset = self._get_dataset(run.dataset_id)
        priority_value = ExperimentPriority.parse(priority).value

        resolved_features = feature_names
        if feature_set_id:
            feature_set = self.repository.get_feature_set(feature_set_id)
            if feature_set is None:
                raise KeyError(f"FeatureSet not found: {feature_set_id}")
            if feature_set.dataset_id != run.dataset_id:
                raise ValueError(
                    f"FeatureSet {feature_set_id} belongs to dataset {feature_set.dataset_id}, "
                    f"not run dataset {run.dataset_id}"
                )
            resolved_features = feature_set.feature_names
        if resolved_features is None:
            registry = self.get_feature_registry(run.dataset_id)
            resolved_features = registry.active_feature_names(dataset.target_column)
        if not resolved_features:
            raise ValueError("An experiment requires at least one feature")

        registry = self.get_feature_registry(run.dataset_id)
        known_features = {feature.name for feature in registry.list()}
        unknown_features = sorted(set(resolved_features) - known_features)
        if unknown_features:
            raise ValueError(f"Unknown features for dataset {run.dataset_id}: {', '.join(unknown_features)}")

        if model_ids:
            self.model_registry.validate_for_task(model_ids, run.config.task_type)

        active_models = model_ids or self.model_registry.resolve_active(
            run.config.models_include,
            run.config.models_exclude,
            task_type=run.config.task_type,
        )
        if not active_models:
            raise ValueError("An experiment requires at least one compatible model")

        resolved_group_col = group_column or run.config.group_column
        resolved_val_strat = validation_strategy or (
            "group_kfold" if (resolved_group_col and run.config.validation_strategy == "holdout")
            else run.config.validation_strategy
        )

        experiment = Experiment(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            run_id=run.id,
            name=name,
            hypothesis=hypothesis or f"Compare models on {', '.join(resolved_features[:5])}",
            feature_names=resolved_features,
            model_ids=active_models,
            metric=run.config.metric,
            validation_strategy=resolved_val_strat,
            priority=priority_value,
            feature_set_id=feature_set_id,
            group_column=resolved_group_col,
        )
        self.repository.save_experiment(experiment)
        self._emit(
            "ExperimentCreated",
            {"experiment_id": experiment.id, "name": name, "priority": priority_value},
            run_id=run.id,
        )
        return experiment

    def run_experiment(
        self,
        run: AutoMLRun,
        experiment: Experiment,
        start_index: int = 0,
        skip_model_ids: set[str] | None = None,
    ) -> list[TrialResult]:
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run.id} is cancelled")
        if experiment.run_id != run.id:
            raise ValueError(f"Experiment {experiment.id} does not belong to run {run.id}")

        dataset = self._get_dataset(run.dataset_id)
        experiment.status = ExperimentStatus.RUNNING
        self.repository.save_experiment(experiment)
        run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
        self.repository.save_run(run)

        trainer = SklearnTrainer(plugin_registry=self.plugin_registry)
        results: list[TrialResult] = []
        model_ids = experiment.model_ids
        if self.execution_progress:
            self.execution_progress(start_index, len(model_ids), "Training models")

        for index, model_id in enumerate(model_ids):
            if index < start_index:
                continue

            if skip_model_ids and model_id in skip_model_ids:
                if self.execution_progress:
                    self.execution_progress(index + 1, len(model_ids), f"Reused {model_id}")
                continue
            if self.execution_check:
                self.execution_check()
            refreshed = self.repository.get_run(run.id)
            if refreshed.status == RunStatus.PAUSED:
                self.repository.save_checkpoint(run.id, experiment.id, index)
                experiment.status = ExperimentStatus.PAUSED
                self.repository.save_experiment(experiment)
                self._emit("RunPaused", {"experiment_id": experiment.id, "next_index": index}, run_id=run.id)
                return results

            if refreshed.status == RunStatus.CANCELLED:
                experiment.status = ExperimentStatus.CANCELLED
                self.repository.save_experiment(experiment)
                self._emit("RunCancelled", {"experiment_id": experiment.id}, run_id=run.id)
                return results

            trial = Trial(
                id=f"trial_{uuid.uuid4().hex[:8]}",
                experiment_id=experiment.id,
                model_id=model_id,
                seed=run.config.random_seed,
            )
            self.repository.save_trial(trial)
            execution = TrialExecution(
                trial=trial,
                experiment=experiment,
                run=run,
                feature_names=experiment.feature_names,
                dataset_path=dataset.path,
                target_column=dataset.target_column,
                task_type=dataset.task_type,
                metric=experiment.metric,
                validation_strategy=experiment.validation_strategy,
                test_size=run.config.test_size,
                cv_folds=run.config.cv_folds,
                random_seed=run.config.random_seed,
                group_column=experiment.group_column,
            )
            result = trainer.run(execution)
            self.repository.save_trial(trial)
            self.repository.save_trial_result(result)
            results.append(result)
            if self.execution_progress:
                self.repository.save_checkpoint(run.id, experiment.id, index + 1)
                self.execution_progress(index + 1, len(model_ids), f"Finished {model_id}")
            self._emit(
                "TrialCompleted" if result.succeeded else "TrialFailed",
                {"trial_id": trial.id, "model_id": model_id, "score": result.primary_score},
                run_id=run.id,
            )

        if self.execution_check:
            self.execution_check()
        self.repository.clear_checkpoint(run.id)
        experiment.status = ExperimentStatus.COMPLETED
        self.repository.save_experiment(experiment)
        run.transition_to(RunStatus.COMPLETED, RunPhase.EVALUATION)
        self.repository.save_run(run)
        self._emit("ExperimentCompleted", {"experiment_id": experiment.id}, run_id=run.id)
        return results

    def resume_run(self, run_id: str) -> list[TrialResult]:
        run = self._get_run(run_id)
        if run.status not in {RunStatus.PAUSED, RunStatus.EXPERIMENTING}:
            raise RuntimeError(f"Run {run_id} is not resumable (status={run.status.value})")

        checkpoint = self.repository.get_checkpoint(run_id)
        if checkpoint is None:
            raise RuntimeError(f"No checkpoint found for run {run_id}")

        experiment_id, start_index = checkpoint
        experiment = self.repository.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment not found: {experiment_id}")

        run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
        self.repository.save_run(run)
        self._emit("RunResumed", {"experiment_id": experiment_id, "start_index": start_index}, run_id=run_id)
        return self.run_experiment(run, experiment, start_index=start_index)

    def pause_run(self, run_id: str) -> AutoMLRun:
        run = self._get_run(run_id)
        run.transition_to(RunStatus.PAUSED, RunPhase.EXPERIMENT_EXECUTION)
        self.repository.save_run(run)
        self._emit("RunPauseRequested", {}, run_id=run_id)
        return run

    def cancel_run(self, run_id: str) -> AutoMLRun:
        run = self._get_run(run_id)
        run.transition_to(RunStatus.CANCELLED, run.current_phase)
        self.repository.save_run(run)
        self.repository.clear_checkpoint(run_id)
        self._emit("RunCancelled", {}, run_id=run_id)
        return run

    def clone_run(self, run_id: str, new_name: str | None = None) -> AutoMLRun:
        source = self._get_run(run_id)
        dataset = self._get_dataset(source.dataset_id)
        cloned_config = RunConfig.from_dict(source.config.to_dict())
        cloned = AutoMLRun(
            id=f"run_{uuid.uuid4().hex[:8]}",
            workspace_id=self.id,
            dataset_id=source.dataset_id,
            config=cloned_config,
            status=RunStatus.CREATED,
            current_phase=RunPhase.EXPERIMENT_PLANNING,
        )
        self._runs[cloned.id] = cloned
        self.repository.save_run(cloned)
        self._emit(
            "RunCloned",
            {"source_run_id": run_id, "new_run_id": cloned.id, "name": new_name or cloned.id},
            run_id=cloned.id,
        )
        return cloned

    def leaderboard(self, run: AutoMLRun) -> list[dict]:
        rows = self.repository.get_leaderboard(run.id)
        return [
            {
                "trial_id": r.trial_id,
                "experiment_id": r.experiment_id,
                "model_id": r.model_id,
                "metric": r.primary_metric,
                "score": round(r.primary_score, 4),
                "training_time_s": round(r.training_time_seconds, 3),
                "failed": not r.succeeded,
            }
            for r in rows
        ]

    def compare_experiments(self, experiment_ids: list[str]) -> list[dict]:
        rows = self.repository.compare_experiments(experiment_ids)
        grouped: dict[str, dict] = {}
        for row in rows:
            exp_id = row["experiment_id"]
            if exp_id not in grouped:
                grouped[exp_id] = {
                    "experiment_id": exp_id,
                    "name": row["name"],
                    "priority": row["priority"],
                    "status": row["status"],
                    "trials": [],
                }
            if row["trial_id"]:
                grouped[exp_id]["trials"].append(
                    {
                        "trial_id": row["trial_id"],
                        "model_id": row["model_id"],
                        "metric": row["primary_metric"],
                        "score": round(float(row["primary_score"]), 4),
                        "training_time_s": round(float(row["training_time_seconds"] or 0), 3),
                        "failed": row["failure_reason"] is not None,
                    }
                )
        return list(grouped.values())

    def list_models(self, run_id: str | None = None, task_type: str | None = None) -> list[dict]:
        if run_id:
            run = self._get_run(run_id)
            task_type = task_type or run.config.task_type
            active = self.model_registry.resolve_active(
                run.config.models_include,
                run.config.models_exclude,
                task_type=task_type,
            )
            return [
                {
                    "id": spec.id,
                    "name": spec.name,
                    "active": spec.id in active,
                    "excluded": spec.id in run.config.models_exclude,
                    "compatible": task_type in spec.task_types,
                    "task_types": spec.task_types,
                }
                for spec in self.model_registry.list(task_type)
            ]
        return [
            {
                "id": spec.id,
                "name": spec.name,
                "task_types": spec.task_types,
                "compatible": task_type in spec.task_types if task_type else True,
            }
            for spec in self.model_registry.list(task_type)
        ]

    def get_task_plan(self, dataset_id: str) -> ProblemDefinition:
        problem = self.repository.get_problem_definition(dataset_id)
        if problem is not None:
            return problem
        dataset = self._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)
        problem = plan_from_dataframe(dataset, df)
        self.repository.save_problem_definition(problem)
        return problem

    def get_dataset_questionnaire(
        self,
        dataset_id: str,
        auto_generate: bool = True,
        llm_provider: Any = None,
    ) -> Any | None:
        """Retrieve dataset framing questionnaire, optionally generating it if not found."""
        q = self.repository.get_dataset_questionnaire(dataset_id)
        if q is not None:
            return q
        if not auto_generate:
            return None
        return self.generate_dataset_questionnaire(dataset_id, llm_provider=llm_provider)

    def generate_dataset_questionnaire(
        self,
        dataset_id: str,
        llm_provider: Any = None,
    ) -> Any:
        """Synthesize and persist a problem context questionnaire for the dataset."""
        dataset = self._get_dataset(dataset_id)
        profile = self.repository.get_dataset_profile(dataset_id)
        profile_summary = profile.to_dict() if profile else None
        columns = [c.name for c in profile.columns] if profile else []
        target_col = getattr(dataset, "target_column", getattr(dataset, "target", ""))

        from automl.engine.planning.questionnaire_advisor import QuestionnaireAdvisor
        advisor = QuestionnaireAdvisor(llm_provider=llm_provider)
        q = advisor.infer(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            column_names=columns,
            target_column=target_col,
            profile_summary=profile_summary,
        )
        self.repository.save_dataset_questionnaire(q)
        return q

    def save_dataset_questionnaire(self, questionnaire: Any) -> None:
        """Explicitly persist an edited or reviewed questionnaire."""
        self.repository.save_dataset_questionnaire(questionnaire)

    def validate_derived_feature(
        self,
        dataset_id: str,
        name: str,
        expression: str,
        expression_type: str = "formula",
    ) -> Any:
        """Validate and evaluate a derived feature definition on dataset without modifying it."""
        from automl.domain.features.derived_feature import (
            DerivedFeatureDefinition,
            DerivedFeatureType,
        )
        from automl.engine.features.generation.derived_feature_engine import DerivedFeatureEngine

        dataset = self._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)

        def_obj = DerivedFeatureDefinition(
            name=name,
            expression_type=DerivedFeatureType(expression_type),
            expression=expression,
        )
        engine = DerivedFeatureEngine()
        result, _ = engine.evaluate(df, def_obj)
        return result

    def apply_derived_feature(
        self,
        dataset_id: str,
        name: str,
        expression: str,
        expression_type: str = "formula",
        description: str = "",
    ) -> tuple[Any, Any]:
        """Evaluate and apply a derived feature to the dataset, re-profiling the dataset and saving changes."""
        from automl.domain.features.derived_feature import (
            DerivedFeatureDefinition,
            DerivedFeatureType,
        )
        from automl.engine.features.generation.derived_feature_engine import DerivedFeatureEngine

        dataset = self._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)

        def_obj = DerivedFeatureDefinition(
            name=name,
            expression_type=DerivedFeatureType(expression_type),
            expression=expression,
            description=description,
        )
        engine = DerivedFeatureEngine()
        updated_df, result = engine.apply(df, def_obj)

        # Save updated dataframe to dataset path
        updated_df.to_csv(dataset.path, index=False)

        # Re-profile dataset so new feature is part of the profile schema and available for experiments
        profile = profile_dataset(dataset, updated_df)
        self.repository.save_dataset_profile(profile)

        return result, profile

    def suggest_derived_features(
        self,
        dataset_id: str,
        llm_provider: Any = None,
    ) -> list[dict[str, Any]]:
        """Suggest candidate derived features tailored to dataset profile and validate them."""
        from automl.application.agents.specialists.feature_advisor import FeatureAdvisor

        dataset = self._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)
        profile = self.repository.get_dataset_profile(dataset_id)

        all_cols = [c.name for c in profile.columns] if profile else list(df.columns)
        numeric_cols = [
            c.name for c in (profile.columns if profile else [])
            if any(term in str(c.dtype).lower() for term in ("numeric", "float", "int"))
        ]
        if not numeric_cols:
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        target_col = getattr(dataset, "target_column", getattr(dataset, "target", ""))
        questionnaire = self.repository.get_dataset_questionnaire(dataset_id)
        domain_hint = questionnaire.domain_hint if questionnaire else "general_tabular"

        advisor = FeatureAdvisor(llm_provider=llm_provider)
        candidates = advisor.propose_features(
            column_names=all_cols,
            numeric_columns=numeric_cols,
            target_column=target_col,
            domain_hint=domain_hint,
        )
        valid_defs, results = advisor.validate_and_filter(df, candidates)

        suggestions = []
        for prop, res in zip(valid_defs, results):
            suggestions.append({
                "definition": prop.to_dict(),
                "evaluation": res.to_dict(),
            })
        return suggestions

    def detect_temporal_structure(self, dataset_id: str) -> dict[str, Any]:
        """Inspects dataset and returns detected temporal periodicities and sequential columns."""
        from automl.engine.profiling.dataset_profiler import detect_sequential_structure

        dataset = self._get_dataset(dataset_id)
        df = load_dataframe(dataset.path)
        target_col = getattr(dataset, "target_column", getattr(dataset, "target", None))
        structure = detect_sequential_structure(df, target_column=target_col)
        return structure.to_dict()

    def generate_temporal_features(
        self,
        run_id: str,
        dataset_id: str | None = None,
        max_lags: int = 1,
        include_lags: bool = True,
        include_deltas: bool = True,
        include_cyclical: bool = True,
        include_rolling: bool = False,
    ) -> list[FeatureSet]:
        """
        Discovers temporal dynamics (lags, deltas, cyclical periodicities) and registers
        candidate FeatureSet instances obeying 'Propose != Accept'.
        """
        from automl.engine.features.generation.temporal_generator import TemporalDynamicsGenerator
        from automl.engine.profiling.dataset_profiler import detect_sequential_structure

        run = self._get_run(run_id)
        effective_dataset_id = dataset_id or run.dataset_id
        dataset = self._get_dataset(effective_dataset_id)
        df = load_dataframe(dataset.path)
        target_col = getattr(dataset, "target_column", getattr(dataset, "target", None))

        base_features = self.get_feature_registry(effective_dataset_id).active_feature_names(target_col)
        structure = detect_sequential_structure(df, target_column=target_col)

        generator = TemporalDynamicsGenerator(
            max_lags=max_lags,
            include_lags=include_lags,
            include_deltas=include_deltas,
            include_cyclical=include_cyclical,
            include_rolling=include_rolling,
        )
        proposed_features = generator.propose_features(
            df=df,
            feature_names=base_features,
            target_column=target_col,
            temporal_structure=structure,
        )

        candidate_feature_sets = generator.propose_candidate_feature_sets(
            base_features=base_features,
            generated_features=proposed_features,
            dataset_id=effective_dataset_id,
        )

        candidates_as_fsc = generator.propose_candidate_sets_as_candidates(
            base_features=base_features,
            generated_features=proposed_features,
        )
        if run.id not in self._candidate_feature_sets:
            self._candidate_feature_sets[run.id] = []
        self._candidate_feature_sets[run.id].extend(candidates_as_fsc)

        created_sets: list[FeatureSet] = []
        for fs in candidate_feature_sets:
            self.repository.save_feature_set(fs)
            created_sets.append(fs)
            self._emit("FeatureSetCreated", fs.to_dict(), run_id=run.id)

        self._emit(
            "TemporalFeaturesGenerated",
            {
                "run_id": run.id,
                "dataset_id": effective_dataset_id,
                "generated_features_count": len(proposed_features),
                "candidate_sets_count": len(created_sets),
            },
            run_id=run.id,
        )
        return created_sets

    def list_task_types(self) -> list[dict]:
        rows = []
        for task in TaskType:
            catalog = TASK_CATALOG[task]
            rows.append(
                {
                    "task_type": task.value,
                    "label": catalog["label"],
                    "description": catalog["description"],
                    "default_metric": catalog["default_metric"],
                    "metrics": catalog["metrics"],
                    "models": catalog["model_ids"],
                }
            )
        return rows

    def exclude_feature(self, dataset_id: str, name: str, run_id: str | None = None) -> None:
        feature = self.get_feature_registry(dataset_id).get(name)
        if feature is None:
            raise KeyError(f"Feature not found: {name}")
        feature.exclude()
        self.repository.save_feature(feature)
        if run_id:
            run = self._get_run(run_id)
            if name not in run.config.features_excluded:
                run.config.features_excluded.append(name)
            self.repository.save_run(run)
        self._emit("FeatureExcluded", {"dataset_id": dataset_id, "feature": name}, run_id=run_id)

    def exclude_identifiers(self, dataset_id: str, run_id: str | None = None) -> list[str]:
        profile = self.repository.get_dataset_profile(dataset_id)
        if profile is None:
            dataset = self._get_dataset(dataset_id)
            profile = profile_dataset(dataset)
            self.repository.save_dataset_profile(profile)

        excluded: list[str] = []
        for name in profile.identifier_column_names:
            if name != profile.target_column:
                self.exclude_feature(dataset_id, name, run_id=run_id)
                excluded.append(name)
        return excluded

    def prioritize_feature(
        self,
        dataset_id: str,
        name: str,
        score: float = 1.0,
        run_id: str | None = None,
    ) -> None:
        feature = self.get_feature_registry(dataset_id).get(name)
        if feature is None:
            raise KeyError(f"Feature not found: {name}")
        feature.prioritize(score)
        self.repository.save_feature(feature)
        if run_id:
            run = self._get_run(run_id)
            if name not in run.config.features_priority:
                run.config.features_priority.append(name)
            self.repository.save_run(run)
        self._emit(
            "FeaturePrioritized",
            {"dataset_id": dataset_id, "feature": name, "score": score},
            run_id=run_id,
        )

    def exclude_model(self, run: AutoMLRun, model_id: str) -> None:
        self.model_registry.validate_for_task([model_id], run.config.task_type)
        if model_id not in run.config.models_exclude:
            run.config.models_exclude.append(model_id)
        self.repository.save_run(run)
        self._emit("ModelExcluded", {"model_id": model_id}, run_id=run.id)

    def include_model(self, run: AutoMLRun, model_id: str) -> None:
        self.model_registry.validate_for_task([model_id], run.config.task_type)
        if model_id not in run.config.models_include:
            run.config.models_include.append(model_id)
        if model_id in run.config.models_exclude:
            run.config.models_exclude.remove(model_id)
        self.repository.save_run(run)
        self._emit("ModelAdded", {"model_id": model_id}, run_id=run.id)

    # --- V0.3 Experiment Planner & Priority Engine ---

    def get_experiment_queue(self, run_id: str) -> ExperimentQueue:
        if run_id not in self._queues:
            self._queues[run_id] = ExperimentQueue()
        return self._queues[run_id]

    def plan_experiments(
        self,
        run_id: str,
        auto_enqueue: bool = True,
        user_priorities: dict[str, str] | None = None,
    ) -> list[ExperimentCandidate]:
        run = self._get_run(run_id)
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run_id} is cancelled")

        run.transition_to(RunStatus.PLANNING, RunPhase.EXPERIMENT_PLANNING)
        self.repository.save_run(run)

        profile = self.repository.get_dataset_profile(run.dataset_id)
        if profile is None:
            dataset = self._get_dataset(run.dataset_id)
            profile = profile_dataset(dataset)
            self.repository.save_dataset_profile(profile)

        feature_registry = self.get_feature_registry(run.dataset_id)
        history = self.repository.get_leaderboard(run.id)

        candidates = self.planner.propose(
            run=run,
            profile=profile,
            feature_registry=feature_registry,
            model_registry=self.model_registry,
            history=history,
        )

        priorities_map = user_priorities or {}
        for candidate in candidates:
            user_prio = (
                priorities_map.get(candidate.id)
                or priorities_map.get(candidate.name)
                or None
            )
            score = self.scorer.score(
                candidate=candidate,
                run=run,
                profile=profile,
                user_priority=user_prio,
            )
            candidate.priority = score

        if auto_enqueue:
            queue = self.get_experiment_queue(run.id)
            queue.enqueue_all(candidates)

        self._emit(
            "ExperimentsPlanned",
            {
                "run_id": run_id,
                "candidate_count": len(candidates),
                "candidates": [c.name for c in candidates],
            },
            run_id=run.id,
        )
        return candidates

    def prioritize_candidate(
        self,
        run_id: str,
        candidate_id: str,
        priority: str,
    ) -> Priority:
        run = self._get_run(run_id)
        queue = self.get_experiment_queue(run_id)
        candidate = queue.get(candidate_id)
        if candidate is None:
            raise KeyError(f"Candidate not found in queue: {candidate_id}")

        profile = self.repository.get_dataset_profile(run.dataset_id)
        if profile is None:
            dataset = self._get_dataset(run.dataset_id)
            profile = profile_dataset(dataset)

        new_priority = self.scorer.score(
            candidate=candidate,
            run=run,
            profile=profile,
            user_priority=priority,
        )
        queue.reprioritize(candidate_id, new_priority)
        self._emit(
            "CandidateReprioritized",
            {
                "candidate_id": candidate_id,
                "priority": priority,
                "effective_score": new_priority.effective_score,
                "is_pinned": new_priority.is_pinned,
            },
            run_id=run.id,
        )
        return new_priority

    def list_candidates(self, run_id: str) -> list[dict]:
        queue = self.get_experiment_queue(run_id)
        return [c.to_dict() for c in queue.list_queued()]

    def execute_next_experiment(
        self,
        run_id: str,
        budget: BudgetPolicy | None = None,
    ) -> tuple[Experiment | None, list[TrialResult]]:
        run = self._get_run(run_id)
        if run.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
            raise RuntimeError(f"Cannot execute next experiment: Run {run_id} is {run.status.value}")

        queue = self.get_experiment_queue(run_id)
        effective_budget = budget or BudgetPolicy()

        existing_experiments = self.repository.list_experiments(run_id)
        candidate = self.scheduler.select_next(
            queue=queue,
            budget=effective_budget,
            executed_experiments=len(existing_experiments),
        )
        if candidate is None:
            return None, []

        prio_value = candidate.priority.level.value if candidate.priority else "normal"
        experiment = self.create_experiment(
            run=run,
            name=candidate.name,
            feature_names=candidate.feature_names,
            feature_set_id=candidate.feature_set_id,
            model_ids=candidate.model_ids,
            hypothesis=candidate.hypothesis,
            priority=prio_value,
        )

        results = self.run_experiment(run, experiment)
        return experiment, results

    def run_scheduled_experiments(
        self,
        run_id: str,
        max_experiments: int | None = None,
        max_trials: int | None = None,
        budget: BudgetPolicy | None = None,
    ) -> list[dict]:
        run = self._get_run(run_id)
        if budget is not None:
            effective_budget = budget
        else:
            existing = len(self.repository.list_experiments(run_id))
            effective_budget = BudgetPolicy(
                max_experiments=existing + max_experiments if max_experiments is not None else None,
                max_trials=max_trials,
            )
        executed: list[dict] = []


        while len(self.get_experiment_queue(run_id)) > 0:
            refreshed = self._get_run(run_id)
            if refreshed.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
                break

            exp, results = self.execute_next_experiment(run_id, budget=effective_budget)
            if exp is None:
                break

            best_score = max([r.primary_score for r in results if r.succeeded], default=0.0)
            executed.append(
                {
                    "experiment_id": exp.id,
                    "name": exp.name,
                    "priority": exp.priority,
                    "trials_count": len(results),
                    "best_score": round(best_score, 4),
                    "succeeded": any(r.succeeded for r in results),
                }
            )

        return executed

    # --- V0.4 Model & Hyperparameter Optimization ---

    def optimize_experiment(
        self,
        run_id: str,
        experiment_id: str,
        model_id: str | None = None,
        optimizer: str = "optuna",
        n_trials: int = 10,
        timeout_seconds: float | None = None,
        patience: int = 5,
        min_delta: float = 0.0001,
    ) -> dict:
        import time

        run = self._get_run(run_id)
        if run.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
            raise RuntimeError(f"Cannot optimize experiment: Run {run_id} is {run.status.value}")

        experiment = self.repository.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment not found: {experiment_id}")
        if experiment.run_id != run.id:
            raise ValueError(f"Experiment {experiment_id} does not belong to run {run_id}")

        dataset = self._get_dataset(run.dataset_id)
        target_model = model_id or experiment.model_ids[0]
        if target_model not in experiment.model_ids:
            raise ValueError(f"Model {target_model} is not part of experiment {experiment_id}")

        experiment.status = ExperimentStatus.RUNNING
        self.repository.save_experiment(experiment)
        run.transition_to(RunStatus.OPTIMIZING, RunPhase.OPTIMIZATION)
        self.repository.save_run(run)

        space = SearchSpaceBuilder.build(target_model, run.config.task_type)
        metric_plugin = self.plugin_registry.get_metric_plugin(run.config.metric)
        if metric_plugin is not None:
            direction = "maximize" if metric_plugin.greater_is_better else "minimize"
        else:
            direction = "minimize" if run.config.metric in {"mae", "rmse"} else "maximize"

        # Check meta-learning warm start priors
        warm_params: dict[str, Any] | None = None
        try:
            meta = self.get_meta_knowledge(run.dataset_id, run_id=run.id)
            if meta.warm_start and meta.warm_start.params:
                valid_warm = {
                    spec.name: meta.warm_start.params[spec.name]
                    for spec in space.list()
                    if spec.name in meta.warm_start.params
                }
                if valid_warm:
                    warm_params = valid_warm
        except Exception:
            pass

        opt: OptimizerPort
        if optimizer.lower() == "optuna":
            opt = OptunaOptimizer(
                seed=run.config.random_seed,
                direction=direction,
                patience=patience,
                min_delta=min_delta,
                warm_start_params=warm_params,
            )
        else:
            opt = RandomSearchOptimizer(
                seed=run.config.random_seed,
                patience=patience,
                min_delta=min_delta,
                mode="max" if direction == "maximize" else "min",
                warm_start_params=warm_params,
            )

        trainer = SklearnTrainer(plugin_registry=self.plugin_registry)
        results: list[TrialResult] = []
        started_at = time.time()
        is_cancelled = False
        is_paused = False

        for trial_idx in range(n_trials):
            if timeout_seconds and (time.time() - started_at) > timeout_seconds:
                break
            if opt.should_stop():
                break

            if self.execution_check:
                self.execution_check()

            refreshed = self.repository.get_run(run.id)
            if refreshed and refreshed.status == RunStatus.PAUSED:
                is_paused = True
                experiment.status = ExperimentStatus.PAUSED
                self.repository.save_experiment(experiment)
                self._emit("RunPaused", {"experiment_id": experiment.id, "trial_idx": trial_idx}, run_id=run.id)
                break

            if refreshed and refreshed.status == RunStatus.CANCELLED:
                is_cancelled = True
                experiment.status = ExperimentStatus.CANCELLED
                self.repository.save_experiment(experiment)
                self._emit("RunCancelled", {"experiment_id": experiment.id, "trial_idx": trial_idx}, run_id=run.id)
                break

            params = opt.suggest(trial_idx, space)
            trial = TrialFactory.create(
                experiment_id=experiment.id,
                model_id=target_model,
                parameters=params,
                seed=run.config.random_seed + trial_idx,
            )
            self.repository.save_trial(trial)

            execution = TrialExecution(
                trial=trial,
                experiment=experiment,
                run=run,
                feature_names=experiment.feature_names,
                dataset_path=dataset.path,
                target_column=dataset.target_column,
                task_type=dataset.task_type,
                metric=experiment.metric,
                validation_strategy=experiment.validation_strategy,
                test_size=run.config.test_size,
                cv_folds=run.config.cv_folds,
                random_seed=run.config.random_seed + trial_idx,
                group_column=experiment.group_column,
            )
            result = trainer.run(execution)
            self.repository.save_trial(trial)
            self.repository.save_trial_result(result)
            opt.observe(trial_idx, params, result.primary_score, result.succeeded)
            results.append(result)

            self._emit(
                "TrialCompleted" if result.succeeded else "TrialFailed",
                {
                    "trial_id": trial.id,
                    "model_id": target_model,
                    "score": result.primary_score,
                    "parameters": params,
                },
                run_id=run.id,
            )

        if not is_cancelled and not is_paused:
            experiment.status = ExperimentStatus.COMPLETED
            self.repository.save_experiment(experiment)
            run.transition_to(RunStatus.COMPLETED, RunPhase.EVALUATION)
            self.repository.save_run(run)

        best_score = opt.best_score()
        best_params = opt.best_parameters()
        best_trial_id = ""
        if results:
            best_res = (
                max(results, key=lambda r: r.primary_score)
                if direction == "maximize"
                else min(results, key=lambda r: r.primary_score)
            )
            best_trial_id = best_res.trial_id

        self._emit(
            "ExperimentOptimized",
            {
                "experiment_id": experiment.id,
                "model_id": target_model,
                "trials_executed": len(results),
                "best_score": round(best_score, 4),
                "best_params": best_params,
                "best_trial_id": best_trial_id,
                "optimizer": optimizer,
            },
            run_id=run.id,
        )

        return {
            "experiment_id": experiment.id,
            "model_id": target_model,
            "optimizer": optimizer,
            "trials_executed": len(results),
            "best_score": round(best_score, 4),
            "best_params": best_params,
            "best_trial_id": best_trial_id,
            "trials": [
                {
                    "trial_id": r.trial_id,
                    "score": round(r.primary_score, 4),
                    "training_time_s": round(r.training_time_seconds, 3),
                    "succeeded": r.succeeded,
                }
                for r in results
            ],
        }

    def get_best_trial(self, experiment_id: str) -> dict | None:
        results = self.repository.list_trial_results(experiment_id)
        if not results:
            return None
        valid = [r for r in results if r.succeeded]
        if not valid:
            return None

        is_minimize = valid[0].primary_metric in {"mae", "rmse"}
        best = min(valid, key=lambda r: r.primary_score) if is_minimize else max(valid, key=lambda r: r.primary_score)

        trial = self.repository.get_trial(best.trial_id)
        params = trial.parameters if trial else {}

        return {
            "trial_id": best.trial_id,
            "experiment_id": best.experiment_id,
            "model_id": best.model_id,
            "metric": best.primary_metric,
            "best_score": round(best.primary_score, 4),
            "parameters": params,
            "training_time_seconds": round(best.training_time_seconds, 3),
        }

    def get_experiment_trials(self, experiment_id: str) -> list[dict]:
        results = self.repository.list_trial_results(experiment_id)
        trials = {r.trial_id: self.repository.get_trial(r.trial_id) for r in results}
        return [
            {
                "trial_id": r.trial_id,
                "experiment_id": r.experiment_id,
                "model_id": r.model_id,
                "metric": r.primary_metric,
                "score": round(r.primary_score, 4),
                "training_time_s": round(r.training_time_seconds, 3),
                "succeeded": r.succeeded,
                "failure_reason": r.failure_reason,
                "parameters": dict(trials[r.trial_id].parameters) if trials[r.trial_id] else {},
            }
            for r in results
        ]

    # --- V0.5 Feature Discovery & Selection ---

    def select_features(
        self,
        run_id: str,
        strategy: FeatureSelectionStrategy | None = None,
    ) -> list[FeatureSetCandidate]:
        run = self._get_run(run_id)
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run_id} is cancelled")

        strat = strategy or FeatureSelectionStrategy()
        dataset = self._get_dataset(run.dataset_id)
        feature_registry = self.get_feature_registry(run.dataset_id)
        active_features = feature_registry.active_feature_names(run.config.target)

        task_type_str = (
            run.config.task_type.value
            if hasattr(run.config.task_type, "value")
            else str(run.config.task_type)
        )
        context = FeatureAnalysisContext(
            dataset_path=dataset.path,
            target_column=dataset.target_column,
            task_type=task_type_str,
            active_feature_names=active_features,
            random_seed=run.config.random_seed,
        )

        selectors: list[FeatureSelectorPort] = []
        for method in strat.methods:
            if method == "mutual_information":
                selectors.append(MutualInformationSelector())
            elif method in ("importance", "tree_importance"):
                selectors.append(TreeImportanceSelector())
            elif method == "correlation":
                selectors.append(CorrelationSelector())
            elif method == "variance":
                selectors.append(VarianceSelector())

        if not selectors:
            selectors = [MutualInformationSelector(), TreeImportanceSelector()]

        all_ranks: dict[str, list[FeatureRank]] = {}
        for sel in selectors:
            sel.fit(context)
            all_ranks[sel.method_id] = sel.rank_features()

        for feature_name in active_features:
            evidence = (
                self.repository.get_feature_evidence(run.id, feature_name)
                or FeatureEvidence(feature_id=feature_name)
            )
            if "mutual_information" in all_ranks:
                for r in all_ranks["mutual_information"]:
                    if r.feature_name == feature_name:
                        evidence.mutual_information = r.score
            if "importance" in all_ranks:
                for r in all_ranks["importance"]:
                    if r.feature_name == feature_name:
                        evidence.shap_importance = r.score
            evidence.confidence = 0.8
            self.repository.save_feature_evidence(run.id, evidence)

        if len(selectors) > 1:
            ensemble_sel = EnsembleRankSelector(
                selectors=selectors,
                combine_method=strat.combine_method,
            )
            combined_ranks = ensemble_sel.combine_rankings(
                all_ranks, combine_method=strat.combine_method
            )
        else:
            combined_ranks = all_ranks[selectors[0].method_id]

        self._feature_ranks[run.id] = combined_ranks

        candidates: list[FeatureSetCandidate] = []
        for k in strat.top_k:
            if k <= len(combined_ranks):
                selected_names = [r.feature_name for r in combined_ranks[:k]]
                candidates.append(
                    FeatureSetCandidate.create(
                        name=f"selected_top_{k}",
                        feature_names=selected_names,
                        method=strat.combine_method if len(selectors) > 1 else selectors[0].method_id,
                        k=k,
                        metadata={
                            "strategy": strat.to_dict(),
                            "features": selected_names,
                        },
                    )
                )

        if strat.include_reduction:
            pca = PCAReducer()
            for var in strat.reduction_variances:
                try:
                    pca.fit(context, n_components=var)
                    candidates.append(
                        FeatureSetCandidate.create(
                            name=f"pca_var_{int(var*100)}",
                            feature_names=[f"PC{i+1}" for i in range(pca.n_components())],
                            method="pca",
                            k=pca.n_components(),
                            metadata={
                                "variance_threshold": var,
                                "explained_variance": pca.explained_variance_ratio(),
                            },
                        )
                    )
                except Exception:
                    pass

        self._candidate_feature_sets[run.id] = candidates

        self._emit(
            "FeaturesSelected",
            {
                "run_id": run_id,
                "strategy": strat.to_dict(),
                "candidates": [c.name for c in candidates],
                "top_features": [r.feature_name for r in combined_ranks[:5]],
            },
            run_id=run.id,
        )
        return candidates

    def plan_ablation_experiments(
        self,
        run_id: str,
        base_feature_names: list[str] | None = None,
        model_ids: list[str] | None = None,
        max_features: int = 5,
        auto_enqueue: bool = True,
    ) -> list[ExperimentCandidate]:
        run = self._get_run(run_id)
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run_id} is cancelled")

        feature_registry = self.get_feature_registry(run.dataset_id)
        all_active = feature_registry.active_feature_names(run.config.target)

        features_base = base_feature_names or all_active
        if not features_base:
            return []

        ranked = self._feature_ranks.get(run.id, [])
        if ranked:
            ranked_names = [r.feature_name for r in ranked if r.feature_name in features_base]
            targets = ranked_names[:max_features]
        else:
            targets = features_base[:max_features]

        chosen_model = (
            model_ids[0]
            if model_ids
            else (run.config.models_include[0] if run.config.models_include else None)
        )

        ablation_planner = AblationPlanner()
        candidates = ablation_planner.propose_ablation(
            run=run,
            base_feature_names=features_base,
            features_to_ablate=targets,
            model_id=chosen_model,
        )

        profile = self.repository.get_dataset_profile(run.dataset_id)
        if profile is None:
            dataset = self._get_dataset(run.dataset_id)
            profile = profile_dataset(dataset)
            self.repository.save_dataset_profile(profile)

        for candidate in candidates:
            score = self.scorer.score(
                candidate=candidate,
                run=run,
                profile=profile,
            )
            candidate.priority = score

        if auto_enqueue:
            queue = self.get_experiment_queue(run.id)
            queue.enqueue_all(candidates)

        self._emit(
            "AblationExperimentsPlanned",
            {
                "run_id": run_id,
                "candidate_count": len(candidates),
                "ablated_features": targets,
            },
            run_id=run.id,
        )
        return candidates

    def promote_candidate_feature_set(
        self,
        run_id: str,
        candidate_id: str,
        new_name: str | None = None,
    ) -> FeatureSet:
        run = self._get_run(run_id)
        candidates = self._candidate_feature_sets.get(run.id, [])
        candidate = next(
            (c for c in candidates if c.id == candidate_id or c.name == candidate_id),
            None,
        )
        if candidate is None:
            raise KeyError(f"Candidate feature set not found: {candidate_id}")

        feature_set_name = new_name or candidate.name
        feature_set = self.create_feature_set(
            dataset_id=run.dataset_id,
            name=feature_set_name,
            feature_names=candidate.feature_names,
            lineage=f"promoted_from_{candidate.method}_{candidate.id}",
        )
        self._emit(
            "FeatureSetPromoted",
            {
                "run_id": run.id,
                "candidate_id": candidate.id,
                "feature_set_id": feature_set.id,
                "feature_names": feature_set.feature_names,
            },
            run_id=run.id,
        )
        return feature_set

    def get_feature_evidence(
        self,
        run_id: str,
        feature_id: str | None = None,
    ) -> FeatureEvidence | list[FeatureEvidence] | None:
        if feature_id:
            return self.repository.get_feature_evidence(run_id, feature_id)
        return self.repository.list_feature_evidence(run_id)

    def list_candidate_feature_sets(self, run_id: str) -> list[dict]:
        candidates = self._candidate_feature_sets.get(run_id, [])
        return [c.to_dict() for c in candidates]

    def get_feature_ranking(self, run_id: str, method: str | None = None) -> list[dict]:
        ranks = self._feature_ranks.get(run_id, [])
        if method:
            ranks = [r for r in ranks if r.method == method]
        return [r.to_dict() for r in ranks]

    # --- V0.6 Plugin System ---

    def list_plugins(
        self,
        plugin_type: str | None = None,
        task_type: str | None = None,
    ) -> list[dict]:
        plugins = self.plugin_registry.list(plugin_type=plugin_type, task_type=task_type)
        items = []
        for p in plugins:
            is_native = getattr(p, "is_native", getattr(p, "is_available", True))
            fallback = getattr(p, "fallback_backend", None)
            if is_native:
                backend_status = "native"
            elif fallback:
                backend_status = f"fallback ({fallback})"
            else:
                backend_status = "unavailable"

            items.append({
                "plugin_id": p.plugin_id,
                "name": p.name,
                "version": p.version,
                "plugin_type": p.plugin_type.value if hasattr(p.plugin_type, "value") else str(p.plugin_type),
                "capabilities": p.capabilities.to_dict(),
                "is_native": bool(is_native),
                "backend_status": backend_status,
                "fallback_backend": fallback,
            })
        return items

    def register_plugin(self, plugin: Any) -> None:
        from automl.domain.plugins.plugin import PluginType
        self.plugin_registry.register(plugin)
        if getattr(plugin, "plugin_type", None) == PluginType.MODEL:
            from automl.domain.models.registry import ModelSpec
            self.model_registry.register(
                ModelSpec(
                    id=plugin.plugin_id,
                    name=plugin.name,
                    task_types=list(plugin.capabilities.supported_tasks),
                )
            )
        self._emit(
            "PluginRegistered",
            {
                "plugin_id": plugin.plugin_id,
                "plugin_type": str(getattr(plugin, "plugin_type", "unknown")),
            },
        )

    def predict(
        self,
        run_id: str,
        test_dataset_path: str | Path,
        experiment_id: str | None = None,
        trial_id: str | None = None,
        predict_proba: bool = False,
        template_path: str | Path | None = None,
        id_column: str | None = None,
    ) -> list[Any]:
        run = self._get_run(run_id)
        dataset = self._get_dataset(run.dataset_id)

        target_trial_id = trial_id
        target_experiment_id = experiment_id
        target_model_id = None
        parameters: dict[str, Any] = {}

        if target_trial_id:
            trial = self.repository.get_trial(target_trial_id)
            if trial is None:
                raise KeyError(f"Trial '{target_trial_id}' not found.")
            target_experiment_id = trial.experiment_id
            target_model_id = trial.model_id
            parameters = dict(trial.parameters or {})
        elif target_experiment_id:
            experiment = self.repository.get_experiment(target_experiment_id)
            if experiment is None:
                raise KeyError(f"Experiment '{target_experiment_id}' not found.")
            lb = self.repository.get_leaderboard(run.id)
            exp_results = [r for r in lb if r.experiment_id == target_experiment_id]
            if exp_results:
                best = exp_results[0]
                target_model_id = best.model_id
                t = self.repository.get_trial(best.trial_id)
                parameters = dict(t.parameters or {}) if t else {}
            else:
                target_model_id = experiment.model_ids[0] if experiment.model_ids else "random_forest"
        else:
            lb = self.repository.get_leaderboard(run.id)
            if not lb:
                raise ValueError(f"Run '{run_id}' has no completed trials in its leaderboard to predict with.")
            best = lb[0]
            target_experiment_id = best.experiment_id
            target_model_id = best.model_id
            t = self.repository.get_trial(best.trial_id)
            parameters = dict(t.parameters or {}) if t else {}

        experiment = self.repository.get_experiment(target_experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment '{target_experiment_id}' not found.")
        if experiment.run_id != run.id:
            if target_trial_id:
                raise ValueError(
                    f"Trial '{target_trial_id}' (experiment '{target_experiment_id}') does not belong to run '{run_id}'."
                )
            raise ValueError(f"Experiment '{target_experiment_id}' does not belong to run '{run_id}'.")

        test_df = load_dataframe(test_dataset_path)
        if experiment.validation_strategy == "oof":
            from automl.application.services.oof_submission import cached_oof_predictions
            raw_preds = cached_oof_predictions(self, run_id, experiment.id, test_dataset_path, predict_proba)
        else:
            feature_names = experiment.feature_names

            train_df = load_dataframe(dataset.path)
            X_train = train_df[feature_names]
            y_train = train_df[dataset.target_column]

            missing_features = [f for f in feature_names if f not in test_df.columns]
            if missing_features:
                raise ValueError(f"Test dataset is missing required features: {missing_features}")
            X_test = test_df[feature_names]

            trainer = SklearnTrainer(plugin_registry=self.plugin_registry)
            preds = trainer.fit_and_predict(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                model_id=target_model_id,
                task_type=dataset.task_type,
                parameters=parameters,
                predict_proba=predict_proba,
            )
            raw_preds = preds.tolist() if hasattr(preds, "tolist") else list(preds)

        if template_path is not None:
            import pandas as pd
            try:
                template_df = load_dataframe(template_path)
            except Exception as e:
                raise ValueError(f"Template CSV is empty or invalid: {e}") from e
            if template_df.empty:
                raise ValueError("Template CSV is empty.")
            if id_column:
                if id_column not in template_df.columns:
                    raise ValueError(f"Specified ID column '{id_column}' not found in template CSV.")
                if id_column not in test_df.columns:
                    raise ValueError(f"Specified ID column '{id_column}' not found in test dataset.")
                actual_id_col = id_column
            else:
                candidate_id = next(
                    (c for c in template_df.columns if c in test_df.columns and (
                        c.lower() in {"id", "passengerid", "customer_id", "guid"} or c.lower().endswith("_id")
                    )),
                    None,
                )
                if candidate_id is None:
                    first_col = template_df.columns[0]
                    if first_col in test_df.columns:
                        candidate_id = first_col
                    else:
                        raise ValueError(
                            f"Could not automatically detect matching ID column between template {list(template_df.columns)} and test dataset {list(test_df.columns)}. Please specify id_column."
                        )
                actual_id_col = candidate_id

            test_preds_series = pd.Series(raw_preds, index=test_df[actual_id_col].values)
            aligned_preds = template_df[actual_id_col].map(test_preds_series)
            if aligned_preds.isna().any():
                missing_mask = aligned_preds.isna()
                missing_sample = template_df[actual_id_col][missing_mask].head(5).tolist()
                raise ValueError(
                    f"Template contains {missing_mask.sum()} IDs not found in the test dataset predictions (e.g. {missing_sample})."
                )
            return aligned_preds.tolist()

        return raw_preds

    def generate_submission(
        self,
        run_id: str,
        test_dataset_path: str | Path,
        output_path: str | Path,
        id_column: str | None = None,
        template_path: str | Path | None = None,
        experiment_id: str | None = None,
        trial_id: str | None = None,
        predict_proba: bool = False,
    ) -> dict[str, Any]:
        preds = self.predict(
            run_id=run_id,
            test_dataset_path=test_dataset_path,
            experiment_id=experiment_id,
            trial_id=trial_id,
            predict_proba=predict_proba,
        )
        return self._write_submission(run_id, test_dataset_path, output_path, preds,
                                      id_column, template_path, predict_proba)

    def export_model_artifact(
        self,
        run_id: str,
        experiment_id: str | None = None,
        trial_id: str | None = None,
    ) -> Any:
        """
        Fits and exports a standalone, portable ModelArtifact for the specified or winning trial.
        """
        run = self._get_run(run_id)
        dataset = self._get_dataset(run.dataset_id)

        target_exp_id = experiment_id
        target_trial_id = trial_id

        if target_exp_id is None:
            leaderboard = self.repository.get_leaderboard(run_id)
            if not leaderboard:
                raise ValueError(
                    f"No completed trials found in run '{run_id}'. Cannot export artifact."
                )
            best_trial_res = leaderboard[0]
            target_exp_id = best_trial_res.experiment_id
            target_trial_id = best_trial_res.trial_id

        experiment = self.repository.get_experiment(target_exp_id)
        if experiment is None:
            raise KeyError(f"Experiment '{target_exp_id}' not found.")

        target_trial = None
        if target_trial_id:
            target_trial = self.repository.get_trial(target_trial_id)
        if target_trial is None:
            trial_results = self.repository.list_trial_results(target_exp_id)
            if trial_results:
                target_trial = self.repository.get_trial(trial_results[0].trial_id)

        if target_trial is None:
            raise ValueError(f"No trial found for experiment '{target_exp_id}'.")

        from automl.engine.profiling.dataset_profiler import load_dataframe

        train_df = load_dataframe(dataset.path)
        feature_names = experiment.feature_names
        X_train = train_df[feature_names]
        y_train = train_df[dataset.target_column]

        from automl.engine.training.sklearn_trainer import SklearnTrainer

        params = dict(target_trial.parameters or {})
        if run.config and run.config.extra:
            if "text_columns" in run.config.extra and "text_columns" not in params:
                params["text_columns"] = run.config.extra["text_columns"]
            if "image_columns" in run.config.extra and "image_columns" not in params:
                params["image_columns"] = run.config.extra["image_columns"]

        trainer = SklearnTrainer(plugin_registry=self.plugin_registry)
        pipeline, adapter = trainer.fit_pipeline(
            X_train=X_train,
            y_train=y_train,
            model_id=target_trial.model_id,
            task_type=dataset.task_type,
            parameters=params,
        )

        from automl.artifacts.model_artifact import ModelArtifact

        score = 0.0
        metric = "score"
        leaderboard = self.repository.get_leaderboard(run_id)
        for res in leaderboard:
            if res.trial_id == target_trial.id:
                score = res.primary_score
                metric = res.primary_metric
                break

        import sys
        from datetime import datetime, timezone
        from importlib.metadata import version, PackageNotFoundError
        from automl import __version__

        dep_versions: dict[str, str] = {"catml": __version__}
        for name in (
            "numpy",
            "pandas",
            "scikit-learn",
            "joblib",
            "lightgbm",
            "xgboost",
            "catboost",
            "optuna",
            "torch",
            "torchvision",
            "timm",
            "pillow",
        ):
            try:
                dep_versions[name] = version(name)
            except PackageNotFoundError:
                pass

        estimator_step = pipeline[-1] if hasattr(pipeline, "__getitem__") and hasattr(pipeline, "steps") else pipeline

        dataset_hash = None
        if hasattr(dataset, "path") and dataset.path:
            dp = Path(dataset.path)
            if dp.is_file():
                import hashlib
                digest = hashlib.sha256()
                with dp.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                dataset_hash = digest.hexdigest()

        provenance = {
            "catml_version": __version__,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "estimator_class": type(estimator_step).__name__,
            "estimator_module": type(estimator_step).__module__,
            "dependencies": dep_versions,
            "dataset_id": dataset.id,
            "dataset_path": str(dataset.path) if hasattr(dataset, "path") else None,
            "dataset_hash": dataset_hash,
            "seed": getattr(getattr(run, "config", None), "random_seed", None),
        }

        return ModelArtifact(
            pipeline=pipeline,
            model_id=target_trial.model_id,
            task_type=dataset.task_type,
            feature_names=feature_names,
            target_name=dataset.target_column,
            target_adapter=adapter,
            metric=metric,
            score=score,
            parameters=target_trial.parameters or {},
            metadata={
                "run_id": run_id,
                "experiment_id": target_exp_id,
                "trial_id": target_trial.id,
            },
            provenance=provenance,
        )

    def _write_submission(
        self, run_id, test_dataset_path, output_path, preds,
        id_column=None, template_path=None, predict_proba=False,
    ) -> dict[str, Any]:
        if self.execution_check:
            self.execution_check()
        run = self._get_run(run_id)
        dataset = self._get_dataset(run.dataset_id)

        from automl.engine.profiling.dataset_profiler import load_dataframe
        import pandas as pd
        test_df = load_dataframe(test_dataset_path)

        template_used = None
        if template_path is not None:
            try:
                template_df = load_dataframe(template_path)
            except Exception as e:
                raise ValueError(f"Template CSV is empty or invalid: {e}") from e
            if template_df.empty:
                raise ValueError("Template CSV is empty.")

            if id_column:
                if id_column not in template_df.columns:
                    raise ValueError(f"Specified ID column '{id_column}' not found in template CSV.")
                if id_column not in test_df.columns:
                    raise ValueError(f"Specified ID column '{id_column}' not found in test dataset.")
                actual_id_col = id_column
            else:
                candidate_id = next(
                    (c for c in template_df.columns if c in test_df.columns and (
                        c.lower() in {"id", "passengerid", "customer_id", "guid"} or c.lower().endswith("_id")
                    )),
                    None,
                )
                if candidate_id is None:
                    first_col = template_df.columns[0]
                    if first_col in test_df.columns:
                        candidate_id = first_col
                    else:
                        raise ValueError(
                            f"Could not automatically detect matching ID column between template {list(template_df.columns)} and test dataset {list(test_df.columns)}. Please pass id_column explicitly."
                        )
                actual_id_col = candidate_id

            target_cols = [c for c in template_df.columns if c != actual_id_col]
            if not target_cols:
                raise ValueError("Template CSV must contain at least one target column in addition to the ID column.")

            test_preds_series = pd.Series(preds, index=test_df[actual_id_col].values)
            aligned_preds = template_df[actual_id_col].map(test_preds_series)
            if aligned_preds.isna().any():
                missing_mask = aligned_preds.isna()
                missing_sample = template_df[actual_id_col][missing_mask].head(5).tolist()
                raise ValueError(
                    f"Template contains {missing_mask.sum()} IDs not found in the test dataset predictions (e.g. {missing_sample})."
                )

            submission_df = template_df.copy()
            target_col = target_cols[0]
            submission_df[target_col] = aligned_preds.values
            submission_df = submission_df[template_df.columns]
            template_used = str(Path(template_path).resolve())
        else:
            if id_column:
                if id_column not in test_df.columns:
                    raise ValueError(f"ID column '{id_column}' not found in test dataset.")
                ids = test_df[id_column]
                actual_id_col = id_column
            else:
                candidate_id = next((c for c in ["id", "Id", "ID", "PassengerId", "customer_id"] if c in test_df.columns), None)
                if candidate_id:
                    ids = test_df[candidate_id]
                    actual_id_col = candidate_id
                else:
                    ids = pd.Series(range(len(test_df)), name="id")
                    actual_id_col = "id"

            target_col = dataset.target_column or "prediction"
            submission_df = pd.DataFrame({
                actual_id_col: ids,
                target_col: preds,
            })

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        submission_df.to_csv(out, index=False)

        self._emit(
            "SubmissionGenerated",
            {
                "run_id": run_id,
                "output_path": str(out.resolve()),
                "row_count": len(submission_df),
            },
            run_id=run_id,
        )

        return {
            "output_path": str(out.resolve()),
            "row_count": len(submission_df),
            "id_column": actual_id_col,
            "target_column": target_col,
            "predict_proba": predict_proba,
            "template_used": template_used,
        }

    def generate_oof_submission(self, command) -> str:
        from automl.application.services.oof_submission import generate_oof_submission
        return generate_oof_submission(self, command)

    def get_oof_result(self, run_id: str, experiment_id: str) -> dict:
        from automl.application.services.oof_submission import get_oof_report
        return get_oof_report(self, run_id, experiment_id)

    def validate_pipeline_graph(self, graph: PipelineGraph) -> None:
        """Validates that a pipeline graph is well-formed, acyclic, and modality-consistent."""
        validator = GraphValidator()
        validator.validate(graph)

    def get_pipeline_execution_order(self, graph: PipelineGraph) -> list[PipelineNode]:
        """Returns the topologically sorted execution order of nodes in the pipeline graph."""
        validator = GraphValidator()
        return validator.get_execution_order(graph)

    def build_multimodal_pipeline(
        self,
        tabular_source_id: str = "tabular_input",
        image_source_id: str = "image_input",
        model_id: str = "random_forest",
        embedding_dim: int = 16,
        pipeline_id: str = "multimodal_pipeline",
        pipeline_name: str = "Multimodal Tabular + Vision Pipeline",
    ) -> PipelineGraph:
        """Constructs a validated standard multimodal DAG with tabular and vision branches."""
        graph = PipelineGraph(id=pipeline_id, name=pipeline_name)

        tab_node = PipelineNode(
            node_id=tabular_source_id,
            node_type=NodeType.SOURCE,
            input_modalities=(Modality.TABULAR,),
            output_modality=Modality.TABULAR,
            name="Tabular Data Source",
        )
        img_node = PipelineNode(
            node_id=image_source_id,
            node_type=NodeType.SOURCE,
            input_modalities=(Modality.IMAGE,),
            output_modality=Modality.IMAGE,
            name="Image Data Source",
        )
        encoder_node = ImageEncoderNode(
            node_id="image_encoder",
            output_dim=embedding_dim,
        ).to_pipeline_node()
        fusion_node = FeatureFusionNode(
            node_id="fusion",
            default_prefix="img_emb",
        ).to_pipeline_node()
        model_node = PipelineNode(
            node_id="model",
            node_type=NodeType.MODEL,
            input_modalities=(Modality.TABULAR,),
            output_modality=Modality.TABULAR,
            name=f"Model({model_id})",
            parameters={"model_id": model_id},
        )

        for n in [tab_node, img_node, encoder_node, fusion_node, model_node]:
            graph.add_node(n)

        graph.add_edge(image_source_id, "image_encoder")
        graph.add_edge(tabular_source_id, "fusion")
        graph.add_edge("image_encoder", "fusion")
        graph.add_edge("fusion", "model")

        self.validate_pipeline_graph(graph)
        return graph

    def execute_pipeline(self, graph: PipelineGraph, inputs: dict[str, Any]) -> Any:
        """Executes a pipeline DAG given modal inputs, returning the terminal node's output."""
        validator = GraphValidator()
        execution_order = validator.get_execution_order(graph)

        node_outputs: dict[str, Any] = {}

        for node in execution_order:
            if node.node_type == NodeType.SOURCE:
                data = inputs.get(node.node_id)
                if data is None:
                    for k, v in inputs.items():
                        if k in node.node_id or (hasattr(node.output_modality, "value") and k == node.output_modality.value):
                            data = v
                            break
                if data is None:
                    raise KeyError(f"Missing required input for source node '{node.node_id}' in pipeline execution.")
                node_outputs[node.node_id] = data

            elif node.node_type == NodeType.ENCODER:
                incoming = graph.get_incoming_nodes(node.node_id)
                if not incoming:
                    raise ValueError(f"Encoder node '{node.node_id}' has no incoming inputs.")
                src_data = node_outputs[incoming[0].node_id]
                encoder = ImageEncoderNode.from_pipeline_node(node)
                node_outputs[node.node_id] = encoder.transform(src_data)

            elif node.node_type == NodeType.FUSION:
                incoming = graph.get_incoming_nodes(node.node_id)
                if not incoming:
                    raise ValueError(f"Fusion node '{node.node_id}' has no incoming inputs.")
                fusion_dict = {inc.node_id: node_outputs[inc.node_id] for inc in incoming}
                fusion = FeatureFusionNode.from_pipeline_node(node)
                node_outputs[node.node_id] = fusion.fuse(fusion_dict)

            elif node.node_type == NodeType.PREPROCESSOR:
                incoming = graph.get_incoming_nodes(node.node_id)
                src_data = node_outputs[incoming[0].node_id]
                node_outputs[node.node_id] = src_data

            elif node.node_type == NodeType.MODEL:
                incoming = graph.get_incoming_nodes(node.node_id)
                src_data = node_outputs[incoming[0].node_id]
                node_outputs[node.node_id] = src_data

        terminal_nodes = graph.get_terminal_nodes()
        if len(terminal_nodes) == 1:
            return node_outputs[terminal_nodes[0].node_id]
        return {tn.node_id: node_outputs[tn.node_id] for tn in terminal_nodes}

    def fit_predict_multimodal(
        self,
        graph: PipelineGraph,
        inputs: dict[str, Any],
        target: Any,
        model_id: str = "random_forest",
        task_type: str = "binary_classification",
    ) -> dict[str, Any]:
        """Trains a model on the fused representations produced by a multimodal DAG pipeline."""
        fused_df = self.execute_pipeline(graph, inputs)
        if not isinstance(fused_df, pd.DataFrame):
            fused_df = pd.DataFrame(fused_df)

        y = np.asarray(target)
        model_plugin = self.plugin_registry.get_model_plugin(model_id)

        if model_plugin is not None:
            model = model_plugin.build_estimator()
            model.fit(fused_df, y)
            preds = model.predict(fused_df)
        else:
            trainer = SklearnTrainer(model_registry=self.model_registry)
            trial_exec = trainer.train(
                task_type=task_type,
                model_id=model_id,
                hyperparameters={},
                X_train=fused_df,
                y_train=y,
                metric=default_metric_for(task_type),
            )
            model = trial_exec.model
            preds = model.predict(fused_df)

        metric_name = default_metric_for(task_type)
        if task_type == "binary_classification":
            from sklearn.metrics import accuracy_score
            score = float(accuracy_score(y, preds))
        elif task_type == "regression":
            from sklearn.metrics import r2_score
            score = float(r2_score(y, preds))
        else:
            from sklearn.metrics import accuracy_score
            score = float(accuracy_score(y, preds))

        return {
            "model_id": model_id,
            "task_type": task_type,
            "metric": metric_name,
            "score": score,
            "fused_feature_names": list(fused_df.columns),
            "feature_count": fused_df.shape[1],
            "sample_count": fused_df.shape[0],
            "predictions": preds,
            "graph_id": graph.id,
        }

    # --- Meta-Learning & Warm Starts ---

    def get_meta_knowledge(
        self,
        dataset_id: str,
        run_id: str | None = None,
    ) -> Any:
        from automl.domain.meta_learning.fingerprint import MetaLearningKnowledge
        from automl.engine.meta_learning.extractor import extract_fingerprint
        from automl.engine.meta_learning.knowledge_base import MetaKnowledgeBase

        dataset = self._get_dataset(dataset_id)
        profile = self.repository.get_dataset_profile(dataset_id)
        if profile is None:
            profile = profile_dataset(dataset)
            self.repository.save_dataset_profile(profile)

        df = None
        try:
            df = load_dataframe(dataset.path)
        except Exception:
            pass

        fingerprint = extract_fingerprint(dataset, profile, df=df)

        # Retrieve any empirical trials in the workspace for this dataset
        trials = []
        try:
            runs = self.repository.list_runs()
            for r in runs:
                if r.dataset_id == dataset_id:
                    results = self.repository.list_trial_results(r.id)
                    trials.extend(results)
        except Exception:
            pass

        kb = MetaKnowledgeBase()
        return kb.synthesize(fingerprint, workspace_trials=trials if trials else None)

    # --- Multi-Model Ensemble Builder ---

    def build_ensemble(
        self,
        run_id: str,
        model_ids: list[str],
        method: str = "average",
        meta_model: str = "ridge",
        folds: int = 5,
        name: str | None = None,
    ) -> Experiment:
        run = self._get_run(run_id)
        dataset = self._get_dataset(run.dataset_id)

        valid_methods = {"average", "rank", "simplex", "stacked"}
        method_norm = method.lower()
        if method_norm not in valid_methods:
            raise ValueError(f"Unknown ensemble method '{method}'. Valid: {sorted(valid_methods)}")

        unique_models: list[str] = []
        for m in model_ids:
            if m not in unique_models and m not in {"voting_ensemble", "oof_blend"}:
                unique_models.append(m)

        if len(unique_models) < 2:
            raise ValueError("Building an ensemble requires at least 2 distinct base models.")

        self.model_registry.validate_for_task(unique_models, dataset.task_type)

        registry = self.get_feature_registry(run.dataset_id)
        features = registry.active_feature_names(dataset.target_column)
        if not features:
            profile = self.repository.get_dataset_profile(run.dataset_id)
            if profile:
                features = [c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column]
            if not features:
                raise ValueError("No features available to train ensemble.")

        exp_name = name or f"Ensemble ({method_norm.title()} - {len(unique_models)} models)"
        experiment = self.create_experiment(
            run=run,
            name=exp_name,
            feature_names=features,
            model_ids=["voting_ensemble"],
            hypothesis=f"Ensemble {method_norm.upper()} blend of [{', '.join(unique_models)}] with {folds} folds",
            validation_strategy="oof" if dataset.task_type == "binary_classification" else "kfold",
        )
        experiment.status = ExperimentStatus.RUNNING
        self.repository.save_experiment(experiment)

        df = load_dataframe(dataset.path)
        X = df[features]
        y = df[dataset.target_column]

        def _get_factory(mid: str):
            plugin = self.plugin_registry.get_model_plugin(mid)
            if plugin:
                return lambda: plugin.build_estimator(task_type=dataset.task_type)
            from automl.plugins.models.sklearn_models import build_sklearn_model
            return lambda: build_sklearn_model(mid, dataset.task_type)

        factories = {m: _get_factory(m) for m in unique_models}

        if dataset.task_type == "binary_classification":
            from automl.engine.ensemble.oof import evaluate_oof
            test_slice = X.iloc[:min(len(X), 10)].copy()
            oof_res = evaluate_oof(
                X=X,
                y=y,
                X_test=test_slice,
                factories=factories,
                folds=folds,
                seed=run.config.random_seed,
                method=method_norm,
                meta_model=meta_model,
            )
            best_score = float(oof_res.scores["oof_blend"])
            model_scores = oof_res.scores
            weights = oof_res.weights or {m: 1.0 / len(unique_models) for m in unique_models}
            cv_std = float(np.std(oof_res.fold_scores["oof_blend"]))
        else:
            from automl.engine.ensemble.voting import VotingEnsembleEstimator
            from automl.engine.training.sklearn_trainer import _build_pipeline, _sklearn_scoring
            from sklearn.model_selection import KFold, cross_val_score

            estimators = [(m, factories[m]()) for m in unique_models]
            voting_mode = "soft"
            opt_w = (method_norm == "simplex")
            if method_norm == "rank" and "classification" in dataset.task_type:
                voting_mode = "rank"
            estimator = VotingEnsembleEstimator(
                estimators=estimators,
                task_type=dataset.task_type,
                voting=voting_mode,
                optimize_weights=opt_w,
            )
            pipeline = _build_pipeline(X, estimator)
            cv_splitter = KFold(n_splits=folds, shuffle=True, random_state=run.config.random_seed)
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv_splitter,
                scoring=_sklearn_scoring(run.config.metric, dataset.task_type),
                n_jobs=1,
            )
            best_score = float(np.mean(scores))
            cv_std = float(np.std(scores))
            model_scores = {"ensemble": best_score}
            weights = {m: 1.0 / len(unique_models) for m in unique_models}

        trial = Trial(
            id=f"trial_{uuid.uuid4().hex[:8]}",
            experiment_id=experiment.id,
            model_id="voting_ensemble",
            seed=run.config.random_seed,
            parameters={
                "models": unique_models,
                "method": method_norm,
                "meta_model": meta_model,
                "folds": folds,
                "weights": weights,
            },
            status=TrialStatus.COMPLETED,
        )
        self.repository.save_trial(trial)

        trial_result = TrialResult(
            trial_id=trial.id,
            experiment_id=experiment.id,
            model_id="voting_ensemble",
            primary_metric=run.config.metric,
            primary_score=best_score,
            secondary_metrics={
                "method": method_norm,
                "meta_model": meta_model,
                "folds": folds,
                "weights": weights,
                "model_scores": model_scores,
                "cv_std": cv_std,
            },
        )
        self.repository.save_trial_result(trial_result)

        experiment.status = ExperimentStatus.COMPLETED
        self.repository.save_experiment(experiment)

        self._emit(
            "EnsembleBuilt",
            {
                "experiment_id": experiment.id,
                "score": best_score,
                "models": unique_models,
                "method": method_norm,
            },
            run_id=run.id,
        )
        return experiment




