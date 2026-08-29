from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from automl.domain.datasets.profile import Dataset
from automl.domain.experiments.priority import ExperimentPriority
from automl.domain.experiments.trial import Experiment, ExperimentStatus, Trial, TrialResult, TrialStatus
from automl.domain.features.feature import Feature
from automl.domain.features.feature_set import FeatureSet
from automl.domain.features.registry import FeatureRegistry
from automl.domain.models.registry import ModelRegistry
from automl.domain.ports import TrialExecution
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus
from automl.domain.tasks.problem_definition import ProblemDefinition
from automl.domain.tasks.task_type import TASK_CATALOG, TaskType, default_metric_for
from automl.engine.planning.task_planner import plan_from_dataframe
from automl.engine.profiling.dataset_profiler import infer_task_type, load_dataframe, profile_dataset
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository
from automl.plugins.models.sklearn_models import default_model_specs

PLATFORM_VERSION = "0.2.0"


@dataclass
class AutoMLWorkspace:
    id: str
    root_dir: Path
    repository: SQLiteExperimentRepository
    model_registry: ModelRegistry = field(default_factory=ModelRegistry)
    _runs: dict[str, AutoMLRun] = field(default_factory=dict)
    _datasets: dict[str, Dataset] = field(default_factory=dict)
    _feature_registries: dict[str, FeatureRegistry] = field(default_factory=dict)

    @classmethod
    def create(cls, name: str, root_dir: str | Path | None = None) -> AutoMLWorkspace:
        base = Path(root_dir or Path.cwd() / ".automl" / name)
        base.mkdir(parents=True, exist_ok=True)
        workspace_id = f"ws_{uuid.uuid4().hex[:8]}"
        repo = SQLiteExperimentRepository(base / "automl.db")
        workspace = cls(id=workspace_id, root_dir=base, repository=repo)
        for spec in default_model_specs():
            workspace.model_registry.register(spec)
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
        if features:
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
            )
            registry.register(feature)
            self.repository.save_feature(feature)
        self._feature_registries[dataset_id] = registry
        self._emit("DatasetRegistered", {"dataset_id": dataset_id, "name": name})
        return dataset

    def create_run(self, dataset: Dataset, metric: str | None = None) -> AutoMLRun:
        problem = self.repository.get_problem_definition(dataset.id)
        if problem is None:
            df = load_dataframe(dataset.path)
            problem = plan_from_dataframe(dataset, df)
            self.repository.save_problem_definition(problem)

        run_metric = metric or problem.default_metric
        config = RunConfig(
            task_type=problem.task_type.value,
            target=dataset.target_column,
            metric=run_metric,
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
    ) -> Experiment:
        dataset = self._get_dataset(run.dataset_id)
        priority_value = ExperimentPriority.parse(priority).value

        resolved_features = feature_names
        if feature_set_id:
            feature_set = self.repository.get_feature_set(feature_set_id)
            if feature_set is None:
                raise KeyError(f"FeatureSet not found: {feature_set_id}")
            resolved_features = feature_set.feature_names
        if resolved_features is None:
            registry = self.get_feature_registry(run.dataset_id)
            resolved_features = registry.active_feature_names(dataset.target_column)

        if model_ids:
            self.model_registry.validate_for_task(model_ids, run.config.task_type)

        active_models = model_ids or self.model_registry.resolve_active(
            run.config.models_include,
            run.config.models_exclude,
            task_type=run.config.task_type,
        )
        experiment = Experiment(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            run_id=run.id,
            name=name,
            hypothesis=hypothesis or f"Compare models on {', '.join(resolved_features[:5])}",
            feature_names=resolved_features,
            model_ids=active_models,
            metric=run.config.metric,
            validation_strategy=run.config.validation_strategy,
            priority=priority_value,
            feature_set_id=feature_set_id,
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
    ) -> list[TrialResult]:
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run.id} is cancelled")

        dataset = self._get_dataset(run.dataset_id)
        experiment.status = ExperimentStatus.RUNNING
        self.repository.save_experiment(experiment)
        run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
        self.repository.save_run(run)

        trainer = SklearnTrainer()
        results: list[TrialResult] = []
        model_ids = experiment.model_ids

        for index, model_id in enumerate(model_ids):
            if index < start_index:
                continue

            refreshed = self._get_run(run.id)
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
            )
            result = trainer.run(execution)
            self.repository.save_trial(trial)
            self.repository.save_trial_result(result)
            results.append(result)
            self._emit(
                "TrialCompleted" if result.succeeded else "TrialFailed",
                {"trial_id": trial.id, "model_id": model_id, "score": result.primary_score},
                run_id=run.id,
            )

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
        if model_id not in run.config.models_exclude:
            run.config.models_exclude.append(model_id)
        self.repository.save_run(run)
        self._emit("ModelExcluded", {"model_id": model_id}, run_id=run.id)

    def include_model(self, run: AutoMLRun, model_id: str) -> None:
        if model_id not in run.config.models_include:
            run.config.models_include.append(model_id)
        if model_id in run.config.models_exclude:
            run.config.models_exclude.remove(model_id)
        self.repository.save_run(run)
        self._emit("ModelAdded", {"model_id": model_id}, run_id=run.id)
