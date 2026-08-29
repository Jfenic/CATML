from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

from automl.domain.datasets.profile import Dataset
from automl.domain.experiments.trial import Experiment, ExperimentStatus, Trial, TrialResult, TrialStatus
from automl.domain.features.feature import Feature
from automl.domain.features.registry import FeatureRegistry
from automl.domain.models.registry import ModelRegistry
from automl.domain.ports import TrialExecution
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus
from automl.engine.profiling.dataset_profiler import infer_task_type, load_dataframe, profile_dataset
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository
from automl.plugins.models.sklearn_models import default_model_specs


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
        inferred = task_type or infer_task_type(df[target])
        dataset = Dataset(
            id=dataset_id,
            workspace_id=self.id,
            name=name,
            path=resolved,
            target_column=target,
            task_type=inferred,
        )
        profile = profile_dataset(dataset)
        self._datasets[dataset_id] = dataset
        self.repository.save_dataset_profile(profile)

        registry = FeatureRegistry()
        for column in profile.columns:
            if column.name == target:
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
        return dataset

    def create_run(self, dataset: Dataset, metric: str | None = None) -> AutoMLRun:
        default_metric = "roc_auc" if dataset.task_type == "binary_classification" else "accuracy"
        if dataset.task_type == "regression":
            default_metric = "r2"
        config = RunConfig(
            task_type=dataset.task_type,
            target=dataset.target_column,
            metric=metric or default_metric,
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
        return run

    def get_feature_registry(self, dataset_id: str) -> FeatureRegistry:
        registry = self._feature_registries.get(dataset_id)
        if registry is None:
            raise KeyError(f"No feature registry for dataset {dataset_id}")
        return registry

    def create_experiment(
        self,
        run: AutoMLRun,
        name: str,
        feature_names: list[str] | None = None,
        model_ids: list[str] | None = None,
        hypothesis: str = "",
        priority: str = "normal",
    ) -> Experiment:
        dataset = self._datasets[run.dataset_id]
        registry = self.get_feature_registry(run.dataset_id)
        active_features = feature_names or registry.active_feature_names(dataset.target_column)
        active_models = model_ids or self.model_registry.resolve_active(
            run.config.models_include,
            run.config.models_exclude,
        )
        experiment = Experiment(
            id=f"exp_{uuid.uuid4().hex[:8]}",
            run_id=run.id,
            name=name,
            hypothesis=hypothesis or f"Compare models on {', '.join(active_features[:5])}",
            feature_names=active_features,
            model_ids=active_models,
            metric=run.config.metric,
            validation_strategy=run.config.validation_strategy,
            priority=priority,
        )
        self.repository.save_experiment(experiment)
        return experiment

    def run_experiment(self, run: AutoMLRun, experiment: Experiment) -> list[TrialResult]:
        dataset = self._datasets[run.dataset_id]
        experiment.status = ExperimentStatus.RUNNING
        self.repository.save_experiment(experiment)
        run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
        self.repository.save_run(run)

        trainer = SklearnTrainer()
        results = []
        for model_id in experiment.model_ids:
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

        experiment.status = ExperimentStatus.COMPLETED
        self.repository.save_experiment(experiment)
        run.transition_to(RunStatus.COMPLETED, RunPhase.EVALUATION)
        self.repository.save_run(run)
        return results

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

    def exclude_feature(self, dataset_id: str, name: str) -> None:
        self.get_feature_registry(dataset_id).exclude(name)

    def prioritize_feature(self, dataset_id: str, name: str, score: float = 1.0) -> None:
        self.get_feature_registry(dataset_id).prioritize(name, score)

    def exclude_model(self, run: AutoMLRun, model_id: str) -> None:
        if model_id not in run.config.models_exclude:
            run.config.models_exclude.append(model_id)
        self.repository.save_run(run)

    def include_model(self, run: AutoMLRun, model_id: str) -> None:
        if model_id not in run.config.models_include:
            run.config.models_include.append(model_id)
        if model_id in run.config.models_exclude:
            run.config.models_exclude.remove(model_id)
        self.repository.save_run(run)
