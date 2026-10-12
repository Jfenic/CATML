"""Application service for running fixed-model experiments (non-HPO)."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from automl.domain.experiments.execution_spec import (
    build_execution_spec,
    trial_execution_from_spec,
)
from automl.domain.experiments.trial import Experiment, ExperimentStatus, Trial, TrialResult
from automl.domain.runs.run import AutoMLRun
from automl.domain.runs.states import RunPhase, RunStatus
from automl.engine.training.sklearn_trainer import SklearnTrainer

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class ExperimentExecutionService:
    """Execute experiments with a fixed model list (no hyperparameter search)."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

    def run_experiment(
        self,
        run: AutoMLRun,
        experiment: Experiment,
        start_index: int = 0,
        skip_model_ids: set[str] | None = None,
    ) -> list[TrialResult]:
        ws = self.workspace
        if run.status == RunStatus.CANCELLED:
            raise RuntimeError(f"Run {run.id} is cancelled")
        if experiment.run_id != run.id:
            raise ValueError(f"Experiment {experiment.id} does not belong to run {run.id}")

        dataset = ws._get_dataset(run.dataset_id)
        experiment.status = ExperimentStatus.RUNNING
        ws.repository.save_experiment(experiment)
        run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
        ws.repository.save_run(run)

        trainer = SklearnTrainer(plugin_registry=ws.plugin_registry)
        results: list[TrialResult] = []
        model_ids = experiment.model_ids
        if ws.execution_progress:
            ws.execution_progress(start_index, len(model_ids), "Training models")

        for index, model_id in enumerate(model_ids):
            if index < start_index:
                continue

            if skip_model_ids and model_id in skip_model_ids:
                if ws.execution_progress:
                    ws.execution_progress(index + 1, len(model_ids), f"Reused {model_id}")
                continue
            if ws.execution_check:
                ws.execution_check()
            refreshed = ws.repository.get_run(run.id)
            if refreshed.status == RunStatus.PAUSED:
                ws.repository.save_checkpoint(run.id, experiment.id, index)
                experiment.status = ExperimentStatus.PAUSED
                ws.repository.save_experiment(experiment)
                ws._emit("RunPaused", {"experiment_id": experiment.id, "next_index": index}, run_id=run.id)
                return results

            if refreshed.status == RunStatus.CANCELLED:
                experiment.status = ExperimentStatus.CANCELLED
                ws.repository.save_experiment(experiment)
                ws._emit("RunCancelled", {"experiment_id": experiment.id}, run_id=run.id)
                return results

            if ws._is_run_budget_exhausted(refreshed):
                run.config.extra["time_budget_exhausted"] = True
                refreshed.config.extra["time_budget_exhausted"] = True
                ws.repository.save_run(refreshed)
                ws._emit(
                    "TimeBudgetExhausted",
                    {
                        "run_id": run.id,
                        "experiment_id": experiment.id,
                        "model_id": model_id,
                        "completed_models": len(results),
                        "total_models": len(model_ids),
                    },
                    run_id=run.id,
                )
                break

            trial = Trial(
                id=f"trial_{uuid.uuid4().hex[:8]}",
                experiment_id=experiment.id,
                model_id=model_id,
                seed=run.config.random_seed,
            )
            ws.repository.save_trial(trial)
            spec = build_execution_spec(
                run,
                experiment,
                dataset,
                model_id=model_id,
                greater_is_better=ws._metric_greater_is_better(experiment.metric),
            )
            execution = trial_execution_from_spec(
                trial=trial,
                experiment=experiment,
                run=run,
                dataset=dataset,
                spec=spec,
            )
            result = trainer.run(execution)
            ws.repository.save_trial(trial)
            ws.repository.save_trial_result(result)
            results.append(result)
            if ws.execution_progress:
                ws.repository.save_checkpoint(run.id, experiment.id, index + 1)
                ws.execution_progress(index + 1, len(model_ids), f"Finished {model_id}")
            ws._emit(
                "TrialCompleted" if result.succeeded else "TrialFailed",
                {"trial_id": trial.id, "model_id": model_id, "score": result.primary_score},
                run_id=run.id,
            )

        if ws.execution_check:
            ws.execution_check()
        ws.repository.clear_checkpoint(run.id)

        succeeded = [r for r in results if getattr(r, "succeeded", False)]
        if not results or not succeeded:
            experiment.status = ExperimentStatus.FAILED
            ws.repository.save_experiment(experiment)
            run.transition_to(RunStatus.FAILED, RunPhase.EVALUATION)
            ws.repository.save_run(run)
            ws._emit(
                "ExperimentFailed",
                {
                    "experiment_id": experiment.id,
                    "completed_trials": len(results),
                    "successful_trials": 0,
                },
                run_id=run.id,
            )
            return results

        latest_run = ws.repository.get_run(run.id) or run
        budget_exhausted = bool((latest_run.config.extra or {}).get("time_budget_exhausted"))
        # Preserve valid trials but do not present a budget stop as a full completion.
        if budget_exhausted and len(succeeded) < len(model_ids):
            experiment.status = ExperimentStatus.PARTIAL
            latest_run.config.extra["completion_reason"] = "BUDGET_EXHAUSTED"
            ws.repository.save_experiment(experiment)
            latest_run.transition_to(RunStatus.COMPLETED, RunPhase.EVALUATION)
            ws.repository.save_run(latest_run)
            run = latest_run
            ws._emit(
                "ExperimentPartial",
                {
                    "experiment_id": experiment.id,
                    "completed_trials": len(results),
                    "successful_trials": len(succeeded),
                    "total_models": len(model_ids),
                    "completion_reason": "BUDGET_EXHAUSTED",
                },
                run_id=run.id,
            )
            return results

        experiment.status = ExperimentStatus.COMPLETED
        ws.repository.save_experiment(experiment)
        run.transition_to(RunStatus.COMPLETED, RunPhase.EVALUATION)
        ws.repository.save_run(run)
        ws._emit(
            "ExperimentCompleted",
            {
                "experiment_id": experiment.id,
                "completed_trials": len(results),
                "successful_trials": len(succeeded),
            },
            run_id=run.id,
        )
        return results
