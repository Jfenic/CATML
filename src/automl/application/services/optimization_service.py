"""Application service for hyperparameter optimization of experiments."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from automl.domain.experiments.execution_spec import (
    build_execution_spec,
    trial_execution_from_spec,
)
from automl.domain.experiments.trial import ExperimentStatus, TrialResult
from automl.domain.ports import OptimizerPort
from automl.domain.runs.states import RunPhase, RunStatus
from automl.engine.optimization.random_search import RandomSearchOptimizer
from automl.engine.optimization.search_space_builder import SearchSpaceBuilder
from automl.engine.optimization.trial_factory import TrialFactory
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.plugins.optimizers.optuna_optimizer import OptunaOptimizer

if TYPE_CHECKING:
    from automl.application.services.workspace import AutoMLWorkspace


class OptimizationService:
    """Run HPO (Optuna / random search) for a single experiment model."""

    def __init__(self, workspace: AutoMLWorkspace) -> None:
        self.workspace = workspace

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

        ws = self.workspace
        run = ws._get_run(run_id)
        if run.status in {RunStatus.PAUSED, RunStatus.CANCELLED}:
            raise RuntimeError(f"Cannot optimize experiment: Run {run_id} is {run.status.value}")

        experiment = ws.repository.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Experiment not found: {experiment_id}")
        if experiment.run_id != run.id:
            raise ValueError(f"Experiment {experiment_id} does not belong to run {run_id}")

        dataset = ws._get_dataset(run.dataset_id)
        target_model = model_id or experiment.model_ids[0]
        if target_model not in experiment.model_ids:
            raise ValueError(f"Model {target_model} is not part of experiment {experiment_id}")

        experiment.transition_to(ExperimentStatus.RUNNING)
        ws.repository.save_experiment(experiment)
        run.transition_to(RunStatus.OPTIMIZING, RunPhase.OPTIMIZATION)
        ws.repository.save_run(run)

        space = SearchSpaceBuilder.build(target_model, run.config.task_type)
        metric_spec = ws._resolve_metric_spec(run.config.metric)
        direction = metric_spec.direction.value

        # Check meta-learning warm start priors
        warm_params: dict[str, Any] | None = None
        try:
            meta = ws.get_meta_knowledge(run.dataset_id, run_id=run.id)
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
        optimizer_key = str(optimizer or "random").lower().strip()
        if optimizer_key == "optuna":
            opt = OptunaOptimizer(
                seed=run.config.random_seed,
                direction=direction,
                patience=patience,
                min_delta=min_delta,
                warm_start_params=warm_params,
            )
        elif optimizer_key in {"random", "random_search", "randomsearch"}:
            opt = RandomSearchOptimizer(
                seed=run.config.random_seed,
                patience=patience,
                min_delta=min_delta,
                mode="max" if direction == "maximize" else "min",
                warm_start_params=warm_params,
            )
        else:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Supported: 'optuna', 'random' "
                f"(aliases: random_search, randomsearch)."
            )

        trainer = SklearnTrainer(plugin_registry=ws.plugin_registry)
        results: list[TrialResult] = []
        started_at = time.time()
        is_cancelled = False
        is_paused = False

        for trial_idx in range(n_trials):
            if timeout_seconds and (time.time() - started_at) > timeout_seconds:
                break
            if opt.should_stop():
                break

            if ws.execution_check:
                ws.execution_check()

            refreshed = ws.repository.get_run(run.id)
            if refreshed and refreshed.status == RunStatus.PAUSED:
                is_paused = True
                experiment.transition_to(ExperimentStatus.PAUSED)
                ws.repository.save_experiment(experiment)
                ws._emit("RunPaused", {"experiment_id": experiment.id, "trial_idx": trial_idx}, run_id=run.id)
                break

            if refreshed and refreshed.status == RunStatus.CANCELLED:
                is_cancelled = True
                experiment.transition_to(ExperimentStatus.CANCELLED)
                ws.repository.save_experiment(experiment)
                ws._emit("RunCancelled", {"experiment_id": experiment.id, "trial_idx": trial_idx}, run_id=run.id)
                break

            params = opt.suggest(trial_idx, space)
            # Estimator seed may vary per trial; CV fold seed stays fixed for fair HPO comparison.
            trial = TrialFactory.create(
                experiment_id=experiment.id,
                model_id=target_model,
                parameters=params,
                seed=run.config.random_seed + trial_idx,
            )
            ws.repository.save_trial(trial)

            # ExecutionSpec.random_seed = run fold seed (fixed); trial.seed may still vary.
            spec = build_execution_spec(
                run,
                experiment,
                dataset,
                model_id=target_model,
                params=params,
                greater_is_better=metric_spec.greater_is_better,
            )
            execution = trial_execution_from_spec(
                trial=trial,
                experiment=experiment,
                run=run,
                dataset=dataset,
                spec=spec,
            )
            # Feed MedianPruner via per-fold scores (no-op for optimizers without report_step).
            report_step = getattr(opt, "report_step", None)
            if callable(report_step):

                def _fold_callback(
                    step: int,
                    score: float,
                    _idx: int = trial_idx,
                    _report=report_step,
                ) -> bool:
                    try:
                        return bool(_report(_idx, step, float(score)))
                    except Exception:
                        return False

                execution.fold_callback = _fold_callback

            result = trainer.run(execution)
            ws.repository.persist_trial_outcome(trial, result)
            was_pruned = bool(
                (result.failure_reason or "").lower() == "pruned"
                or (result.secondary_metrics or {}).get("pruned")
            )
            opt.observe(
                trial_idx,
                params,
                result.primary_score,
                succeeded=result.succeeded and not was_pruned,
                pruned=was_pruned,
            )
            results.append(result)

            ws._emit(
                "TrialPruned" if was_pruned else ("TrialCompleted" if result.succeeded else "TrialFailed"),
                {
                    "trial_id": trial.id,
                    "model_id": target_model,
                    "score": result.primary_score,
                    "parameters": params,
                    "pruned": was_pruned,
                },
                run_id=run.id,
            )

        succeeded = [r for r in results if getattr(r, "succeeded", False)]
        if not is_cancelled and not is_paused:
            if not results or not succeeded:
                experiment.transition_to(ExperimentStatus.FAILED)
                ws.repository.save_experiment(experiment)
                run.transition_to(RunStatus.FAILED, RunPhase.OPTIMIZATION)
                ws.repository.save_run(run)
                ws._emit(
                    "ExperimentFailed",
                    {
                        "experiment_id": experiment.id,
                        "completed_trials": len(results),
                        "successful_trials": 0,
                        "phase": "optimization",
                    },
                    run_id=run.id,
                )
            else:
                experiment.transition_to(ExperimentStatus.COMPLETED)
                ws.repository.save_experiment(experiment)
                run.transition_to(RunStatus.COMPLETED, RunPhase.EVALUATION)
                ws.repository.save_run(run)

        best_score = opt.best_score() if succeeded else None
        best_params = opt.best_parameters() if succeeded else {}
        best_trial_id = ""
        if succeeded:
            best_idx = metric_spec.select_best_index([r.primary_score for r in succeeded])
            best_trial_id = succeeded[best_idx].trial_id

        # Honest HPO importances only when Optuna study can compute them (never fabricated).
        param_importance: list[dict] = []
        study = getattr(opt, "study", None)
        if study is not None and len(succeeded) >= 2:
            try:
                import optuna

                raw = optuna.importance.get_param_importances(study)
                total = float(sum(raw.values())) or 1.0
                param_importance = [
                    {"param": name, "importance": round(float(val) / total, 4)}
                    for name, val in sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
                ]
            except Exception:
                param_importance = []

        ws._emit(
            "ExperimentOptimized",
            {
                "experiment_id": experiment.id,
                "model_id": target_model,
                "trials_executed": len(results),
                "successful_trials": len(succeeded),
                "best_score": None if best_score is None else round(best_score, 4),
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
            "successful_trials": len(succeeded),
            "best_score": None if best_score is None else round(best_score, 4),
            "best_params": best_params,
            "best_trial_id": best_trial_id,
            "param_importance": param_importance,
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
