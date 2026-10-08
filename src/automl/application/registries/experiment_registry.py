from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.commands.workspace_commands import (
    AddModelCommand,
    CancelRunCommand,
    CloneRunCommand,
    CreateExperimentCommand,
    ExcludeModelCommand,
    ExecuteNextExperimentCommand,
    OptimizeExperimentCommand,
    PauseRunCommand,
    PlanExperimentsCommand,
    PrioritizeCandidateCommand,
    ResumeRunCommand,
    RunExperimentCommand,
    RunScheduledExperimentsCommand,
)
from automl.application.queries.workspace_queries import (
    CompareExperimentsQuery,
    GetBestTrialQuery,
    GetExperimentQueueQuery,
    GetExperimentQuery,
    GetExperimentTrialsQuery,
    GetLeaderboardQuery,
    GetTrialQuery,
    ListCandidatesQuery,
    ListExperimentsQuery,
    ListModelsQuery,
    ListTrialResultsQuery,
)

if TYPE_CHECKING:
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.services.workspace import AutoMLWorkspace


def _run_experiment(workspace: AutoMLWorkspace, cmd: RunExperimentCommand):
    experiment = workspace.get_experiment(cmd.experiment_id)
    if experiment is None:
        raise KeyError(f"Experiment not found: {cmd.experiment_id}")
    run = workspace.get_run(cmd.run_id)
    if run is None:
        raise KeyError(f"Run not found: {cmd.run_id}")
    if experiment.run_id != run.id:
        raise ValueError(f"Experiment {experiment.id} does not belong to run {run.id}")
    return workspace.run_experiment(run, experiment)


def register_experiment_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
) -> None:
    """Registers experiment, trial, model inclusion and run execution handlers."""
    # Model inclusion / exclusion
    command_bus.register(
        AddModelCommand,
        lambda cmd: workspace.include_model(workspace.get_run(cmd.run_id), cmd.model_id),
    )
    command_bus.register(
        ExcludeModelCommand,
        lambda cmd: workspace.exclude_model(workspace.get_run(cmd.run_id), cmd.model_id),
    )

    # Experiment lifecycle
    command_bus.register(
        CreateExperimentCommand,
        lambda cmd: workspace.create_experiment(
            workspace.get_run(cmd.run_id),
            name=cmd.name,
            feature_names=cmd.feature_names,
            feature_set_id=cmd.feature_set_id,
            model_ids=cmd.model_ids,
            hypothesis=cmd.hypothesis,
            priority=cmd.priority,
            validation_strategy=cmd.validation_strategy,
            group_column=cmd.group_column,
        ),
    )
    command_bus.register(
        RunExperimentCommand,
        lambda cmd: _run_experiment(workspace, cmd),
    )
    command_bus.register(PauseRunCommand, lambda cmd: workspace.pause_run(cmd.run_id))
    command_bus.register(ResumeRunCommand, lambda cmd: workspace.resume_run(cmd.run_id))
    command_bus.register(CancelRunCommand, lambda cmd: workspace.cancel_run(cmd.run_id))
    command_bus.register(CloneRunCommand, lambda cmd: workspace.clone_run(cmd.run_id, cmd.new_name))

    # Planning & scheduling
    command_bus.register(
        PlanExperimentsCommand,
        lambda cmd: workspace.plan_experiments(
            cmd.run_id,
            auto_enqueue=cmd.auto_enqueue,
            user_priorities=cmd.user_priorities,
        ),
    )
    command_bus.register(
        PrioritizeCandidateCommand,
        lambda cmd: workspace.prioritize_candidate(
            cmd.run_id,
            cmd.candidate_id,
            cmd.priority,
        ),
    )
    command_bus.register(
        ExecuteNextExperimentCommand,
        lambda cmd: workspace.execute_next_experiment(cmd.run_id, budget=cmd.budget),
    )
    command_bus.register(
        RunScheduledExperimentsCommand,
        lambda cmd: workspace.run_scheduled_experiments(
            cmd.run_id,
            max_experiments=cmd.max_experiments,
            max_trials=cmd.max_trials,
            budget=cmd.budget,
        ),
    )
    command_bus.register(
        OptimizeExperimentCommand,
        lambda cmd: workspace.optimize_experiment(
            run_id=cmd.run_id,
            experiment_id=cmd.experiment_id,
            model_id=cmd.model_id,
            optimizer=cmd.optimizer,
            n_trials=cmd.n_trials,
            timeout_seconds=cmd.timeout_seconds,
            patience=cmd.patience,
            min_delta=cmd.min_delta,
        ),
    )

    # Queries
    query_bus.register(ListModelsQuery, lambda q: workspace.list_models(q.run_id, q.task_type))
    query_bus.register(
        GetLeaderboardQuery,
        lambda q: workspace.leaderboard(workspace.get_run(q.run_id)),
    )
    query_bus.register(CompareExperimentsQuery, lambda q: workspace.compare_experiments(q.experiment_ids))
    query_bus.register(ListExperimentsQuery, lambda q: workspace.list_experiments(q.run_id))
    query_bus.register(GetExperimentQuery, lambda q: workspace.get_experiment(q.experiment_id))
    query_bus.register(GetTrialQuery, lambda q: workspace.get_trial(q.trial_id))
    query_bus.register(ListTrialResultsQuery, lambda q: workspace.list_trial_results(q.experiment_id))
    query_bus.register(GetExperimentQueueQuery, lambda q: workspace.get_experiment_queue(q.run_id))
    query_bus.register(ListCandidatesQuery, lambda q: workspace.list_candidates(q.run_id))
    query_bus.register(GetBestTrialQuery, lambda q: workspace.get_best_trial(q.experiment_id))
    query_bus.register(GetExperimentTrialsQuery, lambda q: workspace.get_experiment_trials(q.experiment_id))
