from __future__ import annotations

from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.application.commands.workspace_commands import (
    AddModelCommand,
    CancelRunCommand,
    CloneRunCommand,
    CreateExperimentCommand,
    CreateFeatureSetCommand,
    ExcludeFeatureCommand,
    ExcludeModelCommand,
    PauseRunCommand,
    PrioritizeFeatureCommand,
    ResumeRunCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    CompareExperimentsQuery,
    GetDatasetProfileQuery,
    GetLeaderboardQuery,
    GetTaskPlanQuery,
    ListExperimentsQuery,
    ListFeatureSetsQuery,
    ListModelsQuery,
    ListTaskTypesQuery,
)
from automl.application.services.workspace import AutoMLWorkspace


def _run_experiment(workspace: AutoMLWorkspace, cmd: RunExperimentCommand):
    experiment = workspace.repository.get_experiment(cmd.experiment_id)
    if experiment is None:
        raise KeyError(f"Experiment not found: {cmd.experiment_id}")
    return workspace.run_experiment(workspace._get_run(cmd.run_id), experiment)


def register_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
) -> None:
    command_bus.register(AddModelCommand, lambda cmd: workspace.include_model(workspace._get_run(cmd.run_id), cmd.model_id))
    command_bus.register(
        ExcludeModelCommand,
        lambda cmd: workspace.exclude_model(workspace._get_run(cmd.run_id), cmd.model_id),
    )
    command_bus.register(
        ExcludeFeatureCommand,
        lambda cmd: workspace.exclude_feature(cmd.dataset_id, cmd.feature_name, run_id=cmd.run_id),
    )
    command_bus.register(
        PrioritizeFeatureCommand,
        lambda cmd: workspace.prioritize_feature(
            cmd.dataset_id,
            cmd.feature_name,
            score=cmd.score,
            run_id=cmd.run_id,
        ),
    )
    command_bus.register(
        CreateFeatureSetCommand,
        lambda cmd: workspace.create_feature_set(
            cmd.dataset_id,
            cmd.name,
            cmd.feature_names,
            lineage=cmd.lineage,
        ),
    )
    command_bus.register(
        CreateExperimentCommand,
        lambda cmd: workspace.create_experiment(
            workspace._get_run(cmd.run_id),
            name=cmd.name,
            feature_names=cmd.feature_names,
            feature_set_id=cmd.feature_set_id,
            model_ids=cmd.model_ids,
            hypothesis=cmd.hypothesis,
            priority=cmd.priority,
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

    query_bus.register(ListModelsQuery, lambda q: workspace.list_models(q.run_id, q.task_type))
    query_bus.register(
        GetDatasetProfileQuery,
        lambda q: workspace.repository.get_dataset_profile(q.dataset_id),
    )
    query_bus.register(
        GetLeaderboardQuery,
        lambda q: workspace.leaderboard(workspace._get_run(q.run_id)),
    )
    query_bus.register(CompareExperimentsQuery, lambda q: workspace.compare_experiments(q.experiment_ids))
    query_bus.register(
        ListExperimentsQuery,
        lambda q: workspace.repository.list_experiments(q.run_id),
    )
    query_bus.register(
        ListFeatureSetsQuery,
        lambda q: workspace.repository.list_feature_sets(q.dataset_id),
    )
    query_bus.register(GetTaskPlanQuery, lambda q: workspace.get_task_plan(q.dataset_id))
    query_bus.register(ListTaskTypesQuery, lambda _q: workspace.list_task_types())


def build_application(root_dir: str | None = None) -> tuple[AutoMLWorkspace, CommandBus, QueryBus]:
    workspace = AutoMLWorkspace.load_or_create("default", root_dir=root_dir)
    command_bus = CommandBus()
    query_bus = QueryBus()
    register_handlers(workspace, command_bus, query_bus)
    return workspace, command_bus, query_bus
