from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.queries.workspace_queries import (
    GetDatasetProfileQuery,
    GetDatasetQuery,
    GetRunQuery,
    GetTaskPlanQuery,
    ListDatasetsQuery,
    ListPluginsQuery,
    ListRunsQuery,
    ListTaskTypesQuery,
)

if TYPE_CHECKING:
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.services.workspace import AutoMLWorkspace


def register_core_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
) -> None:
    """Registers core entity queries: runs, datasets, dataset profiles, task types, plugins."""
    query_bus.register(GetRunQuery, lambda q: workspace.get_run(q.run_id))
    query_bus.register(ListRunsQuery, lambda q: workspace.list_runs(dataset_id=q.dataset_id))
    query_bus.register(GetDatasetQuery, lambda q: workspace.get_dataset(q.dataset_id))
    query_bus.register(ListDatasetsQuery, lambda _q: workspace.list_datasets())
    query_bus.register(GetDatasetProfileQuery, lambda q: workspace.get_dataset_profile(q.dataset_id))
    query_bus.register(GetTaskPlanQuery, lambda q: workspace.get_task_plan(q.dataset_id))
    query_bus.register(ListTaskTypesQuery, lambda _q: workspace.list_task_types())
    query_bus.register(
        ListPluginsQuery,
        lambda q: workspace.list_plugins(plugin_type=q.plugin_type, task_type=q.task_type),
    )
