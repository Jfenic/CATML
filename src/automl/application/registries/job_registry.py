from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.commands.job_commands import ControlJobCommand, SubmitJobCommand
from automl.application.queries.job_queries import GetJobQuery, ListJobsQuery

if TYPE_CHECKING:
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.services.jobs import JobService
    from automl.application.services.workspace import AutoMLWorkspace


def register_job_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
    job_service: JobService | None = None,
) -> None:
    """Registers background job commands and queries."""
    if job_service is None:
        from automl.application.services.jobs import JobService
        from automl.infrastructure.database.sqlite_jobs import SQLiteJobRepository

        job_service = JobService(workspace, SQLiteJobRepository(workspace.repository.db_path))

    command_bus.register(SubmitJobCommand, job_service.submit)
    command_bus.register(ControlJobCommand, job_service.control)
    query_bus.register(GetJobQuery, lambda q: job_service.get(q.job_id))
    query_bus.register(ListJobsQuery, lambda q: job_service.list(q.run_id))
