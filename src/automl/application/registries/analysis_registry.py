"""Registration of CQRS command and query handlers for CATML Explore studies."""
from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.analysis.commands import (
    ArchiveStudyCommand,
    CreateHypothesisCommand,
    CreateStudyCommand,
    RunAnalysisCommand,
)
from automl.application.analysis.queries import (
    GetAnalysisRunQuery,
    GetStudyQuery,
    ListAnalysisRunsQuery,
    ListFindingsQuery,
    ListHypothesesQuery,
    ListStudiesQuery,
    ListVisualizationsQuery,
)

if TYPE_CHECKING:
    from automl.application.analysis.study_service import AnalysisStudyService
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.services.workspace import AutoMLWorkspace


def register_analysis_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
    analysis_service: AnalysisStudyService | None = None,
) -> None:
    """Registers exploratory analysis commands and queries."""
    service = analysis_service or workspace.get_analysis_service()

    # Commands
    command_bus.register(CreateStudyCommand, lambda cmd: service.create_study(cmd))
    command_bus.register(RunAnalysisCommand, lambda cmd: service.run_study(cmd))
    command_bus.register(ArchiveStudyCommand, lambda cmd: service.archive_study(cmd))
    command_bus.register(CreateHypothesisCommand, lambda cmd: service.create_hypothesis(cmd))

    # Queries
    query_bus.register(GetStudyQuery, lambda q: service.get_study(q))
    query_bus.register(ListStudiesQuery, lambda q: service.list_studies(q))
    query_bus.register(GetAnalysisRunQuery, lambda q: service.get_analysis_run(q))
    query_bus.register(ListAnalysisRunsQuery, lambda q: service.list_analysis_runs(q))
    query_bus.register(ListFindingsQuery, lambda q: service.list_findings(q))
    query_bus.register(ListVisualizationsQuery, lambda q: service.list_visualizations(q))
    query_bus.register(ListHypothesesQuery, lambda q: service.list_hypotheses(q))
