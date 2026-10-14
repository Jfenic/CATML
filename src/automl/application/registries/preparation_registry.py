"""CQRS registration for Data Preparation handlers."""
from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.preparation.commands import (
    AnalyzeDataQualityCommand,
    ApplyPreparationPlanCommand,
    CreatePreparationPlanCommand,
    PreviewPreparationPlanCommand,
    SuggestPreparationPlanCommand,
    ValidatePreparationPlanCommand,
)
from automl.application.preparation.queries import (
    GetDatasetVersionQuery,
    GetPreparationPlanQuery,
    ListDatasetVersionsQuery,
    ListPreparationPlansQuery,
)
from automl.application.preparation.service import PreparationService

if TYPE_CHECKING:
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.services.workspace import AutoMLWorkspace


def register_preparation_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
) -> None:
    service = PreparationService(workspace)

    command_bus.register(AnalyzeDataQualityCommand, service.analyze)
    command_bus.register(SuggestPreparationPlanCommand, service.suggest)
    command_bus.register(CreatePreparationPlanCommand, service.create_plan)
    command_bus.register(PreviewPreparationPlanCommand, service.preview)
    command_bus.register(ValidatePreparationPlanCommand, service.validate)
    command_bus.register(ApplyPreparationPlanCommand, service.apply)

    query_bus.register(GetPreparationPlanQuery, service.get_plan)
    query_bus.register(ListPreparationPlansQuery, service.list_plans)
    query_bus.register(ListDatasetVersionsQuery, service.list_versions)
    query_bus.register(GetDatasetVersionQuery, service.get_version)
