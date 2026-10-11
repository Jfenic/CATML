"""Registration of CQRS command and query handlers for Experiment Recipes."""
from __future__ import annotations

from typing import TYPE_CHECKING

from automl.application.recipes.commands import (
    ArchiveExperimentRecipeCommand,
    CreateExperimentRecipeFromEvidenceCommand,
    PromoteExperimentRecipeCommand,
    SelectExperimentRecipeCommand,
    SubmitExperimentFromRecipeCommand,
)
from automl.application.recipes.queries import (
    GetExperimentRecipeQuery,
    ListExperimentRecipesQuery,
    ListVerifiedImprovementsQuery,
)

if TYPE_CHECKING:
    from automl.application.bus.command_bus import CommandBus
    from automl.application.bus.query_bus import QueryBus
    from automl.application.recipes.recipe_service import RecipeService
    from automl.application.services.workspace import AutoMLWorkspace


def register_recipe_handlers(
    workspace: AutoMLWorkspace,
    command_bus: CommandBus,
    query_bus: QueryBus,
    recipe_service: RecipeService | None = None,
) -> None:
    """Registers experiment recipe commands and queries."""
    service = recipe_service or workspace.get_recipe_service()

    # Commands
    command_bus.register(CreateExperimentRecipeFromEvidenceCommand, lambda cmd: service.create_recipe_from_evidence(cmd))
    command_bus.register(SubmitExperimentFromRecipeCommand, lambda cmd: service.submit_experiment_from_recipe(cmd))
    command_bus.register(SelectExperimentRecipeCommand, lambda cmd: service.select_recipe(cmd))
    command_bus.register(PromoteExperimentRecipeCommand, lambda cmd: service.promote_recipe(cmd))
    command_bus.register(ArchiveExperimentRecipeCommand, lambda cmd: service.archive_recipe(cmd))

    # Queries
    query_bus.register(GetExperimentRecipeQuery, lambda q: service.get_recipe(q))
    query_bus.register(ListExperimentRecipesQuery, lambda q: service.list_recipes(q))
    query_bus.register(ListVerifiedImprovementsQuery, lambda q: service.list_verified_improvements(q))
