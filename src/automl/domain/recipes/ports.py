from __future__ import annotations

from typing import Protocol

from automl.domain.recipes.recipe import ExperimentRecipe, RecipeStatus


class ExperimentRecipeRepositoryPort(Protocol):
    """Persistence port for ExperimentRecipe entities."""

    def save(self, recipe: ExperimentRecipe) -> None:
        """Save or update an experiment recipe."""
        ...

    def get(self, recipe_id: str) -> ExperimentRecipe | None:
        """Retrieve an experiment recipe by its ID."""
        ...

    def list(
        self,
        dataset_id: str | None = None,
        status: RecipeStatus | str | None = None,
    ) -> list[ExperimentRecipe]:
        """List recipes optionally filtered by dataset ID and status."""
        ...

    def delete(self, recipe_id: str) -> None:
        """Delete an experiment recipe by ID."""
        ...
