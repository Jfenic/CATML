from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CreateExperimentRecipeFromEvidenceCommand:
    """
    Creates a draft reproducible ExperimentRecipe from verified hypothesis evidence.
    Does not execute training automatically (respects Propose ≠ Accept).
    """
    evidence_link_id: str
    recipe_name: str | None = None
    notes: str | None = None
    model_ids: list[str] | None = None
    validation_strategy: str | None = None
    cv_folds: int | None = None
    metric: str | None = None
    random_seed: int | None = None
    extra_lineage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SubmitExperimentFromRecipeCommand:
    """
    Submits and executes one or more experiments using the recipe's reproducible configuration.
    """
    recipe_id: str
    run_id: str | None = None
    model_ids: list[str] | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SelectExperimentRecipeCommand:
    """
    Explicitly transitions a recipe to SELECTED status.
    """
    recipe_id: str
    notes: str | None = None


@dataclass(frozen=True)
class PromoteExperimentRecipeCommand:
    """
    Promotes an experiment recipe after empirical verification across experiments.
    """
    recipe_id: str
    reason: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ArchiveExperimentRecipeCommand:
    """
    Archives an experiment recipe.
    """
    recipe_id: str
