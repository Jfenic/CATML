from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from automl.domain.recipes.recipe import ExperimentRecipe


@dataclass(frozen=True)
class ExperimentRecipeDTO:
    id: str
    name: str
    dataset_id: str
    status: str
    dataset_content_hash: str | None
    feature_names: list[str]
    derived_features: list[dict[str, Any]]
    column_transforms: list[dict[str, Any]]
    drop_features: list[str]
    model_ids: list[str]
    model_params: dict[str, Any]
    validation_strategy: str
    cv_folds: int
    metric: str
    random_seed: int
    evidence_link_id: str | None
    hypothesis_id: str | None
    study_id: str | None
    lineage: dict[str, Any]
    experiment_ids: list[str]
    created_at: str
    updated_at: str

    @classmethod
    def from_domain(cls, recipe: ExperimentRecipe) -> ExperimentRecipeDTO:
        return cls(
            id=recipe.id,
            name=recipe.name,
            dataset_id=recipe.dataset_id,
            status=recipe.status.value,
            dataset_content_hash=recipe.dataset_content_hash,
            feature_names=recipe.feature_names,
            derived_features=recipe.derived_features,
            column_transforms=recipe.column_transforms,
            drop_features=recipe.drop_features,
            model_ids=recipe.model_ids,
            model_params=recipe.model_params,
            validation_strategy=recipe.validation_strategy,
            cv_folds=recipe.cv_folds,
            metric=recipe.metric,
            random_seed=recipe.random_seed,
            evidence_link_id=recipe.evidence_link_id,
            hypothesis_id=recipe.hypothesis_id,
            study_id=recipe.study_id,
            lineage=recipe.lineage,
            experiment_ids=recipe.experiment_ids,
            created_at=recipe.created_at,
            updated_at=recipe.updated_at,
        )


@dataclass(frozen=True)
class VerifiedImprovementDTO:
    evidence_link_id: str
    study_id: str
    hypothesis_id: str
    dataset_id: str
    suggested_action: str
    hypothesis_description: str
    accepted: bool
    delta_primary_metric: float
    baseline_score: float
    candidate_score: float
    metric_name: str
    candidate_experiment_id: str | None
    baseline_experiment_id: str | None
    created_at: str


@dataclass(frozen=True)
class GetExperimentRecipeQuery:
    recipe_id: str


@dataclass(frozen=True)
class ListExperimentRecipesQuery:
    dataset_id: str | None = None
    status: str | None = None


@dataclass(frozen=True)
class ListVerifiedImprovementsQuery:
    study_id: str | None = None
    dataset_id: str | None = None
    accepted_only: bool = True
