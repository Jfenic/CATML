from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


class RecipeStatus(str, Enum):
    DRAFT = "draft"
    SELECTED = "selected"
    SUBMITTED = "submitted"
    PROMOTED = "promoted"
    ARCHIVED = "archived"


@dataclass
class ExperimentRecipe:
    """
    Pure domain representation of a reproducible experiment configuration
    born from statistical findings and hypothesis verification.
    """
    id: str
    name: str
    dataset_id: str
    status: RecipeStatus = RecipeStatus.DRAFT
    dataset_content_hash: str | None = None
    feature_names: list[str] = field(default_factory=list)
    derived_features: list[dict[str, Any]] = field(default_factory=list)
    column_transforms: list[dict[str, Any]] = field(default_factory=list)
    drop_features: list[str] = field(default_factory=list)
    model_ids: list[str] = field(default_factory=list)
    model_params: dict[str, Any] = field(default_factory=dict)
    validation_strategy: str = "k_fold"
    cv_folds: int = 5
    metric: str = "accuracy"
    random_seed: int = 42
    evidence_link_id: str | None = None
    hypothesis_id: str | None = None
    study_id: str | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    experiment_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def create(
        cls,
        name: str,
        dataset_id: str,
        feature_names: list[str],
        derived_features: list[dict[str, Any]] | None = None,
        column_transforms: list[dict[str, Any]] | None = None,
        drop_features: list[str] | None = None,
        model_ids: list[str] | None = None,
        model_params: dict[str, Any] | None = None,
        dataset_content_hash: str | None = None,
        validation_strategy: str = "k_fold",
        cv_folds: int = 5,
        metric: str = "accuracy",
        random_seed: int = 42,
        evidence_link_id: str | None = None,
        hypothesis_id: str | None = None,
        study_id: str | None = None,
        lineage: dict[str, Any] | None = None,
        recipe_id: str | None = None,
        status: RecipeStatus = RecipeStatus.DRAFT,
    ) -> ExperimentRecipe:
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            id=recipe_id or f"rcp_{uuid.uuid4().hex[:10]}",
            name=name,
            dataset_id=dataset_id,
            status=status,
            dataset_content_hash=dataset_content_hash,
            feature_names=list(feature_names),
            derived_features=list(derived_features or []),
            column_transforms=list(column_transforms or []),
            drop_features=list(drop_features or []),
            model_ids=list(model_ids or []),
            model_params=dict(model_params or {}),
            validation_strategy=validation_strategy,
            cv_folds=cv_folds,
            metric=metric,
            random_seed=random_seed,
            evidence_link_id=evidence_link_id,
            hypothesis_id=hypothesis_id,
            study_id=study_id,
            lineage=dict(lineage or {}),
            experiment_ids=[],
            created_at=now,
            updated_at=now,
        )

    def select(self, notes: str | None = None) -> None:
        """Mark recipe as explicitly selected for future runs."""
        self.status = RecipeStatus.SELECTED
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if notes:
            self.lineage.setdefault("history", []).append({
                "action": "selected",
                "timestamp": self.updated_at,
                "notes": notes,
            })

    def submit(self, experiment_id: str) -> None:
        """Associate an executed experiment and transition status."""
        if experiment_id not in self.experiment_ids:
            self.experiment_ids.append(experiment_id)
        if self.status in (RecipeStatus.DRAFT, RecipeStatus.SELECTED):
            self.status = RecipeStatus.SUBMITTED
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def promote(self, reason: str | None = None, notes: str | None = None) -> None:
        """Explicit, auditable promotion of recipe as a reusable high-performing candidate."""
        self.status = RecipeStatus.PROMOTED
        self.updated_at = datetime.now(timezone.utc).isoformat()
        promotion_entry = {
            "action": "promoted",
            "timestamp": self.updated_at,
            "reason": reason,
            "notes": notes,
        }
        self.lineage["promoted_at"] = self.updated_at
        if reason:
            self.lineage["promoted_reason"] = reason
        self.lineage.setdefault("history", []).append(promotion_entry)

    def archive(self) -> None:
        """Archive recipe."""
        self.status = RecipeStatus.ARCHIVED
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecipe:
        d = dict(data)
        if "status" in d:
            if isinstance(d["status"], str):
                d["status"] = RecipeStatus(d["status"].lower())
        return cls(**d)
