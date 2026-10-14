"""CQRS queries for Data Preparation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GetPreparationPlanQuery:
    plan_id: str


@dataclass(frozen=True)
class ListPreparationPlansQuery:
    dataset_id: str | None = None


@dataclass(frozen=True)
class ListDatasetVersionsQuery:
    parent_dataset_id: str


@dataclass(frozen=True)
class GetDatasetVersionQuery:
    version_id: str
