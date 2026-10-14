"""Persistence ports for preparation plans and dataset versions."""
from __future__ import annotations

from typing import Protocol

from automl.domain.preparation.models import DatasetVersion, PreparationPlan


class PreparationPlanRepositoryPort(Protocol):
    def save(self, plan: PreparationPlan) -> None: ...

    def get(self, plan_id: str) -> PreparationPlan | None: ...

    def list_for_dataset(self, dataset_id: str) -> list[PreparationPlan]: ...


class DatasetVersionRepositoryPort(Protocol):
    def save(self, version: DatasetVersion) -> None: ...

    def get(self, version_id: str) -> DatasetVersion | None: ...

    def list_for_parent(self, parent_dataset_id: str) -> list[DatasetVersion]: ...
