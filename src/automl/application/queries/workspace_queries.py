from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ListModelsQuery:
    run_id: str | None = None
    task_type: str | None = None


@dataclass(frozen=True)
class GetDatasetProfileQuery:
    dataset_id: str


@dataclass(frozen=True)
class GetLeaderboardQuery:
    run_id: str


@dataclass(frozen=True)
class CompareExperimentsQuery:
    experiment_ids: list[str]


@dataclass(frozen=True)
class ListExperimentsQuery:
    run_id: str


@dataclass(frozen=True)
class ListFeatureSetsQuery:
    dataset_id: str


@dataclass(frozen=True)
class GetTaskPlanQuery:
    dataset_id: str


@dataclass(frozen=True)
class ListTaskTypesQuery:
    pass


@dataclass(frozen=True)
class GetExperimentQueueQuery:
    run_id: str


@dataclass(frozen=True)
class ListCandidatesQuery:
    run_id: str

