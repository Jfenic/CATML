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


@dataclass(frozen=True)
class GetBestTrialQuery:
    experiment_id: str


@dataclass(frozen=True)
class GetExperimentTrialsQuery:
    experiment_id: str


@dataclass(frozen=True)
class GetFeatureEvidenceQuery:
    run_id: str
    feature_id: str | None = None


@dataclass(frozen=True)
class ListCandidateFeatureSetsQuery:
    run_id: str


@dataclass(frozen=True)
class GetFeatureRankingQuery:
    run_id: str
    method: str | None = None


@dataclass(frozen=True)
class ListPluginsQuery:
    plugin_type: str | None = None
    task_type: str | None = None


@dataclass(frozen=True)
class PredictDatasetQuery:
    run_id: str
    test_dataset_path: str
    experiment_id: str | None = None
    trial_id: str | None = None
    predict_proba: bool = False
    template_path: str | None = None
    id_column: str | None = None


@dataclass(frozen=True)
class ValidatePipelineGraphQuery:
    graph: Any


@dataclass(frozen=True)
class GetPipelineExecutionOrderQuery:
    graph: Any


@dataclass(frozen=True)
class GetOOFResultQuery:
    run_id: str
    experiment_id: str


@dataclass(frozen=True)
class DetectTemporalStructureQuery:
    dataset_id: str


@dataclass(frozen=True)
class GetMetaKnowledgeQuery:
    dataset_id: str
    run_id: str | None = None


@dataclass(frozen=True)
class GetRunQuery:
    run_id: str


@dataclass(frozen=True)
class ListRunsQuery:
    dataset_id: str | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class GetDatasetQuery:
    dataset_id: str


@dataclass(frozen=True)
class ListDatasetsQuery:
    workspace_id: str | None = None


@dataclass(frozen=True)
class GetTrialQuery:
    trial_id: str


@dataclass(frozen=True)
class GetExperimentQuery:
    experiment_id: str


@dataclass(frozen=True)
class ListTrialResultsQuery:
    experiment_id: str

