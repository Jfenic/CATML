"""CQRS queries for CATML Explore studies."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GetStudyQuery:
    """Query to fetch a study specification by ID."""
    study_id: str


@dataclass(frozen=True)
class ListStudiesQuery:
    """Query to list studies in a workspace, optionally filtered by dataset."""
    workspace_id: str = "default"
    dataset_id: str | None = None


@dataclass(frozen=True)
class GetAnalysisRunQuery:
    """Query to fetch an analysis run record by ID."""
    run_id: str


@dataclass(frozen=True)
class ListAnalysisRunsQuery:
    """Query to list all runs associated with a study."""
    study_id: str


@dataclass(frozen=True)
class ListFindingsQuery:
    """Query to retrieve statistical findings for a study or run."""
    study_id: str
    run_id: str | None = None
    finding_type: str | None = None


@dataclass(frozen=True)
class ListVisualizationsQuery:
    """Query to retrieve visualization specs for a study or run."""
    study_id: str
    run_id: str | None = None


@dataclass(frozen=True)
class ListHypothesesQuery:
    """Query to retrieve inferred hypotheses for a study."""
    study_id: str
