"""CATML Explore — Application Layer.

Commands, queries, and application service for exploratory statistical studies.
"""
from automl.application.analysis.commands import (
    ArchiveStudyCommand,
    CreateHypothesisCommand,
    CreateStudyCommand,
    RunAnalysisCommand,
    VerifyHypothesisCommand,
)
from automl.application.analysis.queries import (
    GetAnalysisRunQuery,
    GetEvidenceLinkQuery,
    GetHypothesisQuery,
    GetStudyQuery,
    ListAnalysisRunsQuery,
    ListEvidenceLinksQuery,
    ListFindingsQuery,
    ListHypothesesQuery,
    ListStudiesQuery,
    ListVisualizationsQuery,
)
from automl.application.analysis.reporting import generate_study_markdown_report
from automl.application.analysis.study_service import AnalysisStudyService

__all__ = [
    "ArchiveStudyCommand",
    "CreateHypothesisCommand",
    "CreateStudyCommand",
    "RunAnalysisCommand",
    "VerifyHypothesisCommand",
    "GetAnalysisRunQuery",
    "GetStudyQuery",
    "ListFindingsQuery",
    "ListHypothesesQuery",
    "ListStudiesQuery",
    "ListVisualizationsQuery",
    "GetHypothesisQuery",
    "GetEvidenceLinkQuery",
    "ListEvidenceLinksQuery",
    "AnalysisStudyService",
    "generate_study_markdown_report",
]
