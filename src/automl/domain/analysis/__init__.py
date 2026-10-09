"""CATML Explore — Domain Analysis Context.

Pure Python entities, value objects, and protocols for statistical studies,
exploratory data analysis, findings, visualizations, and evidence links.
Zero external dependencies (no pandas, scipy, sklearn).
"""
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    DataSourceRef,
    EvidenceLink,
    StatisticalFinding,
    StudySpec,
    StudyStatus,
    VisualizationSpec,
)
from automl.domain.analysis.ports import StudyRepositoryPort

__all__ = [
    "AnalysisHypothesis",
    "AnalysisRun",
    "DataSourceRef",
    "EvidenceLink",
    "StatisticalFinding",
    "StudySpec",
    "StudyStatus",
    "VisualizationSpec",
    "StudyRepositoryPort",
]
