"""Domain ports and protocols for CATML Explore."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    EvidenceLink,
    StatisticalFinding,
    StudySpec,
    VisualizationSpec,
)


@runtime_checkable
class StudyRepositoryPort(Protocol):
    """Storage contract for studies, analysis runs, findings and visualizations."""

    def save_study(self, study: StudySpec) -> None:
        """Persist or update a study specification."""
        ...

    def get_study(self, study_id: str) -> StudySpec | None:
        """Retrieve a study by its identifier."""
        ...

    def list_studies(self, workspace_id: str, dataset_id: str | None = None) -> list[StudySpec]:
        """List studies belonging to a workspace, optionally filtered by dataset."""
        ...

    def archive_study(self, study_id: str) -> None:
        """Mark a study as archived."""
        ...

    def save_analysis_run(self, run: AnalysisRun) -> None:
        """Persist or update an analysis run."""
        ...

    def get_analysis_run(self, run_id: str) -> AnalysisRun | None:
        """Retrieve an analysis run by identifier."""
        ...

    def list_analysis_runs(self, study_id: str) -> list[AnalysisRun]:
        """List all execution runs for a given study."""
        ...

    def save_finding(self, finding: StatisticalFinding) -> None:
        """Persist a quantitative statistical finding."""
        ...

    def list_findings(self, study_id: str, run_id: str | None = None) -> list[StatisticalFinding]:
        """Retrieve findings associated with a study or specific run."""
        ...

    def save_visualization_spec(self, spec: VisualizationSpec) -> None:
        """Persist a declarative visualization spec."""
        ...

    def list_visualization_specs(self, study_id: str, run_id: str | None = None) -> list[VisualizationSpec]:
        """Retrieve visualization specs associated with a study or specific run."""
        ...

    def save_hypothesis(self, hypothesis: AnalysisHypothesis) -> None:
        """Persist an inferred hypothesis."""
        ...

    def list_hypotheses(self, study_id: str) -> list[AnalysisHypothesis]:
        """List hypotheses inferred from study findings."""
        ...

    def save_evidence_link(self, link: EvidenceLink) -> None:
        """Persist an evidence verification link."""
        ...

    def get_evidence_link(self, link_id: str) -> EvidenceLink | None:
        """Retrieve an evidence link by identifier."""
        ...


@runtime_checkable
class StatisticalEnginePort(Protocol):
    """Computational contract for exploratory statistical analysis."""

    def analyze(
        self,
        data_source_path: str,
        target_column: str | None = None,
        analysis_types: list[str] | None = None,
        parameters: dict | None = None,
    ) -> tuple[list[StatisticalFinding], list[VisualizationSpec], list[AnalysisHypothesis]]:
        """Compute statistical findings, visualization specifications, and hypotheses from data."""
        ...
