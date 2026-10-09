"""Application service coordinating CATML Explore studies, execution and queries."""
from __future__ import annotations

import logging
from typing import Any, Callable

from automl.application.analysis.commands import (
    ArchiveStudyCommand,
    CreateHypothesisCommand,
    CreateStudyCommand,
    RunAnalysisCommand,
)
from automl.application.analysis.queries import (
    GetAnalysisRunQuery,
    GetStudyQuery,
    ListAnalysisRunsQuery,
    ListFindingsQuery,
    ListHypothesesQuery,
    ListStudiesQuery,
    ListVisualizationsQuery,
)
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    DataSourceRef,
    StudySpec,
    StudyStatus,
)
from automl.domain.analysis.ports import StatisticalEnginePort, StudyRepositoryPort
from automl.engine.analysis.statistical_analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class AnalysisStudyService:
    """Coordinates lifecycle, execution, and queries for exploratory studies."""

    def __init__(
        self,
        repository: StudyRepositoryPort,
        dataset_resolver: Callable[[str], Any | None],
        statistical_engine: StatisticalEnginePort | None = None,
    ):
        self.repository = repository
        self.dataset_resolver = dataset_resolver
        self.statistical_engine = statistical_engine or StatisticalAnalyzer()

    # -------------------------------------------------------------
    # Command Handlers
    # -------------------------------------------------------------

    def create_study(self, cmd: CreateStudyCommand) -> str:
        """Create and persist an exploratory study specification."""
        dataset = self.dataset_resolver(cmd.dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset '{cmd.dataset_id}' not found in workspace.")

        data_source = DataSourceRef(
            dataset_id=dataset.id,
            path=dataset.path,
            content_hash=getattr(dataset, "content_hash", ""),
        )

        target = cmd.target_column if cmd.target_column is not None else getattr(dataset, "target_column", None)

        study = StudySpec.create(
            workspace_id=cmd.workspace_id,
            data_source=data_source,
            name=cmd.name or f"Study {dataset.name or dataset.id}",
            target_column=target,
            analysis_types=list(cmd.analysis_types),
            parameters=cmd.parameters,
            time_budget_seconds=cmd.time_budget_seconds,
        )

        self.repository.save_study(study)
        logger.info("Created study %s for dataset %s", study.id, dataset.id)
        return study.id

    def run_study(self, cmd: RunAnalysisCommand) -> str:
        """Execute mathematical and statistical analysis for a registered study."""
        study = self.repository.get_study(cmd.study_id)
        if study is None:
            raise KeyError(f"Study '{cmd.study_id}' not found.")

        run = AnalysisRun.create(study.id)
        study.status = StudyStatus.RUNNING
        self.repository.save_study(study)
        self.repository.save_analysis_run(run)

        try:
            findings, viz_specs, hypotheses = self.statistical_engine.analyze(
                data_source_path=study.data_source.path,
                target_column=study.target_column,
                analysis_types=list(cmd.analysis_types or study.analysis_types),
                parameters={"study_id": study.id, "run_id": run.id, **study.parameters},
            )

            for finding in findings:
                self.repository.save_finding(finding)

            for viz in viz_specs:
                self.repository.save_visualization_spec(viz)

            for hyp in hypotheses:
                self.repository.save_hypothesis(hyp)

            run.mark_completed(
                findings_count=len(findings),
                visualizations_count=len(viz_specs),
            )
            study.status = StudyStatus.COMPLETED
            self.repository.save_analysis_run(run)
            self.repository.save_study(study)
            logger.info("Successfully executed study %s (run %s): %d findings", study.id, run.id, len(findings))
            return run.id

        except Exception as exc:
            logger.error("Analysis execution failed for study %s: %s", study.id, exc, exc_info=True)
            run.mark_failed(str(exc))
            study.status = StudyStatus.FAILED
            self.repository.save_analysis_run(run)
            self.repository.save_study(study)
            raise

    def archive_study(self, cmd: ArchiveStudyCommand) -> None:
        """Mark a study as archived."""
        self.repository.archive_study(cmd.study_id)

    def create_hypothesis(self, cmd: CreateHypothesisCommand) -> str:
        """Formulate and persist an actionable hypothesis from a finding."""
        hyp = AnalysisHypothesis.create(
            study_id=cmd.study_id,
            finding_id=cmd.finding_id,
            description=cmd.description,
            proposed_action=cmd.proposed_action,
            experiment_delta=cmd.experiment_delta,
        )
        self.repository.save_hypothesis(hyp)
        return hyp.id

    # -------------------------------------------------------------
    # Query Handlers
    # -------------------------------------------------------------

    def get_study(self, qry: GetStudyQuery) -> dict[str, Any] | None:
        """Fetch a study specification by ID."""
        study = self.repository.get_study(qry.study_id)
        return study.to_dict() if study else None

    def list_studies(self, qry: ListStudiesQuery) -> list[dict[str, Any]]:
        """List studies matching workspace and optional dataset filter."""
        studies = self.repository.list_studies(qry.workspace_id, qry.dataset_id)
        return [s.to_dict() for s in studies]

    def get_analysis_run(self, qry: GetAnalysisRunQuery) -> dict[str, Any] | None:
        """Fetch an analysis run by ID."""
        run = self.repository.get_analysis_run(qry.run_id)
        return run.to_dict() if run else None

    def list_analysis_runs(self, qry: ListAnalysisRunsQuery) -> list[dict[str, Any]]:
        """List execution runs for a given study."""
        runs = self.repository.list_analysis_runs(qry.study_id)
        return [r.to_dict() for r in runs]

    def list_findings(self, qry: ListFindingsQuery) -> list[dict[str, Any]]:
        """List findings for a study or specific run."""
        findings = self.repository.list_findings(qry.study_id, qry.run_id)
        if qry.finding_type:
            ft_clean = qry.finding_type.upper()
            findings = [
                f for f in findings
                if (f.finding_type.value if hasattr(f.finding_type, "value") else str(f.finding_type)).upper() == ft_clean
            ]
        return [f.to_dict() for f in findings]

    def list_visualizations(self, qry: ListVisualizationsQuery) -> list[dict[str, Any]]:
        """List visualization specs for a study or specific run."""
        specs = self.repository.list_visualization_specs(qry.study_id, qry.run_id)
        return [v.to_dict() for v in specs]

    def list_hypotheses(self, qry: ListHypothesesQuery) -> list[dict[str, Any]]:
        """List inferred hypotheses for a study."""
        hyps = self.repository.list_hypotheses(qry.study_id)
        return [h.to_dict() for h in hyps]
