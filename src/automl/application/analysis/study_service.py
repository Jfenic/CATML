"""Application service coordinating CATML Explore studies, execution and queries."""
from __future__ import annotations

import logging
from typing import Any, Callable

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
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    DataSourceRef,
    EvidenceLink,
    StudySpec,
    StudyStatus,
)
from automl.domain.analysis.ports import StatisticalEnginePort, StudyRepositoryPort
from automl.engine.analysis.hypothesis_translator import HypothesisExperimentTranslator
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
        """List findings for a study or specific run with optional filters and pagination."""
        findings = self.repository.list_findings(qry.study_id, qry.run_id)

        # Filter by finding_type
        if qry.finding_type:
            ft_clean = qry.finding_type.upper()
            findings = [
                f for f in findings
                if (f.finding_type.value if hasattr(f.finding_type, "value") else str(f.finding_type)).upper() == ft_clean
            ]

        # Filter by category (substring match on finding_type)
        if getattr(qry, "category", None):
            cat_clean = qry.category.lower()
            findings = [
                f for f in findings
                if cat_clean in (f.finding_type.value if hasattr(f.finding_type, "value") else str(f.finding_type)).lower()
            ]

        # Filter by min_significance (threshold on p_value or p_value_fdr)
        if getattr(qry, "min_significance", None) is not None:
            threshold = float(qry.min_significance)
            filtered = []
            for f in findings:
                p_raw = f.p_value
                p_fdr = f.metrics.get("p_value_fdr") if isinstance(f.metrics, dict) else None
                is_significant = False
                if p_raw is not None and p_raw <= threshold:
                    is_significant = True
                elif p_fdr is not None and p_fdr <= threshold:
                    is_significant = True
                if is_significant:
                    filtered.append(f)
            findings = filtered

        # Filter by severity
        if getattr(qry, "severity", None):
            sev_clean = qry.severity.lower()
            findings = [
                f for f in findings
                if isinstance(f.metrics, dict) and str(f.metrics.get("severity", "")).lower() == sev_clean
            ]

        # Apply offset and limit pagination
        offset = max(0, getattr(qry, "offset", 0) or 0)
        findings = findings[offset:]
        limit = getattr(qry, "limit", None)
        if limit is not None and limit > 0:
            findings = findings[:limit]

        return [f.to_dict() for f in findings]

    def list_visualizations(self, qry: ListVisualizationsQuery) -> list[dict[str, Any]]:
        """List visualization specs for a study or specific run with optional filters and pagination."""
        specs = self.repository.list_visualization_specs(qry.study_id, qry.run_id)

        # Filter by specific chart_id
        if getattr(qry, "chart_id", None):
            specs = [s for s in specs if s.id == qry.chart_id]

        # Filter by visualization_type
        if getattr(qry, "visualization_type", None):
            vt_clean = qry.visualization_type.lower()
            specs = [
                s for s in specs
                if (s.chart_type.value if hasattr(s.chart_type, "value") else str(s.chart_type)).lower() == vt_clean
            ]

        # Apply offset and limit pagination
        offset = max(0, getattr(qry, "offset", 0) or 0)
        specs = specs[offset:]
        limit = getattr(qry, "limit", None)
        if limit is not None and limit > 0:
            specs = specs[:limit]

        return [v.to_dict() for v in specs]

    def list_hypotheses(self, qry: ListHypothesesQuery) -> list[dict[str, Any]]:
        """List inferred hypotheses for a study."""
        hyps = self.repository.list_hypotheses(qry.study_id)
        return [h.to_dict() for h in hyps]

    def get_hypothesis(self, qry: GetHypothesisQuery) -> dict[str, Any] | None:
        """Fetch an inferred hypothesis by ID."""
        hyp = self.repository.get_hypothesis(qry.hypothesis_id)
        return hyp.to_dict() if hyp else None

    def get_evidence_link(self, qry: GetEvidenceLinkQuery) -> dict[str, Any] | None:
        """Fetch an evidence link by ID."""
        link = self.repository.get_evidence_link(qry.link_id)
        return link.to_dict() if link else None

    def list_evidence_links(self, qry: ListEvidenceLinksQuery) -> list[dict[str, Any]]:
        """List evidence links matching query filters."""
        if qry.hypothesis_id:
            links = self.repository.list_evidence_links(hypothesis_id=qry.hypothesis_id)
        elif qry.study_id:
            hyps = self.repository.list_hypotheses(qry.study_id)
            links = []
            for h in hyps:
                links.extend(self.repository.list_evidence_links(hypothesis_id=h.id))
        else:
            links = self.repository.list_evidence_links()
        return [l.to_dict() for l in links]

    def verify_hypothesis(self, cmd: VerifyHypothesisCommand, workspace: Any) -> dict[str, Any]:
        """Empirically test a hypothesis against AutoML baseline in identical split (Propose ≠ Accept)."""
        from automl.benchmarks.runner import is_minimizing_metric

        hyp = self.repository.get_hypothesis(cmd.hypothesis_id)
        if not hyp:
            raise KeyError(f"Hypothesis '{cmd.hypothesis_id}' not found.")

        study = self.repository.get_study(hyp.study_id)
        if not study:
            raise KeyError(f"Study '{hyp.study_id}' not found.")

        target_col = study.target_column
        dataset = workspace.get_dataset(study.data_source.dataset_id)
        if not target_col:
            target_col = dataset.target_column

        if not target_col:
            raise ValueError("Cannot verify supervised ML hypothesis on an unsupervised dataset without a target column.")

        # Resolve or create run
        if cmd.run_id:
            run = workspace.get_run(cmd.run_id)
        else:
            existing_runs = workspace.list_runs(dataset_id=dataset.id)
            if existing_runs:
                run = existing_runs[0]
            else:
                run = workspace.create_run(dataset)

        metric = run.config.metric
        profile = workspace.get_dataset_profile(dataset.id)
        if profile and hasattr(profile, "resolve_safe_feature_names"):
            safe_features = profile.resolve_safe_feature_names()
        else:
            safe_features = [c for c in dataset.feature_names if c != target_col]

        # Determine baseline score
        experiments = workspace.repository.list_experiments(run.id)
        baseline_score: float | None = None

        if experiments:
            leaderboard = workspace.repository.get_leaderboard(run.id)
            if leaderboard:
                baseline_row = leaderboard[0]
                if hasattr(baseline_row, "primary_score"):
                    baseline_score = float(baseline_row.primary_score)
                elif isinstance(baseline_row, dict):
                    baseline_score = float(baseline_row.get("score") or baseline_row.get("primary_score", 0.0))

        if baseline_score is None:
            # Run baseline experiment
            default_models = ["ridge", "random_forest"] if dataset.task_type == "regression" else ["logistic_regression", "random_forest"]
            baseline_exp = workspace.create_experiment(
                run=run,
                name="baseline_pre_hypothesis",
                feature_names=safe_features,
                model_ids=default_models,
                hypothesis="Baseline configuration prior to hypothesis verification",
            )
            baseline_results = workspace.run_experiment(run, baseline_exp)
            succeeded_baseline = [r for r in baseline_results if r.succeeded]
            if not succeeded_baseline:
                raise RuntimeError("Failed to establish baseline experiment.")
            if is_minimizing_metric(metric, ws=workspace):
                baseline_score = min(r.primary_score for r in succeeded_baseline)
            else:
                baseline_score = max(r.primary_score for r in succeeded_baseline)

        # Translate hypothesis to candidate experiment
        default_candidate_models = list(cmd.models) if cmd.models else None
        candidate_spec = HypothesisExperimentTranslator.translate(
            hypothesis=hyp,
            available_features=safe_features,
            task_type=dataset.task_type,
            default_models=default_candidate_models,
        )

        candidate_exp = workspace.create_experiment(
            run=run,
            name=f"cand_{candidate_spec.name}",
            feature_names=candidate_spec.feature_names,
            model_ids=candidate_spec.model_ids,
            hypothesis=candidate_spec.hypothesis_text,
        )

        candidate_results = workspace.run_experiment(run, candidate_exp)
        succeeded_candidate = [r for r in candidate_results if r.succeeded]

        is_minimizing = is_minimizing_metric(metric, ws=workspace)
        if not succeeded_candidate:
            candidate_score = float("inf") if is_minimizing else -float("inf")
            delta = -float("inf")
            accepted = False
            notes = "Candidate experiment failed to produce any valid trial."
        else:
            if is_minimizing:
                candidate_score = min(r.primary_score for r in succeeded_candidate)
                delta = baseline_score - candidate_score
            else:
                candidate_score = max(r.primary_score for r in succeeded_candidate)
                delta = candidate_score - baseline_score

            accepted = delta > cmd.min_improvement
            if accepted:
                notes = (
                    f"Accepted: Candidate improved {metric} by +{delta:.4f} "
                    f"over baseline ({baseline_score:.4f} -> {candidate_score:.4f})."
                )
            else:
                notes = (
                    f"Rejected: Candidate delta was {delta:.4f} <= {cmd.min_improvement} "
                    f"on {metric} (baseline: {baseline_score:.4f}, candidate: {candidate_score:.4f})."
                )

        # Update hypothesis state
        hyp.status = "accepted" if accepted else "rejected"
        self.repository.save_hypothesis(hyp)

        # Create and save EvidenceLink
        link = EvidenceLink.create(
            hypothesis_id=hyp.id,
            experiment_id=candidate_exp.id,
            baseline_score=baseline_score,
            candidate_score=candidate_score if abs(candidate_score) != float("inf") else 0.0,
            metric=metric,
            accepted=accepted,
            notes=notes,
        )
        self.repository.save_evidence_link(link)

        return link.to_dict()
