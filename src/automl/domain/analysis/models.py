"""Domain models for CATML Explore studies, runs, findings and evidence."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import uuid


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class StudyStatus(str, Enum):
    CREATED = "CREATED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


@dataclass(frozen=True)
class DataSourceRef:
    """Immutable reference to an ingested dataset or partitioned data source."""
    dataset_id: str
    path: str
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "path": self.path,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataSourceRef:
        return cls(
            dataset_id=str(data.get("dataset_id", "")),
            path=str(data.get("path", "")),
            content_hash=str(data.get("content_hash", "")),
        )


@dataclass
class StudySpec:
    """Specification of an exploratory statistical study."""
    id: str
    workspace_id: str
    data_source: DataSourceRef
    name: str
    target_column: str | None = None
    analysis_types: list[str] = field(default_factory=lambda: ["descriptive", "association"])
    parameters: dict[str, Any] = field(default_factory=dict)
    time_budget_seconds: float | None = None
    status: StudyStatus = StudyStatus.CREATED
    created_at: str = field(default_factory=_now_iso)

    @classmethod
    def create(
        cls,
        workspace_id: str,
        data_source: DataSourceRef,
        name: str,
        target_column: str | None = None,
        analysis_types: list[str] | None = None,
        parameters: dict[str, Any] | None = None,
        time_budget_seconds: float | None = None,
    ) -> StudySpec:
        study_id = f"study_{uuid.uuid4().hex[:8]}"
        return cls(
            id=study_id,
            workspace_id=workspace_id,
            data_source=data_source,
            name=name,
            target_column=target_column,
            analysis_types=list(analysis_types or ["descriptive", "association"]),
            parameters=dict(parameters or {}),
            time_budget_seconds=time_budget_seconds,
            status=StudyStatus.CREATED,
            created_at=_now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "data_source": self.data_source.to_dict(),
            "name": self.name,
            "target_column": self.target_column,
            "analysis_types": list(self.analysis_types),
            "parameters": dict(self.parameters),
            "time_budget_seconds": self.time_budget_seconds,
            "status": self.status.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StudySpec:
        ds_data = data.get("data_source") or {}
        return cls(
            id=str(data.get("id", "")),
            workspace_id=str(data.get("workspace_id", "")),
            data_source=DataSourceRef.from_dict(ds_data),
            name=str(data.get("name", "Exploratory Study")),
            target_column=data.get("target_column"),
            analysis_types=list(data.get("analysis_types", ["descriptive"])),
            parameters=dict(data.get("parameters", {})),
            time_budget_seconds=float(data["time_budget_seconds"]) if data.get("time_budget_seconds") is not None else None,
            status=StudyStatus(data.get("status", StudyStatus.CREATED.value)),
            created_at=str(data.get("created_at", _now_iso())),
        )


@dataclass
class AnalysisRun:
    """Record of a study execution run."""
    id: str
    study_id: str
    status: str = "RUNNING"
    started_at: str = field(default_factory=_now_iso)
    completed_at: str | None = None
    findings_count: int = 0
    visualizations_count: int = 0
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, study_id: str) -> AnalysisRun:
        run_id = f"arun_{uuid.uuid4().hex[:8]}"
        return cls(
            id=run_id,
            study_id=study_id,
            status="RUNNING",
            started_at=_now_iso(),
        )

    def mark_completed(self, findings_count: int, visualizations_count: int) -> None:
        self.status = "COMPLETED"
        self.completed_at = _now_iso()
        self.findings_count = findings_count
        self.visualizations_count = visualizations_count

    def mark_failed(self, error: str) -> None:
        self.status = "FAILED"
        self.completed_at = _now_iso()
        self.error_message = error

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "study_id": self.study_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "findings_count": self.findings_count,
            "visualizations_count": self.visualizations_count,
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisRun:
        return cls(
            id=str(data.get("id", "")),
            study_id=str(data.get("study_id", "")),
            status=str(data.get("status", "RUNNING")),
            started_at=str(data.get("started_at", _now_iso())),
            completed_at=data.get("completed_at"),
            findings_count=int(data.get("findings_count", 0)),
            visualizations_count=int(data.get("visualizations_count", 0)),
            error_message=data.get("error_message"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class StatisticalFinding:
    """A mathematically verified quantitative statistical finding."""
    id: str
    study_id: str
    analysis_run_id: str
    finding_type: str  # e.g., "skewness", "high_missingness", "correlation", "normality_violation", "outlier_density"
    column_name: str | None = None
    secondary_column: str | None = None
    method_name: str = ""
    summary: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    p_value: float | None = None
    effect_size: float | None = None
    limitations: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    @classmethod
    def create(
        cls,
        study_id: str,
        analysis_run_id: str,
        finding_type: str,
        summary: str,
        column_name: str | None = None,
        secondary_column: str | None = None,
        method_name: str = "",
        metrics: dict[str, Any] | None = None,
        p_value: float | None = None,
        effect_size: float | None = None,
        limitations: list[str] | None = None,
    ) -> StatisticalFinding:
        fid = f"find_{uuid.uuid4().hex[:8]}"
        return cls(
            id=fid,
            study_id=study_id,
            analysis_run_id=analysis_run_id,
            finding_type=finding_type,
            summary=summary,
            column_name=column_name,
            secondary_column=secondary_column,
            method_name=method_name,
            metrics=dict(metrics or {}),
            p_value=p_value,
            effect_size=effect_size,
            limitations=list(limitations or []),
            created_at=_now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "study_id": self.study_id,
            "analysis_run_id": self.analysis_run_id,
            "finding_type": self.finding_type,
            "column_name": self.column_name,
            "secondary_column": self.secondary_column,
            "method_name": self.method_name,
            "summary": self.summary,
            "metrics": dict(self.metrics),
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "limitations": list(self.limitations),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StatisticalFinding:
        return cls(
            id=str(data.get("id", "")),
            study_id=str(data.get("study_id", "")),
            analysis_run_id=str(data.get("analysis_run_id", "")),
            finding_type=str(data.get("finding_type", "observation")),
            column_name=data.get("column_name"),
            secondary_column=data.get("secondary_column"),
            method_name=str(data.get("method_name", "")),
            summary=str(data.get("summary", "")),
            metrics=dict(data.get("metrics", {})),
            p_value=float(data["p_value"]) if data.get("p_value") is not None else None,
            effect_size=float(data["effect_size"]) if data.get("effect_size") is not None else None,
            limitations=list(data.get("limitations", [])),
            created_at=str(data.get("created_at", _now_iso())),
        )


@dataclass
class VisualizationSpec:
    """Declarative, client-agnostic visualization specification."""
    id: str
    study_id: str
    analysis_run_id: str
    chart_type: str  # "histogram", "boxplot", "correlation_matrix", "bar", "scatter"
    title: str
    data_series: dict[str, Any] = field(default_factory=dict)
    axes_config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    @classmethod
    def create(
        cls,
        study_id: str,
        analysis_run_id: str,
        chart_type: str,
        title: str,
        data_series: dict[str, Any],
        axes_config: dict[str, Any] | None = None,
    ) -> VisualizationSpec:
        vid = f"viz_{uuid.uuid4().hex[:8]}"
        return cls(
            id=vid,
            study_id=study_id,
            analysis_run_id=analysis_run_id,
            chart_type=chart_type,
            title=title,
            data_series=dict(data_series),
            axes_config=dict(axes_config or {}),
            created_at=_now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "study_id": self.study_id,
            "analysis_run_id": self.analysis_run_id,
            "chart_type": self.chart_type,
            "title": self.title,
            "data_series": dict(self.data_series),
            "axes_config": dict(self.axes_config),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualizationSpec:
        return cls(
            id=str(data.get("id", "")),
            study_id=str(data.get("study_id", "")),
            analysis_run_id=str(data.get("analysis_run_id", "")),
            chart_type=str(data.get("chart_type", "bar")),
            title=str(data.get("title", "Visualization")),
            data_series=dict(data.get("data_series", {})),
            axes_config=dict(data.get("axes_config", {})),
            created_at=str(data.get("created_at", _now_iso())),
        )


@dataclass
class AnalysisHypothesis:
    """Actionable machine learning hypothesis inferred from a statistical finding."""
    id: str
    study_id: str
    finding_id: str
    description: str
    proposed_action: str
    experiment_delta: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"  # "proposed", "accepted", "rejected"
    created_at: str = field(default_factory=_now_iso)

    @classmethod
    def create(
        cls,
        study_id: str,
        finding_id: str,
        description: str,
        proposed_action: str,
        experiment_delta: dict[str, Any] | None = None,
    ) -> AnalysisHypothesis:
        hid = f"hyp_{uuid.uuid4().hex[:8]}"
        return cls(
            id=hid,
            study_id=study_id,
            finding_id=finding_id,
            description=description,
            proposed_action=proposed_action,
            experiment_delta=dict(experiment_delta or {}),
            status="proposed",
            created_at=_now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "study_id": self.study_id,
            "finding_id": self.finding_id,
            "description": self.description,
            "proposed_action": self.proposed_action,
            "experiment_delta": dict(self.experiment_delta),
            "status": self.status,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisHypothesis:
        return cls(
            id=str(data.get("id", "")),
            study_id=str(data.get("study_id", "")),
            finding_id=str(data.get("finding_id", "")),
            description=str(data.get("description", "")),
            proposed_action=str(data.get("proposed_action", "")),
            experiment_delta=dict(data.get("experiment_delta", {})),
            status=str(data.get("status", "proposed")),
            created_at=str(data.get("created_at", _now_iso())),
        )


@dataclass
class EvidenceLink:
    """Empirical verification link tying a hypothesis to an experiment outcome."""
    id: str
    hypothesis_id: str
    experiment_id: str
    baseline_score: float
    candidate_score: float
    metric: str
    accepted: bool = False
    notes: str = ""
    created_at: str = field(default_factory=_now_iso)

    @classmethod
    def create(
        cls,
        hypothesis_id: str,
        experiment_id: str,
        baseline_score: float,
        candidate_score: float,
        metric: str,
        accepted: bool = False,
        notes: str = "",
    ) -> EvidenceLink:
        lid = f"ev_{uuid.uuid4().hex[:8]}"
        return cls(
            id=lid,
            hypothesis_id=hypothesis_id,
            experiment_id=experiment_id,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            metric=metric,
            accepted=accepted,
            notes=notes,
            created_at=_now_iso(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "experiment_id": self.experiment_id,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "metric": self.metric,
            "accepted": self.accepted,
            "notes": self.notes,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceLink:
        return cls(
            id=str(data.get("id", "")),
            hypothesis_id=str(data.get("hypothesis_id", "")),
            experiment_id=str(data.get("experiment_id", "")),
            baseline_score=float(data.get("baseline_score", 0.0)),
            candidate_score=float(data.get("candidate_score", 0.0)),
            metric=str(data.get("metric", "ROC-AUC")),
            accepted=bool(data.get("accepted", False)),
            notes=str(data.get("notes", "")),
            created_at=str(data.get("created_at", _now_iso())),
        )
