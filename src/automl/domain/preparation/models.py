"""Pure domain models for Data Preparation (no pandas/sklearn)."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class PreparationScope(str, Enum):
    """Where a step is allowed to run."""

    DATASET = "dataset"
    TRAINING_PIPELINE = "training_pipeline"


class PreparationOperation(str, Enum):
    """Typed operations only — no arbitrary cleaning scripts."""

    NORMALIZE_WHITESPACE = "normalize_whitespace"
    REPLACE_VALUES = "replace_values"
    CAST_TYPE = "cast_type"
    PARSE_DATES = "parse_dates"
    DROP_COLUMNS = "drop_columns"
    KEEP_COLUMNS = "keep_columns"
    DEDUPE = "dedupe"
    MARK_IMPUTE = "mark_impute"
    MARK_SCALE = "mark_scale"


# Operations that learn statistics from column values → must be training_pipeline.
LEARNING_OPERATIONS: frozenset[PreparationOperation] = frozenset(
    {
        PreparationOperation.MARK_IMPUTE,
        PreparationOperation.MARK_SCALE,
    }
)

# Destructive / semantic dataset ops → require approval by default.
APPROVAL_DEFAULT_OPERATIONS: frozenset[PreparationOperation] = frozenset(
    {
        PreparationOperation.DROP_COLUMNS,
        PreparationOperation.DEDUPE,
        PreparationOperation.REPLACE_VALUES,
        PreparationOperation.CAST_TYPE,
    }
)


class PlanStatus(str, Enum):
    DRAFT = "draft"
    SUGGESTED = "suggested"
    PREVIEWED = "previewed"
    VALIDATED = "validated"
    APPLIED = "applied"
    REJECTED = "rejected"


@dataclass
class PreparationStep:
    """One declarative preparation step."""

    operation: PreparationOperation
    scope: PreparationScope
    columns: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    rationale: str | None = None
    finding_ids: tuple[str, ...] = ()
    step_id: str = field(default_factory=lambda: f"pstep_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation": self.operation.value,
            "scope": self.scope.value,
            "columns": list(self.columns),
            "params": dict(self.params),
            "requires_approval": self.requires_approval,
            "rationale": self.rationale,
            "finding_ids": list(self.finding_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreparationStep:
        op = PreparationOperation(data["operation"])
        scope = PreparationScope(data["scope"])
        requires = data.get("requires_approval")
        if requires is None:
            requires = op in APPROVAL_DEFAULT_OPERATIONS and scope == PreparationScope.DATASET
        return cls(
            step_id=str(data.get("step_id") or f"pstep_{uuid.uuid4().hex[:8]}"),
            operation=op,
            scope=scope,
            columns=tuple(data.get("columns") or ()),
            params=dict(data.get("params") or {}),
            requires_approval=bool(requires),
            rationale=data.get("rationale"),
            finding_ids=tuple(data.get("finding_ids") or ()),
        )


@dataclass
class PreparationPlan:
    """Ordered list of typed preparation steps (Propose ≠ Accept until apply)."""

    id: str
    dataset_id: str
    steps: list[PreparationStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    name: str = ""
    notes: str | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def create(
        cls,
        dataset_id: str,
        steps: list[PreparationStep] | None = None,
        name: str = "",
        notes: str | None = None,
        plan_id: str | None = None,
    ) -> PreparationPlan:
        return cls(
            id=plan_id or f"pplan_{uuid.uuid4().hex[:10]}",
            dataset_id=dataset_id,
            steps=list(steps or []),
            name=name or f"prep_{dataset_id[:8]}",
            notes=notes,
        )

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "status": self.status.value,
            "notes": self.notes,
            "steps": [s.to_dict() for s in self.steps],
            "lineage": dict(self.lineage),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreparationPlan:
        return cls(
            id=data["id"],
            dataset_id=data["dataset_id"],
            name=data.get("name") or "",
            status=PlanStatus(data.get("status") or PlanStatus.DRAFT.value),
            notes=data.get("notes"),
            steps=[PreparationStep.from_dict(s) for s in (data.get("steps") or [])],
            lineage=dict(data.get("lineage") or {}),
            created_at=data.get("created_at") or datetime.now(timezone.utc).isoformat(),
            updated_at=data.get("updated_at") or datetime.now(timezone.utc).isoformat(),
        )


@dataclass
class QualityIssue:
    code: str
    severity: str  # info | warning | error
    message: str
    columns: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "columns": list(self.columns),
            "evidence": dict(self.evidence),
        }


@dataclass
class QualityReport:
    dataset_id: str
    issues: list[QualityIssue] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "issues": [i.to_dict() for i in self.issues],
            "created_at": self.created_at,
        }


@dataclass
class DatasetVersion:
    """Immutable prepared dataset artifact with lineage."""

    id: str
    parent_dataset_id: str
    plan_id: str
    path: str
    content_hash: str | None
    step_digests: list[str] = field(default_factory=list)
    registered_dataset_id: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    lineage: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        parent_dataset_id: str,
        plan_id: str,
        path: str,
        content_hash: str | None,
        step_digests: list[str] | None = None,
        registered_dataset_id: str | None = None,
        lineage: dict[str, Any] | None = None,
        version_id: str | None = None,
    ) -> DatasetVersion:
        return cls(
            id=version_id or f"dsver_{uuid.uuid4().hex[:10]}",
            parent_dataset_id=parent_dataset_id,
            plan_id=plan_id,
            path=path,
            content_hash=content_hash,
            step_digests=list(step_digests or []),
            registered_dataset_id=registered_dataset_id,
            lineage=dict(lineage or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetVersion:
        return cls(
            id=data["id"],
            parent_dataset_id=data["parent_dataset_id"],
            plan_id=data["plan_id"],
            path=data["path"],
            content_hash=data.get("content_hash"),
            step_digests=list(data.get("step_digests") or []),
            registered_dataset_id=data.get("registered_dataset_id"),
            created_at=data.get("created_at") or datetime.now(timezone.utc).isoformat(),
            lineage=dict(data.get("lineage") or {}),
        )
