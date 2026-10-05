from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from automl.domain.modalities.modality import DataSource, Modality
from automl.domain.tasks.task_type import TaskType


@dataclass(frozen=True)
class ModalitySpec:
    """Base specification for an input modality within a ProblemSpec."""
    id: str
    modality: Modality
    columns: list[str] = field(default_factory=list)
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TabularSource(ModalitySpec):
    """Specification for structured tabular columns."""
    id: str = "tabular"
    modality: Modality = Modality.TABULAR
    columns: list[str] = field(default_factory=list)
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextSource(ModalitySpec):
    """Specification for freeform text columns extracted via NLP transformers."""
    id: str = "text"
    modality: Modality = Modality.TEXT
    columns: list[str] = field(default_factory=list)
    path: str | None = None
    max_features: int = 100
    ngram_range: tuple[int, int] = (1, 2)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImageSource(ModalitySpec):
    """Specification for image directories or image path columns."""
    id: str = "image"
    modality: Modality = Modality.IMAGE
    columns: list[str] = field(default_factory=list)
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    """Specification for the prediction target."""
    column: str
    task_type: TaskType | str = TaskType.BINARY_CLASSIFICATION
    positive_class: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_type_str(self) -> str:
        return self.task_type.value if isinstance(self.task_type, TaskType) else str(self.task_type)


@dataclass(frozen=True)
class ValidationSpec:
    """Explicit cross-validation or data split policy."""
    strategy: str = "holdout"  # holdout, kfold, stratified_kfold, time_series, group_kfold
    folds: int = 5
    test_size: float = 0.2
    group_column: str | None = None
    random_seed: int = 42
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Hardware, time, and budget execution constraints."""
    time_budget_seconds: int | None = None
    max_trials: int | None = None
    max_vram_gb: float | None = None
    allow_gpu: bool = True
    early_stopping_rounds: int = 10
    cooperative_cancellation: bool = True


@dataclass(frozen=True)
class BackendCapabilities:
    """Formal declaration of capabilities and hardware requirements for a model/backend."""
    backend_id: str
    supported_modalities: frozenset[Modality] = field(
        default_factory=lambda: frozenset({Modality.TABULAR})
    )
    supported_tasks: frozenset[str] = field(
        default_factory=lambda: frozenset({"binary_classification", "multiclass_classification", "regression"})
    )
    requires_gpu: bool = False
    supports_predict_proba: bool = True
    supports_hpo: bool = True
    export_formats: frozenset[str] = field(
        default_factory=lambda: frozenset({"joblib", "python"})
    )

    def can_handle(self, problem: ProblemSpec) -> tuple[bool, str]:
        """Verify whether this backend can execute the given ProblemSpec."""
        task_str = problem.target.task_type_str if problem.target else "binary_classification"
        if task_str not in self.supported_tasks:
            return False, f"Backend '{self.backend_id}' does not support task '{task_str}'."

        for inp in problem.inputs:
            if inp.modality not in self.supported_modalities:
                return False, f"Backend '{self.backend_id}' does not support modality '{inp.modality}'."

        return True, "Compatible."


@dataclass(frozen=True)
class EvaluationResult:
    """Heterogeneous metric results for any evaluation scenario."""
    primary_metric: str
    primary_score: float
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    resource_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "secondary_metrics": dict(self.secondary_metrics),
            "resource_metrics": dict(self.resource_metrics),
        }


@dataclass
class ProblemSpec:
    """
    Complete, immutable specification of a machine learning problem in CATML.

    Decouples data sources/modalities from the target definition, validation
    strategy, and execution constraints.
    """
    inputs: list[ModalitySpec] = field(default_factory=list)
    target: TargetSpec | None = None
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> set[Modality]:
        return {inp.modality for inp in self.inputs}

    @property
    def text_columns(self) -> list[str]:
        cols: list[str] = []
        for inp in self.inputs:
            if inp.modality == Modality.TEXT:
                cols.extend(inp.columns)
        return cols

    @property
    def tabular_columns(self) -> list[str]:
        cols: list[str] = []
        for inp in self.inputs:
            if inp.modality == Modality.TABULAR:
                cols.extend(inp.columns)
        return cols

    @property
    def all_feature_columns(self) -> list[str]:
        cols: list[str] = []
        for inp in self.inputs:
            cols.extend(inp.columns)
        return cols
