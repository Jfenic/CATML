"""Pure domain entities and value objects for derived feature definitions and evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class DerivedFeatureType(str, Enum):
    """Execution mode of the derived feature definition."""
    FORMULA = "formula"
    PYTHON_CODE = "python_code"


@dataclass
class DerivedFeatureDefinition:
    """Specification of an engineered or derived feature candidate."""
    name: str
    expression_type: DerivedFeatureType
    expression: str
    description: str = ""
    source_columns: list[str] = field(default_factory=list)
    created_by: str = "human"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize definition to dictionary."""
        d = asdict(self)
        d["expression_type"] = self.expression_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DerivedFeatureDefinition:
        """Construct definition from dictionary."""
        return cls(
            name=data["name"],
            expression_type=DerivedFeatureType(data.get("expression_type", DerivedFeatureType.FORMULA.value)),
            expression=data.get("expression", ""),
            description=data.get("description", ""),
            source_columns=list(data.get("source_columns", [])),
            created_by=data.get("created_by", "human"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class FeatureEvaluationResult:
    """Validation and execution diagnostic report for a derived feature candidate."""
    is_valid: bool
    feature_name: str
    sample_values: list[Any] = field(default_factory=list)
    dtype: str = "unknown"
    row_count: int = 0
    null_count: int = 0
    null_percentage: float = 0.0
    is_constant: bool = False
    zero_division_occurred: bool = False
    infinite_values_handled: int = 0
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    summary_stats: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation result to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureEvaluationResult:
        """Construct evaluation result from dictionary."""
        return cls(
            is_valid=bool(data.get("is_valid", False)),
            feature_name=data.get("feature_name", ""),
            sample_values=list(data.get("sample_values", [])),
            dtype=data.get("dtype", "unknown"),
            row_count=int(data.get("row_count", 0)),
            null_count=int(data.get("null_count", 0)),
            null_percentage=float(data.get("null_percentage", 0.0)),
            is_constant=bool(data.get("is_constant", False)),
            zero_division_occurred=bool(data.get("zero_division_occurred", False)),
            infinite_values_handled=int(data.get("infinite_values_handled", 0)),
            warnings=list(data.get("warnings", [])),
            error_message=data.get("error_message"),
            summary_stats=dict(data.get("summary_stats", {})),
        )
