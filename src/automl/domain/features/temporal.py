"""Pure domain entities and value objects for temporal dynamics and sequential features."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TemporalPeriodicityType(str, Enum):
    """Canonical periodicity types for cyclic feature encoding."""
    HOURLY = "hourly"         # 24 hours
    DAILY = "daily"           # 7 days (day of week)
    WEEKLY = "weekly"         # 52 weeks
    MONTHLY = "monthly"       # 12 months
    ANNUAL = "annual"         # 365.25 days
    CUSTOM = "custom"


@dataclass(frozen=True)
class TemporalPeriodicity:
    """Represents a discovered or declared cyclic periodicity."""
    name: str  # e.g. "hourly", "daily", "weekly", "monthly", "annual", "custom"
    period: float  # e.g. 24.0, 7.0, 12.0, 365.25
    column: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalPeriodicity:
        return cls(
            name=str(data["name"]),
            period=float(data["period"]),
            column=str(data["column"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class LagSpec:
    """Specification of an autoregressive lag feature X_{t-k}."""
    column: str
    lag: int
    feature_name: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LagSpec:
        return cls(
            column=str(data["column"]),
            lag=int(data["lag"]),
            feature_name=str(data["feature_name"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class DeltaSpec:
    """Specification of a rate-of-change / trend delta feature X_t - X_{t-k}."""
    column: str
    lag: int
    feature_name: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeltaSpec:
        return cls(
            column=str(data["column"]),
            lag=int(data["lag"]),
            feature_name=str(data["feature_name"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class CyclicalSpec:
    """Specification of trigonometric sine and cosine cyclical projections."""
    column: str
    period: float
    sin_feature_name: str
    cos_feature_name: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CyclicalSpec:
        return cls(
            column=str(data["column"]),
            period=float(data["period"]),
            sin_feature_name=str(data["sin_feature_name"]),
            cos_feature_name=str(data["cos_feature_name"]),
            description=str(data.get("description", "")),
        )


@dataclass(frozen=True)
class RollingWindowSpec:
    """Specification of a moving statistics window over sequential observations."""
    column: str
    window: int
    agg: str  # "mean", "std", "min", "max"
    feature_name: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RollingWindowSpec:
        return cls(
            column=str(data["column"]),
            window=int(data["window"]),
            agg=str(data["agg"]),
            feature_name=str(data["feature_name"]),
            description=str(data.get("description", "")),
        )


@dataclass
class TemporalStructure:
    """Diagnosed temporal and sequential structure of a dataset."""
    is_sequential: bool
    order_column: str | None = None
    detected_periodicities: list[TemporalPeriodicity] = field(default_factory=list)
    temporal_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_sequential": self.is_sequential,
            "order_column": self.order_column,
            "detected_periodicities": [p.to_dict() for p in self.detected_periodicities],
            "temporal_columns": list(self.temporal_columns),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalStructure:
        return cls(
            is_sequential=bool(data.get("is_sequential", False)),
            order_column=data.get("order_column"),
            detected_periodicities=[
                TemporalPeriodicity.from_dict(p)
                for p in data.get("detected_periodicities", [])
            ],
            temporal_columns=list(data.get("temporal_columns", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class GeneratedTemporalFeature:
    """Represents a generated temporal feature (lag, delta, cyclical projection or rolling stat)."""
    name: str
    feature_type: str  # "lag" | "delta" | "cyclical_sin" | "cyclical_cos" | "rolling_mean"
    source_column: str
    lag: int | None = None
    period: float | None = None
    window: int | None = None
    agg: str | None = None
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedTemporalFeature:
        return cls(
            name=str(data["name"]),
            feature_type=str(data["feature_type"]),
            source_column=str(data["source_column"]),
            lag=int(data["lag"]) if data.get("lag") is not None else None,
            period=float(data["period"]) if data.get("period") is not None else None,
            window=int(data["window"]) if data.get("window") is not None else None,
            agg=str(data["agg"]) if data.get("agg") is not None else None,
            description=str(data.get("description", "")),
            metadata=dict(data.get("metadata", {})),
        )
