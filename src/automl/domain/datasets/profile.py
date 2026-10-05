from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    null_count: int
    unique_count: int
    sample_values: list[Any] = field(default_factory=list)
    cardinality_ratio: float = 0.0
    is_identifier: bool = False
    is_high_cardinality: bool = False
    is_text: bool = False
    is_group_candidate: bool = False
    is_image: bool = False
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    median: float | None = None
    q25: float | None = None
    q75: float | None = None
    skew: float | None = None
    target_correlation: float | None = None
    top_categories: list[dict[str, Any]] = field(default_factory=list)
    histogram: dict[str, Any] = field(default_factory=dict)
    box_plot: dict[str, Any] = field(default_factory=dict)


@dataclass
class Dataset:
    id: str
    workspace_id: str
    name: str
    path: str
    target_column: str
    task_type: str


@dataclass
class DatasetProfile:
    dataset_id: str
    row_count: int
    column_count: int
    target_column: str
    task_type: str
    columns: list[ColumnProfile] = field(default_factory=list)
    preview_rows: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    correlation_matrix: dict[str, Any] = field(default_factory=dict)
    temporal_structure: dict[str, Any] = field(default_factory=dict)
    group_leakage_reports: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "target_column": self.target_column,
            "task_type": self.task_type,
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "null_count": c.null_count,
                    "unique_count": c.unique_count,
                    "sample_values": c.sample_values,
                    "cardinality_ratio": c.cardinality_ratio,
                    "is_identifier": c.is_identifier,
                    "is_high_cardinality": c.is_high_cardinality,
                    "is_text": c.is_text,
                    "is_group_candidate": c.is_group_candidate,
                    "is_image": c.is_image,
                    "mean": c.mean,
                    "std": c.std,
                    "min": c.min,
                    "max": c.max,
                    "median": c.median,
                    "q25": c.q25,
                    "q75": c.q75,
                    "skew": c.skew,
                    "target_correlation": c.target_correlation,
                    "top_categories": c.top_categories,
                    "histogram": c.histogram,
                    "box_plot": c.box_plot,
                }
                for c in self.columns
            ],
            "preview_rows": self.preview_rows,
            "recommendations": self.recommendations,
            "correlation_matrix": self.correlation_matrix,
            "temporal_structure": self.temporal_structure,
            "group_leakage_reports": self.group_leakage_reports,
            "group_candidates": self.group_candidates,
            "image_columns": self.image_column_names,
            "has_group_leakage": self.has_group_leakage,
            "has_leakage": self.has_leakage,
            "leakage_columns": self.leakage_column_names,
        }

    @property
    def identifier_column_names(self) -> list[str]:
        return [c.name for c in self.columns if c.is_identifier]

    @property
    def high_cardinality_column_names(self) -> list[str]:
        return [c.name for c in self.columns if c.is_high_cardinality]

    @property
    def text_column_names(self) -> list[str]:
        return [c.name for c in self.columns if c.is_text]

    @property
    def image_column_names(self) -> list[str]:
        return [c.name for c in self.columns if c.is_image]

    @property
    def group_candidates(self) -> list[str]:
        return [c.name for c in self.columns if c.is_group_candidate]

    @property
    def has_group_leakage(self) -> bool:
        return any(r.get("leakage_detected", False) for r in self.group_leakage_reports) or any(
            r.get("type") == "group_leakage" for r in self.recommendations
        )

    @property
    def leakage_column_names(self) -> list[str]:
        return [
            r["column"]
            for r in self.recommendations
            if r.get("type") in {"leakage", "group_leakage"} and r.get("column") != self.target_column
        ]

    @property
    def has_leakage(self) -> bool:
        return any(r.get("type") in {"leakage", "group_leakage"} for r in self.recommendations) or self.has_group_leakage

    @property
    def recommended_feature_names(self) -> list[str]:
        leakage_cols = set(self.leakage_column_names)
        return [
            c.name
            for c in self.columns
            if c.name != self.target_column and not c.is_identifier and c.name not in leakage_cols
        ]
