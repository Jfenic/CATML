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
                }
                for c in self.columns
            ],
        }

    @property
    def identifier_column_names(self) -> list[str]:
        return [c.name for c in self.columns if c.is_identifier]

    @property
    def high_cardinality_column_names(self) -> list[str]:
        return [c.name for c in self.columns if c.is_high_cardinality]

    @property
    def recommended_feature_names(self) -> list[str]:
        return [
            c.name
            for c in self.columns
            if c.name != self.target_column and not c.is_identifier
        ]
