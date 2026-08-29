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
                }
                for c in self.columns
            ],
        }
