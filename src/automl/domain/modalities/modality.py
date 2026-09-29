from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    """Supported data modalities in CATML."""
    TABULAR = "tabular"
    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    TIMESERIES = "timeseries"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class DataSource:
    """Represents a modality-specific data source location and metadata."""
    id: str
    modality: Modality
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)
