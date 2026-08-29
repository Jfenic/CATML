from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureSet:
    id: str
    dataset_id: str
    name: str
    feature_names: list[str]
    version: int = 1
    created_by: str = "user"
    lineage: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "name": self.name,
            "feature_names": list(self.feature_names),
            "version": self.version,
            "created_by": self.created_by,
            "lineage": self.lineage,
        }

    @classmethod
    def from_dict(cls, data: dict) -> FeatureSet:
        return cls(
            id=data["id"],
            dataset_id=data["dataset_id"],
            name=data["name"],
            feature_names=list(data["feature_names"]),
            version=int(data.get("version", 1)),
            created_by=data.get("created_by", "user"),
            lineage=data.get("lineage", ""),
        )
