from __future__ import annotations

from automl.domain.features.feature import Feature, FeatureStatus


class FeatureRegistry:
    def __init__(self, features: list[Feature] | None = None) -> None:
        self._features: dict[str, Feature] = {}
        for feature in features or []:
            self.register(feature)

    def register(self, feature: Feature) -> None:
        self._features[feature.name] = feature

    def get(self, name: str) -> Feature | None:
        return self._features.get(name)

    def list(self) -> list[Feature]:
        return list(self._features.values())

    def exclude(self, name: str) -> None:
        feature = self._require(name)
        feature.exclude()

    def prioritize(self, name: str, score: float = 1.0) -> None:
        feature = self._require(name)
        feature.prioritize(score)

    def active_feature_names(self, target: str) -> list[str]:
        names: list[str] = []
        for feature in self._features.values():
            if feature.name == target:
                continue
            if feature.status == FeatureStatus.EXCLUDED:
                continue
            names.append(feature.name)
        priority = [f for f in self._features.values() if f.status == FeatureStatus.PRIORITY]
        priority.sort(key=lambda f: f.user_priority, reverse=True)
        priority_names = [f.name for f in priority if f.name in names]
        rest = [n for n in names if n not in priority_names]
        return priority_names + rest

    def _require(self, name: str) -> Feature:
        feature = self._features.get(name)
        if feature is None:
            raise KeyError(f"Feature not found: {name}")
        return feature
