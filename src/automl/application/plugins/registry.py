from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from automl.domain.plugins.plugin import PluginType
from automl.domain.ports import (
    MetricPluginPort,
    ModelPluginPort,
    PluginPort,
    PreprocessorPluginPort,
)
from automl.engine.plugins.validator import CompatibilityValidator


@dataclass
class PluginRegistry:
    """Manages discovery, registration, and querying of modular plugins across the platform."""

    _plugins: dict[str, PluginPort] = field(default_factory=dict)

    def register(self, plugin: PluginPort) -> None:
        self._plugins[plugin.plugin_id] = plugin

    def unregister(self, plugin_id: str) -> None:
        self._plugins.pop(plugin_id, None)

    def has(self, plugin_id: str) -> bool:
        return plugin_id in self._plugins

    def get(self, plugin_id: str) -> PluginPort | None:
        return self._plugins.get(plugin_id)

    def require(self, plugin_id: str) -> PluginPort:
        plugin = self.get(plugin_id)
        if plugin is None:
            raise KeyError(f"Plugin '{plugin_id}' not found in registry")
        return plugin

    def __getitem__(self, plugin_id: str) -> PluginPort:
        return self.require(plugin_id)

    def get_model_plugin(self, model_id: str) -> ModelPluginPort | None:
        plugin = self._plugins.get(model_id)
        if plugin is None:
            return None
        if plugin.plugin_type != PluginType.MODEL:
            raise TypeError(f"Plugin '{model_id}' is not a ModelPluginPort (found {plugin.plugin_type})")
        return plugin  # type: ignore[return-value]

    def get_metric_plugin(self, metric_id: str) -> MetricPluginPort | None:
        plugin = self._plugins.get(metric_id)
        if plugin is None:
            return None
        if plugin.plugin_type != PluginType.METRIC:
            raise TypeError(f"Plugin '{metric_id}' is not a MetricPluginPort (found {plugin.plugin_type})")
        return plugin  # type: ignore[return-value]

    def list(
        self,
        plugin_type: PluginType | str | None = None,
        task_type: str | None = None,
    ) -> list[PluginPort]:
        type_str = plugin_type.value if isinstance(plugin_type, PluginType) else plugin_type
        results = list(self._plugins.values())

        if type_str:
            results = [p for p in results if p.plugin_type == type_str]
        if task_type:
            results = [p for p in results if p.capabilities.is_compatible_with_task(task_type)]

        return results

    def validate_plugin_for_task(self, plugin_id: str, task_type: str) -> None:
        plugin = self.get(plugin_id)
        if plugin is None:
            raise KeyError(f"Plugin not found in registry: {plugin_id}")
        CompatibilityValidator.validate_task_compatibility(plugin, task_type)

    def to_summary_list(self) -> list[dict]:
        return [
            {
                "plugin_id": p.plugin_id,
                "name": p.name,
                "version": p.version,
                "plugin_type": p.plugin_type.value if hasattr(p.plugin_type, "value") else str(p.plugin_type),
                "capabilities": p.capabilities.to_dict(),
            }
            for p in self._plugins.values()
        ]
