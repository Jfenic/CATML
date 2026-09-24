from __future__ import annotations

from typing import Any

from automl.domain.optimization.search_space import SearchSpace
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModelPluginPort
from automl.engine.optimization.search_space_builder import SearchSpaceBuilder
from automl.plugins.models.sklearn_models import build_sklearn_model


class SklearnModelPlugin(ModelPluginPort):
    def __init__(
        self,
        model_id: str | None = None,
        name: str = "",
        supported_tasks: list[str] | None = None,
        supports_proba: bool = True,
        version: str = "1.0.0",
        plugin_id: str | None = None,
        estimator_factory: Any = None,
        default_hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        actual_id = plugin_id or model_id or "custom_model"
        self.plugin_id = actual_id
        self.name = name or actual_id
        self.version = version
        self.plugin_type = PluginType.MODEL
        self.capabilities = PluginCapability(
            supported_tasks=list(supported_tasks or ["binary_classification"]),
            supported_modalities=["tabular"],
            requires_gpu=False,
            supports_proba=supports_proba,
        )
        self._model_id = actual_id
        self._estimator_factory = estimator_factory
        self._default_hyperparameters = dict(default_hyperparameters or {})

    def build_estimator(
        self,
        parameters: dict[str, Any] | None = None,
        task_type: str = "binary_classification",
        **kwargs: Any,
    ) -> Any:
        merged = {**self._default_hyperparameters, **(parameters or {}), **kwargs}
        if self._estimator_factory is not None:
            return self._estimator_factory(**merged)
        return build_sklearn_model(self._model_id, task_type=task_type, parameters=merged)

    def get_search_space(self, task_type: str) -> SearchSpace:
        return SearchSpaceBuilder.build_for_model(self._model_id, task_type=task_type)


def create_default_sklearn_plugins() -> list[SklearnModelPlugin]:
    return [
        SklearnModelPlugin(
            model_id="logistic_regression",
            name="Scikit-Learn Logistic Regression",
            supported_tasks=["binary_classification", "multiclass_classification"],
            supports_proba=True,
        ),
        SklearnModelPlugin(
            model_id="random_forest",
            name="Scikit-Learn Random Forest",
            supported_tasks=["binary_classification", "multiclass_classification", "regression"],
            supports_proba=True,
        ),
        SklearnModelPlugin(
            model_id="ridge",
            name="Scikit-Learn Ridge Regression",
            supported_tasks=["regression"],
            supports_proba=False,
        ),
        SklearnModelPlugin(
            model_id="svc",
            name="Scikit-Learn Support Vector Classifier",
            supported_tasks=["binary_classification", "multiclass_classification"],
            supports_proba=True,
        ),
        SklearnModelPlugin(
            model_id="svr",
            name="Scikit-Learn Support Vector Regressor",
            supported_tasks=["regression"],
            supports_proba=False,
        ),
        SklearnModelPlugin(
            model_id="kmeans",
            name="Scikit-Learn K-Means",
            supported_tasks=["clustering"],
            supports_proba=False,
        ),
    ]
