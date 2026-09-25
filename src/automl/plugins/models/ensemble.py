from __future__ import annotations

from typing import Any
import numpy as np

from automl.domain.optimization.search_space import ParameterSpec, SearchSpace
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModelPluginPort
from automl.domain.tasks.task_type import TaskType
from automl.engine.ensemble.blender import blend_predictions, normalize_weights
from automl.engine.ensemble.voting import VotingEnsembleEstimator

# Re-export for public API and backward compatibility
__all__ = [
    "blend_predictions",
    "normalize_weights",
    "VotingEnsembleEstimator",
    "VotingEnsemblePlugin",
]


class VotingEnsemblePlugin(ModelPluginPort):
    """CATML Model Plugin for Voting Ensemble and Blending.

    Acts as an architectural adapter exposing VotingEnsembleEstimator to the CATML PluginRegistry.
    """

    def __init__(self) -> None:
        self.plugin_id = "voting_ensemble"
        self.name = "Voting Ensemble & Blending"
        self.version = "1.0.0"
        self.plugin_type = PluginType.MODEL
        self.capabilities = PluginCapability(
            supported_tasks=[
                TaskType.BINARY_CLASSIFICATION.value,
                TaskType.MULTICLASS_CLASSIFICATION.value,
                TaskType.REGRESSION.value,
            ],
            supported_modalities=["tabular"],
            requires_gpu=False,
            supports_proba=True,
        )

    def build_estimator(
        self,
        parameters: dict[str, Any] | None = None,
        task_type: str = "binary_classification",
        **kwargs: Any,
    ) -> VotingEnsembleEstimator:
        params = dict(parameters or {})
        params.update(kwargs)

        estimators = params.pop("estimators", None)
        weights = params.pop("weights", None)
        voting = params.pop("voting", "soft")
        refit = params.pop("refit", True)
        actual_task_type = params.pop("task_type", task_type)

        default_factory = params.pop("default_estimator_factory", None)
        if default_factory is None and estimators is None:
            from automl.plugins.models.sklearn_models import default_ensemble_factory
            default_factory = default_ensemble_factory

        return VotingEnsembleEstimator(
            estimators=estimators,
            weights=weights,
            task_type=actual_task_type,
            voting=voting,
            refit=refit,
            default_estimator_factory=default_factory,
            **params,
        )

    def get_search_space(self, task_type: str) -> SearchSpace:
        space = SearchSpace()
        if task_type != TaskType.REGRESSION.value:
            space.add(ParameterSpec.categorical("voting", ["soft", "hard"], default="soft"))
        return space

    def blend(
        self,
        predictions: list[np.ndarray | list[float] | list[list[float]]],
        weights: list[float] | None = None,
        task_type: str = "binary_classification",
    ) -> np.ndarray:
        return blend_predictions(predictions, weights=weights, task_type=task_type)

    def from_models(
        self,
        models: list[Any],
        weights: list[float] | None = None,
        task_type: str = "binary_classification",
        refit: bool = False,
    ) -> VotingEnsembleEstimator:
        estimator = VotingEnsembleEstimator(
            estimators=models,
            weights=weights,
            task_type=task_type,
            voting="soft",
            refit=refit,
        )
        if not refit:
            estimator._validate_pre_fitted_estimators(models)
            estimator.fitted_estimators_ = list(models)
            estimator.is_fitted_ = True
        return estimator
