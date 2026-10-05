"""CatBoost Model Plugin for CATML.

Provides plug-and-play ModelPluginPort implementation for CatBoost gradient boosting,
with automatic fallback to scikit-learn HistGradientBoosting when native CatBoost is absent.
"""
from __future__ import annotations

from typing import Any

from automl.domain.optimization.search_space import ParameterSpec, SearchSpace
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModelPluginPort


class CatBoostPlugin(ModelPluginPort):
    """CATML Model Plugin for CatBoost with optional HistGradientBoosting fallback."""

    def __init__(self, use_fallback_if_missing: bool = True) -> None:
        self.plugin_id = "catboost"
        self.name = "CatBoost Gradient Boosting"
        self.version = "1.0.0"
        self.plugin_type = PluginType.MODEL
        self.capabilities = PluginCapability(
            supported_tasks=["binary_classification", "multiclass_classification", "regression"],
            supported_modalities=["tabular"],
            requires_gpu=False,
            supports_proba=True,
        )
        self.use_fallback_if_missing = use_fallback_if_missing
        self.fallback_backend = "HistGradientBoosting"

    @property
    def is_native(self) -> bool:
        return self.is_available

    @property
    def is_available(self) -> bool:
        try:
            import catboost  # noqa: F401
            return True
        except ImportError:
            return False

    def build_estimator(
        self,
        parameters: dict[str, Any] | None = None,
        task_type: str = "binary_classification",
        **kwargs: Any,
    ) -> Any:
        params = dict(parameters or {})
        params.update(kwargs)
        random_state = int(params.pop("random_state", 42))
        is_regression = task_type == "regression"

        if self.is_available:
            import catboost as cb

            # Map n_estimators to iterations if iterations is not explicitly provided
            if "iterations" not in params and "n_estimators" in params:
                params["iterations"] = params.pop("n_estimators")
            if is_regression:
                return cb.CatBoostRegressor(random_seed=random_state, verbose=0, **params)
            return cb.CatBoostClassifier(random_seed=random_state, verbose=0, **params)

        if self.use_fallback_if_missing:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

            max_iter = int(params.get("iterations", params.get("n_estimators", 100)))
            learning_rate = float(params.get("learning_rate", 0.1))
            depth = params.get("depth", params.get("max_depth", None))
            hist_params: dict[str, Any] = {
                "max_iter": max_iter,
                "learning_rate": learning_rate,
                "random_state": random_state,
            }
            if depth is not None:
                hist_params["max_depth"] = int(depth)

            if is_regression:
                return HistGradientBoostingRegressor(**hist_params)
            return HistGradientBoostingClassifier(**hist_params)

        raise ImportError("CatBoost is not installed. Please install it using `pip install catboost`.")

    def get_search_space(self, task_type: str) -> SearchSpace:
        space = SearchSpace()
        space.add(ParameterSpec.int("iterations", 50, 300, step=25, default=100))
        space.add(ParameterSpec.float("learning_rate", 0.01, 0.3, log=True, default=0.1))
        space.add(ParameterSpec.int("depth", 3, 10, default=6))
        return space
