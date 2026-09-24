from __future__ import annotations

from typing import Any

from automl.domain.optimization.search_space import ParameterSpec, SearchSpace
from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import ModelPluginPort


class LightGBMPlugin(ModelPluginPort):
    def __init__(self, use_fallback_if_missing: bool = True) -> None:
        self.plugin_id = "lightgbm"
        self.name = "LightGBM Gradient Boosting"
        self.version = "1.0.0"
        self.plugin_type = PluginType.MODEL
        self.capabilities = PluginCapability(
            supported_tasks=["binary_classification", "multiclass_classification", "regression"],
            supported_modalities=["tabular"],
            requires_gpu=False,
            supports_proba=True,
        )
        self.use_fallback_if_missing = use_fallback_if_missing

    @property
    def is_available(self) -> bool:
        try:
            import lightgbm  # noqa: F401
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
            import lightgbm as lgb
            if is_regression:
                return lgb.LGBMRegressor(random_state=random_state, verbose=-1, **params)
            return lgb.LGBMClassifier(random_state=random_state, verbose=-1, **params)

        if self.use_fallback_if_missing:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            hist_params = {
                "max_iter": int(params.get("n_estimators", 100)),
                "learning_rate": float(params.get("learning_rate", 0.1)),
                "max_leaf_nodes": int(params.get("num_leaves", 31)),
                "random_state": random_state,
            }
            if is_regression:
                return HistGradientBoostingRegressor(**hist_params)
            return HistGradientBoostingClassifier(**hist_params)

        raise ImportError("LightGBM is not installed. Please install it using `pip install lightgbm`.")

    def get_search_space(self, task_type: str) -> SearchSpace:
        space = SearchSpace()
        space.add(ParameterSpec.int("n_estimators", 20, 200, step=10, default=100))
        space.add(ParameterSpec.float("learning_rate", 0.01, 0.3, log=True, default=0.1))
        space.add(ParameterSpec.int("num_leaves", 15, 63, step=4, default=31))
        return space


class XGBoostPlugin(ModelPluginPort):
    def __init__(self, use_fallback_if_missing: bool = True) -> None:
        self.plugin_id = "xgboost"
        self.name = "XGBoost Gradient Boosting"
        self.version = "1.0.0"
        self.plugin_type = PluginType.MODEL
        self.capabilities = PluginCapability(
            supported_tasks=["binary_classification", "multiclass_classification", "regression"],
            supported_modalities=["tabular"],
            requires_gpu=False,
            supports_proba=True,
        )
        self.use_fallback_if_missing = use_fallback_if_missing

    @property
    def is_available(self) -> bool:
        try:
            import xgboost  # noqa: F401
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
            import xgboost as xgb
            if is_regression:
                return xgb.XGBRegressor(random_state=random_state, **params)
            return xgb.XGBClassifier(random_state=random_state, eval_metric="logloss", **params)

        if self.use_fallback_if_missing:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            gb_params = {
                "n_estimators": int(params.get("n_estimators", 100)),
                "learning_rate": float(params.get("learning_rate", 0.1)),
                "max_depth": int(params.get("max_depth", 3)),
                "random_state": random_state,
            }
            if is_regression:
                return GradientBoostingRegressor(**gb_params)
            return GradientBoostingClassifier(**gb_params)

        raise ImportError("XGBoost is not installed. Please install it using `pip install xgboost`.")

    def get_search_space(self, task_type: str) -> SearchSpace:
        space = SearchSpace()
        space.add(ParameterSpec.int("n_estimators", 20, 200, step=10, default=100))
        space.add(ParameterSpec.float("learning_rate", 0.01, 0.3, log=True, default=0.1))
        space.add(ParameterSpec.int("max_depth", 3, 10, default=6))
        return space
