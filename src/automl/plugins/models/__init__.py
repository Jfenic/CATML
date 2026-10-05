from __future__ import annotations

from automl.plugins.models.catboost_plugin import CatBoostPlugin
from automl.plugins.models.ensemble import VotingEnsemblePlugin
from automl.plugins.models.gradient_boosting import LightGBMPlugin, XGBoostPlugin
from automl.plugins.models.sklearn_models import ExtraTreesPlugin, MLPPlugin
from automl.plugins.models.vision_plugin import TimmVisionPlugin

__all__ = [
    "CatBoostPlugin",
    "ExtraTreesPlugin",
    "LightGBMPlugin",
    "MLPPlugin",
    "TimmVisionPlugin",
    "VotingEnsemblePlugin",
    "XGBoostPlugin",
]
