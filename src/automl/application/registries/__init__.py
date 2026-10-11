from __future__ import annotations

from automl.application.registries.analysis_registry import register_analysis_handlers
from automl.application.registries.core_registry import register_core_handlers
from automl.application.registries.experiment_registry import register_experiment_handlers
from automl.application.registries.feature_registry import register_feature_handlers
from automl.application.registries.inference_registry import register_inference_handlers
from automl.application.registries.job_registry import register_job_handlers

__all__ = [
    "register_analysis_handlers",
    "register_core_handlers",
    "register_experiment_handlers",
    "register_feature_handlers",
    "register_inference_handlers",
    "register_job_handlers",
]
