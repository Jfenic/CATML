"""CATML - Modular, reproducible AutoML platform with experiment-first architecture."""

__version__ = "0.9.0-rc1"

from automl.artifacts.model_artifact import ModelArtifact
from automl.facade import AutoML, AutoMLResult

__all__ = [
    "__version__",
    "AutoML",
    "AutoMLResult",
    "ModelArtifact",
]
