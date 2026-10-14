"""Domain exceptions for experiment / validation contracts."""
from __future__ import annotations

from automl.domain.experiments.transitions import InvalidExperimentTransition

__all__ = [
    "InvalidExperimentTransition",
    "InvalidValidationStrategy",
]


class InvalidValidationStrategy(ValueError):
    """Raised when a validation strategy cannot be normalized to a known value."""
