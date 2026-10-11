"""Domain exceptions for experiment / validation contracts."""
from __future__ import annotations


class InvalidValidationStrategy(ValueError):
    """Raised when a validation strategy cannot be normalized to a known value."""
