"""Domain exceptions for metric contracts."""
from __future__ import annotations


class InvalidMetric(ValueError):
    """Raised when a metric name cannot be resolved to a known MetricSpec."""
