"""Metric direction and MetricSpec contracts (pure domain)."""
from automl.domain.metrics.direction import (
    BUILTIN_MINIMIZING_METRICS,
    is_minimizing_metric_name,
)
from automl.domain.metrics.exceptions import InvalidMetric
from automl.domain.metrics.spec import (
    MetricDirection,
    MetricSpec,
    builtin_metric_specs,
    resolve_metric_spec,
)

__all__ = [
    "BUILTIN_MINIMIZING_METRICS",
    "InvalidMetric",
    "MetricDirection",
    "MetricSpec",
    "builtin_metric_specs",
    "is_minimizing_metric_name",
    "resolve_metric_spec",
]
