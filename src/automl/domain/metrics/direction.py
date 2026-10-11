"""Shared metric-direction helpers (builtin names; plugins override in application layer).

Deprecated as a primary API: prefer ``automl.domain.metrics.spec.MetricSpec`` /
``resolve_metric_spec``. This module remains as a thin compatibility shim.
"""
from __future__ import annotations

from automl.domain.metrics.spec import (
    builtin_metric_specs,
    is_minimizing_metric_name,
)

# Names whose builtin direction is minimize (kept for callers importing the frozenset).
BUILTIN_MINIMIZING_METRICS: frozenset[str] = frozenset(
    name for name, spec in builtin_metric_specs().items() if spec.is_minimizing
)

__all__ = ["BUILTIN_MINIMIZING_METRICS", "is_minimizing_metric_name"]
