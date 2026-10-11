"""Shared metric-direction helpers (builtin names; plugins override in application layer)."""
from __future__ import annotations

BUILTIN_MINIMIZING_METRICS: frozenset[str] = frozenset(
    {"mae", "rmse", "mse", "loss", "log_loss"}
)


def is_minimizing_metric_name(metric: str) -> bool:
    """True when lower scores are better for a built-in metric name."""
    return str(metric).lower() in BUILTIN_MINIMIZING_METRICS
