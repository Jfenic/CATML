"""Canonical Workbench / job experiment time-budget labels (single source of truth)."""
from __future__ import annotations

# Labels shown in NewExperimentModal must match these seconds.
EXPERIMENT_BUDGET_SECONDS: dict[str, float] = {
    "quick": 60.0,
    "balanced": 300.0,
    "thorough": 1200.0,
}


def resolve_time_budget_seconds(
    budget: str | float | None = None,
    time_budget_seconds: float | None = None,
    *,
    default: float = 300.0,
) -> float:
    """Prefer an explicit seconds value; otherwise map a budget label."""
    if time_budget_seconds is not None:
        return float(time_budget_seconds)
    if budget is None:
        return float(default)
    if isinstance(budget, (int, float)) and not isinstance(budget, bool):
        return float(budget)
    key = str(budget).lower().strip()
    return float(EXPERIMENT_BUDGET_SECONDS.get(key, default))
