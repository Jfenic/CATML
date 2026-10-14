"""Explicit AutoMLRun status transitions (Clean Arch PR4)."""
from __future__ import annotations

from automl.domain.runs.states import RunStatus


class InvalidRunTransition(ValueError):
    """Raised when a run status change is not allowed."""


# CANCELLED is sticky (must not become COMPLETED via a late worker).
# COMPLETED/FAILED may reopen for additional experiments or retries.
_STICKY_TERMINAL: frozenset[RunStatus] = frozenset({RunStatus.CANCELLED})

ALLOWED_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset(
        {
            RunStatus.PROFILING,
            RunStatus.PLANNING,
            RunStatus.EXPERIMENTING,
            RunStatus.OPTIMIZING,
            RunStatus.PAUSED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
        }
    ),
    RunStatus.PROFILING: frozenset(
        {
            RunStatus.PLANNING,
            RunStatus.EXPERIMENTING,
            RunStatus.PAUSED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
        }
    ),
    RunStatus.PLANNING: frozenset(
        {
            RunStatus.EXPERIMENTING,
            RunStatus.OPTIMIZING,
            RunStatus.PAUSED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
        }
    ),
    RunStatus.EXPERIMENTING: frozenset(
        {
            RunStatus.OPTIMIZING,
            RunStatus.FINALIZING,
            RunStatus.PAUSED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
        }
    ),
    RunStatus.OPTIMIZING: frozenset(
        {
            RunStatus.EXPERIMENTING,
            RunStatus.FINALIZING,
            RunStatus.PAUSED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
        }
    ),
    RunStatus.FINALIZING: frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.PAUSED,
        }
    ),
    RunStatus.PAUSED: frozenset(
        {
            RunStatus.PLANNING,
            RunStatus.EXPERIMENTING,
            RunStatus.OPTIMIZING,
            RunStatus.FINALIZING,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
        }
    ),
    RunStatus.COMPLETED: frozenset(
        {
            RunStatus.PLANNING,
            RunStatus.EXPERIMENTING,
            RunStatus.OPTIMIZING,
            RunStatus.FINALIZING,
            RunStatus.PAUSED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.FAILED: frozenset(
        {
            RunStatus.PLANNING,
            RunStatus.EXPERIMENTING,
            RunStatus.OPTIMIZING,
            RunStatus.PAUSED,
            RunStatus.CANCELLED,
            RunStatus.COMPLETED,
        }
    ),
    RunStatus.CANCELLED: frozenset(),
}


def normalize_run_status(status: RunStatus | str) -> RunStatus:
    if isinstance(status, RunStatus):
        return status
    return RunStatus(str(status))


def can_transition_run(current: RunStatus | str, new: RunStatus | str) -> bool:
    cur = normalize_run_status(current)
    nxt = normalize_run_status(new)
    if cur == nxt:
        return True
    return nxt in ALLOWED_RUN_TRANSITIONS.get(cur, frozenset())


def assert_valid_run_transition(current: RunStatus | str, new: RunStatus | str) -> None:
    cur = normalize_run_status(current)
    nxt = normalize_run_status(new)
    if can_transition_run(cur, nxt):
        return
    raise InvalidRunTransition(
        f"Invalid run transition: {cur.value} → {nxt.value}"
        + (" (cancelled is sticky)" if cur in _STICKY_TERMINAL else "")
    )
