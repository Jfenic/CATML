"""Explicit Experiment status transitions (Clean Arch PR4)."""
from __future__ import annotations

from automl.domain.experiments.trial import ExperimentStatus


class InvalidExperimentTransition(ValueError):
    """Raised when an experiment status change is not allowed."""


# CANCELLED is sticky. COMPLETED/PARTIAL/FAILED may reopen for job retries.
_STICKY_TERMINAL: frozenset[ExperimentStatus] = frozenset({ExperimentStatus.CANCELLED})

ALLOWED_EXPERIMENT_TRANSITIONS: dict[ExperimentStatus, frozenset[ExperimentStatus]] = {
    ExperimentStatus.CREATED: frozenset(
        {
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.COMPLETED,
            ExperimentStatus.PARTIAL,
        }
    ),
    ExperimentStatus.QUEUED: frozenset(
        {
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
        }
    ),
    ExperimentStatus.RUNNING: frozenset(
        {
            ExperimentStatus.EVALUATING,
            ExperimentStatus.PAUSED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.COMPLETED,
            ExperimentStatus.PARTIAL,
        }
    ),
    ExperimentStatus.EVALUATING: frozenset(
        {
            ExperimentStatus.COMPLETED,
            ExperimentStatus.PARTIAL,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.PAUSED,
        }
    ),
    ExperimentStatus.PAUSED: frozenset(
        {
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.COMPLETED,
            ExperimentStatus.PARTIAL,
        }
    ),
    ExperimentStatus.COMPLETED: frozenset(
        {
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.PARTIAL,
        }
    ),
    ExperimentStatus.PARTIAL: frozenset(
        {
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
            ExperimentStatus.FAILED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.COMPLETED,
        }
    ),
    ExperimentStatus.FAILED: frozenset(
        {
            ExperimentStatus.QUEUED,
            ExperimentStatus.RUNNING,
            ExperimentStatus.PAUSED,
            ExperimentStatus.CANCELLED,
            ExperimentStatus.COMPLETED,
            ExperimentStatus.PARTIAL,
        }
    ),
    ExperimentStatus.CANCELLED: frozenset(),
}


def normalize_experiment_status(status: ExperimentStatus | str) -> ExperimentStatus:
    if isinstance(status, ExperimentStatus):
        return status
    return ExperimentStatus(str(status))


def can_transition_experiment(
    current: ExperimentStatus | str, new: ExperimentStatus | str
) -> bool:
    cur = normalize_experiment_status(current)
    nxt = normalize_experiment_status(new)
    if cur == nxt:
        return True
    return nxt in ALLOWED_EXPERIMENT_TRANSITIONS.get(cur, frozenset())


def assert_valid_experiment_transition(
    current: ExperimentStatus | str, new: ExperimentStatus | str
) -> None:
    cur = normalize_experiment_status(current)
    nxt = normalize_experiment_status(new)
    if can_transition_experiment(cur, nxt):
        return
    raise InvalidExperimentTransition(
        f"Invalid experiment transition: {cur.value} → {nxt.value}"
        + (" (cancelled is sticky)" if cur in _STICKY_TERMINAL else "")
    )
