from __future__ import annotations

from enum import Enum


class RunStatus(str, Enum):
    CREATED = "CREATED"
    PROFILING = "PROFILING"
    PLANNING = "PLANNING"
    EXPERIMENTING = "EXPERIMENTING"
    OPTIMIZING = "OPTIMIZING"
    FINALIZING = "FINALIZING"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ACTIVE_RUN_STATUSES: set[RunStatus] = {
    RunStatus.EXPERIMENTING,
    RunStatus.OPTIMIZING,
    RunStatus.FINALIZING,
}


def is_active_run_status(status: RunStatus | str | None) -> bool:
    """Returns True if the run status indicates active execution."""
    if status is None:
        return False
    val = status.value if hasattr(status, "value") else str(status)
    return val in {
        "EXPERIMENTING",
        "OPTIMIZING",
        "FINALIZING",
        "RUNNING",
    }


class RunPhase(str, Enum):
    DATASET_REGISTRATION = "DATASET_REGISTRATION"
    DATASET_PROFILING = "DATASET_PROFILING"
    PROBLEM_DEFINITION = "PROBLEM_DEFINITION"
    FEATURE_ANALYSIS = "FEATURE_ANALYSIS"
    EXPERIMENT_PLANNING = "EXPERIMENT_PLANNING"
    EXPERIMENT_EXECUTION = "EXPERIMENT_EXECUTION"
    EVALUATION = "EVALUATION"
    OPTIMIZATION = "OPTIMIZATION"
    MODEL_SELECTION = "MODEL_SELECTION"
    ENSEMBLE = "ENSEMBLE"
