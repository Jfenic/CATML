"""Durable operation state, independent of execution and storage frameworks."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    CANCEL_REQUESTED = "cancel_requested"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


ACTIVE_STATUSES = {JobStatus.RUNNING, JobStatus.PAUSE_REQUESTED, JobStatus.CANCEL_REQUESTED}


@dataclass
class Job:
    id: str
    operation: str
    run_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str = ""
    status: JobStatus = JobStatus.QUEUED
    completed: int = 0
    total: int = 0
    message: str = "Waiting for worker"
    attempt: int = 0
    retry_failed_models: bool = False
    result: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def final_status(self, requested: JobStatus) -> JobStatus:
        # Controls committed before completion win over the execution outcome.
        if self.status == JobStatus.CANCEL_REQUESTED:
            return JobStatus.CANCELLED
        if self.status == JobStatus.PAUSE_REQUESTED:
            return JobStatus.PAUSED
        return requested

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return {**asdict(self), "status": self.status.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        return cls(**{**data, "status": JobStatus(data["status"])})
