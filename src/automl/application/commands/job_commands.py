from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SubmitJobCommand:
    operation: str
    run_id: str
    payload: dict[str, Any]
    idempotency_key: str


@dataclass(frozen=True)
class ControlJobCommand:
    job_id: str
    action: str
