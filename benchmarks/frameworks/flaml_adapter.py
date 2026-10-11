"""FLAML adapter stub — install flaml for campaign runs; pilots may skip."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExternalAdapterResult:
    status: str
    primary_score: float | None
    wall_time_s: float
    failure_reason: str | None = None


def run_flaml(*_args, **_kwargs) -> ExternalAdapterResult:
    return ExternalAdapterResult(
        status="skipped",
        primary_score=None,
        wall_time_s=0.0,
        failure_reason="flaml_not_wired_in_pilot; install flaml and implement fit API in campaign",
    )
