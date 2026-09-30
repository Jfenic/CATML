from dataclasses import dataclass


@dataclass(frozen=True)
class GetJobQuery:
    job_id: str


@dataclass(frozen=True)
class ListJobsQuery:
    run_id: str | None = None
