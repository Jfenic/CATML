"""Jobs API controller."""
from __future__ import annotations

from http import HTTPStatus
from typing import Any

from automl.application.commands.job_commands import ControlJobCommand, SubmitJobCommand
from automl.application.queries.job_queries import GetJobQuery, ListJobsQuery


def try_get(handler: Any, path: str, query_params: dict, ws: Any, cmd: Any, qry: Any) -> bool:
    """Handle GET routes for this resource. Return True if the request was handled."""
    if path == "/api/jobs" or path.startswith("/api/jobs/"):
        try:
            if path == "/api/jobs":
                result = qry.dispatch(ListJobsQuery(query_params.get("run_id", [None])[0]))
            else:
                result = qry.dispatch(GetJobQuery(path.removeprefix("/api/jobs/")))
            handler._send_json(result)
        except KeyError as exc:
            handler._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        return True
    return False


def try_post(handler: Any, path: str, payload: dict, ws: Any, cmd: Any, qry: Any) -> bool:
    """Handle POST routes for this resource. Return True if the request was handled."""
    if path == "/api/jobs":
        job_id = cmd.dispatch(SubmitJobCommand(**payload))
        handler._send_json({"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}, HTTPStatus.ACCEPTED)
        return True

    if path.startswith("/api/jobs/"):
        parts = path.removeprefix("/api/jobs/").split("/")
        if len(parts) != 2:
            raise ValueError("Expected /api/jobs/{id}/{action}")
        job_id = cmd.dispatch(ControlJobCommand(*parts))
        handler._send_json({"job_id": job_id})
        return True
    return False
