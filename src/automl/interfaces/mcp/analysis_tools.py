"""Analysis and Explore tools for CATML Model Context Protocol (MCP) server."""
from __future__ import annotations

import json
import logging
from typing import Any

try:
    import mcp.types as types
except ImportError:
    types = Any  # type: ignore

from automl.application.analysis.commands import CreateStudyCommand, RunAnalysisCommand
from automl.application.analysis.queries import (
    GetAnalysisRunQuery,
    GetStudyQuery,
    ListFindingsQuery,
    ListHypothesesQuery,
    ListStudiesQuery,
    ListVisualizationsQuery,
)
from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.domain.agents.entities import ToolErrorCode

logger = logging.getLogger("catml.mcp.analysis")


def register_analysis_mcp_tools(
    server: Any,
    query_bus: QueryBus,
    command_bus: CommandBus,
) -> None:
    """Register CATML Explore (Phase E1) tools and resources on an MCPServer instance."""
    if not hasattr(types, "CallToolResult"):
        return

    @server.tool(
        name="analysis_create_study",
        description="Create an exploratory study specification for a dataset (unsupervised or with an optional target column).",
    )
    def analysis_create_study(
        dataset_id: str,
        name: str = "",
        target_column: str | None = None,
    ) -> types.CallToolResult:
        try:
            cmd = CreateStudyCommand(
                dataset_id=dataset_id,
                name=name,
                target_column=target_column if target_column else None,
            )
            study_res = command_bus.dispatch(cmd)
            study_id = study_res if isinstance(study_res, str) else getattr(study_res, "id", str(study_res))
            study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
            data = study if isinstance(study, dict) else (study.to_dict() if hasattr(study, "to_dict") else dict(study))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="analysis_run_study",
        description="Execute deterministic statistical and exploratory analysis on an existing study.",
    )
    def analysis_run_study(study_id: str) -> types.CallToolResult:
        try:
            cmd = RunAnalysisCommand(study_id=study_id)
            run_res = command_bus.dispatch(cmd)
            run_id = run_res if isinstance(run_res, str) else getattr(run_res, "id", str(run_res))
            run_data = query_bus.dispatch(GetAnalysisRunQuery(run_id=run_id))
            findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id, run_id=run_id))
            payload = {
                "study_id": study_id,
                "run_id": run_id,
                "run": run_data,
                "findings_count": len(findings) if findings else 0,
            }
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(payload, default=str, indent=2))],
                is_error=False,
            )
        except KeyError as ke:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] {str(ke)}")],
                is_error=True,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="analysis_get_study",
        description="Retrieve details and configuration of an exploratory study by its ID.",
    )
    def analysis_get_study(study_id: str) -> types.CallToolResult:
        try:
            study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
            if not study:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] Study '{study_id}' not found.")],
                    is_error=True,
                )
            data = study.to_dict() if hasattr(study, "to_dict") else dict(study)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="analysis_list_studies",
        description="List all exploratory studies in the workspace, optionally filtered by dataset ID.",
    )
    def analysis_list_studies(dataset_id: str | None = None) -> types.CallToolResult:
        try:
            studies = query_bus.dispatch(ListStudiesQuery(dataset_id=dataset_id))
            data = [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in (studies or [])]
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="analysis_get_findings",
        description="Retrieve deterministic statistical findings (correlations, outliers, missingness) for a study.",
    )
    def analysis_get_findings(
        study_id: str,
        finding_type: str | None = None,
    ) -> types.CallToolResult:
        try:
            findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id, finding_type=finding_type))
            data = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in (findings or [])]
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    # -----------------------------------------------------------------------
    # Resources
    # -----------------------------------------------------------------------
    @server.resource("catml://studies/{study_id}")
    def study_detail_resource(study_id: str) -> str:
        """Resource exposing study metadata as JSON."""
        try:
            study = query_bus.dispatch(GetStudyQuery(study_id=study_id))
            if not study:
                return json.dumps({"error": f"Study '{study_id}' not found", "code": ToolErrorCode.NOT_FOUND.value})
            data = study.to_dict() if hasattr(study, "to_dict") else dict(study)
            return json.dumps(data, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.INTERNAL_ERROR.value})

    @server.resource("catml://studies/{study_id}/findings")
    def study_findings_resource(study_id: str) -> str:
        """Resource exposing study statistical findings as JSON."""
        try:
            findings = query_bus.dispatch(ListFindingsQuery(study_id=study_id))
            data = [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in (findings or [])]
            return json.dumps(data, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.INTERNAL_ERROR.value})
