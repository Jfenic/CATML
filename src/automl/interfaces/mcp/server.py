"""Model Context Protocol (MCP) server adapter for CATML.

Exposes deterministic inspection queries, feature evidence, and run leaderboards
to MCP clients (e.g. Claude Desktop, Cursor, Gemini CLI) via stdio transport.
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
    import mcp.types as types
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    MCPServer = Any  # type: ignore
    types = Any  # type: ignore

from automl.application.bootstrap import build_application
from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.application.queries.workspace_queries import (
    GetDatasetProfileQuery,
    GetFeatureEvidenceQuery,
    GetFeatureRankingQuery,
    GetLeaderboardQuery,
    ListExperimentsQuery,
    ListModelsQuery,
    ListPluginsQuery,
)
from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.agents.entities import ToolErrorCode

logger = logging.getLogger("catml.mcp")


def create_mcp_server(
    workspace: AutoMLWorkspace | None = None,
    command_bus: CommandBus | None = None,
    query_bus: QueryBus | None = None,
    root_dir: str | None = None,
) -> Any:
    """Create and configure the CATML MCP server instance."""
    if not HAS_MCP:
        raise ImportError(
            "El componente MCP requiere la dependencia opcional 'mcp'. "
            "Instálala con: pip install '.[mcp]'"
        )

    if workspace is None or command_bus is None or query_bus is None:
        workspace, command_bus, query_bus = build_application(root_dir=root_dir)

    server = MCPServer("catml-mcp", version="0.7.0")

    # -----------------------------------------------------------------------
    # Query Tools (Milestone H1)
    # -----------------------------------------------------------------------

    @server.tool(
        name="get_dataset_profile",
        description="Inspect summary profile, rows, columns, and data types of a registered dataset.",
    )
    def get_dataset_profile(dataset_id: str) -> types.CallToolResult:
        try:
            profile = query_bus.dispatch(GetDatasetProfileQuery(dataset_id=dataset_id))
            if profile is None:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] Dataset '{dataset_id}' not found in workspace.")],
                    is_error=True,
                )
            res_dict = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(res_dict, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="list_models",
        description="List registered candidate models and algorithms, optionally filtered by task type.",
    )
    def list_models(task_type: str | None = None) -> types.CallToolResult:
        try:
            models = query_bus.dispatch(ListModelsQuery(task_type=task_type))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(models, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="list_plugins",
        description="List registered plugins in the system, optionally filtered by plugin type or task type.",
    )
    def list_plugins(plugin_type: str | None = None, task_type: str | None = None) -> types.CallToolResult:
        try:
            plugins = query_bus.dispatch(ListPluginsQuery(plugin_type=plugin_type, task_type=task_type))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(plugins, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="list_experiments",
        description="List all experiments and their statuses in a given run.",
    )
    def list_experiments(run_id: str) -> types.CallToolResult:
        try:
            exps = query_bus.dispatch(ListExperimentsQuery(run_id=run_id))
            data = [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in exps]
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(data, default=str, indent=2))],
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
        name="get_leaderboard",
        description="Retrieve the scored leaderboard of experiments for a run, ordered by performance metric.",
    )
    def get_leaderboard(run_id: str, top_k: int | None = None) -> types.CallToolResult:
        try:
            lb = query_bus.dispatch(GetLeaderboardQuery(run_id=run_id))
            if top_k is not None and top_k > 0:
                lb = lb[:top_k]
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(lb, default=str, indent=2))],
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
        name="get_feature_evidence",
        description="Inspect statistical and ablation evidence collected for a feature in a run.",
    )
    def get_feature_evidence(run_id: str, feature_id: str) -> types.CallToolResult:
        try:
            ev = query_bus.dispatch(GetFeatureEvidenceQuery(run_id=run_id, feature_id=feature_id))
            if ev is None:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=f"[{ToolErrorCode.NOT_FOUND.value}] No evidence for feature '{feature_id}' in run '{run_id}'")],
                    is_error=True,
                )
            data = ev.to_dict() if hasattr(ev, "to_dict") else dict(ev)
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
        name="get_feature_ranking",
        description="Get ranked features for a run based on statistical importance or ensemble ranking.",
    )
    def get_feature_ranking(run_id: str, method: str = "ensemble") -> types.CallToolResult:
        try:
            ranking = query_bus.dispatch(GetFeatureRankingQuery(run_id=run_id, method=method))
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=json.dumps(ranking, default=str, indent=2))],
                is_error=False,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    # -----------------------------------------------------------------------
    # Resources (Milestone H1)
    # -----------------------------------------------------------------------

    @server.resource("catml://runs/{run_id}/leaderboard")
    def run_leaderboard_resource(run_id: str) -> str:
        """Resource exposing the live leaderboard of a run as JSON."""
        try:
            lb = query_bus.dispatch(GetLeaderboardQuery(run_id=run_id))
            return json.dumps(lb, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.NOT_FOUND.value})

    @server.resource("catml://datasets/{dataset_id}/profile")
    def dataset_profile_resource(dataset_id: str) -> str:
        """Resource exposing a dataset profile as JSON."""
        try:
            profile = query_bus.dispatch(GetDatasetProfileQuery(dataset_id=dataset_id))
            if profile is None:
                return json.dumps({"error": f"Dataset '{dataset_id}' not found", "code": ToolErrorCode.NOT_FOUND.value})
            res_dict = profile.to_dict() if hasattr(profile, "to_dict") else dict(profile)
            return json.dumps(res_dict, default=str, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "code": ToolErrorCode.INTERNAL_ERROR.value})

    return server


def run_stdio_server(root_dir: str | None = None) -> None:
    """Run the CATML MCP server on standard input/output streams."""
    # Ensure all logging goes to stderr so stdout is strictly preserved for JSON-RPC
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Iniciando CATML MCP Stdio Server (V0.7.0)...")
    server = create_mcp_server(root_dir=root_dir)
    server.run(transport="stdio")
