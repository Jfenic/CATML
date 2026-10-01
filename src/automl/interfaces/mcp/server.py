"""Model Context Protocol (MCP) server adapter for CATML.

Exposes deterministic inspection queries, feature evidence, and run leaderboards
to MCP clients (e.g. Claude Desktop, Cursor, Gemini CLI) via stdio transport.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any
import uuid

try:
    from mcp.server.mcpserver import MCPServer
    import mcp.types as types
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    MCPServer = Any  # type: ignore
    types = Any  # type: ignore

from automl.application.agents.contracts import ToolCallContext, ToolInvocation
from automl.application.agents.executor import ToolExecutor, create_full_tool_registry
from automl.application.agents.policy import AgentPolicyConfig, PolicyEvaluator
from automl.application.agents.ports import AgentLedgerPort
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
from automl.domain.agents.entities import AgentPermission, ToolErrorCode
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger

logger = logging.getLogger("catml.mcp")


def create_mcp_server(
    workspace: AutoMLWorkspace | None = None,
    command_bus: CommandBus | None = None,
    query_bus: QueryBus | None = None,
    root_dir: str | None = None,
    ledger: AgentLedgerPort | None = None,
    policy_evaluator: PolicyEvaluator | None = None,
) -> Any:
    """Create and configure the CATML MCP server instance."""
    if not HAS_MCP:
        raise ImportError(
            "El componente MCP requiere la dependencia opcional 'mcp'. "
            "Instálala con: pip install '.[mcp]'"
        )

    if workspace is None or command_bus is None or query_bus is None:
        workspace, command_bus, query_bus = build_application(root_dir=root_dir)

    if ledger is None:
        if workspace and hasattr(workspace, "root_dir") and workspace.root_dir:
            ledger_path = Path(workspace.root_dir) / "agent_ledger.db"
        elif root_dir:
            ledger_path = Path(root_dir) / "agent_ledger.db"
        else:
            ledger_path = Path.cwd() / ".automl" / "default" / "agent_ledger.db"
        ledger = SqliteAgentLedger(ledger_path)

    if policy_evaluator is None:
        policy_evaluator = PolicyEvaluator(
            AgentPolicyConfig(
                require_approval_for_mutations=True,
                default_mode=AgentPermission.EXECUTE_WITHIN_BUDGET,
            )
        )

    tool_registry = create_full_tool_registry(
        query_bus=query_bus,
        command_bus=command_bus,
        workspace=workspace,
    )
    executor = ToolExecutor(
        registry=tool_registry,
        policy_evaluator=policy_evaluator,
        ledger=ledger,
    )

    ws_path = str(workspace.root_dir) if workspace and hasattr(workspace, "root_dir") else "."

    server = MCPServer("catml-mcp", version="0.7.0")
    server.executor = executor
    server.ledger = ledger

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
            data = []
            for e in exps or []:
                if hasattr(e, "to_dict"):
                    data.append(e.to_dict())
                else:
                    status_val = e.status.value if hasattr(e.status, "value") else str(e.status)
                    data.append({
                        "id": getattr(e, "id", str(e)),
                        "experiment_id": getattr(e, "id", str(e)),
                        "run_id": getattr(e, "run_id", run_id),
                        "name": getattr(e, "name", ""),
                        "model_ids": getattr(e, "model_ids", []),
                        "metric": getattr(e, "metric", ""),
                        "status": status_val,
                    })
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
    # Mutating Tools with Policy & Approval Governance (Milestone H2)
    # -----------------------------------------------------------------------

    @server.tool(
        name="create_experiment",
        description="Propose and create an experiment candidate with specified model and features.",
    )
    def create_experiment(
        run_id: str,
        model_name: str,
        feature_names: list[str] | None = None,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        try:
            arguments = {
                "run_id": run_id,
                "model_name": model_name,
            }
            if feature_names is not None:
                arguments["feature_names"] = feature_names

            inv = ToolInvocation(
                tool_name="create_experiment",
                arguments=arguments,
                context=ToolCallContext(
                    actor="mcp_agent",
                    workspace_path=ws_path,
                    run_id=run_id,
                    correlation_id=f"mcp-{uuid.uuid4().hex[:8]}",
                    permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
                    approval_id=approval_id,
                ),
                idempotency_key=idempotency_key,
                approval_id=approval_id,
            )
            res = executor.execute(inv)
            if res.success:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(res.data, default=str, indent=2))],
                    is_error=False,
                )
            if res.error and res.error.code == ToolErrorCode.APPROVAL_REQUIRED:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "status": "PENDING_APPROVAL",
                                    "approval_id": res.approval_id,
                                    "message": f"Action 'create_experiment' requires human approval. Run: automl agent approve {res.approval_id}",
                                    "details": res.error.details,
                                },
                                indent=2,
                            ),
                        )
                    ],
                    is_error=False,
                )
            err_msg = res.error.message if res.error else "Unknown error"
            err_code = res.error.code.value if res.error else ToolErrorCode.INTERNAL_ERROR.value
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{err_code}] {err_msg}")],
                is_error=True,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="prioritize_feature",
        description="Prioritize a specific feature to adjust its weight in candidate selection.",
    )
    def prioritize_feature(
        run_id: str,
        feature_name: str,
        priority: str,
        dataset_id: str | None = None,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        try:
            arguments = {
                "run_id": run_id,
                "feature_name": feature_name,
                "priority": priority,
            }
            if dataset_id:
                arguments["dataset_id"] = dataset_id

            inv = ToolInvocation(
                tool_name="prioritize_feature",
                arguments=arguments,
                context=ToolCallContext(
                    actor="mcp_agent",
                    workspace_path=ws_path,
                    run_id=run_id,
                    correlation_id=f"mcp-{uuid.uuid4().hex[:8]}",
                    permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
                    approval_id=approval_id,
                ),
                idempotency_key=idempotency_key,
                approval_id=approval_id,
            )
            res = executor.execute(inv)
            if res.success:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(res.data, default=str, indent=2))],
                    is_error=False,
                )
            if res.error and res.error.code == ToolErrorCode.APPROVAL_REQUIRED:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "status": "PENDING_APPROVAL",
                                    "approval_id": res.approval_id,
                                    "message": f"Action 'prioritize_feature' requires human approval. Run: automl agent approve {res.approval_id}",
                                    "details": res.error.details,
                                },
                                indent=2,
                            ),
                        )
                    ],
                    is_error=False,
                )
            err_msg = res.error.message if res.error else "Unknown error"
            err_code = res.error.code.value if res.error else ToolErrorCode.INTERNAL_ERROR.value
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{err_code}] {err_msg}")],
                is_error=True,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{ToolErrorCode.INTERNAL_ERROR.value}] {str(e)}")],
                is_error=True,
            )

    @server.tool(
        name="run_experiment",
        description="Execute training and validation for a configured experiment candidate.",
    )
    def run_experiment(
        run_id: str,
        experiment_id: str,
        idempotency_key: str | None = None,
        approval_id: str | None = None,
    ) -> types.CallToolResult:
        try:
            arguments = {
                "run_id": run_id,
                "experiment_id": experiment_id,
            }
            inv = ToolInvocation(
                tool_name="run_experiment",
                arguments=arguments,
                context=ToolCallContext(
                    actor="mcp_agent",
                    workspace_path=ws_path,
                    run_id=run_id,
                    correlation_id=f"mcp-{uuid.uuid4().hex[:8]}",
                    permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
                    approval_id=approval_id,
                ),
                idempotency_key=idempotency_key,
                approval_id=approval_id,
            )
            res = executor.execute(inv)
            if res.success:
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=json.dumps(res.data, default=str, indent=2))],
                    is_error=False,
                )
            if res.error and res.error.code == ToolErrorCode.APPROVAL_REQUIRED:
                return types.CallToolResult(
                    content=[
                        types.TextContent(
                            type="text",
                            text=json.dumps(
                                {
                                    "status": "PENDING_APPROVAL",
                                    "approval_id": res.approval_id,
                                    "message": f"Action 'run_experiment' requires human approval. Run: automl agent approve {res.approval_id}",
                                    "details": res.error.details,
                                },
                                indent=2,
                            ),
                        )
                    ],
                    is_error=False,
                )
            err_msg = res.error.message if res.error else "Unknown error"
            err_code = res.error.code.value if res.error else ToolErrorCode.INTERNAL_ERROR.value
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"[{err_code}] {err_msg}")],
                is_error=True,
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
