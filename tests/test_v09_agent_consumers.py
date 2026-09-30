"""Tests for Agentic System V0.9/V1.0 Consumers (MCP & Orchestrator).

Persona B Package B0 verification suite.
Exercises consumer-side adaptation of contracts proposed for Persona A (A0).
Tests run entirely isolated without external SDKs, network access, or credentials.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Simulated Contracts for H0 (to be finalized jointly with Persona A)
# ---------------------------------------------------------------------------


class ToolErrorCode(str, Enum):
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    NOT_FOUND = "NOT_FOUND"
    SCOPE_VIOLATION = "SCOPE_VIOLATION"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    CONFLICT = "CONFLICT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ToolError:
    code: ToolErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    retryable: bool = False
    correlation_id: Optional[str] = None


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    version: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    command_or_query: str
    is_mutation: bool
    permission_required: str
    cost_estimate: float = 0.0


@dataclass(frozen=True)
class ToolCallContext:
    actor_id: str
    workspace_path: str
    run_id: str
    request_id: str
    correlation_id: str
    deadline_seconds: Optional[float] = None


@dataclass(frozen=True)
class ToolInvocation:
    tool_name: str
    version: str
    arguments: Dict[str, Any]
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class ToolResult:
    request_id: str
    status: str  # "success", "error", "approval_required"
    data: Optional[Any] = None
    error: Optional[ToolError] = None
    operation_id: Optional[str] = None
    approval_id: Optional[str] = None


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    requires_approval: bool
    reason: str
    effective_limits: Dict[str, Any] = field(default_factory=dict)
    policy_version: str = "1.0"


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    actor_id: str
    run_id: str
    action: str
    arguments_hash: str
    normalized_arguments: Dict[str, Any]
    estimated_cost: float
    status: str  # "pending", "approved", "rejected", "expired", "revoked"
    expires_at: str


@dataclass
class AgentSessionState:
    session_id: str
    version: str
    run_id: str
    goal: str
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    pending_approval_id: Optional[str] = None
    consumed_budget: Dict[str, float] = field(default_factory=dict)
    remaining_budget: Dict[str, float] = field(default_factory=dict)
    iteration_count: int = 0
    max_iterations: int = 5
    stop_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Consumer Adapters (Persona B responsibilities)
# ---------------------------------------------------------------------------


class MCPConsumerAdapter:
    """Adapts application ToolDefinitions and invocations to Model Context Protocol."""

    def __init__(self, tools: Dict[str, ToolDefinition], executor: Callable[[ToolCallContext, ToolInvocation], ToolResult]):
        self._tools = tools
        self._executor = executor

    def list_mcp_tools(self) -> List[Dict[str, Any]]:
        """Exports ToolDefinitions as MCP Tools."""
        mcp_tools = []
        for name, defn in self._tools.items():
            mcp_tools.append({
                "name": defn.name,
                "description": defn.description,
                "inputSchema": defn.input_schema,
            })
        return mcp_tools

    def call_tool(self, server_context: ToolCallContext, tool_name: str, raw_arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handles an MCP tool call request cleanly without leaking server context."""
        if tool_name not in self._tools:
            return {
                "content": [{"type": "text", "text": f"Error: Tool '{tool_name}' not found."}],
                "isError": True,
                "code": ToolErrorCode.NOT_FOUND.value,
            }

        defn = self._tools[tool_name]

        # Prevent LLM/client from tampering with server-level context fields
        sanitized_arguments = {k: v for k, v in raw_arguments.items() if not k.startswith("_")}

        invocation = ToolInvocation(
            tool_name=defn.name,
            version=defn.version,
            arguments=sanitized_arguments,
            idempotency_key=raw_arguments.get("_idempotency_key"),
        )

        result = self._executor(server_context, invocation)

        if result.status == "success":
            return {
                "content": [{"type": "text", "text": json.dumps(result.data, default=str)}],
                "isError": False,
            }
        elif result.status == "approval_required":
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "status": "APPROVAL_REQUIRED",
                        "approval_id": result.approval_id,
                        "message": "This action requires explicit human confirmation before execution.",
                    }),
                }],
                "isError": False,
                "requires_approval": True,
                "approval_id": result.approval_id,
            }
        else:
            err_msg = result.error.message if result.error else "Unknown execution error"
            err_code = result.error.code.value if result.error else ToolErrorCode.INTERNAL_ERROR.value
            return {
                "content": [{"type": "text", "text": f"[{err_code}] {err_msg}"}],
                "isError": True,
                "code": err_code,
            }


class OrchestratorConsumerAdapter:
    """Manages agent execution state transitions and policy enforcement."""

    def __init__(self, policy_evaluator: Callable[[ToolInvocation], PolicyDecision]):
        self._policy_evaluator = policy_evaluator

    def step(self, state: AgentSessionState, proposed_invocation: ToolInvocation) -> AgentSessionState:
        """Executes a cycle step with finite limits and policy gates."""
        if state.stop_reason is not None:
            return state

        if state.iteration_count >= state.max_iterations:
            state.stop_reason = "MAX_ITERATIONS_REACHED"
            return state

        # Check hypothesis deduplication
        args_hash = str(hash(json.dumps(proposed_invocation.arguments, sort_keys=True)))
        for hyp in state.hypotheses:
            if hyp.get("hash") == args_hash:
                state.stop_reason = "REPEATED_HYPOTHESIS"
                return state

        decision = self._policy_evaluator(proposed_invocation)

        if not decision.allowed:
            state.stop_reason = f"POLICY_DENIED: {decision.reason}"
            return state

        if decision.requires_approval:
            state.pending_approval_id = f"appr_{len(state.hypotheses) + 1}"
            state.stop_reason = "AWAITING_APPROVAL"
            return state

        # Proceed
        state.hypotheses.append({"tool": proposed_invocation.tool_name, "hash": args_hash})
        state.iteration_count += 1
        return state


# ---------------------------------------------------------------------------
# Test Suite: Consumer Verification
# ---------------------------------------------------------------------------


def test_mcp_consumer_exports_valid_tool_schemas():
    """Verify MCP adapter correctly translates ToolDefinitions into MCP format."""
    query_tool = ToolDefinition(
        name="get_dataset_profile",
        version="1.0",
        description="Inspect summary profile and column types of a dataset",
        input_schema={
            "type": "object",
            "properties": {"dataset_id": {"type": "string"}},
            "required": ["dataset_id"],
            "additionalProperties": False,
        },
        output_schema={"type": "object"},
        command_or_query="GetDatasetProfileQuery",
        is_mutation=False,
        permission_required="read",
    )

    tools = {query_tool.name: query_tool}
    adapter = MCPConsumerAdapter(tools=tools, executor=lambda ctx, inv: None)

    mcp_tools = adapter.list_mcp_tools()
    assert len(mcp_tools) == 1
    assert mcp_tools[0]["name"] == "get_dataset_profile"
    assert mcp_tools[0]["description"] == query_tool.description
    assert mcp_tools[0]["inputSchema"]["type"] == "object"
    assert "dataset_id" in mcp_tools[0]["inputSchema"]["properties"]


def test_mcp_consumer_query_tool_success():
    """Verify successful query tool invocation produces valid MCP response."""
    query_tool = ToolDefinition(
        name="list_models",
        version="1.0",
        description="List registered candidate models",
        input_schema={"type": "object"},
        output_schema={"type": "array"},
        command_or_query="ListModelsQuery",
        is_mutation=False,
        permission_required="read",
    )

    def mock_executor(ctx: ToolCallContext, inv: ToolInvocation) -> ToolResult:
        assert ctx.run_id == "run_123"
        return ToolResult(
            request_id=ctx.request_id,
            status="success",
            data=["lightgbm", "xgboost", "random_forest"],
        )

    adapter = MCPConsumerAdapter(tools={query_tool.name: query_tool}, executor=mock_executor)
    context = ToolCallContext(
        actor_id="agent_1",
        workspace_path="/tmp/test",
        run_id="run_123",
        request_id="req_001",
        correlation_id="corr_001",
    )

    resp = adapter.call_tool(context, "list_models", {})
    assert resp["isError"] is False
    content_text = resp["content"][0]["text"]
    assert "lightgbm" in content_text


def test_mcp_consumer_mutation_requires_approval_non_blocking():
    """Verify mutation requiring human approval returns approval_id without blocking."""
    mutation_tool = ToolDefinition(
        name="create_experiment",
        version="1.0",
        description="Propose and create an experiment in the current run",
        input_schema={"type": "object", "properties": {"model_name": {"type": "string"}}},
        output_schema={"type": "object"},
        command_or_query="CreateExperimentCommand",
        is_mutation=True,
        permission_required="execute",
    )

    def mock_executor(ctx: ToolCallContext, inv: ToolInvocation) -> ToolResult:
        return ToolResult(
            request_id=ctx.request_id,
            status="approval_required",
            approval_id="appr_999",
        )

    adapter = MCPConsumerAdapter(tools={mutation_tool.name: mutation_tool}, executor=mock_executor)
    context = ToolCallContext(
        actor_id="external_agent",
        workspace_path="/tmp/test",
        run_id="run_123",
        request_id="req_002",
        correlation_id="corr_002",
    )

    resp = adapter.call_tool(context, "create_experiment", {"model_name": "xgboost"})
    assert resp["isError"] is False
    assert resp.get("requires_approval") is True
    assert resp.get("approval_id") == "appr_999"


def test_mcp_consumer_error_propagation():
    """Verify ToolErrors are properly captured and mapped to MCP error responses."""
    tool = ToolDefinition(
        name="get_dataset_profile",
        version="1.0",
        description="Inspect dataset profile",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        command_or_query="GetDatasetProfileQuery",
        is_mutation=False,
        permission_required="read",
    )

    def failing_executor(ctx: ToolCallContext, inv: ToolInvocation) -> ToolResult:
        return ToolResult(
            request_id=ctx.request_id,
            status="error",
            error=ToolError(
                code=ToolErrorCode.SCOPE_VIOLATION,
                message="Dataset does not belong to the active workspace",
                retryable=False,
            ),
        )

    adapter = MCPConsumerAdapter(tools={tool.name: tool}, executor=failing_executor)
    context = ToolCallContext(
        actor_id="user_1",
        workspace_path="/tmp/test",
        run_id="run_1",
        request_id="req_003",
        correlation_id="corr_003",
    )

    resp = adapter.call_tool(context, "get_dataset_profile", {"dataset_id": "other_run_ds"})
    assert resp["isError"] is True
    assert resp["code"] == "SCOPE_VIOLATION"
    assert "SCOPE_VIOLATION" in resp["content"][0]["text"]


def test_orchestrator_state_enforces_max_iterations():
    """Verify agent state stops cleanly when maximum iterations are reached."""
    adapter = OrchestratorConsumerAdapter(
        policy_evaluator=lambda inv: PolicyDecision(allowed=True, requires_approval=False, reason="OK")
    )

    state = AgentSessionState(
        session_id="sess_1",
        version="1.0",
        run_id="run_1",
        goal="optimize ROC-AUC",
        max_iterations=2,
    )

    inv1 = ToolInvocation(tool_name="tool_a", version="1.0", arguments={"step": 1})
    state = adapter.step(state, inv1)
    assert state.iteration_count == 1
    assert state.stop_reason is None

    inv2 = ToolInvocation(tool_name="tool_a", version="1.0", arguments={"step": 2})
    state = adapter.step(state, inv2)
    assert state.iteration_count == 2
    assert state.stop_reason is None

    inv3 = ToolInvocation(tool_name="tool_a", version="1.0", arguments={"step": 3})
    state = adapter.step(state, inv3)
    assert state.iteration_count == 2
    assert state.stop_reason == "MAX_ITERATIONS_REACHED"


def test_orchestrator_state_detects_repeated_hypothesis():
    """Verify agent halts when proposing the exact same configuration hypothesis."""
    adapter = OrchestratorConsumerAdapter(
        policy_evaluator=lambda inv: PolicyDecision(allowed=True, requires_approval=False, reason="OK")
    )

    state = AgentSessionState(
        session_id="sess_1",
        version="1.0",
        run_id="run_1",
        goal="optimize ROC-AUC",
        max_iterations=10,
    )

    inv = ToolInvocation(tool_name="create_experiment", version="1.0", arguments={"model": "lightgbm", "lr": 0.05})
    state = adapter.step(state, inv)
    assert state.iteration_count == 1

    # Exact duplicate invocation
    state = adapter.step(state, inv)
    assert state.stop_reason == "REPEATED_HYPOTHESIS"


def test_orchestrator_state_handles_approval_required():
    """Verify agent halts and sets pending approval when policy requires approval."""
    adapter = OrchestratorConsumerAdapter(
        policy_evaluator=lambda inv: PolicyDecision(
            allowed=True,
            requires_approval=True,
            reason="High compute cost requires human sign-off",
        )
    )

    state = AgentSessionState(
        session_id="sess_1",
        version="1.0",
        run_id="run_1",
        goal="train large ensemble",
        max_iterations=5,
    )

    inv = ToolInvocation(tool_name="run_experiment", version="1.0", arguments={"experiment_id": "exp_big"})
    state = adapter.step(state, inv)
    assert state.stop_reason == "AWAITING_APPROVAL"
    assert state.pending_approval_id is not None
