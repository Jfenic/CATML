"""State translation between public application contracts and internal LangGraph state."""
from __future__ import annotations

import json
from typing import Any, Optional, TypedDict

from automl.application.agents.contracts import AgentSessionState
from automl.application.agents.orchestrator.state_machine import SessionStatus


class GraphAgentState(TypedDict, total=False):
    """Internal LangGraph state representation for autonomous agent cycles."""

    session_id: str
    run_id: str
    goal: str
    version: str
    status: str
    iteration_count: int
    max_iterations: int
    stop_reason: Optional[str]
    session_checkpoint_id: Optional[str]
    budget: dict[str, Any]
    context: Optional[dict[str, Any]]
    proposal: Optional[dict[str, Any]]
    pending_approval_id: Optional[str]
    approval_status: Optional[str]
    latest_result: Optional[dict[str, Any]]
    feedback: Optional[dict[str, Any]]
    operation_id: Optional[str]
    current_hypothesis_id: Optional[str]
    error: Optional[str]
    created_at: str
    updated_at: str


def session_to_graph_state(
    session: AgentSessionState,
    context: Optional[dict[str, Any]] = None,
    proposal: Optional[dict[str, Any]] = None,
    latest_result: Optional[dict[str, Any]] = None,
    feedback: Optional[dict[str, Any]] = None,
    operation_id: Optional[str] = None,
    error: Optional[str] = None,
) -> GraphAgentState:
    """Translate public AgentSessionState into internal GraphAgentState for LangGraph execution."""
    return GraphAgentState(
        session_id=session.session_id,
        run_id=session.run_id,
        goal=session.goal,
        version=session.version,
        status=session.status,
        iteration_count=session.iteration_count,
        max_iterations=session.max_iterations,
        stop_reason=session.stop_reason,
        session_checkpoint_id=session.checkpoint_id,
        budget=dict(session.budget) if session.budget else {},
        context=context,
        proposal=proposal,
        pending_approval_id=session.pending_approval_id,
        approval_status=None,
        latest_result=latest_result,
        feedback=feedback,
        operation_id=operation_id,
        current_hypothesis_id=session.current_hypothesis_id,
        error=error,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def graph_state_to_session(state: GraphAgentState) -> AgentSessionState:
    """Translate internal GraphAgentState back into public AgentSessionState DTO."""
    return AgentSessionState(
        session_id=state.get("session_id", ""),
        run_id=state.get("run_id", ""),
        goal=state.get("goal", ""),
        version=state.get("version", "1.0.0"),
        status=state.get("status", SessionStatus.ACTIVE.value),
        iteration_count=int(state.get("iteration_count", 0)),
        max_iterations=int(state.get("max_iterations", 10)),
        budget=dict(state.get("budget", {})),
        stop_reason=state.get("stop_reason"),
        current_hypothesis_id=state.get("current_hypothesis_id"),
        pending_approval_id=state.get("pending_approval_id"),
        checkpoint_id=state.get("session_checkpoint_id"),
        created_at=state.get("created_at", ""),
        updated_at=state.get("updated_at", ""),
    )


def serialize_graph_state(state: GraphAgentState) -> str:
    """Deterministically serialize GraphAgentState to JSON string."""
    return json.dumps(state, sort_keys=True, default=str)


def deserialize_graph_state(payload: str) -> GraphAgentState:
    """Deserialize JSON string into GraphAgentState."""
    data = json.loads(payload)
    return GraphAgentState(**data)
