"""Agent orchestrator package: deterministic state machine, session management, and autonomous loop."""
from __future__ import annotations

from automl.application.agents.orchestrator.loop import AgentLoop
from automl.application.agents.orchestrator.session_manager import AgentSessionManager
from automl.application.agents.orchestrator.state_machine import (
    AgentStateMachine,
    CycleState,
    SessionStatus,
    StopReason,
)

try:
    from automl.application.agents.orchestrator.graph import (
        LangGraphAgentOrchestrator,
        is_langgraph_available,
    )
except ImportError:  # pragma: no cover
    LangGraphAgentOrchestrator = None  # type: ignore[assignment, misc]

    def is_langgraph_available() -> bool:  # type: ignore[misc]
        return False

__all__ = [
    "AgentLoop",
    "AgentSessionManager",
    "AgentStateMachine",
    "CycleState",
    "LangGraphAgentOrchestrator",
    "SessionStatus",
    "StopReason",
    "is_langgraph_available",
]
