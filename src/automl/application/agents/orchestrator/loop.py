"""Autonomous execution loop and lifecycle runner with hooks for agent sessions."""
from __future__ import annotations

from typing import Any, Callable

from automl.application.agents.contracts import AgentSessionState, SessionStepResult
from automl.application.agents.orchestrator.session_manager import AgentSessionManager
from automl.application.agents.orchestrator.state_machine import SessionStatus


class AgentLoop:
    """Executes an agent session in a controlled loop, emitting lifecycle events."""

    def __init__(
        self,
        session_manager: AgentSessionManager,
        on_step: Callable[[SessionStepResult], None] | None = None,
        on_approval_needed: Callable[[SessionStepResult], None] | None = None,
        on_stop: Callable[[str | None, AgentSessionState], None] | None = None,
    ) -> None:
        self.session_manager = session_manager
        self.on_step = on_step
        self.on_approval_needed = on_approval_needed
        self.on_stop = on_stop

    def run(
        self,
        session_id: str,
        max_steps: int | None = None,
    ) -> list[SessionStepResult]:
        """Execute the agent loop until terminal condition, pause, or max_steps reached."""
        results: list[SessionStepResult] = []
        steps_executed = 0

        while True:
            if max_steps is not None and steps_executed >= max_steps:
                break

            step_res = self.session_manager.step(session_id)
            results.append(step_res)
            steps_executed += 1

            if self.on_step:
                self.on_step(step_res)

            if step_res.state == SessionStatus.WAITING_APPROVAL.value:
                if self.on_approval_needed:
                    self.on_approval_needed(step_res)
                break

            if step_res.state in (
                SessionStatus.COMPLETED.value,
                SessionStatus.STOPPED.value,
                SessionStatus.FAILED.value,
                SessionStatus.PAUSED.value,
            ):
                if self.on_stop:
                    current_session = self.session_manager.get_session(session_id)
                    if current_session:
                        self.on_stop(step_res.stop_reason, current_session)
                break

        return results
