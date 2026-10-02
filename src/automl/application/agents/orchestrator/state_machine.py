"""Deterministic state machine and transition rules for autonomous agent sessions."""
from __future__ import annotations

from enum import Enum
import hashlib
import json
from typing import Any

from automl.domain.agents.entities import AgentBudget


class CycleState(str, Enum):
    """Execution states within a single autonomous cycle step."""
    INIT = "init"
    OBSERVE = "observe"
    PROPOSE = "propose"
    GATE = "gate"
    WAITING_APPROVAL = "waiting_approval"
    EXECUTE = "execute"
    CRITIQUE = "critique"
    CHECK_STOP = "check_stop"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class SessionStatus(str, Enum):
    """High-level lifecycle status for an Agent session."""
    ACTIVE = "active"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    STOPPED = "stopped"
    FAILED = "failed"


class StopReason(str, Enum):
    """Canonical reasons for terminating an autonomous session."""
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TARGET_REACHED = "target_reached"
    REPEATED_HYPOTHESIS = "repeated_hypothesis"
    NO_IMPROVEMENT = "no_improvement"
    MANUALLY_STOPPED = "manually_stopped"
    APPROVAL_REJECTED = "approval_rejected"
    EXECUTION_FAILED = "execution_failed"


# Valid state transitions within the state machine
VALID_TRANSITIONS: dict[CycleState, set[CycleState]] = {
    CycleState.INIT: {CycleState.OBSERVE, CycleState.STOPPED, CycleState.FAILED},
    CycleState.OBSERVE: {CycleState.PROPOSE, CycleState.CHECK_STOP, CycleState.STOPPED, CycleState.FAILED},
    CycleState.PROPOSE: {CycleState.GATE, CycleState.CHECK_STOP, CycleState.STOPPED, CycleState.FAILED},
    CycleState.GATE: {
        CycleState.EXECUTE,
        CycleState.WAITING_APPROVAL,
        CycleState.CHECK_STOP,
        CycleState.STOPPED,
        CycleState.FAILED,
    },
    CycleState.WAITING_APPROVAL: {CycleState.EXECUTE, CycleState.CHECK_STOP, CycleState.STOPPED, CycleState.FAILED},
    CycleState.EXECUTE: {CycleState.CRITIQUE, CycleState.CHECK_STOP, CycleState.STOPPED, CycleState.FAILED},
    CycleState.CRITIQUE: {CycleState.CHECK_STOP, CycleState.STOPPED, CycleState.FAILED},
    CycleState.CHECK_STOP: {
        CycleState.OBSERVE,
        CycleState.COMPLETED,
        CycleState.STOPPED,
        CycleState.FAILED,
    },
    CycleState.COMPLETED: set(),
    CycleState.STOPPED: set(),
    CycleState.FAILED: set(),
}


class AgentStateMachine:
    """Evaluates valid transitions, detects repeated hypotheses, and checks stop conditions."""

    def __init__(self, patience: int = 3) -> None:
        self.patience = patience

    @staticmethod
    def is_valid_transition(current: CycleState, next_state: CycleState) -> bool:
        """Check if transitioning from current to next_state is legally allowed."""
        return next_state in VALID_TRANSITIONS.get(current, set())

    @staticmethod
    def compute_hypothesis_signature(action_type: str, action_payload: dict[str, Any]) -> str:
        """Compute deterministic SHA-256 fingerprint for a proposed action to detect duplicate work."""
        canonical_str = f"{action_type}:{json.dumps(action_payload, sort_keys=True)}"
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    @staticmethod
    def is_hypothesis_duplicate(signature: str, past_signatures: set[str] | list[str]) -> bool:
        """Verify if a proposal signature has already been attempted in this run."""
        return signature in past_signatures

    @staticmethod
    def check_budget_exhausted(budget: AgentBudget, estimated_cost: dict[str, Any] | None = None) -> bool:
        """Verify whether remaining budget is completely exhausted or insufficient for estimated cost."""
        if budget.remaining_experiments() <= 0 and budget.remaining_trials() <= 0:
            return True
        if estimated_cost:
            can_afford, _ = budget.can_afford(estimated_cost)
            return not can_afford
        return False

    @staticmethod
    def check_target_reached(
        current_score: float | None,
        target_score: float | None,
        metric_direction: str = "maximize",
    ) -> bool:
        """Check if empirical performance meets or exceeds target threshold."""
        if current_score is None or target_score is None:
            return False
        if metric_direction.lower() == "minimize":
            return current_score <= target_score
        return current_score >= target_score

    def check_patience_exhausted(self, consecutive_no_improvements: int) -> bool:
        """Check whether consecutive stagnation steps exceed maximum patience."""
        return consecutive_no_improvements >= self.patience
