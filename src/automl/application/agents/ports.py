"""Technical application ports for persistence, audit ledger, and agent state storage."""
from __future__ import annotations

from typing import Any, Protocol

from automl.domain.agents.entities import ApprovalStatus, Hypothesis, OperationStatus, RunLease
from automl.application.agents.contracts import (
    AgentSessionState,
    ApprovalRequest,
    OperationRecord,
)


class AgentLedgerPort(Protocol):
    """Protocol for transactional audit ledger, approval tracking, hypotheses, sessions, and run leases."""

    def record_operation(self, op: OperationRecord) -> None:
        """Record a new operation in pending state with idempotency check."""
        ...

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        """Retrieve an operation record by ID."""
        ...

    def get_operation_by_idempotency_key(
        self, run_id: str, action: str, idempotency_key: str
    ) -> OperationRecord | None:
        """Lookup an operation by its idempotency key and scope."""
        ...

    def update_operation_status(
        self,
        operation_id: str,
        status: OperationStatus,
        consumed: dict[str, Any] | None = None,
        result_ref: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> OperationRecord:
        """Transition operation status, updating consumed budget or error details."""
        ...


    def list_operations(
        self, run_id: str | None = None, status: OperationStatus | None = None
    ) -> list[OperationRecord]:
        """List operation records, optionally filtered by run_id or status."""
        ...

    def request_operation_cancellation(self, operation_id: str) -> OperationRecord:
        """Request cooperative cancellation of a running or queued operation."""
        ...

    def is_cancellation_requested(self, operation_id: str) -> bool:
        """Check if cancellation was requested for the given operation."""
        ...

    def reconcile_operations(
        self, run_id: str | None = None
    ) -> list[OperationRecord]:
        """Scan operations in RUNNING state whose lease expired, transitioning them to RECOVERY_REQUIRED."""
        ...

    def get_active_reserved_budget(self, run_id: str) -> dict[str, Any]:
        """Compute aggregate budget currently reserved by in-flight operations for a run."""
        ...

    def acquire_run_lease(
        self, run_id: str, owner_id: str, ttl_seconds: float = 60.0
    ) -> bool:
        """Attempt to acquire exclusive writer lease for a run. Returns True if acquired, False if held by active owner."""
        ...

    def release_run_lease(self, run_id: str, owner_id: str) -> bool:
        """Release exclusive writer lease for a run if held by owner_id."""
        ...

    def heartbeat_run_lease(
        self, run_id: str, owner_id: str, ttl_seconds: float = 60.0
    ) -> bool:
        """Renew heartbeat and extend expiry for the lease held by owner_id."""
        ...

    def get_run_lease(self, run_id: str) -> RunLease | None:
        """Retrieve current lease for a run."""
        ...

    def save_approval(self, req: ApprovalRequest) -> None:
        """Persist a new approval request."""
        ...

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        """Retrieve an approval request by ID."""
        ...

    def update_approval_status(
        self, approval_id: str, status: ApprovalStatus, reviewer: str | None = None
    ) -> ApprovalRequest:
        """Update approval request status (approved, rejected, revoked)."""
        ...

    def list_approvals(
        self, run_id: str | None = None, status: ApprovalStatus | None = None
    ) -> list[ApprovalRequest]:
        """List approval requests, optionally filtered by run_id or status."""
        ...

    def save_hypothesis(self, hyp: Hypothesis) -> None:
        """Save a proposed or evaluated hypothesis."""
        ...

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        """Retrieve a hypothesis by ID."""
        ...

    def list_hypotheses(self, run_id: str) -> list[Hypothesis]:
        """List all hypotheses associated with a specific run."""
        ...

    def save_session_state(self, state: AgentSessionState) -> None:
        """Save agent session state for durable checkpointing."""
        ...

    def get_session_state(self, session_id: str) -> AgentSessionState | None:
        """Retrieve agent session state by session ID."""
        ...

