"""Application-level data transfer objects (DTOs) and contracts for agent tools and operations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    Hypothesis,
    OperationStatus,
    PolicyDecisionType,
    ToolEffect,
    ToolErrorCode,
)


@dataclass
class ToolCallContext:
    """Security and execution context for tool calls, established by the interface/host."""
    actor: str
    workspace_path: str
    run_id: str
    correlation_id: str
    deadline: float | None = None
    permission: AgentPermission = AgentPermission.READ_ONLY
    approval_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["permission"] = self.permission.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallContext:
        perm = AgentPermission(data.get("permission", AgentPermission.READ_ONLY.value))
        return cls(
            actor=data["actor"],
            workspace_path=data["workspace_path"],
            run_id=data["run_id"],
            correlation_id=data["correlation_id"],
            deadline=data.get("deadline"),
            permission=perm,
            approval_id=data.get("approval_id"),
        )


@dataclass
class ToolDefinition:
    """Explicit definition, contract, effect, and schemas of an Agent Tool."""
    name: str
    version: str
    description: str
    effect: ToolEffect
    permission_required: AgentPermission
    cost_estimate: dict[str, Any] = field(default_factory=dict)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["effect"] = self.effect.value
        d["permission_required"] = self.permission_required.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolDefinition:
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            effect=ToolEffect(data["effect"]),
            permission_required=AgentPermission(data["permission_required"]),
            cost_estimate=data.get("cost_estimate", {}),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
        )


@dataclass
class ToolInvocation:
    """Invocation request for an agent tool."""
    tool_name: str
    arguments: dict[str, Any]
    context: ToolCallContext
    version: str = "1.0.0"
    idempotency_key: str | None = None
    approval_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "version": self.version,
            "arguments": self.arguments,
            "idempotency_key": self.idempotency_key,
            "approval_id": self.approval_id,
            "context": self.context.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolInvocation:
        return cls(
            tool_name=data["tool_name"],
            version=data.get("version", "1.0.0"),
            arguments=data.get("arguments", {}),
            idempotency_key=data.get("idempotency_key"),
            approval_id=data.get("approval_id"),
            context=ToolCallContext.from_dict(data["context"]),
        )


@dataclass
class ToolError:
    """Structured tool error with canonical code and retryability flags."""
    code: ToolErrorCode
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    retryable: bool = False
    correlation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolError:
        return cls(
            code=ToolErrorCode(data["code"]),
            message=data["message"],
            details=data.get("details", {}),
            retryable=data.get("retryable", False),
            correlation_id=data.get("correlation_id"),
        )


@dataclass
class ToolResult:
    """Outcome of a tool execution with structured payload or error."""
    request_id: str
    tool_name: str
    success: bool
    data: Any | None = None
    error: ToolError | None = None
    operation_id: str | None = None
    approval_id: str | None = None
    execution_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
            "error": self.error.to_dict() if self.error else None,
            "operation_id": self.operation_id,
            "approval_id": self.approval_id,
            "execution_time_seconds": self.execution_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        err = ToolError.from_dict(data["error"]) if data.get("error") else None
        return cls(
            request_id=data["request_id"],
            tool_name=data["tool_name"],
            success=data["success"],
            data=data.get("data"),
            error=err,
            operation_id=data.get("operation_id"),
            approval_id=data.get("approval_id"),
            execution_time_seconds=data.get("execution_time_seconds", 0.0),
        )


@dataclass
class PolicyDecision:
    """Outcome of evaluating an action against policy rules and budget limits."""
    decision: PolicyDecisionType
    reason: str
    effective_limits: dict[str, Any] = field(default_factory=dict)
    policy_version: str = "1.0.0"
    approval_required: bool = False
    estimated_cost: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "effective_limits": self.effective_limits,
            "policy_version": self.policy_version,
            "approval_required": self.approval_required,
            "estimated_cost": self.estimated_cost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolicyDecision:
        return cls(
            decision=PolicyDecisionType(data["decision"]),
            reason=data["reason"],
            effective_limits=data.get("effective_limits", {}),
            policy_version=data.get("policy_version", "1.0.0"),
            approval_required=data.get("approval_required", False),
            estimated_cost=data.get("estimated_cost", {}),
        )


@dataclass
class ApprovalRequest:
    """Persistent authorization request for a high-impact or out-of-budget action."""
    approval_id: str
    action: str
    actor: str
    run_id: str
    arguments_hash: str
    arguments: dict[str, Any]
    policy_version: str
    max_cost: dict[str, Any]
    expires_at: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    reviewer: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "action": self.action,
            "actor": self.actor,
            "run_id": self.run_id,
            "arguments_hash": self.arguments_hash,
            "arguments": self.arguments,
            "policy_version": self.policy_version,
            "max_cost": self.max_cost,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "reviewer": self.reviewer,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalRequest:
        return cls(
            approval_id=data["approval_id"],
            action=data["action"],
            actor=data["actor"],
            run_id=data["run_id"],
            arguments_hash=data["arguments_hash"],
            arguments=data.get("arguments", {}),
            policy_version=data.get("policy_version", "1.0.0"),
            max_cost=data.get("max_cost", {}),
            expires_at=data["expires_at"],
            status=ApprovalStatus(data.get("status", ApprovalStatus.PENDING.value)),
            reviewer=data.get("reviewer"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class OperationRecord:
    """Audit ledger entry for an executed or in-progress agent operation."""
    operation_id: str
    run_id: str
    actor: str
    idempotency_key: str
    action: str
    arguments_hash: str
    arguments: dict[str, Any]
    status: OperationStatus = OperationStatus.PENDING
    reserved_budget: dict[str, Any] = field(default_factory=dict)
    consumed_budget: dict[str, Any] = field(default_factory=dict)
    result_ref: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "run_id": self.run_id,
            "actor": self.actor,
            "idempotency_key": self.idempotency_key,
            "action": self.action,
            "arguments_hash": self.arguments_hash,
            "arguments": self.arguments,
            "status": self.status.value,
            "reserved_budget": self.reserved_budget,
            "consumed_budget": self.consumed_budget,
            "result_ref": self.result_ref,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OperationRecord:
        return cls(
            operation_id=data["operation_id"],
            run_id=data["run_id"],
            actor=data["actor"],
            idempotency_key=data["idempotency_key"],
            action=data["action"],
            arguments_hash=data["arguments_hash"],
            arguments=data.get("arguments", {}),
            status=OperationStatus(data.get("status", OperationStatus.PENDING.value)),
            reserved_budget=data.get("reserved_budget", {}),
            consumed_budget=data.get("consumed_budget", {}),
            result_ref=data.get("result_ref"),
            error_code=data.get("error_code"),
            error_message=data.get("error_message"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )


@dataclass
class AgentContext:
    """Structured evidence and state provided to the agent/LLM (never raw data)."""
    run_id: str
    dataset_id: str
    profile_summary: dict[str, Any]
    compatible_models: list[str]
    leaderboard_summary: list[dict[str, Any]] = field(default_factory=list)
    feature_evidence_summary: list[dict[str, Any]] = field(default_factory=list)
    active_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    budget_status: dict[str, Any] = field(default_factory=dict)
    omissions_and_limits: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentContext:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentSessionState:
    """Durable state representation for an autonomous agent session."""
    session_id: str
    run_id: str
    goal: str
    version: str = "1.0.0"
    status: str = "active"
    current_hypothesis_id: str | None = None
    pending_operation_id: str | None = None
    pending_approval_id: str | None = None
    iteration_count: int = 0
    max_iterations: int = 10
    budget: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    checkpoint_id: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSessionState:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
