"""Canonical fixtures for V0.9 contracts, tools, schemas, and ledger testing."""
from __future__ import annotations

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
from automl.application.agents.contracts import (
    AgentContext,
    AgentSessionState,
    ApprovalRequest,
    OperationRecord,
    PolicyDecision,
    ToolCallContext,
    ToolDefinition,
    ToolError,
    ToolInvocation,
    ToolResult,
)
from automl.application.agents.policy import compute_arguments_hash


SAMPLE_VALID_TOOL_PAYLOADS: dict[str, dict[str, Any]] = {
    "get_dataset_profile": {"dataset_id": "dataset-123"},
    "list_models": {"task_type": "binary_classification"},
    "list_plugins": {"plugin_type": "model"},
    "list_experiments": {"run_id": "run-456"},
    "get_leaderboard": {"run_id": "run-456", "top_k": 5},
    "get_feature_evidence": {"run_id": "run-456", "feature_id": "age"},
    "get_feature_ranking": {"run_id": "run-456"},
    "create_experiment": {
        "run_id": "run-456",
        "model_name": "xgboost",
        "feature_names": ["age", "balance", "estimated_salary"],
        "parameters": {"n_estimators": 100, "learning_rate": 0.05},
    },
    "prioritize_feature": {
        "run_id": "run-456",
        "feature_name": "balance",
        "priority": "high",
    },
    "run_experiment": {
        "run_id": "run-456",
        "experiment_id": "exp-789",
    },
    "optimize_experiment": {
        "run_id": "run-456",
        "experiment_id": "exp-789",
        "n_trials": 20,
    },
}

SAMPLE_INVALID_TOOL_PAYLOADS: dict[str, list[dict[str, Any]]] = {
    "get_dataset_profile": [
        {},  # missing dataset_id
        {"dataset_id": 123},  # non-string dataset_id
        {"dataset_id": "d1", "unknown_param": True},  # additional property disallowed
    ],
    "list_models": [
        {"task_type": "invalid_task"},  # not in enum
        {"task_type": 42},  # non-string
    ],
    "list_plugins": [
        {"plugin_type": "arbitrary_type"},  # not in enum
    ],
    "list_experiments": [
        {},  # missing run_id
    ],
    "get_leaderboard": [
        {"run_id": "r1", "top_k": 0},  # below minimum 1
        {"top_k": 5},  # missing run_id
    ],
    "get_feature_evidence": [
        {"run_id": "r1"},  # missing feature_id
        {"feature_id": "f1"},  # missing run_id
    ],
    "get_feature_ranking": [
        {},  # missing run_id
    ],
    "create_experiment": [
        {"run_id": "r1"},  # missing model_name
        {"model_name": "rf"},  # missing run_id
        {"run_id": "r1", "model_name": "rf", "feature_names": [123]},  # invalid item type
    ],
    "prioritize_feature": [
        {"run_id": "r1", "feature_name": "f1", "priority": "ultra"},  # invalid enum
    ],
    "run_experiment": [
        {"run_id": "r1"},  # missing experiment_id
    ],
    "optimize_experiment": [
        {"run_id": "r1", "experiment_id": "e1", "n_trials": 200},  # exceeds max 100
        {"run_id": "r1", "experiment_id": "e1", "n_trials": 0},  # below min 1
    ],
}


def make_sample_context(
    actor: str = "agent-unit-test",
    run_id: str = "run-test-001",
    permission: AgentPermission = AgentPermission.EXECUTE_WITHIN_BUDGET,
    workspace_path: str = "/tmp/catml_workspace",
    correlation_id: str = "corr-test-123",
) -> ToolCallContext:
    return ToolCallContext(
        actor=actor,
        workspace_path=workspace_path,
        run_id=run_id,
        correlation_id=correlation_id,
        deadline=None,
        permission=permission,
    )


def make_sample_invocation(
    tool_name: str = "get_dataset_profile",
    arguments: dict[str, Any] | None = None,
    context: ToolCallContext | None = None,
    idempotency_key: str | None = None,
) -> ToolInvocation:
    ctx = context or make_sample_context()
    args = arguments if arguments is not None else {"dataset_id": "dataset-sample"}
    return ToolInvocation(
        tool_name=tool_name,
        arguments=args,
        context=ctx,
        version="1.0.0",
        idempotency_key=idempotency_key,
    )


def make_sample_approval(
    approval_id: str = "appr-001",
    action: str = "run_experiment",
    actor: str = "agent-test",
    run_id: str = "run-test-001",
    arguments: dict[str, Any] | None = None,
) -> ApprovalRequest:
    args = arguments or {"run_id": run_id, "experiment_id": "exp-001"}
    arg_hash = compute_arguments_hash(action, actor, run_id, args, "1.0.0", {"experiments": 1})
    return ApprovalRequest(
        approval_id=approval_id,
        action=action,
        actor=actor,
        run_id=run_id,
        arguments_hash=arg_hash,
        arguments=args,
        policy_version="1.0.0",
        max_cost={"experiments": 1, "trials": 1},
        expires_at="2026-12-31T23:59:59Z",
        status=ApprovalStatus.PENDING,
    )


def make_sample_operation(
    operation_id: str = "op-001",
    run_id: str = "run-test-001",
    actor: str = "agent-test",
    idempotency_key: str = "idem-op-001",
    action: str = "create_experiment",
    arguments: dict[str, Any] | None = None,
) -> OperationRecord:
    args = arguments or {"run_id": run_id, "model_name": "xgboost"}
    arg_hash = compute_arguments_hash(action, actor, run_id, args, "1.0.0", {"experiments": 1})
    return OperationRecord(
        operation_id=operation_id,
        run_id=run_id,
        actor=actor,
        idempotency_key=idempotency_key,
        action=action,
        arguments_hash=arg_hash,
        arguments=args,
        status=OperationStatus.PENDING,
        reserved_budget={"experiments": 1},
        consumed_budget={"experiments": 0},
    )


def make_sample_hypothesis(
    hypothesis_id: str = "hyp-001",
    run_id: str = "run-test-001",
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        run_id=run_id,
        reasoning="Testing gradient boosting with target encoded high cardinality feature",
        candidate_config={"model": "xgboost", "n_estimators": 100},
        target_metric="roc_auc",
        metric_direction="maximize",
        baseline_metric=0.852,
        verification_criteria={"min_improvement": 0.005, "cv_folds": 5},
    )


def make_sample_session_state(
    session_id: str = "sess-001",
    run_id: str = "run-test-001",
) -> AgentSessionState:
    return AgentSessionState(
        session_id=session_id,
        run_id=run_id,
        goal="Maximize ROC-AUC on Kaggle Churn dataset",
        version="1.0.0",
        status="active",
        iteration_count=1,
        max_iterations=5,
        budget={"experiments_remaining": 4},
    )
