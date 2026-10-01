"""Unit and integration tests for Package A3 (Persona A).

Covers:
- Run leases (exclusive ownership, heartbeats, TTL expiration takeover)
- Operation reconciliation (recovering dangling operations without active leases)
- Cooperative cancellation (status transitions and trial loop interruption)
- Atomic active budget reservations (concurrency protection)
- optimize_experiment tool execution and dynamic budget deduction
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import uuid

import pytest

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    OperationStatus,
    RunLease,
    ToolEffect,
    ToolErrorCode,
)
from automl.application.agents.contracts import (
    OperationRecord,
    ToolCallContext,
    ToolInvocation,
)
from automl.application.agents.policy import (
    AgentPolicyConfig,
    PolicyEvaluator,
)
from automl.application.agents.executor import (
    ToolExecutor,
    create_full_tool_registry,
)
from automl.application.bus.command_bus import CommandBus
from automl.application.bus.query_bus import QueryBus
from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    OptimizeExperimentCommand,
)
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus
from automl.domain.experiments.trial import Experiment, ExperimentStatus
from automl.domain.datasets.profile import Dataset
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger


@pytest.fixture
def temp_ledger(tmp_path: Path) -> SqliteAgentLedger:
    db_file = tmp_path / "test_agent_a3_ledger.db"
    return SqliteAgentLedger(db_file)


@pytest.fixture
def mock_buses():
    q_bus = QueryBus()
    c_bus = CommandBus()
    return q_bus, c_bus


# =========================================================================
# 1. Run Lease Tests
# =========================================================================

def test_run_lease_acquire_and_exclusive_ownership(temp_ledger):
    run_id = "run-lease-1"
    worker_a = "worker-node-a"
    worker_b = "worker-node-b"

    # Worker A acquires lease
    acquired = temp_ledger.acquire_run_lease(run_id, worker_a, ttl_seconds=30.0)
    assert acquired is True

    # Worker B tries to acquire same run lease -> denied
    acquired_b = temp_ledger.acquire_run_lease(run_id, worker_b, ttl_seconds=30.0)
    assert acquired_b is False

    # Worker A re-acquires / renews -> succeeds
    renewed = temp_ledger.acquire_run_lease(run_id, worker_a, ttl_seconds=60.0)
    assert renewed is True

    lease = temp_ledger.get_run_lease(run_id)
    assert lease is not None
    assert lease.run_id == run_id
    assert lease.owner_id == worker_a
    assert not lease.is_expired()


def test_run_lease_heartbeat_and_release(temp_ledger):
    run_id = "run-lease-2"
    worker_a = "worker-node-a"
    worker_b = "worker-node-b"

    temp_ledger.acquire_run_lease(run_id, worker_a, ttl_seconds=10.0)

    # Worker A sends heartbeat -> succeeds
    assert temp_ledger.heartbeat_run_lease(run_id, worker_a, ttl_seconds=20.0) is True

    # Worker B tries to heartbeat Worker A's lease -> fails
    assert temp_ledger.heartbeat_run_lease(run_id, worker_b, ttl_seconds=20.0) is False

    # Worker B tries to release Worker A's lease -> fails
    assert temp_ledger.release_run_lease(run_id, worker_b) is False

    # Worker A releases lease -> succeeds
    assert temp_ledger.release_run_lease(run_id, worker_a) is True

    # Now Worker B can acquire
    assert temp_ledger.acquire_run_lease(run_id, worker_b, ttl_seconds=15.0) is True
    lease = temp_ledger.get_run_lease(run_id)
    assert lease is not None
    assert lease.owner_id == worker_b


def test_run_lease_ttl_expiration_takeover(temp_ledger):
    run_id = "run-lease-3"
    worker_a = "worker-crashed"
    worker_b = "worker-takeover"

    # Worker A acquires short lease (0.05 seconds)
    temp_ledger.acquire_run_lease(run_id, worker_a, ttl_seconds=0.05)
    time.sleep(0.08)

    lease = temp_ledger.get_run_lease(run_id)
    assert lease is not None
    assert lease.is_expired() is True

    # Worker B can now take over the expired lease
    acquired = temp_ledger.acquire_run_lease(run_id, worker_b, ttl_seconds=10.0)
    assert acquired is True

    active_lease = temp_ledger.get_run_lease(run_id)
    assert active_lease is not None
    assert active_lease.owner_id == worker_b
    assert not active_lease.is_expired()


# =========================================================================
# 2. Reconciliation Tests
# =========================================================================

def test_reconcile_operations_recovers_dangling_tasks(temp_ledger):
    run_crashed = "run-crashed-1"
    run_active = "run-active-2"

    # run_crashed has an expired lease
    temp_ledger.acquire_run_lease(run_crashed, "crashed-worker", ttl_seconds=0.05)

    # run_active has an active lease
    temp_ledger.acquire_run_lease(run_active, "alive-worker", ttl_seconds=60.0)

    # Record dangling operation on run_crashed
    op_dangling = OperationRecord(
        operation_id="op-dang-001",
        run_id=run_crashed,
        actor="agent-user",
        idempotency_key="key-dangling-1",
        action="optimize_experiment",
        arguments_hash="hash-1",
        arguments={"experiment_id": "exp-1"},
        status=OperationStatus.RUNNING,
        reserved_budget={"trials": 10},
    )
    temp_ledger.record_operation(op_dangling)

    # Record legitimate running operation on run_active
    op_healthy = OperationRecord(
        operation_id="op-health-002",
        run_id=run_active,
        actor="agent-user",
        idempotency_key="key-healthy-1",
        action="optimize_experiment",
        arguments_hash="hash-2",
        arguments={"experiment_id": "exp-2"},
        status=OperationStatus.RUNNING,
        reserved_budget={"trials": 5},
    )
    temp_ledger.record_operation(op_healthy)

    time.sleep(0.08)  # ensure crashed lease is expired

    # Reconcile operations
    recovered = temp_ledger.reconcile_operations()
    assert len(recovered) == 1
    assert recovered[0].operation_id == "op-dang-001"
    assert recovered[0].status == OperationStatus.RECOVERY_REQUIRED
    assert recovered[0].error_code == ToolErrorCode.RECOVERY_REQUIRED.value

    # Verify healthy operation is untouched
    healthy_check = temp_ledger.get_operation("op-health-002")
    assert healthy_check is not None
    assert healthy_check.status == OperationStatus.RUNNING


# =========================================================================
# 3. Cooperative Cancellation Tests
# =========================================================================

def test_request_operation_cancellation_transitions(temp_ledger):
    # 1. PENDING operation -> transitions immediately to CANCELLED
    op_pending = OperationRecord(
        operation_id="op-pending-1",
        run_id="run-cancel",
        actor="agent-user",
        idempotency_key="k1",
        action="optimize_experiment",
        arguments_hash="h1",
        arguments={},
        status=OperationStatus.PENDING,
    )
    temp_ledger.record_operation(op_pending)
    res_pending = temp_ledger.request_operation_cancellation("op-pending-1")
    assert res_pending.status == OperationStatus.CANCELLED
    assert res_pending.error_code == ToolErrorCode.CANCELLED.value
    assert temp_ledger.is_cancellation_requested("op-pending-1") is True

    # 2. RUNNING operation -> transitions to CANCEL_REQUESTED
    op_running = OperationRecord(
        operation_id="op-running-1",
        run_id="run-cancel",
        actor="agent-user",
        idempotency_key="k2",
        action="optimize_experiment",
        arguments_hash="h2",
        arguments={},
        status=OperationStatus.RUNNING,
    )
    temp_ledger.record_operation(op_running)
    res_running = temp_ledger.request_operation_cancellation("op-running-1")
    assert res_running.status == OperationStatus.CANCEL_REQUESTED
    assert temp_ledger.is_cancellation_requested("op-running-1") is True

    # 3. Terminal SUCCEEDED operation -> does not transition
    op_succeeded = OperationRecord(
        operation_id="op-succ-1",
        run_id="run-cancel",
        actor="agent-user",
        idempotency_key="k3",
        action="optimize_experiment",
        arguments_hash="h3",
        arguments={},
        status=OperationStatus.SUCCEEDED,
    )
    temp_ledger.record_operation(op_succeeded)
    res_succ = temp_ledger.request_operation_cancellation("op-succ-1")
    assert res_succ.status == OperationStatus.SUCCEEDED
    assert temp_ledger.is_cancellation_requested("op-succ-1") is False


# =========================================================================
# 4. Atomic Budget Reservations Tests
# =========================================================================

def test_active_reserved_budget_aggregation(temp_ledger):
    run_id = "run-budget-test"

    # op1: running with 5 trials
    temp_ledger.record_operation(
        OperationRecord(
            operation_id="op-b1",
            run_id=run_id,
            actor="agent",
            idempotency_key="bk1",
            action="optimize_experiment",
            arguments_hash="bh1",
            arguments={},
            status=OperationStatus.RUNNING,
            reserved_budget={"trials": 5, "experiments": 1},
        )
    )

    # op2: cancel_requested with 3 trials
    temp_ledger.record_operation(
        OperationRecord(
            operation_id="op-b2",
            run_id=run_id,
            actor="agent",
            idempotency_key="bk2",
            action="optimize_experiment",
            arguments_hash="bh2",
            arguments={},
            status=OperationStatus.CANCEL_REQUESTED,
            reserved_budget={"trials": 3},
        )
    )

    # op3: completed succeeded with 10 trials -> should NOT count as active reserved
    temp_ledger.record_operation(
        OperationRecord(
            operation_id="op-b3",
            run_id=run_id,
            actor="agent",
            idempotency_key="bk3",
            action="optimize_experiment",
            arguments_hash="bh3",
            arguments={},
            status=OperationStatus.SUCCEEDED,
            reserved_budget={"trials": 10},
        )
    )

    active_res = temp_ledger.get_active_reserved_budget(run_id)
    assert active_res["trials"] == 8
    assert active_res["experiments"] == 1


def test_executor_concurrent_budget_reservation_blocks_over_budget(mock_buses, temp_ledger):
    q_bus, c_bus = mock_buses
    registry = create_full_tool_registry(q_bus, c_bus)
    policy = AgentPolicyConfig(
        require_approval_for_mutations=False,
        default_mode=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    evaluator = PolicyEvaluator(policy)
    executor = ToolExecutor(registry=registry, policy_evaluator=evaluator, ledger=temp_ledger)

    run_id = "run-concurrent-budget"

    # Already active operation reserving 15 trials
    temp_ledger.record_operation(
        OperationRecord(
            operation_id="op-existing-flight",
            run_id=run_id,
            actor="agent-1",
            idempotency_key="k-flight",
            action="optimize_experiment",
            arguments_hash="h-flight",
            arguments={},
            status=OperationStatus.RUNNING,
            reserved_budget={"trials": 15},
        )
    )

    # Global budget has max_trials = 20
    budget = AgentBudget(max_trials=20, consumed_trials=0)

    # Request requires 10 trials -> 15 (active) + 10 (new) = 25 > 20 -> must be denied
    invocation = ToolInvocation(
        tool_name="optimize_experiment",
        arguments={"run_id": run_id, "experiment_id": "exp_abc", "n_trials": 10},
        context=ToolCallContext(
            actor="agent-2",
            run_id=run_id,
            workspace_path="/tmp/test",
            correlation_id="corr-1",
            permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
        ),
    )

    res = executor.execute(invocation, budget=budget)
    assert res.success is False
    assert res.error is not None
    assert res.error.code == ToolErrorCode.BUDGET_EXCEEDED
    assert "Trial budget exceeded" in res.error.message


# =========================================================================
# 5. ToolExecutor optimize_experiment & Cooperative Cancellation Tests
# =========================================================================

def test_executor_optimize_experiment_success_and_dynamic_consumption(mock_buses, temp_ledger):
    q_bus, c_bus = mock_buses

    dispatched = []

    def _handle_opt(cmd: OptimizeExperimentCommand):
        dispatched.append(cmd)
        return {
            "best_trial_id": "trial_best_99",
            "best_score": 0.9452,
            "trials_executed": cmd.n_trials,
        }

    c_bus.register(OptimizeExperimentCommand, _handle_opt)

    registry = create_full_tool_registry(q_bus, c_bus)
    policy = AgentPolicyConfig(
        require_approval_for_mutations=False,
        default_mode=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    evaluator = PolicyEvaluator(policy)
    executor = ToolExecutor(registry=registry, policy_evaluator=evaluator, ledger=temp_ledger)

    budget = AgentBudget(max_trials=20)
    invocation = ToolInvocation(
        tool_name="optimize_experiment",
        arguments={"run_id": "run-opt-1", "experiment_id": "exp-test-opt", "n_trials": 7},
        context=ToolCallContext(
            actor="agent-user",
            run_id="run-opt-1",
            workspace_path="/tmp/test",
            correlation_id="corr-2",
            permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
        ),
    )

    res = executor.execute(invocation, budget=budget)
    assert res.success is True
    assert res.data["best_trial_id"] == "trial_best_99"
    assert res.data["best_score"] == 0.9452

    # Budget consumed 7 trials
    assert budget.consumed_trials == 7

    # Verify ledger status
    op_record = temp_ledger.get_operation(res.operation_id)
    assert op_record is not None
    assert op_record.status == OperationStatus.SUCCEEDED
    assert op_record.consumed_budget == {"trials": 7}


def test_executor_cancellation_cooperative_abort(mock_buses, temp_ledger):
    q_bus, c_bus = mock_buses

    # If executed, handler would fail test
    def _handle_opt(cmd: OptimizeExperimentCommand):
        pytest.fail("Handler should not have been called after cancellation was requested")

    c_bus.register(OptimizeExperimentCommand, _handle_opt)

    registry = create_full_tool_registry(q_bus, c_bus)
    policy = AgentPolicyConfig(
        require_approval_for_mutations=False,
        default_mode=AgentPermission.EXECUTE_WITHIN_BUDGET,
    )
    evaluator = PolicyEvaluator(policy)
    executor = ToolExecutor(registry=registry, policy_evaluator=evaluator, ledger=temp_ledger)

    idemp_key = "idemp-cancel-test"
    run_id = "run-cancel-test"

    # Pre-record operation with cancel_requested
    arg_hash = "fakehash"
    op_rec = OperationRecord(
        operation_id="op-pre-cancelled",
        run_id=run_id,
        actor="agent-user",
        idempotency_key=idemp_key,
        action="optimize_experiment",
        arguments_hash=arg_hash,
        arguments={"run_id": run_id, "experiment_id": "exp-1", "n_trials": 5},
        status=OperationStatus.CANCEL_REQUESTED,
    )
    temp_ledger.record_operation(op_rec)

    invocation = ToolInvocation(
        tool_name="optimize_experiment",
        idempotency_key=idemp_key,
        arguments={"run_id": run_id, "experiment_id": "exp-1", "n_trials": 5},
        context=ToolCallContext(
            actor="agent-user",
            run_id=run_id,
            workspace_path="/tmp/test",
            correlation_id="corr-3",
            permission=AgentPermission.EXECUTE_WITHIN_BUDGET,
        ),
    )

    res = executor.execute(invocation)
    assert res.success is False
    assert res.error is not None
    assert res.error.code in (ToolErrorCode.CANCELLED, ToolErrorCode.CONFLICT)


def test_workspace_optimize_experiment_stops_trials_upon_cancellation(tmp_path):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws_opt_cancel"))

    # Create dummy dataset file with enough rows for CV
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("x1,x2,target\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n11,12,1\n" * 10)

    dataset = ws.register_dataset("cancel_ds", str(csv_file), target="target")
    run = ws.create_run(dataset, metric="f1")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="exp_opt_cancel",
            feature_names=["x1", "x2"],
            model_ids=["logistic_regression"],
        )
    )

    # Cooperative cancellation check: cancel run after 1 trial executed
    trial_count = 0

    def cancellation_check():
        nonlocal trial_count
        trial_count += 1
        if trial_count >= 2:
            r = ws.repository.get_run(run.id)
            r.transition_to(RunStatus.CANCELLED, RunPhase.OPTIMIZATION)
            ws.repository.save_run(r)

    ws.execution_check = cancellation_check

    # Execute 10 trials
    result = ws.optimize_experiment(
        run_id=run.id,
        experiment_id=exp.id,
        n_trials=10,
    )

    # Must have stopped early upon observing cancellation
    assert result["trials_executed"] < 10
    assert result["trials_executed"] == 1
    assert "best_trial_id" in result

    # Experiment should be marked CANCELLED
    exp_after = ws.repository.get_experiment(exp.id)
    assert exp_after.status == ExperimentStatus.CANCELLED
