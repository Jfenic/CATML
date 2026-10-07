"""Unit and integration tests for Milestone H5 Package B5: LangGraph durable orchestrator and SQLite checkpointer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
import pytest

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    Hypothesis,
    OperationStatus,
    PolicyDecisionType,
)
from automl.application.agents.contracts import (
    AgentSessionState,
    CandidateProposal,
    EvaluationFeedback,
    OperationRecord,
    PolicyDecision,
    ToolCallContext,
    ToolInvocation,
    ToolResult,
)
from automl.application.agents.orchestrator.state_machine import (
    AgentStateMachine,
    CycleState,
    SessionStatus,
    StopReason,
)
from automl.application.agents.orchestrator.graph import (
    LangGraphAgentOrchestrator,
    is_langgraph_available,
)
from automl.application.agents.state import (
    GraphAgentState,
    deserialize_graph_state,
    graph_state_to_session,
    serialize_graph_state,
    session_to_graph_state,
)
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from automl.infrastructure.database.sqlite_checkpoint_saver import SqliteCheckpointSaver
from automl.interfaces.cli.agent_session_cli import (
    session_resume_cli,
    session_start_cli,
    session_status_cli,
)


@pytest.fixture
def temp_ledger(tmp_path: Path) -> SqliteAgentLedger:
    db_file = tmp_path / "agent_ledger.db"
    return SqliteAgentLedger(db_file)


@pytest.fixture
def mock_workspace():
    ws = MagicMock()
    run = MagicMock()
    run.id = "run-b5-001"
    run.dataset_id = "ds-b5-001"
    ws.get_run.return_value = run

    dataset = MagicMock()
    dataset.id = "ds-b5-001"
    dataset.name = "credit_risk"
    dataset.target_column = "default"
    ws.get_dataset.return_value = dataset

    profile = MagicMock()
    profile.n_rows = 500
    profile.n_columns = 6
    profile.numeric_columns = ["income", "debt", "score"]
    profile.categorical_columns = ["purpose"]
    profile.task_type = "binary_classification"
    ws.get_profile.return_value = profile

    ws.get_leaderboard.return_value = []
    ws.list_feature_sets.return_value = []
    return ws


# =====================================================================
# 1. State Mapping & Serialization Tests
# =====================================================================

def test_session_state_to_graph_state_roundtrip():
    session = AgentSessionState(
        session_id="sess-roundtrip-1",
        run_id="run-001",
        goal="Maximize ROC-AUC",
        version="1.0.0",
        status=SessionStatus.ACTIVE.value,
        iteration_count=2,
        max_iterations=8,
        budget={"experiments": 3, "trials": 10},
        stop_reason=None,
        current_hypothesis_id="hyp-123",
        pending_approval_id="appr-456",
        checkpoint_id="chk-sess-roundtrip-1-2",
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:05:00Z",
    )

    graph_state = session_to_graph_state(session)
    assert graph_state["session_id"] == "sess-roundtrip-1"
    assert graph_state["run_id"] == "run-001"
    assert graph_state["iteration_count"] == 2
    assert graph_state["max_iterations"] == 8
    assert graph_state["budget"]["experiments"] == 3
    assert graph_state["current_hypothesis_id"] == "hyp-123"
    assert graph_state["pending_approval_id"] == "appr-456"

    restored = graph_state_to_session(graph_state)
    assert restored.session_id == session.session_id
    assert restored.run_id == session.run_id
    assert restored.goal == session.goal
    assert restored.iteration_count == session.iteration_count
    assert restored.max_iterations == session.max_iterations
    assert restored.budget == session.budget
    assert restored.current_hypothesis_id == session.current_hypothesis_id
    assert restored.pending_approval_id == session.pending_approval_id


def test_graph_state_serialization():
    state: GraphAgentState = {
        "session_id": "sess-json-1",
        "run_id": "run-001",
        "goal": "Test goal",
        "iteration_count": 1,
        "max_iterations": 5,
        "status": "active",
        "budget": {"experiments": 2},
    }
    payload = serialize_graph_state(state)
    assert isinstance(payload, str)
    deserialized = deserialize_graph_state(payload)
    assert deserialized["session_id"] == "sess-json-1"
    assert deserialized["budget"] == {"experiments": 2}


# =====================================================================
# 2. Durable SQLite Checkpoint Saver Tests
# =====================================================================

def test_sqlite_checkpoint_saver_in_memory():
    with SqliteCheckpointSaver(":memory:") as cp:
        assert cp.saver is not None
        assert cp.connection is not None
        # Verify empty initial state
        assert cp.get_tuple("thread-nonexistent") is None
        assert cp.get_latest_state("thread-nonexistent") is None


def test_sqlite_checkpoint_saver_file_persistence(tmp_path: Path):
    db_file = tmp_path / "checkpoints.db"

    # First session write
    with SqliteCheckpointSaver(db_file) as cp1:
        assert cp1.db_path == db_file
        # Save a manual checkpoint tuple through saver
        config = {"configurable": {"thread_id": "thread-durable-1", "checkpoint_ns": ""}}
        checkpoint = {
            "v": 1,
            "id": "chk-001",
            "ts": "2026-10-06T12:00:00Z",
            "channel_values": {"iteration_count": 3, "score": 0.885},
            "channel_versions": {"iteration_count": "1", "score": "1"},
            "versions_seen": {},
        }
        metadata = {"source": "test"}
        cp1.saver.put(config, checkpoint, metadata, {"iteration_count": "1", "score": "1"})

    # Reconnect to same database file and verify persistence
    with SqliteCheckpointSaver(db_file) as cp2:
        tup = cp2.get_tuple("thread-durable-1")
        assert tup is not None
        assert tup.checkpoint["id"] == "chk-001"
        assert tup.checkpoint["channel_values"]["iteration_count"] == 3
        assert tup.checkpoint["channel_values"]["score"] == 0.885

        latest = cp2.get_latest_state("thread-durable-1")
        assert latest is not None
        assert latest["score"] == 0.885


# =====================================================================
# 3. LangGraph Orchestrator Full Cycle & Stop Conditions
# =====================================================================

def test_langgraph_orchestrator_max_iterations(temp_ledger: SqliteAgentLedger, mock_workspace):
    session = AgentSessionState(
        session_id="sess-lg-iter",
        run_id="run-b5-001",
        goal="Autonomous testing",
        version="1.0.0",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=3,
        budget=AgentBudget(max_experiments=10, max_trials=20).to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    temp_ledger.save_session_state(session)

    checkpointer = SqliteCheckpointSaver(":memory:")
    orchestrator = LangGraphAgentOrchestrator(
        workspace=mock_workspace,
        session_store=temp_ledger,
        checkpointer=checkpointer.saver,
        approval_store=temp_ledger,
        operation_store=temp_ledger,
        patience=5,
    )

    final_state = orchestrator.run(session)
    assert final_state["status"] == SessionStatus.COMPLETED.value
    assert final_state["stop_reason"] == StopReason.MAX_ITERATIONS_REACHED.value
    assert final_state["iteration_count"] == 3

    # Verify session state was synced to session_store
    persisted_session = temp_ledger.get_session_state("sess-lg-iter")
    assert persisted_session is not None
    assert persisted_session.status == SessionStatus.COMPLETED.value
    assert persisted_session.iteration_count == 3
    assert persisted_session.checkpoint_id == "chk-sess-lg-iter-3"

    # Verify SQLite checkpointer tuple persisted
    state_from_cp = orchestrator.get_latest_graph_state("sess-lg-iter")
    assert state_from_cp is not None
    assert state_from_cp["iteration_count"] == 3


def test_langgraph_orchestrator_target_reached(temp_ledger: SqliteAgentLedger, mock_workspace):
    budget_dict = AgentBudget(max_experiments=10, max_trials=20).to_dict()
    budget_dict["target_score"] = 0.82  # simulated score reaches 0.82 on step 1

    session = AgentSessionState(
        session_id="sess-lg-target",
        run_id="run-b5-001",
        goal="Reach 0.82",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=10,
        budget=budget_dict,
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    temp_ledger.save_session_state(session)

    checkpointer = SqliteCheckpointSaver(":memory:")
    orchestrator = LangGraphAgentOrchestrator(
        workspace=mock_workspace,
        session_store=temp_ledger,
        checkpointer=checkpointer.saver,
    )

    final_state = orchestrator.run(session)
    assert final_state["status"] == SessionStatus.COMPLETED.value
    assert final_state["stop_reason"] == StopReason.TARGET_REACHED.value
    assert final_state["iteration_count"] == 1


def test_langgraph_orchestrator_budget_exhausted(temp_ledger: SqliteAgentLedger, mock_workspace):
    budget = AgentBudget(max_experiments=1, max_trials=1)  # only 1 experiment allowed
    session = AgentSessionState(
        session_id="sess-lg-budget",
        run_id="run-b5-001",
        goal="Test budget stop",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=10,
        budget=budget.to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    temp_ledger.save_session_state(session)

    checkpointer = SqliteCheckpointSaver(":memory:")
    orchestrator = LangGraphAgentOrchestrator(
        workspace=mock_workspace,
        session_store=temp_ledger,
        checkpointer=checkpointer.saver,
    )

    final_state = orchestrator.run(session)
    assert final_state["status"] == SessionStatus.STOPPED.value
    assert final_state["stop_reason"] == StopReason.BUDGET_EXHAUSTED.value


# =====================================================================
# 4. Human Approval Interruption & Resumption
# =====================================================================

def test_langgraph_orchestrator_human_approval_approved(temp_ledger: SqliteAgentLedger, mock_workspace):
    session = AgentSessionState(
        session_id="sess-lg-approval",
        run_id="run-b5-001",
        goal="Test approval requirement",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=5,
        budget=AgentBudget(max_experiments=5, max_trials=10).to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    temp_ledger.save_session_state(session)

    checkpointer = SqliteCheckpointSaver(":memory:")
    orchestrator = LangGraphAgentOrchestrator(
        workspace=mock_workspace,
        session_store=temp_ledger,
        checkpointer=checkpointer.saver,
        approval_store=temp_ledger,
        permission=AgentPermission.PROPOSE_ONLY,  # forces human approval
    )

    # 1. Initial invocation halts at interrupt
    interrupted_state = orchestrator.run(session)
    assert "__interrupt__" in interrupted_state
    interrupt_info = interrupted_state["__interrupt__"][0]
    assert "approval_id" in interrupt_info.value

    # Verify approval request was saved in ledger
    approvals = temp_ledger.list_approvals(run_id="run-b5-001")
    assert len(approvals) == 1
    appr_id = approvals[0].approval_id

    # 2. Human approves via resume
    resumed_state = orchestrator.resume(
        session_id=session.session_id,
        human_decision={"approved": True},
    )
    # Check that execution advanced past gate to execute and check_stop
    assert resumed_state["iteration_count"] >= 1

    # Verify approval status in ledger updated to approved
    updated_appr = temp_ledger.get_approval(appr_id)
    assert updated_appr is not None
    assert updated_appr.status == ApprovalStatus.APPROVED


def test_langgraph_orchestrator_human_approval_rejected(temp_ledger: SqliteAgentLedger, mock_workspace):
    session = AgentSessionState(
        session_id="sess-lg-reject",
        run_id="run-b5-001",
        goal="Test approval rejection",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=3,
        budget=AgentBudget(max_experiments=5, max_trials=10).to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    temp_ledger.save_session_state(session)

    checkpointer = SqliteCheckpointSaver(":memory:")
    orchestrator = LangGraphAgentOrchestrator(
        workspace=mock_workspace,
        session_store=temp_ledger,
        checkpointer=checkpointer.saver,
        approval_store=temp_ledger,
        permission=AgentPermission.PROPOSE_ONLY,
    )

    interrupted_state = orchestrator.run(session)
    assert "__interrupt__" in interrupted_state

    # Human rejects the proposal
    resumed_state = orchestrator.resume(
        session_id=session.session_id,
        human_decision={"approved": False},
    )
    # Proposal rejected routes back to observe without executing
    approvals = temp_ledger.list_approvals(run_id="run-b5-001")
    assert len(approvals) >= 1
    assert approvals[0].status == ApprovalStatus.REJECTED


# =====================================================================
# 5. Crash Recovery & Non-Duplication Tests
# =====================================================================

def test_langgraph_crash_recovery_does_not_repeat_command(temp_ledger: SqliteAgentLedger, mock_workspace, tmp_path: Path):
    """Verify criterion: Caída después de efecto no repite Command."""
    db_file = tmp_path / "durable_ledger.db"
    ledger = SqliteAgentLedger(db_file)

    session = AgentSessionState(
        session_id="sess-crash-recovery",
        run_id="run-b5-001",
        goal="Test crash recovery",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=1,
        budget=AgentBudget(max_experiments=5, max_trials=10).to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    ledger.save_session_state(session)

    # Set up mock executor tracking execution count
    mock_executor = MagicMock()
    mock_executor.execute.return_value = ToolResult(
        request_id="req-exec-001",
        tool_name="create_experiment",
        success=True,
        operation_id="op-exec-001",
        data={"experiment_id": "exp-001", "score": 0.85},
    )

    # Pre-record that this exact proposal action was ALREADY executed and SUCCEEDED
    # (simulating a crash right after execution succeeded, before critique)
    sig = AgentStateMachine.compute_hypothesis_signature(
        "create_experiment",
        {"model_id": "baseline_ridge", "hyperparameters": {}},
    )
    op_record = OperationRecord(
        operation_id="op-pre-executed",
        run_id="run-b5-001",
        actor="agent_orchestrator",
        idempotency_key=sig,
        action="create_experiment",
        arguments_hash=sig,
        arguments={"model_id": "baseline_ridge", "hyperparameters": {}},
        status=OperationStatus.SUCCEEDED,
        result_ref="exp-001",
    )
    ledger.record_operation(op_record)

    # Planner proposes baseline_ridge
    mock_planner = MagicMock()
    mock_planner.analyze.return_value = CandidateProposal(
        proposal_id="prop-1",
        specialist_name="planner",
        run_id="run-b5-001",
        hypothesis="Evaluate baseline ridge",
        action_type="create_experiment",
        action_payload={"model_id": "baseline_ridge", "hyperparameters": {}},
        estimated_cost={"experiments": 1, "trials": 1},
    )

    # Launch orchestrator
    checkpointer = SqliteCheckpointSaver(db_file)
    orchestrator = LangGraphAgentOrchestrator(
        workspace=mock_workspace,
        session_store=ledger,
        checkpointer=checkpointer.saver,
        approval_store=ledger,
        operation_store=ledger,
        planner=mock_planner,
        executor=mock_executor,
    )

    final_state = orchestrator.run(session)
    assert final_state["status"] == SessionStatus.COMPLETED.value

    # Crucial check: mock_executor.execute must NOT have been called because
    # the operation was already completed and idempotently reused!
    assert mock_executor.execute.call_count == 0
    assert final_state["latest_result"]["status"] == "reused"
    assert final_state["latest_result"]["result_ref"] == "exp-001"


def test_langgraph_reconciles_inflight_operations_on_resumption(temp_ledger: SqliteAgentLedger, mock_workspace):
    """Verify that restarting the orchestrator reconciles orphan operations left in RUNNING."""
    op_stale = OperationRecord(
        operation_id="op-stale-1",
        run_id="run-b5-001",
        actor="agent",
        idempotency_key="stale-key",
        action="optimize_experiment",
        arguments_hash="hash-1",
        arguments={},
        status=OperationStatus.RUNNING,
    )
    temp_ledger.record_operation(op_stale)

    session = AgentSessionState(
        session_id="sess-reconcile",
        run_id="run-b5-001",
        goal="Test reconciliation",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=1,
        budget=AgentBudget(max_experiments=5, max_trials=10).to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    temp_ledger.save_session_state(session)

    checkpointer = SqliteCheckpointSaver(":memory:")
    orchestrator = LangGraphAgentOrchestrator(
        workspace=mock_workspace,
        session_store=temp_ledger,
        checkpointer=checkpointer.saver,
        operation_store=temp_ledger,
    )

    orchestrator.run(session)

    # Verify stale operation was reconciled to RECOVERY_REQUIRED
    reconciled_op = temp_ledger.get_operation("op-stale-1")
    assert reconciled_op is not None
    assert reconciled_op.status == OperationStatus.RECOVERY_REQUIRED


# =====================================================================
# 6. CLI Parity Tests
# =====================================================================

def test_cli_session_resume_langgraph_engine(temp_ledger: SqliteAgentLedger, tmp_path: Path, mock_workspace, monkeypatch):
    db_file = tmp_path / "cli_agent_ledger.db"
    ledger = SqliteAgentLedger(db_file)
    from automl.application.agents.orchestrator.session_manager import AgentSessionManager
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=ledger,
        approval_store=ledger,
    )
    monkeypatch.setattr(
        "automl.interfaces.cli.agent_session_cli._resolve_workspace_and_ledger",
        lambda ws: (mock_workspace, ledger, manager),
    )

    start_args = argparse.Namespace(
        run_id="run-b5-001",
        goal="Test CLI LangGraph",
        max_iterations=2,
        target_score=None,
        patience=3,
        workspace=str(db_file),
        json=True,
    )
    assert session_start_cli(start_args) == 0

    # Read created session ID
    created_session = ledger.list_sessions(run_id="run-b5-001")[0]
    session_id = created_session.session_id

    # Resume via LangGraph engine
    resume_args = argparse.Namespace(
        session_id=session_id,
        max_steps=None,
        engine="langgraph",
        workspace=str(db_file),
        json=False,
    )
    assert session_resume_cli(resume_args) == 0

    # Status check via LangGraph engine
    status_args = argparse.Namespace(
        session_id=session_id,
        engine="langgraph",
        workspace=str(db_file),
        json=True,
    )
    assert session_status_cli(status_args) == 0


def test_is_langgraph_available():
    assert is_langgraph_available() is True
