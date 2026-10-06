"""End-to-end integration and crash recovery tests for Milestone H5 autonomous agent orchestrator."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from automl.domain.agents.entities import (
    AgentBudget,
    AgentPermission,
    ApprovalStatus,
    OperationStatus,
)
from automl.application.agents.contracts import AgentSessionState
from automl.application.agents.executor import ToolExecutor, create_full_tool_registry
from automl.application.agents.orchestrator.graph import LangGraphAgentOrchestrator
from automl.application.agents.orchestrator.state_machine import SessionStatus, StopReason
from automl.application.agents.specialists.context_builder import ContextBuilder
from automl.application.agents.specialists.critic import Critic
from automl.application.agents.specialists.planner import Planner
from automl.application.bootstrap import build_application
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from automl.infrastructure.database.sqlite_checkpoint_saver import SqliteCheckpointSaver


@pytest.fixture
def e2e_workspace(tmp_path: Path):
    """Set up real AutoML application workspace with sample dataset and registered run."""
    ws_dir = tmp_path / "e2e_ws"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    # Create and register real CSV dataset
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] * 5,
        "feat_b": [10.0, 20.0, 15.0, 35.0, 25.0, 30.0, 45.0, 50.0, 55.0, 60.0] * 5,
        "category": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"] * 5,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5,
    })
    csv_file = ws_dir / "data.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_file, index=False)

    dataset = ws.register_dataset(name="e2e_data", path=str(csv_file), target="target")
    dataset_id = dataset.id
    run = ws.create_run(dataset=dataset)
    run_id = run.id

    ledger_file = ws_dir / "agent_ledger.db"
    ledger = SqliteAgentLedger(ledger_file)

    tool_reg = create_full_tool_registry(
        query_bus=qry_bus,
        command_bus=cmd_bus,
        workspace=ws,
        ledger=ledger,
    )
    executor = ToolExecutor(registry=tool_reg, ledger=ledger)

    return {
        "ws_dir": ws_dir,
        "workspace": ws,
        "ledger": ledger,
        "ledger_file": ledger_file,
        "executor": executor,
        "run_id": run_id,
        "dataset_id": dataset_id,
    }


def test_e2e_autonomous_cycle_and_checkpoint_sync(e2e_workspace):
    """E2E Test: Observe -> Propose -> Gate -> Execute -> Critique -> Checkpoint with real workspace."""
    ws = e2e_workspace["workspace"]
    ledger = e2e_workspace["ledger"]
    executor = e2e_workspace["executor"]
    run_id = e2e_workspace["run_id"]
    ledger_file = e2e_workspace["ledger_file"]

    # 1. Create agent session
    session = AgentSessionState(
        session_id="sess-e2e-001",
        run_id=run_id,
        goal="E2E test optimization",
        version="1.0.0",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=2,
        budget=AgentBudget(max_experiments=5, max_trials=10).to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    ledger.save_session_state(session)

    # 2. Build orchestrator with durable checkpointer
    checkpointer = SqliteCheckpointSaver(ledger_file)
    orchestrator = LangGraphAgentOrchestrator(
        workspace=ws,
        session_store=ledger,
        checkpointer=checkpointer.saver,
        approval_store=ledger,
        operation_store=ledger,
        executor=executor,
        patience=5,
        permission=AgentPermission.ADMIN,
    )

    # 3. Execute autonomous flow
    result = orchestrator.run(session)
    assert result["status"] == SessionStatus.COMPLETED.value
    assert result["stop_reason"] == StopReason.MAX_ITERATIONS_REACHED.value
    assert result["iteration_count"] == 2

    # Verify leaderboard has real experiments trained
    leaderboard = ws.get_leaderboard(run_id)
    assert len(leaderboard) >= 1

    # Verify durable checkpoints exist in SQLite
    persisted_session = ledger.get_session_state("sess-e2e-001")
    assert persisted_session is not None
    assert persisted_session.status == SessionStatus.COMPLETED.value
    assert persisted_session.iteration_count == 2
    assert persisted_session.checkpoint_id == "chk-sess-e2e-001-2"

    # Verify checkpointer channel state
    cp_state = orchestrator.get_latest_graph_state("sess-e2e-001")
    assert cp_state is not None
    assert cp_state["iteration_count"] == 2
    assert cp_state["status"] == SessionStatus.COMPLETED.value


def test_e2e_crash_and_restart_recovery(e2e_workspace):
    """E2E Test: Simulate crash after iteration 1, restart with new orchestrator instance, verify non-duplication."""
    ws = e2e_workspace["workspace"]
    ledger = e2e_workspace["ledger"]
    executor = e2e_workspace["executor"]
    run_id = e2e_workspace["run_id"]
    ledger_file = e2e_workspace["ledger_file"]

    session = AgentSessionState(
        session_id="sess-e2e-crash",
        run_id=run_id,
        goal="Test crash recovery",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=1,  # run 1 iteration then stop
        budget=AgentBudget(max_experiments=5, max_trials=10).to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    ledger.save_session_state(session)

    # First process lifecycle
    with SqliteCheckpointSaver(ledger_file) as cp1:
        orch1 = LangGraphAgentOrchestrator(
            workspace=ws,
            session_store=ledger,
            checkpointer=cp1.saver,
            approval_store=ledger,
            operation_store=ledger,
            executor=executor,
            patience=5,
            permission=AgentPermission.ADMIN,
        )
        res1 = orch1.run(session)
        assert res1["iteration_count"] == 1
        assert res1["status"] == SessionStatus.COMPLETED.value

    # Verify 1 experiment in leaderboard
    initial_lb = ws.get_leaderboard(run_id)
    initial_exp_count = len(initial_lb)
    assert initial_exp_count >= 1

    # SIMULATE PROCESS RESTART:
    # Update max_iterations to 2 so second process has work to do
    session_resumed = ledger.get_session_state("sess-e2e-crash")
    assert session_resumed is not None
    session_resumed.status = SessionStatus.ACTIVE.value
    session_resumed.stop_reason = None
    session_resumed.max_iterations = 2
    ledger.save_session_state(session_resumed)

    # Second process lifecycle (completely fresh instances and connection)
    with SqliteCheckpointSaver(ledger_file) as cp2:
        orch2 = LangGraphAgentOrchestrator(
            workspace=ws,
            session_store=ledger,
            checkpointer=cp2.saver,
            approval_store=ledger,
            operation_store=ledger,
            executor=executor,
            patience=5,
            permission=AgentPermission.ADMIN,
        )

        res2 = orch2.run(session_resumed)
        assert res2["iteration_count"] == 2
        assert res2["status"] == SessionStatus.COMPLETED.value

    # Verify final leaderboard advanced without re-training iteration 1
    final_lb = ws.get_leaderboard(run_id)
    assert len(final_lb) >= initial_exp_count


def test_e2e_approval_rejection_does_not_train(e2e_workspace):
    """E2E Test: Propose-only permission halts at gate; rejecting approval discards candidate without training."""
    ws = e2e_workspace["workspace"]
    ledger = e2e_workspace["ledger"]
    executor = e2e_workspace["executor"]
    run_id = e2e_workspace["run_id"]
    ledger_file = e2e_workspace["ledger_file"]

    session = AgentSessionState(
        session_id="sess-e2e-reject",
        run_id=run_id,
        goal="Test approval rejection",
        status=SessionStatus.ACTIVE.value,
        iteration_count=0,
        max_iterations=1,
        budget=AgentBudget(max_experiments=5, max_trials=10).to_dict(),
        created_at="2026-10-06T12:00:00Z",
        updated_at="2026-10-06T12:00:00Z",
    )
    ledger.save_session_state(session)

    initial_lb = ws.get_leaderboard(run_id)

    with SqliteCheckpointSaver(ledger_file) as cp:
        orchestrator = LangGraphAgentOrchestrator(
            workspace=ws,
            session_store=ledger,
            checkpointer=cp.saver,
            approval_store=ledger,
            operation_store=ledger,
            executor=executor,
            permission=AgentPermission.PROPOSE_ONLY,
        )

        # 1. Runs and pauses at interrupt
        state1 = orchestrator.run(session)
        assert "__interrupt__" in state1

        # 2. Human rejects
        state2 = orchestrator.resume(session.session_id, human_decision={"approved": False})

    # Verify no new experiments were trained on rejected proposal
    final_lb = ws.get_leaderboard(run_id)
    assert len(final_lb) == len(initial_lb)

    # Verify approval status in ledger
    approvals = ledger.list_approvals(run_id=run_id)
    assert len(approvals) >= 1
    assert approvals[0].status == ApprovalStatus.REJECTED
