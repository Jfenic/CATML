"""Unit and integration tests for Milestone H4 orchestrator, state machine, and session management."""
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
    PolicyDecisionType,
)
from automl.application.agents.contracts import (
    AgentSessionState,
    CandidateProposal,
    EvaluationFeedback,
    PolicyDecision,
    SessionStepResult,
    ToolCallContext,
    ToolResult,
)
from automl.application.agents.orchestrator.loop import AgentLoop
from automl.application.agents.orchestrator.session_manager import AgentSessionManager
from automl.application.agents.orchestrator.state_machine import (
    AgentStateMachine,
    CycleState,
    SessionStatus,
    StopReason,
)
from automl.application.agents.specialists.context_builder import ContextBuilder
from automl.application.agents.specialists.critic import Critic
from automl.application.agents.specialists.planner import Planner
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from automl.infrastructure.llm.fake_provider import FakeLLMProvider
from automl.interfaces.cli.agent_session_cli import (
    session_resume_cli,
    session_start_cli,
    session_status_cli,
    session_step_cli,
    session_stop_cli,
)


@pytest.fixture
def temp_ledger(tmp_path: Path) -> SqliteAgentLedger:
    db_file = tmp_path / "test_agent_ledger.db"
    return SqliteAgentLedger(db_file)


@pytest.fixture
def mock_workspace():
    ws = MagicMock()
    run = MagicMock()
    run.id = "run-orch-001"
    run.dataset_id = "ds-orch-001"
    ws.get_run.return_value = run

    dataset = MagicMock()
    dataset.id = "ds-orch-001"
    dataset.name = "credit_data"
    dataset.target_column = "default"
    ws.get_dataset.return_value = dataset

    profile = MagicMock()
    profile.n_rows = 1000
    profile.n_columns = 8
    profile.numeric_columns = ["income", "age", "loan_amount"]
    profile.categorical_columns = ["employment_type"]
    profile.task_type = "binary_classification"
    ws.get_profile.return_value = profile

    ws.get_leaderboard.return_value = []
    ws.list_feature_sets.return_value = []
    return ws


# =====================================================================
# 1. State Machine Tests
# =====================================================================

def test_state_machine_valid_transitions():
    sm = AgentStateMachine()
    assert sm.is_valid_transition(CycleState.INIT, CycleState.OBSERVE)
    assert sm.is_valid_transition(CycleState.OBSERVE, CycleState.PROPOSE)
    assert sm.is_valid_transition(CycleState.PROPOSE, CycleState.GATE)
    assert sm.is_valid_transition(CycleState.GATE, CycleState.EXECUTE)
    assert sm.is_valid_transition(CycleState.GATE, CycleState.WAITING_APPROVAL)
    assert sm.is_valid_transition(CycleState.EXECUTE, CycleState.CRITIQUE)
    assert sm.is_valid_transition(CycleState.CRITIQUE, CycleState.CHECK_STOP)
    assert sm.is_valid_transition(CycleState.CHECK_STOP, CycleState.OBSERVE)
    assert sm.is_valid_transition(CycleState.CHECK_STOP, CycleState.COMPLETED)

    # Invalid transitions
    assert not sm.is_valid_transition(CycleState.INIT, CycleState.EXECUTE)
    assert not sm.is_valid_transition(CycleState.COMPLETED, CycleState.OBSERVE)
    assert not sm.is_valid_transition(CycleState.STOPPED, CycleState.PROPOSE)


def test_state_machine_duplicate_hypothesis_fingerprint():
    sm = AgentStateMachine()
    sig1 = sm.compute_hypothesis_signature("create_experiment", {"model_id": "lightgbm", "hyperparameters": {}})
    sig2 = sm.compute_hypothesis_signature("create_experiment", {"hyperparameters": {}, "model_id": "lightgbm"})
    sig3 = sm.compute_hypothesis_signature("create_experiment", {"model_id": "random_forest", "hyperparameters": {}})

    assert sig1 == sig2
    assert sig1 != sig3
    assert sm.is_hypothesis_duplicate(sig1, [sig2])
    assert not sm.is_hypothesis_duplicate(sig3, [sig1])


def test_state_machine_stop_conditions():
    sm = AgentStateMachine(patience=3)

    # Budget
    budget = AgentBudget(max_experiments=2, consumed_experiments=2, max_trials=5, consumed_trials=5)
    assert sm.check_budget_exhausted(budget)

    budget_ok = AgentBudget(max_experiments=5, consumed_experiments=1)
    assert not sm.check_budget_exhausted(budget_ok, estimated_cost={"experiments": 1})
    assert sm.check_budget_exhausted(budget_ok, estimated_cost={"experiments": 10})

    # Target score
    assert sm.check_target_reached(0.92, 0.90, "maximize")
    assert not sm.check_target_reached(0.85, 0.90, "maximize")
    assert sm.check_target_reached(0.12, 0.15, "minimize")
    assert not sm.check_target_reached(0.20, 0.15, "minimize")

    # Patience
    assert not sm.check_patience_exhausted(2)
    assert sm.check_patience_exhausted(3)


# =====================================================================
# 2. Session Manager Lifecycle & Step Tests
# =====================================================================

def test_session_manager_create_and_get(mock_workspace, temp_ledger):
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
    )
    session = manager.create_session(
        run_id="run-orch-001",
        goal="Maximize ROC-AUC on credit data",
        max_iterations=5,
        target_score=0.90,
    )

    assert session.session_id.startswith("sess-")
    assert session.run_id == "run-orch-001"
    assert session.status == SessionStatus.ACTIVE.value
    assert session.iteration_count == 0
    assert session.max_iterations == 5

    loaded = manager.get_session(session.session_id)
    assert loaded is not None
    assert loaded.session_id == session.session_id
    assert loaded.goal == session.goal


def test_session_manager_deterministic_step(mock_workspace, temp_ledger):
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=3)

    # Step 1: Baseline proposal (logistic_regression)
    res1 = manager.step(session.session_id)
    assert res1.step_number == 1
    assert res1.state == SessionStatus.ACTIVE.value
    assert res1.proposal is not None
    assert res1.proposal.action_payload["model_id"] == "logistic_regression"
    assert res1.feedback is not None

    # Step 2: Next family (random_forest)
    mock_workspace.get_leaderboard.return_value = [
        {"model_id": "logistic_regression", "score": 0.80, "metric": "roc_auc", "experiment_id": "exp-1"}
    ]
    res2 = manager.step(session.session_id)
    assert res2.step_number == 2
    assert res2.proposal.action_payload["model_id"] == "random_forest"

    # Step 3: Next family (lightgbm)
    mock_workspace.get_leaderboard.return_value = [
        {"model_id": "random_forest", "score": 0.85, "metric": "roc_auc", "experiment_id": "exp-2"},
        {"model_id": "logistic_regression", "score": 0.80, "metric": "roc_auc", "experiment_id": "exp-1"},
    ]
    res3 = manager.step(session.session_id)
    assert res3.step_number == 3
    assert res3.proposal.action_payload["model_id"] == "lightgbm"
    # Max iterations reached
    assert res3.state == SessionStatus.COMPLETED.value
    assert res3.stop_reason == StopReason.MAX_ITERATIONS_REACHED.value


def test_session_manager_stop_budget_exhausted(mock_workspace, temp_ledger):
    # Tight budget: 1 experiment
    tight_budget = AgentBudget(max_experiments=1, max_trials=1)
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
    )
    session = manager.create_session(
        run_id="run-orch-001", max_iterations=5, budget=tight_budget
    )

    res1 = manager.step(session.session_id)
    assert res1.step_number == 1
    # Next step: budget exhausted
    res2 = manager.step(session.session_id)
    assert res2.state == SessionStatus.STOPPED.value
    assert res2.stop_reason == StopReason.BUDGET_EXHAUSTED.value


def test_session_manager_stop_target_reached(mock_workspace, temp_ledger):
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
    )
    session = manager.create_session(
        run_id="run-orch-001", max_iterations=10, target_score=0.88
    )

    # First step establishes baseline with score 0.90 (exceeding target 0.88)
    mock_workspace.get_leaderboard.return_value = [
        {"model_id": "logistic_regression", "score": 0.91, "metric": "roc_auc", "experiment_id": "exp-1"}
    ]
    res = manager.step(session.session_id)
    assert res.state == SessionStatus.COMPLETED.value
    assert res.stop_reason == StopReason.TARGET_REACHED.value


def test_session_manager_stop_repeated_hypothesis(mock_workspace, temp_ledger):
    planner = MagicMock()
    # Planner stubbornly returns the exact same proposal every time
    planner.analyze.return_value = CandidateProposal(
        proposal_id="prop-static-1",
        specialist_name="planner",
        run_id="run-orch-001",
        hypothesis="Try static decision tree",
        action_type="create_experiment",
        action_payload={"model_id": "decision_tree"},
    )
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
        planner=planner,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=5)

    res1 = manager.step(session.session_id)
    assert res1.state == SessionStatus.ACTIVE.value

    # Second step encounters duplicate proposal
    res2 = manager.step(session.session_id)
    assert res2.state == SessionStatus.STOPPED.value
    assert res2.stop_reason == StopReason.REPEATED_HYPOTHESIS.value


def test_session_manager_stop_no_improvement_patience(mock_workspace, temp_ledger):
    planner = MagicMock()
    planner.analyze.side_effect = [
        CandidateProposal(
            proposal_id="p1",
            specialist_name="planner",
            run_id="run-orch-001",
            hypothesis="Try model 1",
            action_type="create_experiment",
            action_payload={"model_id": "model_1"},
        ),
        CandidateProposal(
            proposal_id="p2",
            specialist_name="planner",
            run_id="run-orch-001",
            hypothesis="Try model 2",
            action_type="create_experiment",
            action_payload={"model_id": "model_2"},
        ),
    ]
    critic = MagicMock()
    # Critic always returns degradation (is_improvement=False)
    critic.analyze.return_value = EvaluationFeedback(
        feedback_id="feed-no-imp",
        run_id="run-orch-001",
        specialist_name="critic",
        is_improvement=False,
        metric_name="roc_auc",
        current_score=0.70,
        baseline_score=0.80,
        recommendation="reject",
    )
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
        planner=planner,
        critic=critic,
        patience=2,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=10, patience=2)

    res1 = manager.step(session.session_id)
    assert res1.state == SessionStatus.ACTIVE.value

    # Second consecutive non-improvement triggers patience exhaustion
    res2 = manager.step(session.session_id)
    assert res2.state == SessionStatus.STOPPED.value
    assert res2.stop_reason == StopReason.NO_IMPROVEMENT.value


def test_session_manager_approval_workflow(mock_workspace, temp_ledger):
    # Enforce PROPOSE_ONLY permission to mandate human approvals
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
        approval_store=temp_ledger,
        permission=AgentPermission.PROPOSE_ONLY,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=5)

    # Step triggers approval requirement
    res1 = manager.step(session.session_id)
    assert res1.state == SessionStatus.WAITING_APPROVAL.value
    current_session = manager.get_session(session.session_id)
    assert current_session.status == SessionStatus.WAITING_APPROVAL.value
    appr_id = current_session.pending_approval_id
    assert appr_id is not None

    # Stepping again while pending remains in waiting_approval
    res_waiting = manager.step(session.session_id)
    assert res_waiting.state == SessionStatus.WAITING_APPROVAL.value

    # User approves request
    temp_ledger.update_approval_status(appr_id, ApprovalStatus.APPROVED, reviewer="reviewer_bob")

    # Stepping resumes execution of approved action
    res_approved = manager.step(session.session_id)
    assert res_approved.state == SessionStatus.ACTIVE.value
    assert manager.get_session(session.session_id).pending_approval_id is None


def test_session_manager_resume_loop(mock_workspace, temp_ledger):
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=3)

    results = manager.resume(session.session_id, max_steps=10)
    assert len(results) == 3
    assert results[-1].state == SessionStatus.COMPLETED.value
    assert results[-1].stop_reason == StopReason.MAX_ITERATIONS_REACHED.value


def test_session_manager_pause_and_stop(mock_workspace, temp_ledger):
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=5)

    paused = manager.pause_session(session.session_id)
    assert paused.status == SessionStatus.PAUSED.value

    # Step while paused returns paused state
    res = manager.step(session.session_id)
    assert res.state == SessionStatus.PAUSED.value

    stopped = manager.stop_session(session.session_id, reason="user_aborted")
    assert stopped.status == SessionStatus.STOPPED.value
    assert stopped.stop_reason == "user_aborted"


# =====================================================================
# 3. AgentLoop Tests with Lifecycle Hooks
# =====================================================================

def test_agent_loop_hooks(mock_workspace, temp_ledger):
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=2)

    step_events = []
    stop_events = []

    loop = AgentLoop(
        session_manager=manager,
        on_step=lambda res: step_events.append(res.step_number),
        on_stop=lambda reason, s: stop_events.append((reason, s.session_id)),
    )

    results = loop.run(session.session_id)
    assert len(results) == 2
    assert step_events == [1, 2]
    assert len(stop_events) == 1
    assert stop_events[0][0] == StopReason.MAX_ITERATIONS_REACHED.value


# =====================================================================
# 4. Orchestration with FakeLLMProvider
# =====================================================================

def test_orchestrator_with_fake_llm_provider(mock_workspace, temp_ledger):
    provider = FakeLLMProvider()
    provider.register_response(
        r".*",
        {
            "hypothesis": "Test gradient boosting via FakeLLMProvider",
            "action_type": "create_experiment",
            "model_id": "lightgbm",
            "hyperparameters": {"learning_rate": 0.03},
            "estimated_trials": 1,
        },
    )
    planner = Planner(llm_provider=provider)
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
        planner=planner,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=1)

    result = manager.step(session.session_id)
    assert result.proposal.action_payload["model_id"] == "lightgbm"
    assert "FakeLLMProvider" in result.proposal.hypothesis
    assert provider.total_tokens_consumed > 0


# =====================================================================
# 5. CLI Session Commands Tests
# =====================================================================

def test_cli_session_start_step_status_stop(tmp_path: Path, monkeypatch, capsys):
    workspace_dir = tmp_path / "cli_ws"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = workspace_dir / "agent_ledger.db"

    # 1. Start CLI
    args_start = argparse.Namespace(
        run_id="run-cli-001",
        goal="CLI Autonomous session",
        max_iterations=3,
        target_score=None,
        patience=3,
        workspace=str(ledger_path),
        json=True,
    )
    code_start = session_start_cli(args_start)
    assert code_start == 0
    out_start = capsys.readouterr().out
    session_data = json.loads(out_start)
    session_id = session_data["session_id"]

    # 2. Status CLI
    args_status = argparse.Namespace(
        session_id=session_id,
        workspace=str(ledger_path),
        json=True,
    )
    code_status = session_status_cli(args_status)
    assert code_status == 0
    out_status = capsys.readouterr().out
    status_data = json.loads(out_status)
    assert status_data["session_id"] == session_id
    assert status_data["status"] == "active"

    # 3. Stop CLI
    args_stop = argparse.Namespace(
        session_id=session_id,
        reason="manual_cancel",
        workspace=str(ledger_path),
        json=True,
    )
    code_stop = session_stop_cli(args_stop)
    assert code_stop == 0
    out_stop = capsys.readouterr().out
    stop_data = json.loads(out_stop)
    assert stop_data["status"] == "stopped"
    assert stop_data["stop_reason"] == "manual_cancel"


def test_session_manager_rejected_approval_recovery(mock_workspace, temp_ledger):
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
        approval_store=temp_ledger,
        permission=AgentPermission.PROPOSE_ONLY,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=5)

    # Step 1: waiting approval
    res1 = manager.step(session.session_id)
    assert res1.state == SessionStatus.WAITING_APPROVAL.value
    appr_id = manager.get_session(session.session_id).pending_approval_id

    # User rejects proposal
    temp_ledger.update_approval_status(appr_id, ApprovalStatus.REJECTED, reviewer="admin")

    # Step 2: session detects rejection, clears pending approval, and advances to next proposal
    # (mock leaderboard to simulate different model proposed next)
    mock_workspace.get_leaderboard.return_value = [
        {"model_id": "logistic_regression", "score": 0.79, "metric": "roc_auc", "experiment_id": "exp-1"}
    ]
    res2 = manager.step(session.session_id)
    assert res2.state == SessionStatus.WAITING_APPROVAL.value
    assert manager.get_session(session.session_id).pending_approval_id != appr_id


def test_session_manager_unknown_session_error(mock_workspace, temp_ledger):
    manager = AgentSessionManager(workspace=mock_workspace, session_store=temp_ledger)
    with pytest.raises(KeyError, match="not found"):
        manager.step("non-existent-session-id")


def test_session_manager_list_sessions(mock_workspace, temp_ledger):
    manager = AgentSessionManager(workspace=mock_workspace, session_store=temp_ledger)
    sess1 = manager.create_session(run_id="run-001")
    sess2 = manager.create_session(run_id="run-002")

    listed = manager.list_sessions(sess1.session_id)
    assert len(listed) >= 1
    assert listed[0].session_id == sess1.session_id


def test_cli_session_step_and_resume(monkeypatch, capsys):
    mock_manager = MagicMock()
    step_result = SessionStepResult(
        session_id="sess-test",
        step_number=1,
        state="active",
        action_taken="create_experiment",
        proposal=CandidateProposal(
            proposal_id="p1",
            specialist_name="planner",
            run_id="run-1",
            hypothesis="Baseline proposal",
            action_type="create_experiment",
            action_payload={"model_id": "logistic_regression"},
        ),
        feedback=EvaluationFeedback(
            feedback_id="f1",
            run_id="run-1",
            specialist_name="critic",
            is_improvement=True,
            metric_name="roc_auc",
            current_score=0.82,
            variance_observation="Low spread",
        ),
    )
    mock_manager.step.return_value = step_result
    mock_manager.resume.return_value = [step_result]
    monkeypatch.setattr(
        "automl.interfaces.cli.agent_session_cli._resolve_workspace_and_ledger",
        lambda ws: (MagicMock(), MagicMock(), mock_manager),
    )

    # 1. Step CLI (JSON)
    args_step_json = argparse.Namespace(session_id="sess-test", workspace=None, json=True)
    assert session_step_cli(args_step_json) == 0
    out_step_json = capsys.readouterr().out
    assert json.loads(out_step_json)["step_number"] == 1

    # 2. Step CLI (Text)
    args_step_text = argparse.Namespace(session_id="sess-test", workspace=None, json=False)
    assert session_step_cli(args_step_text) == 0
    out_step_text = capsys.readouterr().out
    assert "Session Step 1" in out_step_text

    # 3. Resume CLI (JSON)
    args_resume_json = argparse.Namespace(session_id="sess-test", max_steps=5, workspace=None, json=True)
    assert session_resume_cli(args_resume_json) == 0
    out_resume_json = capsys.readouterr().out
    assert len(json.loads(out_resume_json)) == 1

    # 4. Resume CLI (Text)
    args_resume_text = argparse.Namespace(session_id="sess-test", max_steps=5, workspace=None, json=False)
    assert session_resume_cli(args_resume_text) == 0
    out_resume_text = capsys.readouterr().out
    assert "Resumed Session" in out_resume_text


def test_feature_advisor_fallback_on_duplicate_hypothesis(mock_workspace, temp_ledger):
    planner = MagicMock()
    # Planner proposes same static hypothesis
    planner.analyze.return_value = CandidateProposal(
        proposal_id="p-dup",
        specialist_name="planner",
        run_id="run-orch-001",
        hypothesis="Repeat model",
        action_type="create_experiment",
        action_payload={"model_id": "repeat_model"},
    )
    feature_advisor = MagicMock()
    # Advisor proposes an alternative valid feature set
    feature_advisor.analyze.return_value = CandidateProposal(
        proposal_id="p-adv",
        specialist_name="feature_advisor",
        run_id="run-orch-001",
        hypothesis="Try alternative feature differences",
        action_type="propose_feature_set",
        action_payload={"feature_set_name": "alt_features"},
    )
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
        planner=planner,
        feature_advisor=feature_advisor,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=5)

    # Step 1: uses planner proposal
    res1 = manager.step(session.session_id)
    assert res1.proposal.action_payload["model_id"] == "repeat_model"

    # Step 2: planner duplicates, falls back to feature advisor
    res2 = manager.step(session.session_id)
    assert res2.state == SessionStatus.ACTIVE.value
    assert res2.proposal.specialist_name == "feature_advisor"
    assert res2.proposal.action_payload["feature_set_name"] == "alt_features"


def test_session_manager_with_executor(mock_workspace, temp_ledger):
    executor = MagicMock()
    executor.execute.side_effect = [
        # 1. create_experiment
        ToolResult(
            request_id="r1",
            tool_name="create_experiment",
            success=True,
            data={"experiment_id": "exp-101"},
            operation_id="op-101",
        ),
        # 2. run_experiment
        ToolResult(
            request_id="r2",
            tool_name="run_experiment",
            success=True,
            data={"experiment_id": "exp-101", "score": 0.88},
            operation_id="op-102",
        ),
        # 3. optimize_experiment
        ToolResult(
            request_id="r3",
            tool_name="optimize_experiment",
            success=True,
            data={"best_score": 0.92, "best_trial_id": "tr-1"},
            operation_id="op-103",
        ),
        # 4. prioritize_feature
        ToolResult(
            request_id="r4",
            tool_name="prioritize_feature",
            success=True,
            data={"status": "prioritized"},
            operation_id="op-104",
        ),
    ]

    planner = MagicMock()
    planner.analyze.side_effect = [
        CandidateProposal(
            proposal_id="p1",
            specialist_name="planner",
            run_id="run-orch-001",
            hypothesis="Create baseline",
            action_type="create_experiment",
            action_payload={"model_id": "lgbm"},
        ),
        CandidateProposal(
            proposal_id="p2",
            specialist_name="planner",
            run_id="run-orch-001",
            hypothesis="Optimize hyperparameters",
            action_type="tune_hyperparameters",
            action_payload={"experiment_id": "exp-101", "n_trials": 5},
        ),
        CandidateProposal(
            proposal_id="p3",
            specialist_name="planner",
            run_id="run-orch-001",
            hypothesis="Prioritize features",
            action_type="prioritize_feature",
            action_payload={"feature_set_name": "interactions"},
        ),
    ]

    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
        planner=planner,
        executor=executor,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=5)

    res1 = manager.step(session.session_id)
    assert res1.operation_id == "op-101"

    res2 = manager.step(session.session_id)
    assert res2.operation_id == "op-103"

    res3 = manager.step(session.session_id)
    assert res3.operation_id == "op-104"


def test_session_manager_policy_evaluator_require_approval(mock_workspace, temp_ledger):
    evaluator = MagicMock()
    evaluator.evaluate_action.return_value = PolicyDecision(
        decision=PolicyDecisionType.REQUIRE_APPROVAL,
        reason="Action cost exceeds budget threshold",
    )
    manager = AgentSessionManager(
        workspace=mock_workspace,
        session_store=temp_ledger,
        approval_store=temp_ledger,
        policy_evaluator=evaluator,
    )
    session = manager.create_session(run_id="run-orch-001", max_iterations=5)

    res = manager.step(session.session_id)
    assert res.state == SessionStatus.WAITING_APPROVAL.value
    assert manager.get_session(session.session_id).pending_approval_id is not None


def test_session_manager_pause_and_step_terminal(mock_workspace, temp_ledger):
    manager = AgentSessionManager(workspace=mock_workspace, session_store=temp_ledger)
    session = manager.create_session(run_id="run-orch-001")

    # Stop session
    manager.stop_session(session.session_id)
    # Pause stopped session returns self
    p = manager.pause_session(session.session_id)
    assert p.status == SessionStatus.STOPPED.value

    # Stepping a stopped session returns stopped result
    res = manager.step(session.session_id)
    assert res.state == SessionStatus.STOPPED.value


def test_cli_subparser_registration_and_text_output(tmp_path: Path, capsys):
    from automl.interfaces.cli.agent_session_cli import register_agent_session_subparser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="subcommand")
    agent_p = sub.add_parser("agent")
    agent_sub = agent_p.add_subparsers(dest="agent_subcommand")

    register_agent_session_subparser(agent_p)

    # Verify session subcommands can be parsed
    parsed = parser.parse_args(["agent", "session", "start", "--run-id", "run-1", "--target-score", "0.95"])
    assert parsed.run_id == "run-1"
    assert parsed.target_score == 0.95

    # Test text output branches of CLI
    workspace_dir = tmp_path / "cli_text_ws"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = workspace_dir / "agent_ledger.db"

    # Start text
    args_start = argparse.Namespace(
        run_id="run-cli-text",
        goal="Text output test",
        max_iterations=5,
        target_score=0.90,
        patience=3,
        workspace=str(ledger_path),
        json=False,
    )
    assert session_start_cli(args_start) == 0
    out_start = capsys.readouterr().out
    assert "Agent Session Created" in out_start
    assert "Target Score:" in out_start

    # Status text
    sess_id = [line.split()[-1] for line in out_start.splitlines() if "Session ID:" in line][0]
    args_status = argparse.Namespace(session_id=sess_id, workspace=str(ledger_path), json=False)
    assert session_status_cli(args_status) == 0
    out_status = capsys.readouterr().out
    assert "Agent Session Details" in out_status

    # Stop text
    args_stop = argparse.Namespace(session_id=sess_id, reason="finished", workspace=str(ledger_path), json=False)
    assert session_stop_cli(args_stop) == 0
    out_stop = capsys.readouterr().out
    assert "stopped (finished)" in out_stop
