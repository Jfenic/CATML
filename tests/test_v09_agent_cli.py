"""Tests for CATML agent CLI commands (automl agent approvals list, approve)."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import uuid

import pytest

from automl.domain.agents.entities import ApprovalStatus, OperationStatus
from automl.application.agents.contracts import ApprovalRequest, OperationRecord
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from automl.interfaces.cli.agent_cli import (
    approvals_list_cli,
    approve_cli,
    operations_cancel_cli,
    operations_get_cli,
    operations_list_cli,
    register_agent_subparser,
)


@pytest.fixture
def temp_ledger(tmp_path: Path) -> SqliteAgentLedger:
    ledger_db = tmp_path / "test_agent_ledger.db"
    return SqliteAgentLedger(ledger_db)


def _seed_approval(
    ledger: SqliteAgentLedger,
    approval_id: str,
    action: str = "run_experiment",
    run_id: str = "run_123",
    status: ApprovalStatus = ApprovalStatus.PENDING,
) -> ApprovalRequest:
    req = ApprovalRequest(
        approval_id=approval_id,
        action=action,
        actor="test_agent",
        run_id=run_id,
        arguments_hash="hash_abc123",
        arguments={"run_id": run_id, "experiment_id": "exp_1"},
        policy_version="1.0.0",
        max_cost={"trials": 1},
        expires_at="2099-12-31T23:59:59Z",
        status=status,
    )
    ledger.save_approval(req)
    return req


def test_agent_subparser_registration():
    """Verify that agent subparser is registered correctly with all subcommands."""
    parser = argparse.ArgumentParser(prog="automl")
    sub = parser.add_subparsers(dest="command")
    agent_parser = sub.add_parser("agent")
    register_agent_subparser(agent_parser)

    # Test approvals list args
    args1 = parser.parse_args(["agent", "approvals", "list", "--run-id", "run_1", "--status", "PENDING"])
    assert args1.command == "agent"
    assert args1.agent_subcommand == "approvals"
    assert args1.approvals_subcommand == "list"
    assert args1.run_id == "run_1"
    assert args1.status == "PENDING"

    # Test direct approve args
    args2 = parser.parse_args(["agent", "approve", "appr-123", "--reject", "--reviewer", "alice"])
    assert args2.command == "agent"
    assert args2.agent_subcommand == "approve"
    assert args2.approval_id == "appr-123"
    assert args2.reject is True
    assert args2.reviewer == "alice"

    # Test operations list args
    args3 = parser.parse_args(["agent", "operations", "list", "--run-id", "run_2", "--status", "running"])
    assert args3.command == "agent"
    assert args3.agent_subcommand == "operations"
    assert args3.operations_subcommand == "list"
    assert args3.run_id == "run_2"
    assert args3.status == "running"

    # Test operations get args
    args4 = parser.parse_args(["agent", "operations", "get", "op-123"])
    assert args4.command == "agent"
    assert args4.agent_subcommand == "operations"
    assert args4.operations_subcommand == "get"
    assert args4.operation_id == "op-123"

    # Test operations cancel args
    args5 = parser.parse_args(["agent", "operations", "cancel", "op-123", "--reason", "user request", "--force"])
    assert args5.command == "agent"
    assert args5.agent_subcommand == "operations"
    assert args5.operations_subcommand == "cancel"
    assert args5.operation_id == "op-123"
    assert args5.reason == "user request"
    assert args5.force is True

    # Test direct cancel alias args
    args6 = parser.parse_args(["agent", "cancel", "op-123", "--reason", "quick cancel"])
    assert args6.command == "agent"
    assert args6.agent_subcommand == "cancel"
    assert args6.operation_id == "op-123"
    assert args6.reason == "quick cancel"


def test_approvals_list_empty(temp_ledger, monkeypatch):
    """Verify approvals list with no records."""
    stdout_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)

    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        run_id=None,
        status="pending",
        json=False,
    )
    ret = approvals_list_cli(args)
    assert ret == 0
    assert "No approval requests found with status 'pending'." in stdout_buf.getvalue()


def test_approvals_list_populated_and_filtering(temp_ledger, monkeypatch):
    """Verify approvals list displays formatted table and respects run_id and status filters."""
    _seed_approval(temp_ledger, "appr-001", action="run_experiment", run_id="run_a", status=ApprovalStatus.PENDING)
    _seed_approval(temp_ledger, "appr-002", action="create_experiment", run_id="run_b", status=ApprovalStatus.PENDING)
    _seed_approval(temp_ledger, "appr-003", action="prioritize_feature", run_id="run_a", status=ApprovalStatus.APPROVED)

    # Filter by run_a and PENDING
    stdout_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)
    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        run_id="run_a",
        status="PENDING",
        json=False,
    )
    assert approvals_list_cli(args) == 0
    out = stdout_buf.getvalue()
    assert "appr-001" in out
    assert "appr-002" not in out
    assert "appr-003" not in out

    # Filter with status=all and JSON format
    stdout_buf_json = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf_json)
    args_json = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        run_id=None,
        status="all",
        json=True,
    )
    assert approvals_list_cli(args_json) == 0
    data = json.loads(stdout_buf_json.getvalue())
    assert len(data) == 3
    ids = {d["approval_id"] for d in data}
    assert ids == {"appr-001", "appr-002", "appr-003"}


def test_approvals_list_invalid_status(temp_ledger, monkeypatch):
    """Verify error on invalid status argument."""
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        run_id=None,
        status="INVALID_STATUS",
        json=False,
    )
    assert approvals_list_cli(args) == 1
    assert "Invalid status filter" in stderr_buf.getvalue()


def test_approve_cli_success(temp_ledger, monkeypatch):
    """Verify approving an approval request via CLI."""
    _seed_approval(temp_ledger, "appr-100", status=ApprovalStatus.PENDING)

    stdout_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)

    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        approval_id="appr-100",
        reject=False,
        reviewer="lead_engineer",
        notes="Risk assessment verified",
        json=False,
    )
    ret = approve_cli(args)
    assert ret == 0
    assert "successfully approved by 'lead_engineer (Risk assessment verified)'" in stdout_buf.getvalue()

    # Verify status in ledger
    updated = temp_ledger.get_approval("appr-100")
    assert updated is not None
    assert updated.status == ApprovalStatus.APPROVED
    assert updated.reviewer == "lead_engineer (Risk assessment verified)"


def test_approve_cli_reject_json(temp_ledger, monkeypatch):
    """Verify rejecting an approval request via CLI with JSON output."""
    _seed_approval(temp_ledger, "appr-200", status=ApprovalStatus.PENDING)

    stdout_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)

    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        approval_id="appr-200",
        reject=True,
        reviewer="auditor_bob",
        notes=None,
        json=True,
    )
    ret = approve_cli(args)
    assert ret == 0

    res = json.loads(stdout_buf.getvalue())
    assert res["approval_id"] == "appr-200"
    assert res["status"] == "rejected"
    assert res["reviewer"] == "auditor_bob"

    updated = temp_ledger.get_approval("appr-200")
    assert updated.status == ApprovalStatus.REJECTED


def test_approve_cli_nonexistent(temp_ledger, monkeypatch):
    """Verify error handling when approval_id does not exist."""
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        approval_id="nonexistent-id",
        reject=False,
        reviewer="user",
        notes=None,
        json=False,
    )
    ret = approve_cli(args)
    assert ret == 1
    assert "Approval request 'nonexistent-id' not found" in stderr_buf.getvalue()


def _seed_operation(
    ledger: SqliteAgentLedger,
    operation_id: str,
    action: str = "run_experiment",
    run_id: str = "run_123",
    status: OperationStatus = OperationStatus.RUNNING,
    result_ref: str | None = None,
    error_message: str | None = None,
) -> OperationRecord:
    op = OperationRecord(
        operation_id=operation_id,
        run_id=run_id,
        actor="test_agent",
        idempotency_key=f"idemp_{operation_id}",
        action=action,
        arguments_hash="hash_op_123",
        arguments={"run_id": run_id, "experiment_id": "exp_1"},
        status=status,
        reserved_budget={"trials": 1},
        consumed_budget={"trials": 1} if status == OperationStatus.SUCCEEDED else {},
        result_ref=result_ref,
        error_message=error_message,
    )
    ledger.record_operation(op)
    return op


def test_operations_list_empty(temp_ledger, monkeypatch):
    """Verify operations list with no records."""
    stdout_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)

    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        run_id=None,
        status="all",
        json=False,
    )
    ret = operations_list_cli(args)
    assert ret == 0
    assert "No operations found." in stdout_buf.getvalue()


def test_operations_list_populated_and_filtering(temp_ledger, monkeypatch):
    """Verify operations list displays formatted table and respects run_id and status filters."""
    _seed_operation(temp_ledger, "op-001", action="run_experiment", run_id="run_a", status=OperationStatus.RUNNING)
    _seed_operation(temp_ledger, "op-002", action="create_experiment", run_id="run_b", status=OperationStatus.SUCCEEDED)
    _seed_operation(temp_ledger, "op-003", action="run_experiment", run_id="run_a", status=OperationStatus.CANCELLED)

    # Filter by run_a and running
    stdout_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)
    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        run_id="run_a",
        status="running",
        json=False,
    )
    assert operations_list_cli(args) == 0
    out = stdout_buf.getvalue()
    assert "op-001" in out
    assert "op-002" not in out
    assert "op-003" not in out

    # Filter with status=all and JSON format
    stdout_buf_json = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf_json)
    args_json = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        run_id=None,
        status="all",
        json=True,
    )
    assert operations_list_cli(args_json) == 0
    data = json.loads(stdout_buf_json.getvalue())
    assert len(data) == 3
    ids = {d["operation_id"] for d in data}
    assert ids == {"op-001", "op-002", "op-003"}


def test_operations_list_invalid_status(temp_ledger, monkeypatch):
    """Verify error on invalid status filter argument."""
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        run_id=None,
        status="INVALID_STATUS",
        json=False,
    )
    assert operations_list_cli(args) == 1
    assert "Invalid status filter" in stderr_buf.getvalue()


def test_operations_get_success_and_json(temp_ledger, monkeypatch):
    """Verify inspecting operation details via text output and JSON."""
    _seed_operation(
        temp_ledger,
        "op-get-100",
        action="run_experiment",
        run_id="run_x",
        status=OperationStatus.SUCCEEDED,
        result_ref=json.dumps({"trial_id": "trial_01", "score": 0.95}),
    )

    # Text output
    stdout_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)
    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        operation_id="op-get-100",
        json=False,
    )
    assert operations_get_cli(args) == 0
    out = stdout_buf.getvalue()
    assert "Operation ID:     op-get-100" in out
    assert "Action:           run_experiment" in out
    assert "Status:           succeeded" in out

    # JSON output
    stdout_buf_json = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf_json)
    args_json = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        operation_id="op-get-100",
        json=True,
    )
    assert operations_get_cli(args_json) == 0
    data = json.loads(stdout_buf_json.getvalue())
    assert data["operation_id"] == "op-get-100"
    assert data["status"] == "succeeded"


def test_operations_get_nonexistent(temp_ledger, monkeypatch):
    """Verify error message when operation_id does not exist."""
    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    args = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        operation_id="nonexistent-op",
        json=False,
    )
    assert operations_get_cli(args) == 1
    assert "Operation 'nonexistent-op' not found" in stderr_buf.getvalue()


def test_operations_cancel_running_and_force(temp_ledger, monkeypatch):
    """Verify cancelling running operation to cancel_requested, and cancelling with --force to cancelled."""
    _seed_operation(temp_ledger, "op-cancel-1", status=OperationStatus.RUNNING)
    _seed_operation(temp_ledger, "op-cancel-2", status=OperationStatus.RUNNING)

    # 1. Cooperative cancel -> cancel_requested
    stdout_buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf)
    args1 = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        operation_id="op-cancel-1",
        reason="Budget threshold reached",
        force=False,
        json=False,
    )
    assert operations_cancel_cli(args1) == 0
    assert "status updated to 'cancel_requested'" in stdout_buf.getvalue()

    op1 = temp_ledger.get_operation("op-cancel-1")
    assert op1.status == OperationStatus.CANCEL_REQUESTED
    assert "Budget threshold reached" in op1.error_message

    # 2. Force cancel -> cancelled with JSON output
    stdout_buf_json = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout_buf_json)
    args2 = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        operation_id="op-cancel-2",
        reason="Admin emergency stop",
        force=True,
        json=True,
    )
    assert operations_cancel_cli(args2) == 0
    res = json.loads(stdout_buf_json.getvalue())
    assert res["operation_id"] == "op-cancel-2"
    assert res["status"] == "cancelled"

    op2 = temp_ledger.get_operation("op-cancel-2")
    assert op2.status == OperationStatus.CANCELLED


def test_operations_cancel_terminal_and_nonexistent(temp_ledger, monkeypatch):
    """Verify cancelling terminal operations (succeeded, failed, etc.) and nonexistent operation errors cleanly."""
    _seed_operation(temp_ledger, "op-term-1", status=OperationStatus.SUCCEEDED)

    stderr_buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf)

    # Cannot cancel succeeded operation
    args_term = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        operation_id="op-term-1",
        reason=None,
        force=False,
        json=False,
    )
    assert operations_cancel_cli(args_term) == 1
    assert "Cannot cancel operation 'op-term-1' with terminal status 'succeeded'" in stderr_buf.getvalue()

    # Nonexistent
    stderr_buf2 = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr_buf2)
    args_nonexistent = argparse.Namespace(
        workspace=str(temp_ledger.db_path),
        operation_id="op-missing",
        reason=None,
        force=False,
        json=False,
    )
    assert operations_cancel_cli(args_nonexistent) == 1
    assert "Operation 'op-missing' not found" in stderr_buf2.getvalue()

