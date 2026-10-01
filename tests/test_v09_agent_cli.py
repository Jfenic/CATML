"""Tests for CATML agent CLI commands (automl agent approvals list, approve)."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import uuid

import pytest

from automl.domain.agents.entities import ApprovalStatus
from automl.application.agents.contracts import ApprovalRequest
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from automl.interfaces.cli.agent_cli import approvals_list_cli, approve_cli, register_agent_subparser


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
