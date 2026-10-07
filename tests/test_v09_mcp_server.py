"""Unit and integration tests for CATML Model Context Protocol (MCP) server.

Persona B Package B1 verification suite.
Tests cover server lifecycle, tool listing, query execution, resource reading,
and missing dependency handling.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import shutil
import tempfile
from unittest.mock import patch

import pytest

from pathlib import Path

from automl import __version__
from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    PlanExperimentsCommand,
)
from automl.application.agents.contracts import OperationRecord
from automl.domain.agents.entities import ApprovalStatus, OperationStatus
from automl.interfaces.cli.mcp_cli import run_mcp_cli
from automl.interfaces.mcp.server import HAS_MCP, create_mcp_server


@pytest.fixture
def sample_dataset_path():
    return Path(__file__).resolve().parents[1] / "examples" / "data" / "customers_churn.csv"


@pytest.fixture
def temp_workspace(tmp_path: Path, sample_dataset_path: Path):
    """Create a temporary workspace with registered dataset and run for MCP tests."""
    ws, cb, qb = build_application(root_dir=str(tmp_path / "ws"))
    dataset = ws.register_dataset(
        name="customers",
        path=sample_dataset_path,
        target="churn",
        task_type="binary_classification",
    )
    run = ws.create_run(dataset, metric="roc_auc")
    return ws, cb, qb, dataset, run


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_server_initialization_and_tool_listing(temp_workspace):
    """Verify MCP server exposes all canonical H1 query tools with valid schemas."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}

    expected_tools = {
        "get_dataset_profile",
        "list_models",
        "list_plugins",
        "list_experiments",
        "get_leaderboard",
        "get_feature_evidence",
        "get_feature_ranking",
        "create_experiment",
        "prioritize_feature",
        "run_experiment",
        "get_operation_status",
        "list_operations",
        "cancel_operation",
    }
    assert expected_tools.issubset(tool_names)
    assert len(expected_tools) == 13

    # Verify input schemas
    for t in tools:
        if t.name in expected_tools:
            assert t.description is not None
            assert t.input_schema is not None
            assert t.input_schema["type"] == "object"


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_tool_list_models(temp_workspace):
    """Verify list_models tool returns registered models through MCP."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    res = asyncio.run(server.call_tool("list_models", {}))
    assert res.is_error is False
    assert len(res.content) == 1

    models = json.loads(res.content[0].text)
    assert isinstance(models, list)
    model_ids = [m["id"] if isinstance(m, dict) else m for m in models]
    assert "logistic_regression" in model_ids
    assert "random_forest" in model_ids


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_tool_list_plugins(temp_workspace):
    """Verify list_plugins tool returns registered plugins through MCP."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    res = asyncio.run(server.call_tool("list_plugins", {}))
    assert res.is_error is False

    plugins = json.loads(res.content[0].text)
    assert isinstance(plugins, list)
    assert len(plugins) > 0


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_tool_get_dataset_profile(temp_workspace):
    """Verify get_dataset_profile returns profile or NOT_FOUND error."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    # Nonexistent dataset
    res_err = asyncio.run(server.call_tool("get_dataset_profile", {"dataset_id": "nonexistent_ds"}))
    assert res_err.is_error is True
    assert "NOT_FOUND" in res_err.content[0].text

    # Existing dataset query
    res_ok = asyncio.run(server.call_tool("get_dataset_profile", {"dataset_id": dataset.id}))
    assert res_ok.is_error is False
    profile_data = json.loads(res_ok.content[0].text)
    assert "n_rows" in profile_data or "shape" in profile_data or "columns" in profile_data


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_tool_experiments_and_leaderboard(temp_workspace):
    """Verify list_experiments and get_leaderboard tools."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    # Create experiment in run
    cb.dispatch(CreateExperimentCommand(run.id, "exp_1", model_ids=["logistic_regression"]))

    # Test list_experiments
    res_exps = asyncio.run(server.call_tool("list_experiments", {"run_id": run.id}))
    assert res_exps.is_error is False
    exps = json.loads(res_exps.content[0].text)
    assert len(exps) == 1

    # Test get_leaderboard
    res_lb = asyncio.run(server.call_tool("get_leaderboard", {"run_id": run.id, "top_k": 5}))
    assert res_lb.is_error is False
    lb = json.loads(res_lb.content[0].text)
    assert isinstance(lb, list)

    # Test leaderboard on unknown run
    res_unknown = asyncio.run(server.call_tool("get_leaderboard", {"run_id": "unknown_run_xyz"}))
    assert res_unknown.is_error is True
    assert "NOT_FOUND" in res_unknown.content[0].text


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_tool_feature_evidence_and_ranking(temp_workspace):
    """Verify feature evidence and ranking queries."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    # Feature ranking on empty run returns empty list
    res_rank = asyncio.run(server.call_tool("get_feature_ranking", {"run_id": run.id, "method": "ensemble"}))
    assert res_rank.is_error is False
    ranking = json.loads(res_rank.content[0].text)
    assert isinstance(ranking, list)

    # Feature evidence on nonexistent feature returns NOT_FOUND
    res_ev = asyncio.run(server.call_tool("get_feature_evidence", {"run_id": run.id, "feature_id": "feat_xyz"}))
    assert res_ev.is_error is True
    assert "NOT_FOUND" in res_ev.content[0].text


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_resources(temp_workspace):
    """Verify MCP resource endpoints for live leaderboards and dataset profiles."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    # Read leaderboard resource
    lb_res = asyncio.run(server.read_resource(f"catml://runs/{run.id}/leaderboard"))
    assert len(lb_res) == 1
    lb_data = json.loads(lb_res[0].content)
    assert isinstance(lb_data, list)

    # Read dataset profile resource
    ds_res = asyncio.run(server.read_resource(f"catml://datasets/{dataset.id}/profile"))
    assert len(ds_res) == 1
    ds_data = json.loads(ds_res[0].content)
    assert "n_rows" in ds_data or "columns" in ds_data


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_mutating_tools_and_governance(temp_workspace):
    """Verify MCP mutating tools require approval, don't block terminal, and execute once authorized."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    # 1. Propose candidate without approval -> returns PENDING_APPROVAL
    res_prop = asyncio.run(
        server.call_tool(
            "create_experiment",
            {"run_id": run.id, "model_name": "logistic_regression"},
        )
    )
    assert res_prop.is_error is False
    data_prop = json.loads(res_prop.content[0].text)
    assert data_prop["status"] == "PENDING_APPROVAL"
    assert "approval_id" in data_prop
    appr_id = data_prop["approval_id"]

    # 2. Approve via server's ledger
    server.ledger.update_approval_status(appr_id, status=ApprovalStatus.APPROVED, reviewer="test_reviewer")

    # 3. Call again with approval_id -> succeeds and returns experiment_id
    res_auth = asyncio.run(
        server.call_tool(
            "create_experiment",
            {"run_id": run.id, "model_name": "logistic_regression", "approval_id": appr_id},
        )
    )
    assert res_auth.is_error is False
    data_auth = json.loads(res_auth.content[0].text)
    assert "experiment_id" in data_auth
    exp_id = data_auth["experiment_id"]

    # 4. Propose run_experiment -> returns PENDING_APPROVAL
    res_run_prop = asyncio.run(
        server.call_tool(
            "run_experiment",
            {"run_id": run.id, "experiment_id": exp_id},
        )
    )
    assert res_run_prop.is_error is False
    data_run_prop = json.loads(res_run_prop.content[0].text)
    assert data_run_prop["status"] == "PENDING_APPROVAL"
    run_appr_id = data_run_prop["approval_id"]

    # 5. Approve run_experiment and execute
    server.ledger.update_approval_status(run_appr_id, status=ApprovalStatus.APPROVED, reviewer="test_reviewer")
    res_run_exec = asyncio.run(
        server.call_tool(
            "run_experiment",
            {"run_id": run.id, "experiment_id": exp_id, "approval_id": run_appr_id},
        )
    )
    assert res_run_exec.is_error is False
    data_run = json.loads(res_run_exec.content[0].text)
    assert "trial_id" in data_run
    assert "metric_value" in data_run


def test_mcp_server_missing_dependency():
    """Verify create_mcp_server raises clean ImportError when mcp is absent."""
    with patch("automl.interfaces.mcp.server.HAS_MCP", False):
        with pytest.raises(ImportError, match="pip install '\\.\\[mcp\\]'"):
            create_mcp_server()


def test_mcp_cli_missing_dependency():
    """Verify automl mcp CLI exits with code 1 and writes to stderr when mcp is absent."""
    with patch("automl.interfaces.mcp.server.HAS_MCP", False):
        stderr_buf = io.StringIO()
        with patch("sys.stderr", stderr_buf):
            ret = run_mcp_cli(argparse.Namespace(workspace=None))
            assert ret == 1
            assert "pip install '.[mcp]'" in stderr_buf.getvalue()


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_cli_subprocess_stdio_handshake(tmp_path: Path):
    """Verify real subprocess stdio lifecycle: stdout contains only pure JSON-RPC and logs go to stderr."""
    import subprocess
    import sys

    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0.0"},
        },
    }

    proc = subprocess.Popen(
        [sys.executable, "-m", "automl.interfaces.cli.main", "mcp", "--workspace", str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(input=json.dumps(req) + "\n", timeout=25)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise

    assert proc.returncode == 0
    # STDOUT must parse as valid JSON-RPC
    assert len(stdout.strip()) > 0
    resp = json.loads(stdout.strip())
    assert resp.get("jsonrpc") == "2.0"
    assert resp.get("id") == 1
    assert "result" in resp
    assert resp["result"]["serverInfo"]["name"] == "catml-mcp"
    assert resp["result"]["serverInfo"]["version"] == __version__

    # STDERR must contain the server initialization log
    assert "catml.mcp" in stderr


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_operation_tools(temp_workspace):
    """Verify get_operation_status, list_operations, and cancel_operation tools in MCP."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    # Seed an operation in the server's ledger
    op = OperationRecord(
        operation_id="op-test-mcp-1",
        run_id=run.id,
        actor="mcp_tester",
        idempotency_key="idemp_mcp_1",
        action="run_experiment",
        arguments_hash="hash_1",
        arguments={"run_id": run.id, "experiment_id": "exp_1"},
        status=OperationStatus.RUNNING,
    )
    server.ledger.record_operation(op)

    # 1. get_operation_status
    res_get = asyncio.run(server.call_tool("get_operation_status", {"operation_id": "op-test-mcp-1"}))
    assert res_get.is_error is False
    data_get = json.loads(res_get.content[0].text)
    assert data_get["operation_id"] == "op-test-mcp-1"
    assert data_get["status"] == "running"

    # Nonexistent
    res_get_missing = asyncio.run(server.call_tool("get_operation_status", {"operation_id": "op-missing"}))
    assert res_get_missing.is_error is True
    assert "NOT_FOUND" in res_get_missing.content[0].text

    # 2. list_operations
    res_list = asyncio.run(server.call_tool("list_operations", {"run_id": run.id}))
    assert res_list.is_error is False
    ops = json.loads(res_list.content[0].text)
    assert len(ops) >= 1
    assert ops[0]["operation_id"] == "op-test-mcp-1"

    # Invalid status filter
    res_list_bad = asyncio.run(server.call_tool("list_operations", {"run_id": run.id, "status": "INVALID"}))
    assert res_list_bad.is_error is True
    assert "INVALID_ARGUMENT" in res_list_bad.content[0].text

    # 3. cancel_operation: cooperative cancellation -> cancel_requested
    res_cancel = asyncio.run(
        server.call_tool("cancel_operation", {"operation_id": "op-test-mcp-1", "reason": "User stop"})
    )
    assert res_cancel.is_error is False
    cancel_data = json.loads(res_cancel.content[0].text)
    assert cancel_data["operation_id"] == "op-test-mcp-1"
    assert cancel_data["status"] == "cancel_requested"

    # Verify in ledger
    updated_op = server.ledger.get_operation("op-test-mcp-1")
    assert updated_op.status == OperationStatus.CANCEL_REQUESTED

    # 4. Force cancel -> cancelled
    res_force = asyncio.run(
        server.call_tool("cancel_operation", {"operation_id": "op-test-mcp-1", "force": True})
    )
    assert res_force.is_error is False
    assert json.loads(res_force.content[0].text)["status"] == "cancelled"

    # 5. Cannot cancel terminal operation
    res_term = asyncio.run(
        server.call_tool("cancel_operation", {"operation_id": "op-test-mcp-1"})
    )
    assert res_term.is_error is True
    assert "CONFLICT" in res_term.content[0].text

    # 6. Nonexistent operation cancellation
    res_cancel_missing = asyncio.run(
        server.call_tool("cancel_operation", {"operation_id": "nonexistent-op"})
    )
    assert res_cancel_missing.is_error is True
    assert "NOT_FOUND" in res_cancel_missing.content[0].text


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_operation_resources(temp_workspace):
    """Verify catml://runs/{run_id}/operations and catml://operations/{op_id} resources."""
    ws, cb, qb, dataset, run = temp_workspace
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb)

    op = OperationRecord(
        operation_id="op-resource-test",
        run_id=run.id,
        actor="agent",
        idempotency_key="idemp_res",
        action="create_experiment",
        arguments_hash="hash_res",
        arguments={"run_id": run.id, "model_name": "rf"},
        status=OperationStatus.SUCCEEDED,
    )
    server.ledger.record_operation(op)

    # 1. Read run operations resource
    ops_res = asyncio.run(server.read_resource(f"catml://runs/{run.id}/operations"))
    assert len(ops_res) == 1
    ops_data = json.loads(ops_res[0].content)
    assert isinstance(ops_data, list)
    assert any(o["operation_id"] == "op-resource-test" for o in ops_data)

    # 2. Read single operation resource
    op_res = asyncio.run(server.read_resource("catml://operations/op-resource-test"))
    assert len(op_res) == 1
    op_data = json.loads(op_res[0].content)
    assert op_data["operation_id"] == "op-resource-test"
    assert op_data["status"] == "succeeded"

    # Nonexistent
    missing_res = asyncio.run(server.read_resource("catml://operations/missing-op"))
    assert len(missing_res) == 1
    missing_data = json.loads(missing_res[0].content)
    assert "error" in missing_data


def test_mcp_cli_transport_args():
    """Verify run_mcp_cli passes transport configuration cleanly to run_mcp_service."""
    with patch("automl.interfaces.mcp.server.HAS_MCP", True), patch("automl.interfaces.mcp.server.run_mcp_service") as mock_service:
        args = argparse.Namespace(
            workspace="/tmp/test_ws",
            transport="streamable-http",
            host="0.0.0.0",
            port=9000,
            path="/api/mcp",
        )
        ret = run_mcp_cli(args)
        assert ret == 0
        mock_service.assert_called_once_with(
            root_dir="/tmp/test_ws",
            transport="streamable-http",
            host="0.0.0.0",
            port=9000,
            streamable_http_path="/api/mcp",
        )

