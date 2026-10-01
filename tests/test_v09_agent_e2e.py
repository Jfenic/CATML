"""End-to-end integration tests for Milestone H2: V0.9 Local Agent Delivery.

Validates the full lifecycle:
1. Inspect an existing run and dataset via MCP query tools.
2. Propose a new experiment candidate via MCP (returns PENDING_APPROVAL).
3. Verify terminal does not block and candidate is not yet trained.
4. Resolve approval via CLI (automl agent approve <approval_id>).
5. Execute candidate once via authorized run_experiment.
6. Verify evaluated trial metric is consultable on MCP leaderboard.
7. Verify idempotency deduplication and rejection handling.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import tempfile

import pytest

try:
    from mcp.server.mcpserver import MCPServer
    import mcp.types as types
    HAS_MCP = True
except ImportError:
    HAS_MCP = False

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    PlanExperimentsCommand,
    RunExperimentCommand,
)
from automl.domain.agents.entities import ApprovalStatus, ToolErrorCode
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
from automl.interfaces.cli.agent_cli import approvals_list_cli, approve_cli
from automl.interfaces.mcp.server import create_mcp_server


@pytest.fixture
def sample_churn_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(
        "id,tenure,monthly_charges,churn\n"
        "1,12,65.5,0\n"
        "2,2,80.0,1\n"
        "3,48,20.0,0\n"
        "4,5,95.2,1\n"
        "5,24,45.0,0\n"
        "6,1,70.0,1\n"
        "7,36,55.0,0\n"
        "8,8,85.0,1\n"
        "9,18,40.0,0\n"
        "10,3,90.0,1\n"
    )
    return csv_path


@pytest.fixture
def e2e_environment(tmp_path: Path, sample_churn_csv: Path):
    ws_dir = tmp_path / "e2e_ws"
    ws, cb, qb = build_application(root_dir=str(ws_dir))
    dataset = ws.register_dataset(
        name="churn_data",
        path=sample_churn_csv,
        target="churn",
        task_type="binary_classification",
    )
    run = ws.create_run(dataset, metric="roc_auc")
    ledger_file = ws_dir / "agent_ledger.db"
    ledger = SqliteAgentLedger(ledger_file)
    return ws, cb, qb, dataset, run, ledger, ws_dir


@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_h2_full_agent_e2e_flow(e2e_environment):
    """Full H2 verification: MCP inspection -> Proposal -> CLI Approval -> Execution -> Leaderboard."""
    ws, cb, qb, dataset, run, ledger, ws_dir = e2e_environment
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, ledger=ledger)

    # -------------------------------------------------------------------------
    # Step 1: Inspect dataset profile and candidate models via MCP
    # -------------------------------------------------------------------------
    profile_res = asyncio.run(server.call_tool("get_dataset_profile", {"dataset_id": dataset.id}))
    assert profile_res.is_error is False
    profile_data = json.loads(profile_res.content[0].text)
    assert profile_data["row_count"] == 10
    column_names = [c["name"] for c in profile_data["columns"]]
    assert "churn" in column_names

    models_res = asyncio.run(server.call_tool("list_models", {}))
    assert models_res.is_error is False
    models_list = json.loads(models_res.content[0].text)
    assert len(models_list) > 0
    available_model_names = [m["id"] if isinstance(m, dict) else m for m in models_list]
    assert "logistic_regression" in available_model_names

    # -------------------------------------------------------------------------
    # Step 2: Propose candidate experiment via MCP create_experiment
    # -------------------------------------------------------------------------
    prop_res = asyncio.run(
        server.call_tool(
            "create_experiment",
            {
                "run_id": run.id,
                "model_name": "logistic_regression",
            },
        )
    )
    # Policy requires approval -> returns PENDING_APPROVAL without blocking terminal
    assert prop_res.is_error is False
    prop_data = json.loads(prop_res.content[0].text)
    assert prop_data["status"] == "PENDING_APPROVAL"
    assert "approval_id" in prop_data
    create_approval_id = prop_data["approval_id"]
    assert create_approval_id.startswith("appr-")

    # Verify candidate is NOT created yet in workspace (query via MCP)
    exps_res0 = asyncio.run(server.call_tool("list_experiments", {"run_id": run.id}))
    assert exps_res0.is_error is False
    assert len(json.loads(exps_res0.content[0].text)) == 0

    # -------------------------------------------------------------------------
    # Step 3: Human reviews & approves create_experiment via CLI
    # -------------------------------------------------------------------------
    cli_appr_args = argparse.Namespace(
        workspace=str(ws_dir),
        approval_id=create_approval_id,
        reject=False,
        reviewer="lead_data_scientist",
        notes="Approved candidate architecture",
        json=True,
    )
    assert approve_cli(cli_appr_args) == 0

    # Re-call create_experiment with authorized approval_id
    prop_authorized_res = asyncio.run(
        server.call_tool(
            "create_experiment",
            {
                "run_id": run.id,
                "model_name": "logistic_regression",
                "approval_id": create_approval_id,
            },
        )
    )
    assert prop_authorized_res.is_error is False
    created_data = json.loads(prop_authorized_res.content[0].text)
    assert "experiment_id" in created_data
    experiment_id = created_data["experiment_id"]
    assert experiment_id is not None

    # Verify experiment candidate exists in workspace (query via MCP), but has not yet trained
    exps_res1 = asyncio.run(server.call_tool("list_experiments", {"run_id": run.id}))
    assert exps_res1.is_error is False
    exps1 = json.loads(exps_res1.content[0].text)
    assert len(exps1) == 1
    assert exps1[0]["id"] == experiment_id

    # -------------------------------------------------------------------------
    # Step 4: Propose execution via MCP run_experiment
    # -------------------------------------------------------------------------
    run_prop_res = asyncio.run(
        server.call_tool(
            "run_experiment",
            {
                "run_id": run.id,
                "experiment_id": experiment_id,
            },
        )
    )
    assert run_prop_res.is_error is False
    run_prop_data = json.loads(run_prop_res.content[0].text)
    assert run_prop_data["status"] == "PENDING_APPROVAL"
    run_approval_id = run_prop_data["approval_id"]
    assert run_approval_id.startswith("appr-")

    # Verify no trials were trained yet (leaderboard is empty)
    lb_before_res = asyncio.run(server.call_tool("get_leaderboard", {"run_id": run.id}))
    assert lb_before_res.is_error is False
    assert len(json.loads(lb_before_res.content[0].text)) == 0

    # -------------------------------------------------------------------------
    # Step 5: Check CLI approvals list and approve execution
    # -------------------------------------------------------------------------
    cli_list_args = argparse.Namespace(
        workspace=str(ws_dir),
        run_id=run.id,
        status="pending",
        json=True,
    )
    approvals_in_ledger = ledger.list_approvals(run_id=run.id, status=ApprovalStatus.PENDING)
    assert any(a.approval_id == run_approval_id for a in approvals_in_ledger)

    cli_exec_appr_args = argparse.Namespace(
        workspace=str(ws_dir),
        approval_id=run_approval_id,
        reject=False,
        reviewer="lead_data_scientist",
        notes="Authorized 1 trial execution",
        json=True,
    )
    assert approve_cli(cli_exec_appr_args) == 0

    # -------------------------------------------------------------------------
    # Step 6: Authorized execution produces trial and metrics
    # -------------------------------------------------------------------------
    exec_res = asyncio.run(
        server.call_tool(
            "run_experiment",
            {
                "run_id": run.id,
                "experiment_id": experiment_id,
                "approval_id": run_approval_id,
                "idempotency_key": "idemp_e2e_trial_1",
            },
        )
    )
    assert exec_res.is_error is False
    exec_data = json.loads(exec_res.content[0].text)
    assert "trial_id" in exec_data
    assert "metric_value" in exec_data
    assert isinstance(exec_data["metric_value"], float)
    trial_id = exec_data["trial_id"]

    # -------------------------------------------------------------------------
    # Step 7: Verify live metrics consultable via MCP leaderboard & resource
    # -------------------------------------------------------------------------
    lb_res = asyncio.run(server.call_tool("get_leaderboard", {"run_id": run.id}))
    assert lb_res.is_error is False
    lb_data = json.loads(lb_res.content[0].text)
    assert len(lb_data) >= 1
    assert lb_data[0]["trial_id"] == trial_id

    # Also verify via MCP resource
    resource_res = asyncio.run(server.read_resource(f"catml://runs/{run.id}/leaderboard"))
    assert len(resource_res) == 1
    resource_lb = json.loads(resource_res[0].content)
    assert len(resource_lb) >= 1
    assert resource_lb[0]["trial_id"] == trial_id

    # -------------------------------------------------------------------------
    # Step 8: Verify Idempotency deduplication ("Duplicados no repiten efectos")
    # -------------------------------------------------------------------------
    exec_duplicate_res = asyncio.run(
        server.call_tool(
            "run_experiment",
            {
                "run_id": run.id,
                "experiment_id": experiment_id,
                "approval_id": run_approval_id,
                "idempotency_key": "idemp_e2e_trial_1",
            },
        )
    )
    assert exec_duplicate_res.is_error is False
    dup_data = json.loads(exec_duplicate_res.content[0].text)
    assert dup_data["trial_id"] == trial_id
    # Leaderboard still has exactly the same trials (no redundant execution)
    lb_after = json.loads(asyncio.run(server.call_tool("get_leaderboard", {"run_id": run.id})).content[0].text)
    assert len(lb_after) == len(lb_data)

    # -------------------------------------------------------------------------
    # Step 9: Verify Rejection handling
    # -------------------------------------------------------------------------
    reject_prop_res = asyncio.run(
        server.call_tool(
            "prioritize_feature",
            {
                "run_id": run.id,
                "feature_name": "tenure",
                "priority": "high",
            },
        )
    )
    assert reject_prop_res.is_error is False
    rej_approval_id = json.loads(reject_prop_res.content[0].text)["approval_id"]

    # Reject via CLI
    cli_reject_args = argparse.Namespace(
        workspace=str(ws_dir),
        approval_id=rej_approval_id,
        reject=True,
        reviewer="risk_officer",
        notes="Feature prioritization rejected",
        json=True,
    )
    assert approve_cli(cli_reject_args) == 0

    # Attempt to execute with rejected approval_id -> denied
    denied_res = asyncio.run(
        server.call_tool(
            "prioritize_feature",
            {
                "run_id": run.id,
                "feature_name": "tenure",
                "priority": "high",
                "approval_id": rej_approval_id,
            },
        )
    )
    assert denied_res.is_error is True
    assert "PERMISSION_DENIED" in denied_res.content[0].text
    assert "rejected" in denied_res.content[0].text
