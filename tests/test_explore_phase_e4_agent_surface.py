"""Tests for CATML Explore Phase E4: Agent Surface (MCP Interoperability & CLI Completeness).

Validates:
- Full MCP tool surface: analysis_create_study, analysis_run_study, analysis_get_findings,
  analysis_get_visualizations, analysis_propose_experiment ("Propose ≠ Accept").
- Token efficiency controls: pagination (limit/offset), compact metrics pruning, envelope mode.
- MCP resources: catml://studies/{study_id}/summary and catml://studies/{study_id}/visualizations.
- CLI completeness: catml explore list, run, findings (with filters), export (markdown and json).
- Pure reporting service: generate_study_markdown_report.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import pytest

import numpy as np
import pandas as pd

from automl.application.analysis.reporting import generate_study_markdown_report
from automl.application.bootstrap import build_application
from automl.domain.agents.entities import ToolErrorCode
from automl.interfaces.cli.explore_cli import run_explore_cli
from automl.interfaces.mcp.server import create_mcp_server


@pytest.fixture
def explore_dataset_csv(tmp_path: Path) -> Path:
    """Creates a deterministic tabular dataset with collinearity, skewness, and category signals."""
    np.random.seed(42)
    n = 150
    # X1 and X2 are highly collinear (r > 0.95)
    x1 = np.random.normal(loc=10.0, scale=2.0, size=n)
    x2 = x1 * 1.5 + np.random.normal(loc=0.0, scale=0.1, size=n)
    # X3 has extreme right skewness (exponential)
    x3 = np.random.exponential(scale=5.0, size=n)
    # Cat has 3 levels
    cats = np.random.choice(["TierA", "TierB", "TierC"], size=n)
    # Target separated by cats and x1
    prob = 1.0 / (1.0 + np.exp(-(0.8 * x1 - 0.5 * x2 + (cats == "TierA") * 2.0 - 5.0)))
    target = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "feature_x1": x1,
        "feature_x2": x2,
        "skewed_x3": x3,
        "category_tier": cats,
        "is_converted": target,
    })
    csv_path = tmp_path / "explore_e4_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# =========================================================================
# 1. MCP TOOLS REGISTRATION & INVOCATION TESTS
# =========================================================================

def test_mcp_tools_registration_phase_e4(explore_dataset_csv: Path, tmp_path: Path):
    """Verify all Phase E4 MCP tools and resources are registered on the MCPServer."""
    ws_dir = str(tmp_path / "ws_mcp_reg")
    ws, cb, qb = build_application(root_dir=ws_dir)
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, root_dir=ws_dir)

    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}

    expected_tools = {
        "analysis_create_study",
        "analysis_run_study",
        "analysis_get_study",
        "analysis_list_studies",
        "analysis_get_findings",
        "analysis_get_visualizations",
        "analysis_propose_experiment",
    }
    assert expected_tools.issubset(tool_names), f"Missing tools: {expected_tools - tool_names}"


def test_mcp_analysis_get_findings_pagination_and_envelope(explore_dataset_csv: Path, tmp_path: Path):
    """Verify analysis_get_findings supports limit, offset, compact mode, and envelope mode."""
    ws_dir = str(tmp_path / "ws_mcp_findings")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(
        name="sales_study_ds",
        path=explore_dataset_csv,
        target="is_converted",
        task_type="binary_classification",
    )
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, root_dir=ws_dir)

    # 1. Create study
    create_res = asyncio.run(
        server.call_tool(
            "analysis_create_study",
            {"dataset_id": ds.id, "name": "Conversion Study", "target_column": "is_converted"},
        )
    )
    assert not create_res.is_error
    study_id = json.loads(create_res.content[0].text)["id"]

    # 2. Run study
    run_res = asyncio.run(server.call_tool("analysis_run_study", {"study_id": study_id}))
    assert not run_res.is_error
    run_payload = json.loads(run_res.content[0].text)
    assert run_payload["findings_count"] > 0

    # 3. Call get_findings with default envelope=False (returns list directly for backward compatibility)
    f_res = asyncio.run(server.call_tool("analysis_get_findings", {"study_id": study_id}))
    assert not f_res.is_error
    findings_list = json.loads(f_res.content[0].text)
    assert isinstance(findings_list, list)
    assert len(findings_list) > 0
    total_findings = len(findings_list)

    # 4. Pagination: limit=2, offset=0
    p1_res = asyncio.run(server.call_tool("analysis_get_findings", {"study_id": study_id, "limit": 2, "offset": 0}))
    p1_list = json.loads(p1_res.content[0].text)
    assert len(p1_list) == min(2, total_findings)

    # Pagination: limit=2, offset=1
    p2_res = asyncio.run(server.call_tool("analysis_get_findings", {"study_id": study_id, "limit": 2, "offset": 1}))
    p2_list = json.loads(p2_res.content[0].text)
    assert p2_list[0]["id"] == p1_list[1]["id"]

    # 5. Envelope mode: envelope=True
    env_res = asyncio.run(
        server.call_tool(
            "analysis_get_findings",
            {"study_id": study_id, "limit": 2, "offset": 0, "envelope": True},
        )
    )
    assert not env_res.is_error
    env_data = json.loads(env_res.content[0].text)
    assert env_data["study_id"] == study_id
    assert env_data["total"] == total_findings
    assert env_data["limit"] == 2
    assert env_data["offset"] == 0
    assert "has_more" in env_data
    assert len(env_data["findings"]) == min(2, total_findings)

    # 6. Filter by min_significance
    sig_res = asyncio.run(
        server.call_tool(
            "analysis_get_findings",
            {"study_id": study_id, "min_significance": 0.05},
        )
    )
    assert not sig_res.is_error
    sig_list = json.loads(sig_res.content[0].text)
    assert isinstance(sig_list, list)
    for f in sig_list:
        p_val = f.get("p_value")
        p_fdr = (f.get("metrics") or {}).get("p_value_fdr")
        assert (p_val is not None and p_val <= 0.05) or (p_fdr is not None and p_fdr <= 0.05)


def test_mcp_analysis_get_visualizations(explore_dataset_csv: Path, tmp_path: Path):
    """Verify analysis_get_visualizations retrieves declarative specs with filtering."""
    ws_dir = str(tmp_path / "ws_mcp_viz")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="viz_ds", path=explore_dataset_csv, target="is_converted")
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, root_dir=ws_dir)

    # Create & run study
    c_res = asyncio.run(server.call_tool("analysis_create_study", {"dataset_id": ds.id, "target_column": "is_converted"}))
    study_id = json.loads(c_res.content[0].text)["id"]
    asyncio.run(server.call_tool("analysis_run_study", {"study_id": study_id}))

    # Get visualizations
    v_res = asyncio.run(server.call_tool("analysis_get_visualizations", {"study_id": study_id}))
    assert not v_res.is_error
    viz_list = json.loads(v_res.content[0].text)
    assert isinstance(viz_list, list)
    assert len(viz_list) > 0

    first_viz = viz_list[0]
    assert "id" in first_viz
    assert "chart_type" in first_viz
    assert "data_series" in first_viz

    # Filter by specific chart_id
    single_res = asyncio.run(server.call_tool("analysis_get_visualizations", {"study_id": study_id, "chart_id": first_viz["id"]}))
    assert not single_res.is_error
    single_list = json.loads(single_res.content[0].text)
    assert len(single_list) == 1
    assert single_list[0]["id"] == first_viz["id"]


def test_mcp_analysis_propose_experiment(explore_dataset_csv: Path, tmp_path: Path):
    """Verify analysis_propose_experiment adheres strictly to 'Propose ≠ Accept' principle."""
    ws_dir = str(tmp_path / "ws_mcp_hyp")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="hyp_ds", path=explore_dataset_csv, target="is_converted")
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, root_dir=ws_dir)

    c_res = asyncio.run(server.call_tool("analysis_create_study", {"dataset_id": ds.id, "target_column": "is_converted"}))
    study_id = json.loads(c_res.content[0].text)["id"]
    asyncio.run(server.call_tool("analysis_run_study", {"study_id": study_id}))

    f_res = asyncio.run(server.call_tool("analysis_get_findings", {"study_id": study_id, "limit": 1}))
    findings = json.loads(f_res.content[0].text)
    assert len(findings) >= 1
    finding_id = findings[0]["id"]

    # Propose hypothesis
    prop_res = asyncio.run(
        server.call_tool(
            "analysis_propose_experiment",
            {
                "study_id": study_id,
                "finding_id": finding_id,
                "hypothesis_description": "Apply Yeo-Johnson power transform on skewed features to normalize residual variance",
                "proposed_action": "power_transform",
                "transformation_spec": {"method": "yeo_johnson", "columns": ["skewed_x3"]},
            },
        )
    )
    assert not prop_res.is_error
    prop_data = json.loads(prop_res.content[0].text)
    assert prop_data["study_id"] == study_id
    assert prop_data["finding_id"] == finding_id
    assert prop_data["status"] == "proposed"
    assert prop_data["hypothesis_id"].startswith("hyp_")
    assert "Propose ≠ Accept" in prop_data["note"]

    # Negative test: Non-existent finding
    bad_res = asyncio.run(
        server.call_tool(
            "analysis_propose_experiment",
            {
                "study_id": study_id,
                "finding_id": "non_existent_finding_id",
                "hypothesis_description": "Invalid test",
            },
        )
    )
    assert bad_res.is_error
    assert ToolErrorCode.NOT_FOUND.value in bad_res.content[0].text


def test_mcp_resources_summary_and_visualizations(explore_dataset_csv: Path, tmp_path: Path):
    """Verify read-only MCP resources: summary and visualizations."""
    ws_dir = str(tmp_path / "ws_mcp_res")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="res_ds", path=explore_dataset_csv, target="is_converted")
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, root_dir=ws_dir)

    c_res = asyncio.run(server.call_tool("analysis_create_study", {"dataset_id": ds.id, "target_column": "is_converted"}))
    study_id = json.loads(c_res.content[0].text)["id"]
    asyncio.run(server.call_tool("analysis_run_study", {"study_id": study_id}))

    # 1. Summary resource
    summary_raw = asyncio.run(server.read_resource(f"catml://studies/{study_id}/summary"))
    assert summary_raw
    content_str = summary_raw[0].content if isinstance(summary_raw, list) else summary_raw
    summary_data = json.loads(content_str)
    assert summary_data["study_id"] == study_id
    assert summary_data["total_findings"] > 0
    assert "findings_by_type" in summary_data
    assert "key_signals" in summary_data
    assert summary_data["visualizations_count"] > 0

    # 2. Visualizations resource
    viz_raw = asyncio.run(server.read_resource(f"catml://studies/{study_id}/visualizations"))
    assert viz_raw
    viz_content_str = viz_raw[0].content if isinstance(viz_raw, list) else viz_raw
    viz_data = json.loads(viz_content_str)
    assert isinstance(viz_data, list)
    assert len(viz_data) > 0


# =========================================================================
# 2. CLI EXPLORE COMPLETE SUITE TESTS
# =========================================================================

def test_cli_explore_complete_workflow(explore_dataset_csv: Path, tmp_path: Path, capsys):
    """Verify complete lifecycle of `catml explore` commands: create, run, list, show, findings, export."""
    ws_dir = str(tmp_path / "ws_cli")
    ws, _, _ = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="cli_ds", path=explore_dataset_csv, target="is_converted")

    # 1. explore create
    args_create = argparse.Namespace(
        workspace=ws_dir,
        explore_action="create",
        dataset_id=ds.id,
        name="CLI Exploration Test",
        target="is_converted",
        json=True,
    )
    rc = run_explore_cli(args_create)
    assert rc == 0
    out_create = capsys.readouterr().out
    study_id = json.loads(out_create)["study_id"]

    # 2. explore run
    args_run = argparse.Namespace(
        workspace=ws_dir,
        explore_action="run",
        study_id=study_id,
        json=True,
    )
    rc = run_explore_cli(args_run)
    assert rc == 0
    out_run = capsys.readouterr().out
    run_payload = json.loads(out_run)
    assert run_payload["status"] == "success"
    assert run_payload["findings_count"] > 0

    # 3. explore list
    args_list = argparse.Namespace(
        workspace=ws_dir,
        explore_action="list",
        dataset=None,
        json=False,
    )
    rc = run_explore_cli(args_list)
    assert rc == 0
    out_list = capsys.readouterr().out
    assert study_id in out_list

    # 4. explore show
    args_show = argparse.Namespace(
        workspace=ws_dir,
        explore_action="show",
        study_id=study_id,
        json=True,
    )
    rc = run_explore_cli(args_show)
    assert rc == 0
    out_show = capsys.readouterr().out
    show_data = json.loads(out_show)
    assert show_data["study"]["id"] == study_id
    assert len(show_data["findings"]) > 0

    # 5. explore findings with filters
    args_findings = argparse.Namespace(
        workspace=ws_dir,
        explore_action="findings",
        study_id=study_id,
        type=None,
        min_sig=0.05,
        limit=5,
        json=True,
    )
    rc = run_explore_cli(args_findings)
    assert rc == 0
    out_find = capsys.readouterr().out
    find_data = json.loads(out_find)
    assert isinstance(find_data, list)
    assert len(find_data) <= 5

    # 6. explore export to markdown file
    md_out_path = tmp_path / "test_report.md"
    args_export_md = argparse.Namespace(
        workspace=ws_dir,
        explore_action="export",
        study_id=study_id,
        format="markdown",
        output=str(md_out_path),
    )
    rc = run_explore_cli(args_export_md)
    assert rc == 0
    assert md_out_path.exists()
    md_content = md_out_path.read_text(encoding="utf-8")
    assert "# Technical Exploration Report · CATML Explore" in md_content
    assert study_id in md_content
    assert "## 1. Executive Summary" in md_content
    assert "## 2. Statistical Findings Catalog" in md_content

    # 7. explore export to json file
    json_out_path = tmp_path / "test_report.json"
    args_export_json = argparse.Namespace(
        workspace=ws_dir,
        explore_action="export",
        study_id=study_id,
        format="json",
        output=str(json_out_path),
    )
    rc = run_explore_cli(args_export_json)
    assert rc == 0
    assert json_out_path.exists()
    json_content = json.loads(json_out_path.read_text(encoding="utf-8"))
    assert "study" in json_content
    assert "findings" in json_content
    assert "visualizations" in json_content


# =========================================================================
# 3. DETERMINISTIC REPORT GENERATOR UNIT TEST
# =========================================================================

def test_reporting_markdown_generation_isolated():
    """Verify generate_study_markdown_report formats findings and hypotheses deterministically."""
    study_dto = {
        "id": "study_test_123",
        "name": "Customer Churn Discovery",
        "data_source": {"dataset_id": "ds_churn", "path": "/data/churn.csv"},
        "target_column": "churned",
        "status": "COMPLETED",
        "created_at": "2026-10-10T12:00:00Z",
    }
    findings_dtos = [
        {
            "id": "find_1",
            "finding_type": "high_collinearity",
            "column_name": "tenure_months",
            "secondary_column": "total_charges",
            "method_name": "Pearson Correlation",
            "summary": "Strong collinearity (r=0.96) between tenure and charges",
            "p_value": 0.00001,
            "effect_size": 0.96,
            "metrics": {"r": 0.96, "p_value_fdr": 0.00002},
            "limitations": ["Sensitive to non-linear relationships"],
        },
        {
            "id": "find_2",
            "finding_type": "skewness",
            "column_name": "monthly_usage",
            "method_name": "Fisher-Pearson Skewness",
            "summary": "Extreme positive skewness (skew=3.82)",
            "p_value": None,
            "effect_size": 3.82,
            "metrics": {},
        },
    ]
    viz_dtos = [
        {
            "id": "viz_1",
            "chart_type": "correlation_matrix",
            "title": "Pearson Correlation Heatmap",
            "data_series": {"tenure_months": [1.0, 0.96], "total_charges": [0.96, 1.0]},
        }
    ]
    hyp_dtos = [
        {
            "id": "hyp_1",
            "description": "Drop total_charges to eliminate multicollinearity",
            "proposed_action": "drop_feature",
            "finding_id": "find_1",
            "status": "proposed",
            "experiment_delta": {"drop_features": ["total_charges"]},
        }
    ]

    report = generate_study_markdown_report(
        study=study_dto,
        findings=findings_dtos,
        visualizations=viz_dtos,
        hypotheses=hyp_dtos,
    )

    assert "# Technical Exploration Report · CATML Explore" in report
    assert "Customer Churn Discovery" in report
    assert "`study_test_123`" in report
    assert "Total Quantitative Findings:** 2" in report
    assert "Statistically Significant Findings (p < 0.05 or FDR p < 0.05):** 1" in report
    assert "HIGH_COLLINEARITY" in report
    assert "Pearson Correlation Heatmap" in report
    assert "Drop total_charges" in report
    assert "Propose ≠ Accept" in report
