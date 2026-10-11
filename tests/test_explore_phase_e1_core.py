"""Comprehensive test suite for CATML Explore (Phase E1 Core).

Validates domain entities, SQLite study repository, deterministic statistical engine,
CQRS application services, CLI explore commands, MCP tools, and Web REST API endpoints.
"""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from automl.application.analysis.commands import (
    ArchiveStudyCommand,
    CreateStudyCommand,
    RunAnalysisCommand,
)
from automl.application.analysis.queries import (
    GetAnalysisRunQuery,
    GetStudyQuery,
    ListAnalysisRunsQuery,
    ListFindingsQuery,
    ListHypothesesQuery,
    ListStudiesQuery,
    ListVisualizationsQuery,
)
from automl.application.analysis.study_service import AnalysisStudyService
from automl.application.bootstrap import build_application
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    DataSourceRef,
    StatisticalFinding,
    StudySpec,
    StudyStatus,
    VisualizationSpec,
)
from automl.engine.analysis.statistical_analyzer import StatisticalAnalyzer
from automl.infrastructure.database.sqlite_studies import SQLiteStudyRepository
from automl.interfaces.cli.main import main
from automl.interfaces.mcp.server import HAS_MCP, create_mcp_server


@pytest.fixture
def synthetic_dataset_csv(tmp_path: Path) -> Path:
    """Generate a deterministic synthetic dataset with collinearity, separation, and outliers."""
    np.random.seed(42)
    n = 100
    # price and monthly_views are highly collinear (r > 0.90)
    price = np.linspace(10.0, 500.0, n)
    views = price * 2.5 + np.random.normal(0, 5.0, n)

    # class separation: high views -> label 1
    label = (views > np.median(views)).astype(int)

    # variable with extreme outliers
    feature_outliers = np.random.normal(50.0, 2.0, n)
    feature_outliers[0] = 500.0
    feature_outliers[1] = 600.0
    feature_outliers[2] = 700.0
    feature_outliers[3] = 800.0
    feature_outliers[4] = 900.0
    feature_outliers[5] = 1000.0
    feature_outliers[6] = 1200.0
    feature_outliers[7] = 1500.0

    # ID column
    ids = [f"id_{i:04d}" for i in range(n)]

    df = pd.DataFrame({
        "item_id": ids,
        "price_usd": price,
        "monthly_views": views,
        "outlier_col": feature_outliers,
        "is_bestseller": label,
    })
    csv_path = tmp_path / "synthetic_catalog.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# =========================================================================
# 1. DOMAIN LAYER TESTS
# =========================================================================

def test_domain_study_spec_creation():
    """Verify StudySpec initialization and defaults."""
    ds = DataSourceRef(dataset_id="ds_123", path="/path/to/data.csv")
    study = StudySpec.create(
        workspace_id="default",
        data_source=ds,
        name="Test Study",
        target_column="is_bestseller",
        analysis_types=["descriptive", "association"],
    )

    assert study.id.startswith("study_")
    assert study.status == StudyStatus.CREATED
    assert study.name == "Test Study"
    assert study.target_column == "is_bestseller"
    assert study.created_at is not None

    d = study.to_dict()
    assert d["id"] == study.id
    assert d["data_source"]["dataset_id"] == "ds_123"
    assert d["status"] == "CREATED"


def test_domain_study_spec_unsupervised_support():
    """Verify StudySpec supports unsupervised studies without target column."""
    ds = DataSourceRef(dataset_id="ds_unsupervised", path="/path/to/raw.csv")
    study = StudySpec.create(
        workspace_id="ws_unsupervised",
        data_source=ds,
        name="Unsupervised Clustering",
        target_column=None,
    )
    assert study.target_column is None
    d = study.to_dict()
    assert d["target_column"] is None


def test_domain_analysis_run_and_findings():
    """Verify AnalysisRun and StatisticalFinding creation and serialization."""
    run = AnalysisRun.create("study_abc")
    assert run.id.startswith("arun_")
    assert run.study_id == "study_abc"
    assert run.status == "RUNNING"

    finding = StatisticalFinding.create(
        study_id="study_abc",
        analysis_run_id=run.id,
        finding_type="high_collinearity",
        summary="High collinearity detected",
        column_name="price",
        secondary_column="views",
        metrics={"r": 0.92, "threshold": 0.70},
        p_value=0.0001,
        effect_size=None,
    )
    assert finding.id.startswith("find_")
    assert finding.finding_type == "high_collinearity"
    assert finding.column_name == "price"
    assert finding.secondary_column == "views"

    f_dict = finding.to_dict()
    assert f_dict["id"] == finding.id
    assert f_dict["column_name"] == "price"
    assert f_dict["metrics"]["r"] == 0.92


# =========================================================================
# 2. INFRASTRUCTURE LAYER TESTS (SQLiteStudyRepository)
# =========================================================================

def test_sqlite_study_repository_crud(tmp_path: Path):
    """Verify SQLiteStudyRepository CRUD operations and query filters."""
    repo = SQLiteStudyRepository(":memory:")

    ds = DataSourceRef(dataset_id="ds_1", path="/data/test.csv")
    study1 = StudySpec.create(workspace_id="ws_default", data_source=ds, name="Study 1", target_column="target")
    study2 = StudySpec.create(workspace_id="ws_default", data_source=ds, name="Study 2", target_column=None)
    study3 = StudySpec.create(workspace_id="ws_other", data_source=ds, name="Study 3", target_column=None)

    repo.save_study(study1)
    repo.save_study(study2)
    repo.save_study(study3)

    # Query by workspace
    ws_studies = repo.list_studies("ws_default")
    assert len(ws_studies) == 2
    assert {s.id for s in ws_studies} == {study1.id, study2.id}

    # Query single study
    fetched = repo.get_study(study1.id)
    assert fetched is not None
    assert fetched.name == "Study 1"
    assert fetched.target_column == "target"

    # Save Run
    run1 = AnalysisRun.create(study1.id)
    repo.save_analysis_run(run1)

    runs = repo.list_analysis_runs(study1.id)
    assert len(runs) == 1
    assert runs[0].id == run1.id

    # Save Findings
    f1 = StatisticalFinding.create(
        study_id=study1.id,
        analysis_run_id=run1.id,
        finding_type="high_collinearity",
        column_name="c1",
        secondary_column="c2",
        summary="High collinearity",
    )
    f2 = StatisticalFinding.create(
        study_id=study1.id,
        analysis_run_id=run1.id,
        finding_type="outlier_density",
        column_name="c3",
        summary="Outliers detected",
    )
    repo.save_finding(f1)
    repo.save_finding(f2)

    findings = repo.list_findings(study1.id)
    assert len(findings) == 2

    # Save Visualization Spec
    viz = VisualizationSpec.create(
        study_id=study1.id,
        analysis_run_id=run1.id,
        chart_type="correlation_matrix",
        title="Correlation Heatmap",
        data_series={"columns": ["c1", "c2"], "matrix": [[1.0, 0.8], [0.8, 1.0]]},
    )
    repo.save_visualization_spec(viz)
    viz_list = repo.list_visualization_specs(study1.id)
    assert len(viz_list) == 1
    assert viz_list[0].chart_type == "correlation_matrix"

    # Save Hypothesis
    hyp = AnalysisHypothesis.create(
        study_id=study1.id,
        finding_id=f1.id,
        proposed_action="drop_collinear_feature",
        description="Drop c1 to mitigate collinearity",
    )
    repo.save_hypothesis(hyp)
    hyps = repo.list_hypotheses(study1.id)
    assert len(hyps) == 1
    assert hyps[0].proposed_action == "drop_collinear_feature"


# =========================================================================
# 3. ENGINE LAYER TESTS (StatisticalAnalyzer)
# =========================================================================

def test_statistical_analyzer_deterministic_findings(synthetic_dataset_csv: Path):
    """Verify StatisticalAnalyzer computes true bivariate collinearity, ANOVA, and outliers."""
    analyzer = StatisticalAnalyzer()
    findings, visualizations, hypotheses = analyzer.analyze(
        data_source_path=str(synthetic_dataset_csv),
        target_column="is_bestseller",
        analysis_types=["descriptive", "association"],
    )

    finding_types = {f.finding_type for f in findings}

    # Should detect collinearity between price_usd and monthly_views (r > 0.90)
    assert "high_collinearity" in finding_types

    # Should detect target separation between monthly_views and is_bestseller
    assert "target_class_separation" in finding_types

    # Should detect outlier density on outlier_col
    assert "outlier_density" in finding_types

    # Should detect item_id as high cardinality identifier candidate
    assert "high_cardinality_identifier" in finding_types

    # Visualizations must include correlation matrix
    viz_types = {v.chart_type for v in visualizations}
    assert "correlation_matrix" in viz_types

    # Hypotheses should be formulated
    assert len(hypotheses) > 0
    actions = {h.proposed_action for h in hypotheses}
    assert "resolve_collinearity" in actions


def test_statistical_analyzer_unsupervised_mode(synthetic_dataset_csv: Path):
    """Verify StatisticalAnalyzer runs cleanly without target column (target_column=None)."""
    analyzer = StatisticalAnalyzer()
    findings, visualizations, hypotheses = analyzer.analyze(
        data_source_path=str(synthetic_dataset_csv),
        target_column=None,
        analysis_types=["descriptive", "association"],
    )

    finding_types = {f.finding_type for f in findings}
    assert "high_collinearity" in finding_types
    # Target class separation should not be present when target is None
    assert "target_class_separation" not in finding_types
    assert len(visualizations) > 0


# =========================================================================
# 4. APPLICATION LAYER & CQRS TESTS
# =========================================================================

def test_analysis_cqrs_bus_dispatch(synthetic_dataset_csv: Path, tmp_path: Path):
    """Verify full CQRS dispatching through CommandBus and QueryBus."""
    ws, cb, qb = build_application(root_dir=str(tmp_path / "app_ws"))
    ds = ws.register_dataset(
        name="synthetic",
        path=synthetic_dataset_csv,
        target="is_bestseller",
        task_type="binary_classification",
    )

    # 1. Dispatch CreateStudyCommand
    cmd_create = CreateStudyCommand(
        dataset_id=ds.id,
        name="CQRS Study",
        target_column="is_bestseller",
    )
    study_id = cb.dispatch(cmd_create)
    assert isinstance(study_id, str)
    assert study_id.startswith("study_")

    # 2. Query GetStudyQuery
    study_data = qb.dispatch(GetStudyQuery(study_id=study_id))
    assert study_data is not None
    assert study_data["id"] == study_id
    assert study_data["name"] == "CQRS Study"

    # 3. Query ListStudiesQuery
    studies_list = qb.dispatch(ListStudiesQuery())
    assert len(studies_list) >= 1
    assert any(s["id"] == study_id for s in studies_list)

    # 4. Dispatch RunAnalysisCommand
    cmd_run = RunAnalysisCommand(study_id=study_id)
    run_id = cb.dispatch(cmd_run)
    assert isinstance(run_id, str)
    assert run_id.startswith("arun_")

    # 5. Query ListFindingsQuery
    findings = qb.dispatch(ListFindingsQuery(study_id=study_id))
    assert len(findings) >= 2

    # Query with finding_type filter
    collinear_findings = qb.dispatch(ListFindingsQuery(study_id=study_id, finding_type="high_collinearity"))
    assert len(collinear_findings) >= 1
    assert all(f["finding_type"] == "high_collinearity" for f in collinear_findings)

    # 6. Query ListVisualizationsQuery
    visualizations = qb.dispatch(ListVisualizationsQuery(study_id=study_id))
    assert len(visualizations) >= 1

    # 7. Query ListHypothesesQuery
    hyps = qb.dispatch(ListHypothesesQuery(study_id=study_id))
    assert len(hyps) >= 1

    # 8. Dispatch ArchiveStudyCommand
    cb.dispatch(ArchiveStudyCommand(study_id=study_id))
    archived_study = qb.dispatch(GetStudyQuery(study_id=study_id))
    assert archived_study["status"] == "ARCHIVED"


# =========================================================================
# 5. CLI INTERFACE TESTS
# =========================================================================

def test_cli_explore_lifecycle(synthetic_dataset_csv: Path, tmp_path: Path, capsys):
    """Verify 'automl explore' commands (create, run, list, show, findings)."""
    ws_dir = str(tmp_path / "cli_ws")
    ws, _, _ = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(
        name="cli_dataset",
        path=synthetic_dataset_csv,
        target="is_bestseller",
        task_type="binary_classification",
    )

    # explore create
    exit_code = main(["explore", "create", ds.id, "--name", "CLI Study", "--workspace", ws_dir])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Created study" in captured.out
    study_id = [w for w in captured.out.split() if w.startswith("study_")][0]

    # explore list
    exit_code = main(["explore", "list", "--workspace", ws_dir])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert study_id in captured.out

    # explore run
    exit_code = main(["explore", "run", study_id, "--workspace", ws_dir])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"Executed study {study_id}" in captured.out
    assert "Findings detected" in captured.out

    # explore show
    exit_code = main(["explore", "show", study_id, "--workspace", ws_dir])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "COMPLETED" in captured.out

    # explore findings
    exit_code = main(["explore", "findings", study_id, "--workspace", ws_dir])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "HIGH_COLLINEARITY" in captured.out or "Separación" in captured.out


# =========================================================================
# 6. MCP SERVER TOOLS TESTS
# =========================================================================

@pytest.mark.skipif(not HAS_MCP, reason="mcp extra required")
def test_mcp_explore_tools_integration(synthetic_dataset_csv: Path, tmp_path: Path):
    """Verify MCP server exposes and executes CATML Explore tools."""
    ws_dir = str(tmp_path / "mcp_ws")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(
        name="mcp_dataset",
        path=synthetic_dataset_csv,
        target="is_bestseller",
        task_type="binary_classification",
    )
    server = create_mcp_server(workspace=ws, command_bus=cb, query_bus=qb, root_dir=ws_dir)

    tools = asyncio.run(server.list_tools())
    tool_names = {t.name for t in tools}

    assert "analysis_create_study" in tool_names
    assert "analysis_run_study" in tool_names
    assert "analysis_get_findings" in tool_names
    assert "analysis_list_studies" in tool_names
    assert "analysis_get_study" in tool_names

    # Call analysis_create_study tool
    res = asyncio.run(
        server.call_tool(
            "analysis_create_study",
            {"dataset_id": ds.id, "name": "MCP Exploratory Study", "target_column": "is_bestseller"},
        )
    )
    assert not res.is_error
    created_study = json.loads(res.content[0].text)
    study_id = created_study["id"]
    assert study_id.startswith("study_")

    # Call analysis_run_study tool
    run_res = asyncio.run(server.call_tool("analysis_run_study", {"study_id": study_id}))
    assert not run_res.is_error
    run_payload = json.loads(run_res.content[0].text)
    assert run_payload["study_id"] == study_id
    assert run_payload["findings_count"] >= 1

    # Call analysis_get_findings tool
    findings_res = asyncio.run(server.call_tool("analysis_get_findings", {"study_id": study_id}))
    assert not findings_res.is_error
    findings_list = json.loads(findings_res.content[0].text)
    assert len(findings_list) >= 1


# =========================================================================
# 7. WEB REST API ENDPOINT TESTS
# =========================================================================

def test_web_api_analysis_endpoints(synthetic_dataset_csv: Path, tmp_path: Path):
    """Verify HTTP REST endpoints for CATML Explore studies, execution and findings."""
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from automl.interfaces.web.server import AutoMLWebHandler

    ws_dir = str(tmp_path / "web_ws")
    ws, _, _ = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(
        name="web_dataset",
        path=synthetic_dataset_csv,
        target="is_bestseller",
        task_type="binary_classification",
    )

    AutoMLWebHandler.workspace_dir = ws_dir
    AutoMLWebHandler.require_auth = False
    AutoMLWebHandler.auth_token = None

    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address
    base_url = f"http://{host}:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # 1. GET /api/analysis/studies (initially empty)
        with urlopen(f"{base_url}/api/analysis/studies") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data == []

        # 2. POST /api/analysis/studies/create
        req_body = json.dumps({
            "dataset_id": ds.id,
            "name": "Web Exploratory Study",
            "target_column": "is_bestseller",
        }).encode("utf-8")
        req = Request(f"{base_url}/api/analysis/studies/create", data=req_body, headers={"Content-Type": "application/json"})
        with urlopen(req) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res["status"] == "success"
            study_id = res["study_id"]
            assert study_id.startswith("study_")

        # 3. GET /api/analysis/studies (now contains 1 study)
        with urlopen(f"{base_url}/api/analysis/studies") as resp:
            assert resp.status == 200
            studies = json.loads(resp.read().decode("utf-8"))
            assert len(studies) == 1
            assert studies[0]["id"] == study_id

        # 4. POST /api/analysis/studies/run
        run_body = json.dumps({"study_id": study_id}).encode("utf-8")
        req_run = Request(f"{base_url}/api/analysis/studies/run", data=run_body, headers={"Content-Type": "application/json"})
        with urlopen(req_run) as resp:
            assert resp.status == 200
            run_res = json.loads(resp.read().decode("utf-8"))
            assert run_res["status"] == "success"
            assert run_res["findings_count"] >= 1

        # 5. GET /api/analysis/study?id=...
        with urlopen(f"{base_url}/api/analysis/study?id={study_id}") as resp:
            assert resp.status == 200
            study_detail = json.loads(resp.read().decode("utf-8"))
            assert study_detail["id"] == study_id
            assert len(study_detail["findings"]) >= 1
            assert len(study_detail["visualizations"]) >= 1

        # 6. GET /api/analysis/findings?study_id=...
        with urlopen(f"{base_url}/api/analysis/findings?study_id={study_id}") as resp:
            assert resp.status == 200
            findings_data = json.loads(resp.read().decode("utf-8"))
            assert len(findings_data) >= 1

        # 7. GET /api/analysis/visualizations?study_id=...
        with urlopen(f"{base_url}/api/analysis/visualizations?study_id={study_id}") as resp:
            assert resp.status == 200
            viz_data = json.loads(resp.read().decode("utf-8"))
            assert len(viz_data) >= 1

        # 8. POST /api/analysis/studies/archive
        arch_body = json.dumps({"study_id": study_id}).encode("utf-8")
        req_arch = Request(f"{base_url}/api/analysis/studies/archive", data=arch_body, headers={"Content-Type": "application/json"})
        with urlopen(req_arch) as resp:
            assert resp.status == 200
            arch_res = json.loads(resp.read().decode("utf-8"))
            assert arch_res["archived"] is True

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
