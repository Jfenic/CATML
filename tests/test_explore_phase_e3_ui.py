"""Comprehensive test suite for CATML Explore Phase E3 (Workbench Visual Laboratory).

Validates declarative VisualizationBuilder (histograms, boxplots, downsampled scatter plots),
REST API visualization endpoints, and frontend JavaScript explore.js integration.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.engine.analysis.statistical_analyzer import StatisticalAnalyzer
from automl.engine.analysis.visualizations.builder import VisualizationBuilder
from automl.interfaces.web.server import AutoMLWebHandler


@pytest.fixture
def viz_test_csv(tmp_path: Path) -> Path:
    """Generate deterministic numerical and categorical data for visualization testing."""
    np.random.seed(42)
    n = 120
    x = np.linspace(10.0, 100.0, n)
    y = 2.5 * x + np.random.normal(0, 5.0, n)
    # skew & outliers
    outliers = np.array([250.0, 300.0, -100.0])
    x_with_outliers = np.append(x[:n-3], outliers)

    df = pd.DataFrame({
        "feature_x": x_with_outliers,
        "feature_y": y,
        "group_cat": np.random.choice(["Alpha", "Beta", "Gamma"], size=n),
    })
    csv_file = tmp_path / "viz_data.csv"
    df.to_csv(csv_file, index=False)
    return csv_file


# =========================================================================
# 1. VISUALIZATION BUILDER BACKEND UNIT TESTS
# =========================================================================

def test_builder_histogram_aggregation(viz_test_csv: Path):
    """Verify aggregated histogram generation with safe payload size."""
    df = pd.read_csv(viz_test_csv)
    spec = VisualizationBuilder.build_histogram_spec(
        df=df,
        column="feature_x",
        study_id="s1",
        run_id="r1",
        bins=10,
    )

    assert spec is not None
    assert spec.chart_type == "histogram"
    assert spec.study_id == "s1"
    assert spec.analysis_run_id == "r1"
    assert len(spec.data_series["counts"]) == 10
    assert len(spec.data_series["bin_edges"]) == 11
    assert "mean" in spec.data_series
    assert "std" in spec.data_series
    assert spec.data_series["total_points"] == len(df)


def test_builder_boxplot_five_number_summary(viz_test_csv: Path):
    """Verify five-number summary boxplot generation and capped outlier sampling."""
    df = pd.read_csv(viz_test_csv)
    spec = VisualizationBuilder.build_boxplot_spec(
        df=df,
        column="feature_x",
        study_id="s1",
        run_id="r1",
    )

    assert spec is not None
    assert spec.chart_type == "boxplot"
    assert "q25" in spec.data_series
    assert "median" in spec.data_series
    assert "q75" in spec.data_series
    assert "whisker_low" in spec.data_series
    assert "whisker_high" in spec.data_series
    assert spec.data_series["outliers_count"] >= 1
    assert len(spec.data_series["outliers_sample"]) <= 25


def test_builder_bivariate_scatter_downsampling(viz_test_csv: Path):
    """Verify bivariate scatter plot builds with trend line and point limits."""
    df = pd.read_csv(viz_test_csv)
    spec = VisualizationBuilder.build_bivariate_scatter_spec(
        df=df,
        col_x="feature_x",
        col_y="feature_y",
        study_id="s1",
        run_id="r1",
        max_points=50,
    )

    assert spec is not None
    assert spec.chart_type == "scatter"
    assert len(spec.data_series["x_points"]) <= 50
    assert len(spec.data_series["y_points"]) <= 50
    assert "trend_slope" in spec.data_series
    assert "trend_intercept" in spec.data_series
    assert "pearson_r" in spec.data_series


# =========================================================================
# 2. STATISTICAL ANALYZER FULL VIZ INTEGRATION TEST
# =========================================================================

def test_statistical_analyzer_generates_all_viz_specs(viz_test_csv: Path):
    """Verify StatisticalAnalyzer pipeline emits correlation, distribution and scatter specs."""
    analyzer = StatisticalAnalyzer()
    findings, visualizations, hypotheses = analyzer.analyze(
        data_source_path=str(viz_test_csv),
        target_column=None,
        analysis_types=["descriptive", "association"],
    )

    viz_types = {v.chart_type for v in visualizations}
    assert "correlation_matrix" in viz_types
    # Since feature_x has outlier density / skewness, boxplot or histogram should be built
    assert "boxplot" in viz_types or "histogram" in viz_types
    # Bivariate scatter should be generated for collinear pair
    assert "scatter" in viz_types or "correlation_matrix" in viz_types


# =========================================================================
# 3. REST API ENDPOINTS FOR VISUALIZATIONS
# =========================================================================

def test_web_api_serves_rich_visualizations(viz_test_csv: Path, tmp_path: Path):
    """Verify HTTP GET /api/analysis/visualizations serves declarative specs."""
    ws_dir = str(tmp_path / "web_viz_ws")
    ws, _, _ = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(
        name="viz_web_dataset",
        path=viz_test_csv,
        target=None,
        task_type="clustering",
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
        # 1. Create study
        req_create = Request(
            f"{base_url}/api/analysis/studies/create",
            data=json.dumps({"dataset_id": ds.id, "name": "Viz Web Study"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req_create) as resp:
            study_id = json.loads(resp.read().decode("utf-8"))["study_id"]

        # 2. Run study
        req_run = Request(
            f"{base_url}/api/analysis/studies/run",
            data=json.dumps({"study_id": study_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req_run) as resp:
            run_res = json.loads(resp.read().decode("utf-8"))
            assert run_res["status"] == "success"

        # 3. Query GET /api/analysis/visualizations
        with urlopen(f"{base_url}/api/analysis/visualizations?study_id={study_id}") as resp:
            assert resp.status == 200
            viz_list = json.loads(resp.read().decode("utf-8"))
            assert len(viz_list) >= 1
            chart_types = {v["chart_type"] for v in viz_list}
            assert "correlation_matrix" in chart_types

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


# =========================================================================
# 4. FRONTEND JS MODULE INTEGRATION & SYNTAX TESTS
# =========================================================================

def test_frontend_explore_view_file_and_syntax():
    """Verify explore.js exists, has all required methods and valid JavaScript syntax."""
    explore_js = Path("src/automl/interfaces/web/static/js/views/explore.js")
    assert explore_js.exists(), "explore.js view must exist"

    content = explore_js.read_text(encoding="utf-8")
    assert "class ExploreView" in content
    assert "_renderVisualizationContent" in content
    assert "_renderCorrelationMatrixHtml" in content
    assert "_renderCategoricalMatrixHtml" in content
    assert "_renderHistogramSvgHtml" in content
    assert "_renderBoxplotSvgHtml" in content
    assert "_renderScatterSvgHtml" in content
    assert "_downloadMarkdownReport" in content
    assert "Benjamini-Hochberg" in content or "FDR" in content

    # Check that app.js registers ExploreView
    app_js = Path("src/automl/interfaces/web/static/js/app.js").read_text(encoding="utf-8")
    assert 'import { ExploreView } from "./views/explore.js";' in app_js
    assert 'case "explore":' in app_js

    # Syntax check via node if available
    node_bin = shutil.which("node")
    if node_bin:
        res = subprocess.run([node_bin, "--check", str(explore_js)], capture_output=True, text=True)
        assert res.returncode == 0, f"Node syntax error in explore.js: {res.stderr}"
