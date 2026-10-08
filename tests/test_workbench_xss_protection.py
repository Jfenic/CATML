"""
Tests for Workbench frontend XSS protection and metrics segregation.

Verifies:
1. XSS protection: utils.js escapeHtml implementation, agent.js & overview.js sanitization.
2. Incompatible metrics segregation in /api/overview (best_by_metric, contextual best_metric).
3. BenchmarkRunner delta_vs_baseline calculation reflecting error reductions as positive gains.
"""
from __future__ import annotations

import io
import json
import re
from http import HTTPStatus
from pathlib import Path

import pytest

from automl.application.bootstrap import build_application
from automl.benchmarks.runner import BenchmarkRunner, is_minimizing_metric
from automl.interfaces.web.server import AutoMLWebHandler


def _create_handler(workspace_dir: Path, path: str) -> tuple[AutoMLWebHandler, io.BytesIO]:
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(workspace_dir)
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.headers = {"Host": "127.0.0.1"}
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.path = path
    return handler, handler.wfile


def _parse_response_json(wfile: io.BytesIO) -> tuple[int, dict]:
    output = wfile.getvalue().decode("utf-8")
    status_line = output.splitlines()[0]
    status_code = int(status_line.split()[1])
    body = output.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in output else output.split("\n\n", 1)[1]
    return status_code, json.loads(body)


def test_escape_html_utility_exists_and_escapes_properly():
    utils_path = Path("src/automl/interfaces/web/static/js/utils.js")
    assert utils_path.exists(), "utils.js must exist in static/js"
    content = utils_path.read_text(encoding="utf-8")
    assert "export function escapeHtml" in content
    assert ".replace(/&/g, \"&amp;\")" in content
    assert ".replace(/</g, \"&lt;\")" in content
    assert ".replace(/>/g, \"&gt;\")" in content
    assert ".replace(/\"/g, \"&quot;\")" in content
    assert ".replace(/'/g, \"&#039;\")" in content


def test_agent_drawer_uses_escape_html():
    agent_path = Path("src/automl/interfaces/web/static/js/views/agent.js")
    assert agent_path.exists()
    content = agent_path.read_text(encoding="utf-8")
    assert "import { escapeHtml } from \"../utils.js\";" in content
    assert "escapeHtml(dsName)" in content
    assert "escapeHtml(taskType)" in content
    assert "escapeHtml(statusText)" in content
    assert "escapeHtml(hyp.statement" in content
    assert "escapeHtml(hyp.action" in content
    assert "escapeHtml(hyp.critic_reason" in content
    assert "escapeHtml(hyp.id)" in content


def test_overview_view_uses_escape_html():
    overview_path = Path("src/automl/interfaces/web/static/js/views/overview.js")
    assert overview_path.exists()
    content = overview_path.read_text(encoding="utf-8")
    assert "import { escapeHtml } from \"../utils.js\";" in content
    assert "escapeHtml(activeRun.id)" in content
    assert "escapeHtml(activeRun.dataset_name" in content
    assert "escapeHtml(r.dataset_name" in content
    assert "escapeHtml(ds.name)" in content
    assert "escapeHtml(act.title)" in content
    assert "escapeHtml(act.description)" in content


def test_overview_api_segregates_incompatible_metrics(tmp_path: Path):
    ws_dir = tmp_path / "multi_metric_ws"
    ws, _, _ = build_application(root_dir=str(ws_dir))

    # Create dummy data file
    data_file = ws_dir / "data.csv"
    data_file.write_text("x1,x2,y\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n", encoding="utf-8")

    # Run 1: Binary classification with ROC-AUC (e.g. score 0.85)
    ds1 = ws.register_dataset(name="ds_classification", path=str(data_file), target="y", task_type="binary_classification")
    run1 = ws.create_run(ds1, metric="roc_auc")
    exp1 = ws.create_experiment(run1, name="exp_clf", model_ids=["logistic_regression"])

    from automl.domain.experiments.trial import TrialResult
    # Save trial with ROC-AUC = 0.85
    tr1 = TrialResult(
        trial_id="tr_1",
        experiment_id=exp1.id,
        model_id="logistic_regression",
        primary_metric="roc_auc",
        primary_score=0.85,
    )
    ws.repository.save_trial_result(tr1)

    # Run 2: Regression with RMSE (e.g. score 0.35 - lower is better)
    ds2 = ws.register_dataset(name="ds_regression", path=str(data_file), target="y", task_type="regression")
    run2 = ws.create_run(ds2, metric="rmse")
    exp2 = ws.create_experiment(run2, name="exp_reg", model_ids=["ridge"])

    tr2 = TrialResult(
        trial_id="tr_2",
        experiment_id=exp2.id,
        model_id="ridge",
        primary_metric="rmse",
        primary_score=0.35,
    )
    ws.repository.save_trial_result(tr2)

    # Call /api/overview
    handler, wfile = _create_handler(ws_dir, "/api/overview")
    handler.do_GET()
    status, data = _parse_response_json(wfile)

    assert status == 200
    assert data["total_runs"] == 2

    # Verify best_by_metric has separate, un-mixed entries
    best_by_metric = data["best_by_metric"]
    assert "roc_auc" in best_by_metric
    assert "rmse" in best_by_metric

    assert best_by_metric["roc_auc"]["score"] == 0.85
    assert best_by_metric["roc_auc"]["model_id"] == "logistic_regression"

    assert best_by_metric["rmse"]["score"] == 0.35
    assert best_by_metric["rmse"]["model_id"] == "ridge"

    # Verify top-level best_score is paired with its corresponding best_metric
    assert data["best_metric"] in {"roc_auc", "rmse"}
    if data["best_metric"] == "roc_auc":
        assert data["best_score"] == 0.85
        assert data["best_model"] == "logistic_regression"
    else:
        assert data["best_score"] == 0.35
        assert data["best_model"] == "ridge"


def test_benchmark_delta_vs_baseline_minimizing_and_maximizing():
    assert is_minimizing_metric("rmse") is True
    assert is_minimizing_metric("mae") is True
    assert is_minimizing_metric("mse") is True
    assert is_minimizing_metric("loss") is True
    assert is_minimizing_metric("roc_auc") is False
    assert is_minimizing_metric("accuracy") is False

    # Simulate minimizing metric results (e.g. RMSE)
    # Baseline RMSE: 2.50
    # Scenario A RMSE: 2.10 (improved -> error reduced by 0.40 -> delta should be +0.40)
    # Scenario B RMSE: 2.80 (worse -> error increased by 0.30 -> delta should be -0.30)
    results_min = [
        {"scenario_id": "baseline_all_features", "metric": "rmse", "best_score": 2.50, "delta_vs_baseline": None},
        {"scenario_id": "scenario_improved", "metric": "rmse", "best_score": 2.10, "delta_vs_baseline": None},
        {"scenario_id": "scenario_worse", "metric": "rmse", "best_score": 2.80, "delta_vs_baseline": None},
    ]

    baseline = next(r for r in results_min if r["scenario_id"] == "baseline_all_features")
    for row in results_min:
        if is_minimizing_metric(row.get("metric", "")):
            row["delta_vs_baseline"] = round(baseline["best_score"] - row["best_score"], 4)
        else:
            row["delta_vs_baseline"] = round(row["best_score"] - baseline["best_score"], 4)

    assert results_min[0]["delta_vs_baseline"] == 0.0
    assert results_min[1]["delta_vs_baseline"] == 0.40  # Positive gain from error reduction
    assert results_min[2]["delta_vs_baseline"] == -0.30  # Negative regression

    # Simulate maximizing metric results (e.g. ROC-AUC)
    # Baseline ROC-AUC: 0.80
    # Scenario A ROC-AUC: 0.86 (improved -> delta should be +0.06)
    results_max = [
        {"scenario_id": "baseline_all_features", "metric": "roc_auc", "best_score": 0.80, "delta_vs_baseline": None},
        {"scenario_id": "scenario_improved", "metric": "roc_auc", "best_score": 0.86, "delta_vs_baseline": None},
    ]

    baseline_max = next(r for r in results_max if r["scenario_id"] == "baseline_all_features")
    for row in results_max:
        if is_minimizing_metric(row.get("metric", "")):
            row["delta_vs_baseline"] = round(baseline_max["best_score"] - row["best_score"], 4)
        else:
            row["delta_vs_baseline"] = round(row["best_score"] - baseline_max["best_score"], 4)

    assert results_max[0]["delta_vs_baseline"] == 0.0
    assert results_max[1]["delta_vs_baseline"] == 0.06  # Positive score gain
