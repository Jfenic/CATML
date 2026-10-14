"""Comprehensive tests for Workbench Sprint 3: Demo project & Executive results export."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.interfaces.web.server import AutoMLWebHandler


def _create_handler(
    workspace_dir: Path,
    path: str,
    method: str = "GET",
    body: dict | None = None,
) -> tuple[AutoMLWebHandler, io.BytesIO]:
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(workspace_dir)
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = io.BytesIO(json.dumps(body or {}).encode("utf-8") if body else b"")
    handler.wfile = io.BytesIO()
    handler.headers = {
        "Host": "127.0.0.1",
        "Content-Length": str(len(handler.rfile.getvalue())),
        "Content-Type": "application/json",
    }
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.path = path
    return handler, handler.wfile


def _parse_response(wfile: io.BytesIO) -> tuple[int, dict[str, str], str]:
    raw = wfile.getvalue().decode("utf-8", errors="ignore")
    lines = raw.split("\r\n")
    status_line = lines[0]
    status_code = int(status_line.split()[1])

    headers = {}
    i = 1
    while i < len(lines) and lines[i]:
        if ": " in lines[i]:
            k, v = lines[i].split(": ", 1)
            headers[k.lower()] = v
        i += 1

    body = "\r\n".join(lines[i + 1 :]) if i < len(lines) else ""
    return status_code, headers, body


def test_post_project_demo_bootstraps_complete_project(tmp_path: Path):
    ws_dir = tmp_path / "ws_demo"
    ws_dir.mkdir(parents=True, exist_ok=True)

    handler, wfile = _create_handler(ws_dir, "/api/project/demo", method="POST", body={"quick_train": True})
    handler.do_POST()
    status, headers, body = _parse_response(wfile)

    assert status == 200
    res = json.loads(body)
    assert res["status"] == "success"
    assert "dataset_id" in res and res["dataset_id"]
    assert "run_id" in res and res["run_id"]
    assert "experiment_id" in res and res["experiment_id"]
    assert res["dataset_name"] == "Demo Churn de Clientes"

    # Verify workspace state
    ws, cmd, qry = build_application(root_dir=str(ws_dir))
    datasets = ws.list_datasets()
    assert any(d.id == res["dataset_id"] for d in datasets)

    runs = ws.list_runs(dataset_id=res["dataset_id"])
    assert any(r.id == res["run_id"] for r in runs)

    experiments = ws.list_experiments(res["run_id"])
    assert len(experiments) >= 1

    trials = ws.repository.list_trial_results(experiments[0].id)
    assert len(trials) >= 1

    # Idempotence: calling again succeeds without duplicating or erroring
    handler2, wfile2 = _create_handler(ws_dir, "/api/project/demo", method="POST", body={"quick_train": True})
    handler2.do_POST()
    status2, _, body2 = _parse_response(wfile2)
    assert status2 == 200
    res2 = json.loads(body2)
    assert res2["dataset_id"] == res["dataset_id"]
    assert res2["run_id"] == res["run_id"]


def test_executive_report_html_md_json(tmp_path: Path):
    ws_dir = tmp_path / "ws_report"
    ws_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bootstrap demo
    h_demo, w_demo = _create_handler(ws_dir, "/api/project/demo", method="POST", body={"quick_train": True})
    h_demo.do_POST()
    _, _, b_demo = _parse_response(w_demo)
    res_demo = json.loads(b_demo)
    run_id = res_demo["run_id"]

    # 2. Test HTML report
    h_html, w_html = _create_handler(ws_dir, f"/api/reports/executive?run_id={run_id}&format=html")
    h_html.do_GET()
    status, headers, body = _parse_response(w_html)
    assert status == 200
    assert "text/html" in headers.get("content-type", "")
    assert "Demo Churn de Clientes" in body
    assert "Informe Ejecutivo de Resultados" in body
    assert "Resumen del Rendimiento Óptimo" in body
    assert "Leaderboard Completo de Ensayos" in body
    assert "no es un certificado" in body
    assert "in-fold" in body

    # 3. Test Markdown report
    h_md, w_md = _create_handler(ws_dir, f"/api/reports/executive?run_id={run_id}&format=md")
    h_md.do_GET()
    status_md, headers_md, body_md = _parse_response(w_md)
    assert status_md == 200
    assert "text/markdown" in headers_md.get("content-type", "")
    assert "# CATML — Informe Ejecutivo de Resultados" in body_md
    assert "Demo Churn de Clientes" in body_md
    assert "| Rango | Modelo |" in body_md

    # 4. Test JSON report with download disposition
    h_json, w_json = _create_handler(ws_dir, f"/api/reports/executive?run_id={run_id}&format=json&download=1")
    h_json.do_GET()
    status_json, headers_json, body_json = _parse_response(w_json)
    assert status_json == 200
    assert "application/json" in headers_json.get("content-type", "")
    assert "attachment" in headers_json.get("content-disposition", "")
    data = json.loads(body_json)
    assert data["metadata"]["run_id"] == run_id
    assert data["dataset"]["name"] == "Demo Churn de Clientes"
    assert data["champion"]["model_id"]
    assert "leaderboard" in data
    assert len(data["leaderboard"]) >= 1


def test_executive_report_summary_endpoint(tmp_path: Path):
    ws_dir = tmp_path / "ws_summary"
    ws_dir.mkdir(parents=True, exist_ok=True)

    h_demo, w_demo = _create_handler(ws_dir, "/api/project/demo", method="POST", body={"quick_train": True})
    h_demo.do_POST()
    _, _, b_demo = _parse_response(w_demo)
    run_id = json.loads(b_demo)["run_id"]

    h_sum, w_sum = _create_handler(ws_dir, f"/api/reports/executive-summary?run_id={run_id}")
    h_sum.do_GET()
    status, headers, body = _parse_response(w_sum)
    assert status == 200
    assert "application/json" in headers.get("content-type", "")
    data = json.loads(body)
    assert data["metadata"]["run_id"] == run_id
    assert "champion" in data
    assert "validation" in data
    assert data["validation"]["anti_leakage_guaranteed"] is False
    assert data["validation"]["in_fold_preprocessing"] is True
    assert "delta_vs_worst_succeeded" in data["champion"]


def test_executive_report_errors_on_invalid_run(tmp_path: Path):
    ws_dir = tmp_path / "ws_err"
    ws_dir.mkdir(parents=True, exist_ok=True)

    # Missing run_id on empty workspace
    h1, w1 = _create_handler(ws_dir, "/api/reports/executive")
    h1.do_GET()
    s1, _, b1 = _parse_response(w1)
    assert s1 == 400

    # Non-existent run_id
    h2, w2 = _create_handler(ws_dir, "/api/reports/executive?run_id=non_existent_run_999")
    h2.do_GET()
    s2, _, b2 = _parse_response(w2)
    assert s2 == 404


def test_frontend_files_contain_demo_and_export_hooks():
    api_js = Path("src/automl/interfaces/web/static/js/api.js").read_text(encoding="utf-8")
    assert "async loadDemoProject" in api_js
    assert "getExecutiveReportUrl" in api_js
    assert "getExecutiveReportSummary" in api_js

    home_js = Path("src/automl/interfaces/web/static/js/views/home.js").read_text(encoding="utf-8")
    assert "btnHomeLoadDemoEmpty" in home_js
    assert "api.getExecutiveReportUrl" in home_js
    assert "Informe ejecutivo" in home_js

    modal_js = Path("src/automl/interfaces/web/static/js/views/register_dataset_modal.js").read_text(encoding="utf-8")
    assert "btnModalLoadDemoDirect" in modal_js
    assert "loadDemoProject" in modal_js

    studio_js = Path("src/automl/interfaces/web/static/js/views/studio.js").read_text(encoding="utf-8")
    assert "/api/reports/executive?run_id=" in studio_js
    assert "Informe ejecutivo" in studio_js

    overview_js = Path("src/automl/interfaces/web/static/js/views/overview.js").read_text(encoding="utf-8")
    assert "/api/reports/executive?run_id=" in overview_js
    assert "Informe ejecutivo" in overview_js


def test_executive_report_failed_trials_and_duration_extraction(tmp_path: Path):
    from automl.domain.experiments.trial import TrialResult

    ws_dir = tmp_path / "ws_failed_test"
    ws, cmd, qry = build_application(root_dir=str(ws_dir))
    csv_path = ws_dir / "data.csv"
    pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 1, 0, 1]}).to_csv(csv_path, index=False)
    ds = ws.register_dataset("ds_test", str(csv_path), target="y", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")

    # 1. Empty run test (0 trials)
    h_empty, w_empty = _create_handler(ws_dir, f"/api/reports/executive?run_id={run.id}&format=html")
    h_empty.do_GET()
    s_emp, _, b_emp = _parse_response(w_empty)
    assert s_emp == 200
    assert "No se registraron modelos completados exitosamente" in b_emp
    assert "Sin ensayos evaluados" in b_emp

    h_md_empty, w_md_empty = _create_handler(ws_dir, f"/api/reports/executive?run_id={run.id}&format=md")
    h_md_empty.do_GET()
    _, _, b_md_emp = _parse_response(w_md_empty)
    assert "No se registraron modelos completados exitosamente" in b_md_emp
    assert "Sin ensayos evaluados" in b_md_emp

    # 2. Add an experiment with 1 failed trial (1.5s duration) and 1 successful trial (2.5s duration)
    exp = ws.create_experiment(
        run=run,
        name="Mixed Exp",
        hypothesis="Testing mixed trial outcomes",
        feature_names=["x"],
        model_ids=["logistic_regression", "random_forest"],
    )

    tr_fail = TrialResult(
        trial_id="tr_fail",
        experiment_id=exp.id,
        model_id="logistic_regression",
        primary_metric="roc_auc",
        primary_score=0.99,
        training_time_seconds=1.5,
        failure_reason="Runtime out of memory error",
    )
    tr_ok = TrialResult(
        trial_id="tr_ok",
        experiment_id=exp.id,
        model_id="random_forest",
        primary_metric="roc_auc",
        primary_score=0.88,
        training_time_seconds=2.5,
    )
    ws.repository.save_trial_result(tr_fail)
    ws.repository.save_trial_result(tr_ok)

    # 3. Check JSON report
    h_json, w_json = _create_handler(ws_dir, f"/api/reports/executive?run_id={run.id}&format=json")
    h_json.do_GET()
    s_j, _, b_j = _parse_response(w_json)
    assert s_j == 200
    data = json.loads(b_j)

    # Champion must be random_forest (the ONLY successful trial, despite logistic_regression having higher raw score)
    assert data["champion"]["model_id"] == "random_forest"
    assert data["champion"]["score"] == 0.88
    assert data["champion"]["training_time_seconds"] == 2.5
    assert data["champion"]["delta_vs_worst_succeeded"] is None  # only 1 successful trial
    assert data["champion"]["delta_vs_baseline"] is None  # legacy alias

    # Leaderboard row durations must NOT be 0.0s
    lb = data["leaderboard"]
    assert len(lb) == 2
    for row in lb:
        t = row.get("training_time_s")
        assert t is not None and t > 0

    # 4. Check HTML report contains error badge for failed trial and ok badge for successful
    h_html, w_html = _create_handler(ws_dir, f"/api/reports/executive?run_id={run.id}&format=html")
    h_html.do_GET()
    _, _, b_html = _parse_response(w_html)
    assert "● ERROR" in b_html
    assert "● OK" in b_html
    assert "2.50s" in b_html
    assert "1.50s" in b_html

    # 5. Check Markdown report displays Fallido and Completado
    h_md, w_md = _create_handler(ws_dir, f"/api/reports/executive?run_id={run.id}&format=md")
    h_md.do_GET()
    _, _, b_md = _parse_response(w_md)
    assert "Fallido" in b_md
    assert "Completado" in b_md


def test_post_project_demo_empty_body_and_workspace_close(tmp_path: Path):
    from automl.application.services.workspace import AutoMLWorkspace

    ws_dir = tmp_path / "ws_empty_body"
    ws_dir.mkdir(parents=True, exist_ok=True)

    # POST with None body must default to quick_train=True without erroring
    h, w = _create_handler(ws_dir, "/api/project/demo", method="POST", body=None)
    h.do_POST()
    status, _, body = _parse_response(w)
    assert status == 200
    res = json.loads(body)
    assert res["status"] == "success"

    # Verify AutoMLWorkspace.close() alias
    ws = AutoMLWorkspace.create("test_ws_close", root_dir=tmp_path / "ws_close")
    s_repo = ws.get_study_repository()
    r_repo = ws.get_recipe_repository()
    assert ws._cached_study_repo is not None
    assert ws._cached_recipe_repo is not None
    ws.close()
    assert ws._cached_study_repo is None
    assert ws._cached_recipe_repo is None

