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
    assert "cero fuga de datos" in body

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
    assert data["validation"]["anti_leakage_guaranteed"] is True


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
