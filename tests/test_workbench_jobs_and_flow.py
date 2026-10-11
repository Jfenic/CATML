"""
Tests for Workbench PR 3 (Fase P1):
- Guided 4-Stage Workflow: Datos -> Exploración -> Experimentos -> Resultados
- Honest Job and Run Lifecycle State Fidelity (queued, running, pause_requested, cancel_requested, paused, completed, failed, cancelled, interrupted)
- Discrete Floating Jobs Bar (jobs.js) with non-intrusive 0-jobs trigger and clear active vs history separation
- Collapsible technical details in Home hero card
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from automl.interfaces.web.server import AutoMLWebHandler


@pytest.fixture
def running_web_server(tmp_path: Path):
    AutoMLWebHandler.workspace_dir = str(tmp_path)
    AutoMLWebHandler.require_auth = False
    AutoMLWebHandler.auth_token = None

    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    try:
        yield {"base_url": base_url, "tmp_path": tmp_path}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        AutoMLWebHandler.require_auth = False
        AutoMLWebHandler.auth_token = None


def test_utils_job_lifecycle_state_fidelity():
    utils_js = Path("src/automl/interfaces/web/static/js/utils.js").read_text(encoding="utf-8")
    
    # Assert job status exports
    assert "export const ACTIVE_JOB_STATUSES" in utils_js
    assert "export function isJobActive" in utils_js
    assert "export const JOB_STATUS_LABELS" in utils_js
    assert "export const JOB_ACTION_LABELS" in utils_js
    assert "export function getJobStatusLabel" in utils_js

    # Verify all 9 distinct states are mapped in Spanish
    assert 'queued: "En cola"' in utils_js
    assert 'running: "En ejecución"' in utils_js
    assert 'pause_requested: "Pausando..."' in utils_js
    assert 'cancel_requested: "Cancelando..."' in utils_js
    assert 'paused: "Pausado"' in utils_js
    assert 'cancelled: "Cancelado"' in utils_js
    assert 'completed: "Completado"' in utils_js
    assert 'failed: "Fallido"' in utils_js
    assert 'interrupted: "Interrumpido"' in utils_js


def test_jobs_panel_discrete_floating_bar_and_separation():
    jobs_js = Path("src/automl/interfaces/web/static/js/views/jobs.js").read_text(encoding="utf-8")

    # Class and lifecycle
    assert "export class JobsPanel" in jobs_js
    assert "this.timer = setInterval(" in jobs_js
    assert "clearInterval(this.timer)" in jobs_js

    # Discrete trigger when 0 active jobs (does not occupy 360px permanent space)
    assert "_renderDiscreetTrigger" in jobs_js
    assert "btnToggleJobsDrawer" in jobs_js
    assert "Trabajos (0 activos)" in jobs_js

    # Active floating banner when jobs are in progress
    assert "_renderActiveBanner" in jobs_js
    assert "btnMinimizeJobsBar" in jobs_js
    assert "btnExpandJobsModal" in jobs_js

    # Full expanded modal with separate tabs for Active vs History
    assert "_renderExpandedModal" in jobs_js
    assert "tabJobsActive" in jobs_js
    assert "tabJobsHistory" in jobs_js
    assert "_renderActiveList" in jobs_js
    assert "_renderHistoryList" in jobs_js
    assert "En curso" in jobs_js
    assert "Historial" in jobs_js


def test_home_view_guided_workflow_and_honest_execution_states():
    home_js = Path("src/automl/interfaces/web/static/js/views/home.js").read_text(encoding="utf-8")

    # 4-stage guided workflow banner
    assert "_renderGuidedWorkflowBanner" in home_js
    assert "01 DATOS" in home_js
    assert "02 EXPLORACIÓN" in home_js
    assert "03 EXPERIMENTOS" in home_js
    assert "04 RESULTADOS" in home_js
    assert "cardFlowData" in home_js
    assert "cardFlowExplore" in home_js
    assert "cardFlowExp" in home_js
    assert "cardFlowResults" in home_js

    # Honest execution states: distinction between active running vs idle engine
    assert "MOTOR EN REPOSO" in home_js
    assert "ENTRENANDO" in home_js
    assert "LISTO PARA ENTRENAR" in home_js
    assert "isJobActive" in home_js

    # Collapsible technical details in hero card (Section 5.4)
    assert "<details" in home_js
    assert "<summary" in home_js
    assert "Detalles técnicos del entorno de datos" in home_js

    # State-based primary CTAs (Section 5.3)
    assert "btnHomeContinue" in home_js
    assert "btnHomeStartFirstExpInner" in home_js
    assert "btnHomeGoToExperiments" in home_js


def test_index_html_guided_lifecycle_navigation():
    index_html = Path("src/automl/interfaces/web/static/index.html").read_text(encoding="utf-8")

    # Workflow header and 4-step sequence
    assert "Flujo de Trabajo" in index_html
    assert 'data-nav="home"' in index_html
    assert 'data-nav="dataset"' in index_html
    assert 'data-nav="explore"' in index_html
    assert 'data-nav="experiments"' in index_html
    assert 'data-nav="evidence"' in index_html
    assert "00" in index_html
    assert "01" in index_html
    assert "02" in index_html
    assert "03" in index_html
    assert "04" in index_html


def test_app_js_imports_and_job_active_check():
    app_js = Path("src/automl/interfaces/web/static/js/app.js").read_text(encoding="utf-8")

    assert "JobsPanel" in app_js
    assert "isJobActive" in app_js
    assert "isRunActive" in app_js


def test_static_jobs_serving(running_web_server):
    base = running_web_server["base_url"]

    # Verify static jobs view is served
    with urlopen(f"{base}/static/js/views/jobs.js") as resp:
        assert resp.status == 200
        assert "javascript" in resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8")
        assert "JobsPanel" in body
        assert "_renderDiscreetTrigger" in body

    # Verify api/jobs endpoint returns list
    with urlopen(f"{base}/api/jobs") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert isinstance(data, list)
