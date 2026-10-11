"""
Tests for Workbench Single Dataset & Run Context Consistency (PR 1 / Fase P0).

Verifies the Section 3 Acceptance Criteria:
1. Centralized Context: activeDatasetId and activeRunId as single source of truth.
2. If a dataset has no runs, activeRunId must be null (NEVER silently select a run from another dataset).
3. If user selects a run from another dataset, context updates coherently.
4. Static frontend asset guard: no leaks of runs[0] or deceptive fallbacks across views.
5. Multi-dataset server isolation: /api/runs and /api/experiments strictly scoped with no crosstalk.
"""
from __future__ import annotations

import io
import json
import re
from pathlib import Path
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import CreateExperimentCommand
from automl.interfaces.web.server import AutoMLWebHandler


def _create_handler(workspace_dir: Path, path: str, method: str = "GET", body: dict | None = None) -> tuple[AutoMLWebHandler, io.BytesIO]:
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(workspace_dir)
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = io.BytesIO(json.dumps(body).encode("utf-8") if body else b"")
    handler.wfile = io.BytesIO()
    handler.headers = {
        "Host": "127.0.0.1",
        "Content-Length": str(len(json.dumps(body).encode("utf-8"))) if body else "0",
        "Content-Type": "application/json",
    }
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.path = path
    return handler, handler.wfile


def _parse_response_json(wfile: io.BytesIO) -> tuple[int, any]:
    raw = wfile.getvalue().decode("utf-8", errors="ignore")
    status_line = raw.split("\r\n")[0]
    status_code = int(status_line.split()[1])
    body = raw.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in raw else "{}"
    return status_code, json.loads(body)


# ---------------------------------------------------------------------------
# 1. Static Frontend Context Architecture Guards
# ---------------------------------------------------------------------------

def test_frontend_store_exports_centralized_context_methods():
    js_dir = Path(__file__).resolve().parent.parent / "src" / "automl" / "interfaces" / "web" / "static" / "js"
    store_js = (js_dir / "store.js").read_text(encoding="utf-8")

    # Centralized context resolution methods must exist in store.js
    assert "resolveContext" in store_js
    assert "setActiveDataset" in store_js
    assert "setActiveRun" in store_js
    assert "getActiveDataset" in store_js
    assert "getActiveRun" in store_js
    assert "getRunsForActiveDataset" in store_js
    assert "getContextVersion" in store_js
    assert "catml_active_dataset_id" in store_js


def test_no_cross_dataset_runs_zero_fallback_in_views():
    """Views must not blindly fall back to runs[0] when activeRunId is null."""
    views_dir = Path(__file__).resolve().parent.parent / "src" / "automl" / "interfaces" / "web" / "static" / "js" / "views"
    
    files_to_check = [
        "agent.js",
        "compare.js",
        "studio.js",
        "overview.js",
        "home.js",
        "kaggle.js",
        "knowledge.js",
        "pipeline.js",
        "new_experiment.js",
    ]

    for fname in files_to_check:
        content = (views_dir / fname).read_text(encoding="utf-8")
        # Ensure '|| runs[0]' or '|| state.runs[0]' pattern is eliminated
        assert "|| runs[0]" not in content, f"Forbidden fallback '|| runs[0]' detected in {fname}"
        assert "|| state.runs[0]" not in content, f"Forbidden fallback '|| state.runs[0]' detected in {fname}"


def test_agent_drawer_implements_explicit_context_states():
    """AgentDrawer must implement explicit states: no dataset, dataset without runs, active run."""
    views_dir = Path(__file__).resolve().parent.parent / "src" / "automl" / "interfaces" / "web" / "static" / "js" / "views"
    agent_js = (views_dir / "agent.js").read_text(encoding="utf-8")

    assert "Sin dataset seleccionado" in agent_js
    assert "Sin ejecución activa" in agent_js
    assert "Contexto Activo" in agent_js
    assert "Proponer ≠ Aceptar" in agent_js


def test_studio_eliminates_deceptive_overview_fallbacks():
    """Studio must not display global overview metrics when local run has no results."""
    views_dir = Path(__file__).resolve().parent.parent / "src" / "automl" / "interfaces" / "web" / "static" / "js" / "views"
    studio_js = (views_dir / "studio.js").read_text(encoding="utf-8")

    assert "overview.best_score" not in studio_js
    assert "overview.best_model" not in studio_js
    assert "overview.total_trials" not in studio_js
    assert "Sin resultados" in studio_js


# ---------------------------------------------------------------------------
# 2. Multi-Dataset Server Isolation & Consistency Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_dataset_env(tmp_path: Path):
    ws_dir = tmp_path / "multi_ds_ws"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    # Dataset A (Classification)
    df_a = pd.DataFrame({
        "feat_a1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "feat_a2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 1, 0, 1, 0, 1],
    })
    ds_a = ws.register_dataset("dataset_alpha", df_a, target="target")
    run_a = ws.create_run(ds_a, metric="roc_auc")
    exp_a = cmd_bus.dispatch(
        CreateExperimentCommand(
            run_id=run_a.id,
            name="exp_alpha_rf",
            feature_names=["feat_a1", "feat_a2"],
            model_ids=["random_forest"],
        )
    )

    # Dataset B (Regression)
    df_b = pd.DataFrame({
        "feat_b1": [100.0, 200.0, 300.0, 400.0],
        "target_reg": [5.5, 6.2, 7.8, 8.1],
    })
    ds_b = ws.register_dataset("dataset_beta", df_b, target="target_reg")
    run_b = ws.create_run(ds_b, metric="rmse")
    exp_b = cmd_bus.dispatch(
        CreateExperimentCommand(
            run_id=run_b.id,
            name="exp_beta_ridge",
            feature_names=["feat_b1"],
            model_ids=["ridge"],
        )
    )

    # Dataset C (Empty / No runs)
    df_c = pd.DataFrame({
        "col_c": [1, 2, 3],
        "target_c": [0, 1, 0],
    })
    ds_c = ws.register_dataset("dataset_gamma_empty", df_c, target="target_c")

    return {
        "ws_dir": ws_dir,
        "ds_a": ds_a,
        "run_a": run_a,
        "exp_a": exp_a,
        "ds_b": ds_b,
        "run_b": run_b,
        "exp_b": exp_b,
        "ds_c": ds_c,
    }


def test_runs_endpoint_scopes_strictly_by_dataset(multi_dataset_env):
    ws_dir = multi_dataset_env["ws_dir"]
    ds_a = multi_dataset_env["ds_a"]
    ds_b = multi_dataset_env["ds_b"]
    ds_c = multi_dataset_env["ds_c"]
    run_a = multi_dataset_env["run_a"]
    run_b = multi_dataset_env["run_b"]

    # 1. Query all runs
    handler, wfile = _create_handler(ws_dir, "/api/runs")
    handler.do_GET()
    status, runs = _parse_response_json(wfile)
    assert status == 200
    assert len(runs) == 2

    # 2. Query runs strictly for Dataset A
    handler_a, wfile_a = _create_handler(ws_dir, f"/api/runs?dataset_id={ds_a.id}")
    handler_a.do_GET()
    status_a, runs_a = _parse_response_json(wfile_a)
    assert status_a == 200
    assert len(runs_a) == 1
    assert runs_a[0]["id"] == run_a.id
    assert runs_a[0]["dataset_id"] == ds_a.id

    # 3. Query runs strictly for Dataset B
    handler_b, wfile_b = _create_handler(ws_dir, f"/api/runs?dataset_id={ds_b.id}")
    handler_b.do_GET()
    status_b, runs_b = _parse_response_json(wfile_b)
    assert status_b == 200
    assert len(runs_b) == 1
    assert runs_b[0]["id"] == run_b.id
    assert runs_b[0]["dataset_id"] == ds_b.id

    # 4. Query runs for Dataset C (empty, zero runs)
    handler_c, wfile_c = _create_handler(ws_dir, f"/api/runs?dataset_id={ds_c.id}")
    handler_c.do_GET()
    status_c, runs_c = _parse_response_json(wfile_c)
    assert status_c == 200
    assert len(runs_c) == 0, "Empty dataset must return 0 runs, never runs from other datasets!"


def test_experiments_endpoint_isolation_between_runs(multi_dataset_env):
    ws_dir = multi_dataset_env["ws_dir"]
    run_a = multi_dataset_env["run_a"]
    run_b = multi_dataset_env["run_b"]
    exp_a = multi_dataset_env["exp_a"]
    exp_b = multi_dataset_env["exp_b"]

    # Query experiments for Run A
    handler_a, wfile_a = _create_handler(ws_dir, f"/api/experiments?run_id={run_a.id}")
    handler_a.do_GET()
    status_a, exps_a = _parse_response_json(wfile_a)
    assert status_a == 200
    assert len(exps_a) == 1
    assert exps_a[0]["id"] == exp_a.id
    assert exps_a[0]["name"] == "exp_alpha_rf"

    # Query experiments for Run B
    handler_b, wfile_b = _create_handler(ws_dir, f"/api/experiments?run_id={run_b.id}")
    handler_b.do_GET()
    status_b, exps_b = _parse_response_json(wfile_b)
    assert status_b == 200
    assert len(exps_b) == 1
    assert exps_b[0]["id"] == exp_b.id
    assert exps_b[0]["name"] == "exp_beta_ridge"


def test_dataset_questionnaire_isolation_between_datasets(multi_dataset_env):
    ws_dir = multi_dataset_env["ws_dir"]
    ds_a = multi_dataset_env["ds_a"]
    ds_b = multi_dataset_env["ds_b"]

    # Save questionnaire on Dataset A
    payload_a = {
        "dataset_id": ds_a.id,
        "questionnaire": {
            "error_cost": "avoid_false_positives",
            "latency_constraint": "ultra_low_realtime",
            "user_confirmed": True,
        },
    }
    handler_save_a, wfile_save_a = _create_handler(ws_dir, "/api/dataset/questionnaire", method="POST", body=payload_a)
    handler_save_a.do_POST()
    status, res = _parse_response_json(wfile_save_a)
    assert status == 200
    assert res["status"] == "success"

    # Verify GET for Dataset A has user-confirmed values
    h_get_a, w_get_a = _create_handler(ws_dir, f"/api/dataset/questionnaire?dataset_id={ds_a.id}")
    h_get_a.do_GET()
    _, q_a = _parse_response_json(w_get_a)
    assert q_a["error_cost"] == "avoid_false_positives"
    assert q_a["user_confirmed"] is True

    # Verify GET for Dataset B has NOT been overwritten by Dataset A
    h_get_b, w_get_b = _create_handler(ws_dir, f"/api/dataset/questionnaire?dataset_id={ds_b.id}")
    h_get_b.do_GET()
    _, q_b = _parse_response_json(w_get_b)
    assert q_b.get("error_cost") != "avoid_false_positives" or q_b.get("user_confirmed") is not True
