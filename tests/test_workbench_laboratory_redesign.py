from __future__ import annotations

import re
from pathlib import Path
import pytest


def test_index_html_laboratory_sidebar_and_navigation():
    index_path = Path("src/automl/interfaces/web/static/index.html")
    assert index_path.exists(), "index.html must exist"
    content = index_path.read_text(encoding="utf-8")

    # 1. Laboratorio local header & active dataset widget
    assert "Laboratorio local" in content
    assert 'id="sidebarActiveDatasetName"' in content
    assert 'id="btnSidebarSwitchDataset"' in content

    # 2. Simplified 4-view primary navigation
    assert 'data-nav="home"' in content
    assert 'data-nav="dataset"' in content
    assert 'data-nav="experiments"' in content
    assert 'data-nav="evidence"' in content

    # 3. Spanish technical labels
    assert "Inicio" in content
    assert "Dataset" in content
    assert "Experimentos" in content
    assert "Evidencia" in content


def test_frontend_laboratory_modules_exist_and_integrate():
    views_dir = Path("src/automl/interfaces/web/static/js/views")
    home_js = views_dir / "home.js"
    switch_modal_js = views_dir / "switch_dataset_modal.js"
    datasets_js = views_dir / "datasets.js"
    store_js = Path("src/automl/interfaces/web/static/js/store.js")
    app_js = Path("src/automl/interfaces/web/static/js/app.js")

    assert home_js.exists(), "home.js view must exist"
    assert switch_modal_js.exists(), "switch_dataset_modal.js must exist"
    assert datasets_js.exists(), "datasets.js must exist"
    assert store_js.exists(), "store.js must exist"
    assert app_js.exists(), "app.js must exist"

    # Store active dataset management
    store_content = store_js.read_text(encoding="utf-8")
    assert "setActiveDataset" in store_content
    assert "getActiveDataset" in store_content
    assert "activeDatasetId" in store_content
    assert "activeDataset" in store_content

    # Home view implementation
    home_content = home_js.read_text(encoding="utf-8")
    assert "class HomeView" in home_content
    assert "_renderActiveDatasetHero" in home_content
    assert "btnHomeContinue" in home_content
    assert "btnHomeAddDataset" in home_content
    assert "btnHomeOpenDataset" in home_content
    assert "Ejecuciones en curso" in home_content
    assert "Trabajos recientes" in home_content

    # Switch dataset modal implementation
    modal_content = switch_modal_js.read_text(encoding="utf-8")
    assert "class SwitchDatasetModal" in modal_content
    assert "Cambiar dataset activo" in modal_content
    assert "btn-select-ds" in modal_content

    # DatasetsView Resumen, Columnas and Exploracion tabs
    ds_content = datasets_js.read_text(encoding="utf-8")
    assert "_renderResumenTab" in ds_content
    assert "_renderColumnasTab" in ds_content
    assert "_renderHealthCard" in ds_content
    assert "_renderExplorationTab" in ds_content
    assert "btnNewExperimentFromDS" in ds_content
    assert "Anti-Leakage Guardian" in ds_content


def test_app_js_routes_home_and_laboratory_views():
    app_js = Path("src/automl/interfaces/web/static/js/app.js")
    content = app_js.read_text(encoding="utf-8")

    assert 'import { HomeView } from "./views/home.js";' in content
    assert 'import { SwitchDatasetModal } from "./views/switch_dataset_modal.js";' in content
    assert 'case "home":' in content
    assert 'new HomeView()' in content
    assert 'case "dataset":' in content
    assert 'case "experiments":' in content
    assert 'case "evidence":' in content
    assert "_updateSidebarActiveDataset" in content


def test_javascript_syntax_validity():
    import shutil
    import subprocess

    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js not installed on system")

    js_dir = Path("src/automl/interfaces/web/static/js")
    js_files = list(js_dir.glob("*.js")) + list((js_dir / "views").glob("*.js"))
    assert len(js_files) > 0

    for js_file in js_files:
        result = subprocess.run([node_bin, "--check", str(js_file)], capture_output=True, text=True)
        assert result.returncode == 0, f"Syntax error in {js_file}:\n{result.stderr}"


def test_javascript_views_render_without_reference_errors():
    import shutil
    import subprocess

    node_bin = shutil.which("node")
    if not node_bin:
        pytest.skip("Node.js not installed on system")

    script = """
globalThis.window = { addEventListener: () => {}, location: { search: '', hash: '', pathname: '' }, history: { replaceState: () => {} } };
globalThis.document = { getElementById: () => null, querySelectorAll: () => [] };
globalThis.localStorage = { getItem: () => null, setItem: () => {} };
globalThis.sessionStorage = { getItem: () => null, setItem: () => {} };

const views = [
  { name: 'HomeView', path: './src/automl/interfaces/web/static/js/views/home.js' },
  { name: 'OverviewView', path: './src/automl/interfaces/web/static/js/views/overview.js' },
  { name: 'StudioView', path: './src/automl/interfaces/web/static/js/views/studio.js' },
  { name: 'CompareView', path: './src/automl/interfaces/web/static/js/views/compare.js' },
  { name: 'PipelineView', path: './src/automl/interfaces/web/static/js/views/pipeline.js' },
  { name: 'KnowledgeView', path: './src/automl/interfaces/web/static/js/views/knowledge.js' },
  { name: 'KaggleView', path: './src/automl/interfaces/web/static/js/views/kaggle.js' },
  { name: 'DatasetsView', path: './src/automl/interfaces/web/static/js/views/datasets.js' },
  { name: 'RegisterDatasetModal', path: './src/automl/interfaces/web/static/js/views/register_dataset_modal.js' },
];

async function run() {
  for (const v of views) {
    const mod = await import(v.path);
    const Cls = mod[v.name];
    const inst = new Cls();
    inst.container = { innerHTML: '', querySelector: () => null, querySelectorAll: () => [] };
    if (typeof inst.render === 'function') {
      inst.render();
    }
  }
}
run();
"""
    result = subprocess.run([node_bin, "-e", script], capture_output=True, text=True)
    assert result.returncode == 0, f"View render runtime error:\n{result.stderr}"


def test_file_browse_and_inspect_api_endpoints(tmp_path):
    import json
    import base64
    from unittest.mock import MagicMock
    from automl.interfaces.web.server import AutoMLWebHandler

    # Mock handler
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(tmp_path)
    handler.require_auth = False
    handler.auth_token = None
    handler.headers = {}

    sent_data = {}
    sent_status = []

    def mock_send_json(data, status=200):
        sent_data.clear()
        sent_data.update(data if isinstance(data, dict) else {"items": data})
        sent_status.append(status)

    handler._send_json = mock_send_json

    # Create dummy data file in tmp_path
    sample_file = tmp_path / "sample.csv"
    sample_file.write_text("a,b,target\n1,2,0\n3,4,1\n")

    # 1. Inspect file
    handler.path = "/api/dataset/inspect-file"
    handler.rfile = MagicMock()
    body = json.dumps({"path": str(sample_file)}).encode("utf-8")
    handler.rfile.read.return_value = body
    handler.headers = {"Content-Length": str(len(body))}
    handler.do_POST()

    assert sent_data.get("status") == "success"
    assert sent_data.get("columns") == ["a", "b", "target"]
    assert sent_data.get("suggested_target") == "target"

    # 2. Upload file via base64
    handler.path = "/api/dataset/upload"
    b64_content = base64.b64encode(b"feature1,feature2,churn\n10,20,yes\n").decode("utf-8")
    body = json.dumps({"filename": "customer_churn.csv", "content_base64": b64_content}).encode("utf-8")
    handler.rfile.read.return_value = body
    handler.headers = {"Content-Length": str(len(body))}
    handler.do_POST()

    assert sent_data.get("status") == "success"
    assert sent_data.get("columns") == ["feature1", "feature2", "churn"]
    assert sent_data.get("suggested_target") == "churn"



