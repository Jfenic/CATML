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
