"""
Tests for Workbench PR 4 (Fase P1):
- Results-focused Leaderboard: Winner & Evaluation Context Hero Card shown first
- Unambiguous trial identification with individual Trial IDs and params
- Honest baseline comparison with metric direction interpretation
- Dedicated secondary tabs for HPO, AutoML Plan, Recipes, Resources and Logs
- 5 explicit transformation lifecycle stages: Propuesta, Verificada, Seleccionada, Aplicada, Promovida
- Unified feature selection contract between datasets.js and new_experiment.js
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


def test_studio_winner_hero_and_evaluation_context():
    studio_js = Path("src/automl/interfaces/web/static/js/views/studio.js").read_text(encoding="utf-8")

    # Winner Hero Card presentation (Section 7.1)
    assert "Modelo Mejor Evaluado en esta Ejecución" in studio_js
    assert "1. Modelo Ganador" in studio_js
    assert "2. Puntuación" in studio_js
    assert "3. Validación" in studio_js
    assert "4. Comparación vs Baseline" in studio_js
    assert "5. Pipeline de Datos" in studio_js
    assert "Mejor de esta ejecución" in studio_js

    # Honest baseline comparison handling
    assert "baselineTrial" in studio_js
    assert "Sin baseline evaluado en esta ejecución" in studio_js
    assert "formatMetricComparison" in studio_js


def test_studio_unambiguous_trial_identification_and_tabs():
    studio_js = Path("src/automl/interfaces/web/static/js/views/studio.js").read_text(encoding="utf-8")

    # Leaderboard table headers and distinct trial fields (Section 7.2)
    assert "Ensayo / Trial ID" in studio_js
    assert "Δ vs Baseline" in studio_js
    assert "Pipeline de Datos / Receta" in studio_js
    assert "btn-inspect-model-params" in studio_js
    assert "data-inspect-trial" in studio_js

    # Dedicated secondary tabs (Section 7.1)
    assert 'data-tab="models"' in studio_js
    assert 'data-tab="hpo"' in studio_js
    assert 'data-tab="planner"' in studio_js
    assert 'data-tab="recipes"' in studio_js
    assert 'data-tab="resources"' in studio_js
    assert 'data-tab="logs"' in studio_js

    # Secondary tab rendering methods
    assert "_renderLeaderboardTabHtml" in studio_js
    assert "_renderHpoTabHtml" in studio_js
    assert "_renderPlannerTabHtml" in studio_js
    assert "_renderRecipesTabHtml" in studio_js
    assert "_renderResourcesTabHtml" in studio_js
    assert "_renderLogsTabHtml" in studio_js


def test_transformation_lifecycle_stages_definition_and_explicability():
    utils_js = Path("src/automl/interfaces/web/static/js/utils.js").read_text(encoding="utf-8")
    explore_js = Path("src/automl/interfaces/web/static/js/views/explore.js").read_text(encoding="utf-8")
    studio_js = Path("src/automl/interfaces/web/static/js/views/studio.js").read_text(encoding="utf-8")

    # All 5 explicit stages defined in utils.js (Section 8.1)
    assert "export const TRANSFORMATION_LIFECYCLE_STAGES" in utils_js
    assert "export function getTransformationStageInfo" in utils_js
    assert 'label: "Propuesta"' in utils_js
    assert 'label: "Verificada"' in utils_js
    assert 'label: "Seleccionada"' in utils_js
    assert 'label: "Aplicada"' in utils_js
    assert 'label: "Promovida"' in utils_js

    # Explore and Studio consume getTransformationStageInfo
    assert "getTransformationStageInfo" in explore_js
    assert "getTransformationStageInfo" in studio_js
    assert "Estado del Ciclo" in studio_js


def test_feature_selection_contract_unification():
    from workbench_static_sources import datasets_view_sources

    datasets_js = datasets_view_sources()
    new_exp_js = Path("src/automl/interfaces/web/static/js/views/new_experiment.js").read_text(encoding="utf-8")

    # Feature selector synchronizes both customFeatures and selectedFeaturesByDataset (Section 8.3)
    assert (
        "selectedFeaturesByDataset: { ...currentMap, [this.activeDatasetId]: feats }" in datasets_js
        or "selectedFeaturesByDataset: { ...currentMap, [view.activeDatasetId]: feats }" in datasets_js
    )
    assert "customFeatures: feats" in datasets_js
    assert "selectedFeatures: feats" in datasets_js

    # new_experiment.js resolves features and forwards feature_names
    assert "state.selectedFeaturesByDataset" in new_exp_js
    assert "state.customFeatures" in new_exp_js
    assert "state.selectedFeatures" in new_exp_js
    assert "feature_names: this.customFeatures" in new_exp_js


def test_static_studio_serving_and_winner_elements(running_web_server):
    base = running_web_server["base_url"]

    with urlopen(f"{base}/static/js/views/studio.js") as resp:
        assert resp.status == 200
        assert "javascript" in resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8")
        assert "Modelo Mejor Evaluado en esta Ejecución" in body
        assert "_renderLeaderboardTabHtml" in body
