"""Clean Architecture PR3: web API controllers + DatasetsView modular split."""
from __future__ import annotations

from pathlib import Path

from workbench_static_sources import datasets_view_sources


def test_api_controllers_package_exists():
    api = Path("src/automl/interfaces/web/api")
    assert (api / "__init__.py").is_file()
    assert (api / "_helpers.py").is_file()
    for name in ("jobs", "datasets", "experiments", "analysis"):
        assert (api / f"{name}.py").is_file()
        mod = __import__(f"automl.interfaces.web.api.{name}", fromlist=["try_get", "try_post"])
        assert callable(mod.try_get)
        assert callable(mod.try_post)


def test_server_is_thin_http_glue():
    server = Path("src/automl/interfaces/web/server.py").read_text(encoding="utf-8")
    assert "from automl.interfaces.web.api import analysis, datasets, experiments, jobs" in server
    assert "jobs.try_get(" in server
    assert "datasets.try_post(" in server
    # Route bodies should not remain inlined as giant elif chains
    assert server.count('elif path == "/api/') < 3
    assert len(server.splitlines()) < 400


def test_datasets_view_modules_match_plan():
    root = Path("src/automl/interfaces/web/static/js/views")
    assert (root / "datasets.js").is_file()
    barrel = (root / "datasets.js").read_text(encoding="utf-8")
    assert 'from "./datasets/datasets_view.js"' in barrel

    ds = root / "datasets"
    expected = {
        "datasets_view.js",
        "dataset_summary.js",
        "dataset_columns.js",
        "data_quality_panel.js",
        "feature_selector.js",
        "correlations_view.js",
        "questionnaire.js",
        "derived_features.js",
    }
    assert expected.issubset({p.name for p in ds.glob("*.js")})

    sources = datasets_view_sources()
    assert "export class DatasetsView" in sources
    assert "_renderHealthCard" in sources
    assert "_renderResumenTab" in sources
    assert "_updateSelectionUi" in sources
    assert "escapeHtml" in sources
