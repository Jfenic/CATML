from __future__ import annotations

import json
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer
from urllib.request import urlopen

import pytest

from automl import __version__
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


def test_workbench_index_html_version_and_navigation():
    index_path = Path("src/automl/interfaces/web/static/index.html")
    assert index_path.exists()
    content = index_path.read_text(encoding="utf-8")

    # Dynamic version IDs present
    assert 'id="sidebarVersionLabel"' in content
    assert 'id="sidebarAgentBadge"' in content
    assert f"v{__version__}" in content

    # All primary 5 views present in sidebar
    assert 'data-nav="home"' in content
    assert 'data-nav="dataset"' in content
    assert 'data-nav="explore"' in content
    assert 'data-nav="experiments"' in content
    assert 'data-nav="evidence"' in content


def test_workbench_datasets_view_explore_linkage():
    datasets_js = Path("src/automl/interfaces/web/static/js/views/datasets.js")
    assert datasets_js.exists()
    content = datasets_js.read_text(encoding="utf-8")

    # Has full canvas exploration shortcut button
    assert 'id="btnOpenFullExploreCanvas"' in content
    assert 'Lienzo Completo' in content
    assert 'store.setNav("explore")' in content


def test_workbench_app_router_canonical_nav():
    app_js = Path("src/automl/interfaces/web/static/js/app.js")
    assert app_js.exists()
    content = app_js.read_text(encoding="utf-8")

    # Canonical navigation aliases
    assert 'canonicalNav' in content
    assert 'sidebarVersionLabel' in content
    assert 'sidebarAgentBadge' in content
    assert 'ExploreView' in content


def test_workbench_static_assets_serving(running_web_server):
    base = running_web_server["base_url"]

    # Verify static explore view is served
    with urlopen(f"{base}/static/js/views/explore.js") as resp:
        assert resp.status == 200
        assert "javascript" in resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8")
        assert "ExploreView" in body

    # Verify static datasets view is served
    with urlopen(f"{base}/static/js/views/datasets.js") as resp:
        assert resp.status == 200
        assert "javascript" in resp.headers.get("Content-Type", "")
        body = resp.read().decode("utf-8")
        assert "btnOpenFullExploreCanvas" in body

    # Verify API overview returns version matching __version__
    with urlopen(f"{base}/api/overview") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("version") == __version__
