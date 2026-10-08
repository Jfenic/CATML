from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from automl.interfaces.web.server import AutoMLWebHandler


@pytest.fixture
def vision_web_server(tmp_path: Path):
    # Create sample image files
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    sample_png = img_dir / "sample_1.png"
    sample_png.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4")

    sample_jpg = img_dir / "sample_2.jpg"
    sample_jpg.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")

    # Create dataset referencing images and text
    n = 10
    df = pd.DataFrame({
        "id": range(n),
        "img_path": [str(sample_png) if i % 2 == 0 else str(sample_jpg) for i in range(n)],
        "rel_img": ["images/sample_1.png" if i % 2 == 0 else "images/sample_2.jpg" for i in range(n)],
        "notes": [f"Patient note description for case number {i}" for i in range(n)],
        "value": np.random.randn(n),
        "target": [0, 1] * (n // 2),
    })
    csv_path = tmp_path / "vision_dataset.csv"
    df.to_csv(csv_path, index=False)

    workspace_dir = str(tmp_path / "vision_workspace")
    AutoMLWebHandler.workspace_dir = workspace_dir

    from automl.application.bootstrap import build_application
    ws, _, _ = build_application(root_dir=workspace_dir)
    ws.register_dataset(name="Vision Fixture Dataset", path=csv_path, target="target")

    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    yield {
        "base_url": base_url,
        "csv_path": str(csv_path),
        "sample_png": sample_png,
        "sample_jpg": sample_jpg,
        "tmp_path": tmp_path,
    }

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_media_preview_endpoint_success(vision_web_server):
    base_url = vision_web_server["base_url"]
    png_path = str(vision_web_server["sample_png"])

    # Test PNG preview
    url = f"{base_url}/api/media/preview?path={quote(png_path)}"
    with urlopen(url) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/png"
        payload = resp.read()
        assert payload.startswith(b"\x89PNG")

    # Test JPG preview
    jpg_path = str(vision_web_server["sample_jpg"])
    url_jpg = f"{base_url}/api/media/preview?path={quote(jpg_path)}"
    with urlopen(url_jpg) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/jpeg"
        payload_jpg = resp.read()
        assert payload_jpg.startswith(b"\xff\xd8")


def test_media_preview_validation_and_errors(vision_web_server):
    base_url = vision_web_server["base_url"]

    # 1. Missing path parameter -> 400 Bad Request
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{base_url}/api/media/preview")
    assert exc_info.value.code == 400

    # 2. Unsupported extension (.txt, .py) -> 400 Bad Request
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{base_url}/api/media/preview?path={quote('script.py')}")
    assert exc_info.value.code == 400

    # 3. Non-existent image file -> 404 Not Found
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{base_url}/api/media/preview?path={quote('/nonexistent/image.png')}")
    assert exc_info.value.code == 404

    # 4. Traversal attempt to system file -> 400 or 403 or 404
    with pytest.raises(HTTPError) as exc_info:
        urlopen(f"{base_url}/api/media/preview?path={quote('/etc/passwd')}")
    assert exc_info.value.code in (400, 403, 404)


def test_media_preview_relative_to_dataset(vision_web_server):
    base_url = vision_web_server["base_url"]
    csv_path = vision_web_server["csv_path"]

    # Register dataset first
    reg_url = f"{base_url}/api/dataset/register"
    req = Request(
        reg_url,
        data=json.dumps({"name": "Vision Dataset", "path": csv_path, "target_column": "target"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        reg_data = json.loads(resp.read().decode("utf-8"))
        dataset_id = reg_data["dataset_id"]

    # Test resolving relative path via dataset_id
    rel_path = "images/sample_1.png"
    url = f"{base_url}/api/media/preview?path={quote(rel_path)}&dataset_id={quote(dataset_id)}"
    with urlopen(url) as resp:
        assert resp.status == 200
        assert resp.headers.get("Content-Type") == "image/png"
        payload = resp.read()
        assert payload.startswith(b"\x89PNG")


def test_dataset_profile_action_enrichment_for_images_and_text(vision_web_server):
    base_url = vision_web_server["base_url"]
    csv_path = vision_web_server["csv_path"]

    # Register dataset
    reg_url = f"{base_url}/api/dataset/register"
    req = Request(
        reg_url,
        data=json.dumps({"name": "Vision Profiling Test", "path": csv_path, "target_column": "target"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        dataset_id = json.loads(resp.read().decode("utf-8"))["dataset_id"]

    # Get dataset profile
    profile_url = f"{base_url}/api/dataset/profile?dataset_id={quote(dataset_id)}"
    with urlopen(profile_url) as resp:
        assert resp.status == 200
        p = json.loads(resp.read().decode("utf-8"))

        columns = {c["name"]: c for c in p["columns"]}

        # Check img_path column
        assert "img_path" in columns
        img_col = columns["img_path"]
        assert img_col.get("is_image") is True
        assert img_col.get("catml_action") == "Vision Embedding"
        assert "timm" in img_col.get("action_reason", "")

        # Check notes column (text)
        assert "notes" in columns
        notes_col = columns["notes"]
        if notes_col.get("is_text"):
            assert notes_col.get("catml_action") == "NLP Tokenize & TF-IDF"
