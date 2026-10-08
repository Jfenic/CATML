"""
Security tests for Workbench file confinement and path sanitization (PR B).

Verifies strict confinement and traversal protection across:
1. /api/kaggle/download
2. /api/media/preview
3. /static/
"""
from __future__ import annotations

import io
import json
import os
from http import HTTPStatus
from pathlib import Path
import pytest

from automl.application.bootstrap import build_application
from automl.interfaces.web.server import AutoMLWebHandler


class DummyRequest:
    def makefile(self, *args, **kwargs):
        return io.BytesIO(b"")

    def sendall(self, *args, **kwargs):
        pass


def _create_handler(workspace_dir: Path, path: str) -> tuple[AutoMLWebHandler, io.BytesIO]:
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(workspace_dir)
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.headers = {"Host": "127.0.0.1"}
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.path = path
    return handler, handler.wfile


def test_kaggle_download_confinement(tmp_path: Path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    submissions_dir = ws_dir / "submissions"
    submissions_dir.mkdir()

    # 1. Valid submission file inside submissions/
    valid_sub = submissions_dir / "valid_submission.csv"
    valid_sub.write_text("id,pred\n1,0.95\n2,0.12\n", encoding="utf-8")

    # Outside secret file
    outside_dir = tmp_path / "outside_forbidden"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret_leak.csv"
    outside_file.write_text("leaked_secret_data", encoding="utf-8")

    # A. Valid download with relative path inside submissions
    handler, wfile = _create_handler(ws_dir, "/api/kaggle/download?file=valid_submission.csv")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "200 OK" in resp
    assert "id,pred" in resp
    assert "Content-Disposition" in resp

    # B. Valid download with absolute path inside workspace
    handler, wfile = _create_handler(ws_dir, f"/api/kaggle/download?file={valid_sub.resolve()}")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "200 OK" in resp
    assert "id,pred" in resp

    # C. Path traversal attempt with relative ../../
    handler, wfile = _create_handler(ws_dir, f"/api/kaggle/download?file=../../outside_forbidden/secret_leak.csv")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp
    assert "leaked_secret_data" not in resp

    # D. Absolute path outside workspace
    handler, wfile = _create_handler(ws_dir, f"/api/kaggle/download?file={outside_file.resolve()}")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp
    assert "leaked_secret_data" not in resp

    # E. Symlink pointing outside workspace
    symlink_file = submissions_dir / "symlink_leak.csv"
    try:
        os.symlink(str(outside_file), str(symlink_file))
        handler, wfile = _create_handler(ws_dir, f"/api/kaggle/download?file={symlink_file.name}")
        handler.do_GET()
        resp = wfile.getvalue().decode("utf-8", errors="ignore")
        assert "403 Forbidden" in resp or "Access denied" in resp
        assert "leaked_secret_data" not in resp
    except OSError:
        pass  # Skip symlink test if OS lacks symlink privilege

    # F. Non-whitelisted file extension inside workspace (e.g. database or script)
    db_file = ws_dir / "database.db"
    db_file.write_text("sqlite format 3", encoding="utf-8")
    handler, wfile = _create_handler(ws_dir, f"/api/kaggle/download?file=database.db")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp

    # G. Missing file param
    handler, wfile = _create_handler(ws_dir, "/api/kaggle/download")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "400 Bad Request" in resp

    # H. Non-existent file inside workspace
    handler, wfile = _create_handler(ws_dir, "/api/kaggle/download?file=nonexistent.csv")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "404 Not Found" in resp

    # I. Valid CSV file directly in workspace root (outside submissions/) must be rejected
    root_csv = ws_dir / "root_data.csv"
    root_csv.write_text("sensitive_data,123\n", encoding="utf-8")
    handler, wfile = _create_handler(ws_dir, f"/api/kaggle/download?file={root_csv.resolve()}")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp
    assert "sensitive_data" not in resp

    # J. Relative traversal pointing into workspace root from submissions must be rejected
    handler, wfile = _create_handler(ws_dir, "/api/kaggle/download?file=../root_data.csv")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp
    assert "sensitive_data" not in resp


def test_media_preview_confinement_and_parent_exclusion(tmp_path: Path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()

    # Image inside workspace
    inside_img = ws_dir / "inside.png"
    inside_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 30)

    # Image in parent directory of workspace (tmp_path)
    parent_img = tmp_path / "parent_secret.png"
    parent_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 30)

    # Image outside parent
    outside_dir = tmp_path.parent / "completely_outside"
    outside_dir.mkdir(exist_ok=True)
    outside_img = outside_dir / "external_secret.png"
    outside_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 30)

    # A. Serving image inside workspace must succeed
    handler, wfile = _create_handler(ws_dir, "/api/media/preview?path=inside.png")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "200 OK" in resp
    assert "image/png" in resp

    # B. Serving image in ws_root.parent must be FORBIDDEN (verifying audit finding fix)
    handler, wfile = _create_handler(ws_dir, f"/api/media/preview?path={parent_img.resolve()}")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp

    # C. Serving image completely outside must be FORBIDDEN
    handler, wfile = _create_handler(ws_dir, f"/api/media/preview?path={outside_img.resolve()}")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp

    # D. Symlink in workspace pointing to outside image must be FORBIDDEN
    symlink_img = ws_dir / "symlink.png"
    try:
        os.symlink(str(parent_img), str(symlink_img))
        handler, wfile = _create_handler(ws_dir, "/api/media/preview?path=symlink.png")
        handler.do_GET()
        resp = wfile.getvalue().decode("utf-8", errors="ignore")
        assert "403 Forbidden" in resp or "Access denied" in resp or "Image file not found" in resp
    except OSError:
        pass


def test_media_preview_registered_dataset_directory(tmp_path: Path):
    ws_dir = tmp_path / "ws_dataset_test"
    ws_dir.mkdir()

    # External dataset directory
    ds_dir = tmp_path / "external_dataset"
    ds_dir.mkdir()
    ds_img = ds_dir / "photo.jpg"
    ds_img.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    ds_csv = ds_dir / "dataset.csv"
    ds_csv.write_text(f"id,img_path,target\n1,{ds_img},0\n", encoding="utf-8")

    ws, _, _ = build_application(root_dir=str(ws_dir))
    dataset = ws.register_dataset(name="External Vision DS", path=ds_csv, target="target")

    # A. Serving image from registered dataset directory must succeed
    handler, wfile = _create_handler(ws_dir, f"/api/media/preview?path={ds_img.resolve()}&dataset_id={dataset.id}")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "200 OK" in resp
    assert "image/jpeg" in resp

    # B. Relative path relative to registered dataset
    handler, wfile = _create_handler(ws_dir, f"/api/media/preview?path=photo.jpg&dataset_id={dataset.id}")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "200 OK" in resp
    assert "image/jpeg" in resp


def test_static_files_directory_confinement(tmp_path: Path):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()

    # A. Path traversal attempt using /static/../../server.py must return 403 Forbidden
    handler, wfile = _create_handler(ws_dir, "/static/../../server.py")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp

    # B. Path traversal attempt using /static/../../../etc/passwd must return 403 Forbidden
    handler, wfile = _create_handler(ws_dir, "/static/../../../etc/passwd")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403 Forbidden" in resp or "Access denied" in resp

    # C. Valid static file index.html must return 200 OK
    handler, wfile = _create_handler(ws_dir, "/static/index.html")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "200 OK" in resp

    # D. Non-existent file inside static dir must return 404 Not Found
    handler, wfile = _create_handler(ws_dir, "/static/nonexistent_file_12345.js")
    handler.do_GET()
    resp = wfile.getvalue().decode("utf-8", errors="ignore")
    assert "404 Not Found" in resp
