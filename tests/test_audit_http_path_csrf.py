"""Audit remediations: /api/predict path confinement + mutable HTTP origin guards."""
from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from automl.interfaces.web import server as web_server
from automl.interfaces.web.server import AutoMLWebHandler, _is_loopback_origin


def _handler(*, workspace_dir: Path, path: str = "/api/predict") -> AutoMLWebHandler:
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(workspace_dir)
    handler.require_auth = False
    handler.auth_token = None
    handler.path = path
    return handler


def test_is_loopback_origin():
    assert _is_loopback_origin("http://127.0.0.1:8080")
    assert _is_loopback_origin("http://localhost:3000")
    assert _is_loopback_origin("https://localhost")
    assert not _is_loopback_origin("https://evil.example")
    assert not _is_loopback_origin("null")


def test_post_rejects_non_json_content_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        web_server,
        "build_application",
        lambda root_dir: (MagicMock(), MagicMock(), MagicMock()),
    )
    handler = _handler(workspace_dir=tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    body = b'{"run_id":"r"}'
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    sent: dict = {}

    def mock_send_json(data, status=200):
        sent["data"] = data
        sent["status"] = status

    handler._send_json = mock_send_json
    handler.do_POST()
    assert sent["status"] == HTTPStatus.UNSUPPORTED_MEDIA_TYPE


def test_post_rejects_cross_site_sec_fetch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        web_server,
        "build_application",
        lambda root_dir: (MagicMock(), MagicMock(), MagicMock()),
    )
    handler = _handler(workspace_dir=tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    body = b"{}"
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
        "Sec-Fetch-Site": "cross-site",
        "Origin": "https://evil.example",
    }
    sent: dict = {}

    def mock_send_json(data, status=200):
        sent["data"] = data
        sent["status"] = status

    handler._send_json = mock_send_json
    handler.do_POST()
    assert sent["status"] == HTTPStatus.FORBIDDEN
    assert "Cross-site" in sent["data"].get("message", "")


def test_post_rejects_non_loopback_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        web_server,
        "build_application",
        lambda root_dir: (MagicMock(), MagicMock(), MagicMock()),
    )
    handler = _handler(workspace_dir=tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    body = b"{}"
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
        "Origin": "https://evil.example",
        "Host": "127.0.0.1:8080",
    }
    sent: dict = {}

    def mock_send_json(data, status=200):
        sent["data"] = data
        sent["status"] = status

    handler._send_json = mock_send_json
    handler.do_POST()
    assert sent["status"] == HTTPStatus.FORBIDDEN
    assert "Origin" in sent["data"].get("message", "")


def test_post_allows_loopback_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cmd = MagicMock()
    qry = MagicMock()
    monkeypatch.setattr(
        web_server,
        "build_application",
        lambda root_dir: (MagicMock(), cmd, qry),
    )
    handler = _handler(workspace_dir=tmp_path / "ws", path="/api/unknown")
    (tmp_path / "ws").mkdir()
    body = b"{}"
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
        "Origin": "http://127.0.0.1:8080",
        "Host": "127.0.0.1:8080",
    }
    sent: dict = {}

    def mock_send_json(data, status=200):
        sent["data"] = data
        sent["status"] = status

    handler._send_json = mock_send_json
    handler.do_POST()
    assert sent["status"] == HTTPStatus.NOT_FOUND


def test_predict_rejects_output_path_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from automl.interfaces.web.api import experiments as experiments_api

    monkeypatch.chdir(tmp_path)
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    inside = tmp_path / "test.csv"
    inside.write_text("a,b\n1,2\n")
    outside = Path("/tmp") / f"catml_audit_escape_{tmp_path.name}.csv"

    handler = _handler(workspace_dir=ws_dir)
    sent: dict = {}

    def mock_send_json(data, status=200):
        sent["data"] = data
        sent["status"] = status

    handler._send_json = mock_send_json
    cmd = MagicMock()
    qry = MagicMock()
    handled = experiments_api.try_post(
        handler,
        "/api/predict",
        {
            "run_id": "run_1",
            "test_dataset_path": str(inside),
            "output_path": str(outside),
        },
        MagicMock(),
        cmd,
        qry,
    )
    assert handled is True
    assert sent["status"] == HTTPStatus.FORBIDDEN
    assert "Access denied" in sent["data"].get("error", "")
    cmd.dispatch.assert_not_called()


def test_predict_rejects_test_path_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from automl.interfaces.web.api import experiments as experiments_api

    monkeypatch.chdir(tmp_path)
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    outside = Path("/etc/passwd")

    handler = _handler(workspace_dir=ws_dir)
    sent: dict = {}

    def mock_send_json(data, status=200):
        sent["data"] = data
        sent["status"] = status

    handler._send_json = mock_send_json
    cmd = MagicMock()
    handled = experiments_api.try_post(
        handler,
        "/api/predict",
        {
            "run_id": "run_1",
            "test_dataset_path": str(outside),
            "output_path": "submission.csv",
        },
        MagicMock(),
        cmd,
        MagicMock(),
    )
    assert handled is True
    assert sent["status"] == HTTPStatus.FORBIDDEN
    cmd.dispatch.assert_not_called()


def test_predict_relative_output_confined_to_workspace_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from automl.interfaces.web.api import experiments as experiments_api

    monkeypatch.chdir(tmp_path)
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("id,a\n1,2\n")

    handler = _handler(workspace_dir=ws_dir)
    sent: dict = {}

    def mock_send_json(data, status=200):
        sent["data"] = data
        sent["status"] = status

    handler._send_json = mock_send_json
    cmd = MagicMock()
    cmd.dispatch.return_value = {
        "output_path": str(ws_dir / "exports" / "submission.csv"),
        "row_count": 1,
        "id_column": "id",
        "target_column": "target",
    }
    handled = experiments_api.try_post(
        handler,
        "/api/predict",
        {
            "run_id": "run_1",
            "test_dataset_path": str(test_csv),
            "output_path": "submissions/submission.csv",
        },
        MagicMock(),
        cmd,
        MagicMock(),
    )
    assert handled is True
    assert sent.get("status", HTTPStatus.OK) == HTTPStatus.OK
    assert cmd.dispatch.called
    call_cmd = cmd.dispatch.call_args[0][0]
    out = Path(call_cmd.output_path)
    assert out.is_relative_to(ws_dir.resolve())
    assert out.parent.name == "exports"
    assert out.name == "submission.csv"


def test_static_woff2_mime_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    static_dir = tmp_path / "static"
    fonts = static_dir / "fonts"
    fonts.mkdir(parents=True)
    font_file = fonts / "Geist.woff2"
    font_file.write_bytes(b"w2")

    handler = _handler(workspace_dir=ws_dir)
    handler.static_dir = static_dir
    handler.path = "/static/fonts/Geist.woff2"
    handler.headers = {}
    recorded: dict = {}

    def send_response(code):
        recorded["code"] = code

    def send_header(key, value):
        recorded.setdefault("headers", {})[key] = value

    def end_headers():
        recorded["ended"] = True

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = end_headers
    handler.wfile = MagicMock()
    handler._require_auth_or_reject = lambda: True
    handler.do_GET()
    assert recorded.get("code") == HTTPStatus.OK
    assert recorded["headers"]["Content-Type"] == "font/woff2"
