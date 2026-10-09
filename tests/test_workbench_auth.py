from __future__ import annotations

import json
from http import HTTPStatus
from pathlib import Path
import threading
import urllib.request
import urllib.error
import pytest

from automl.interfaces.web.server import AutoMLWebHandler, run_web_dashboard


def test_workbench_auth_local_requires_no_token():
    # Verify localhost handler default configuration
    AutoMLWebHandler.require_auth = False
    AutoMLWebHandler.auth_token = None

    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.headers = {}
    handler.path = "/api/overview"
    assert handler._is_authenticated() is True


def test_workbench_auth_remote_rejects_unauthenticated():
    AutoMLWebHandler.require_auth = True
    AutoMLWebHandler.auth_token = "secret_workbench_key_123"

    try:
        # Request without token
        handler_no_auth = AutoMLWebHandler.__new__(AutoMLWebHandler)
        handler_no_auth.headers = {}
        handler_no_auth.path = "/api/overview"
        assert handler_no_auth._is_authenticated() is False

        # Request with Bearer header
        handler_bearer = AutoMLWebHandler.__new__(AutoMLWebHandler)
        handler_bearer.headers = {"Authorization": "Bearer secret_workbench_key_123"}
        handler_bearer.path = "/api/overview"
        assert handler_bearer._is_authenticated() is True

        # Request with incorrect Bearer token
        handler_wrong = AutoMLWebHandler.__new__(AutoMLWebHandler)
        handler_wrong.headers = {"Authorization": "Bearer invalid_key"}
        handler_wrong.path = "/api/overview"
        assert handler_wrong._is_authenticated() is False

        # Request with URL query token
        handler_query = AutoMLWebHandler.__new__(AutoMLWebHandler)
        handler_query.headers = {}
        handler_query.path = "/api/overview?token=secret_workbench_key_123"
        assert handler_query._is_authenticated() is True

    finally:
        AutoMLWebHandler.require_auth = False
        AutoMLWebHandler.auth_token = None


def test_workbench_auth_insecure_no_auth_flag():
    AutoMLWebHandler.require_auth = False
    AutoMLWebHandler.auth_token = None

    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.headers = {}
    handler.path = "/api/overview"
    assert handler._is_authenticated() is True


def test_workbench_auth_query_token_rejected_on_post():
    AutoMLWebHandler.require_auth = True
    AutoMLWebHandler.auth_token = "secret_workbench_key_123"

    try:
        handler_post = AutoMLWebHandler.__new__(AutoMLWebHandler)
        handler_post.command = "POST"
        handler_post.headers = {}
        handler_post.path = "/api/run/create?token=secret_workbench_key_123"
        # Query tokens are strictly rejected on state-mutating operations
        assert handler_post._is_authenticated() is False

        # But Bearer header succeeds on POST
        handler_post_bearer = AutoMLWebHandler.__new__(AutoMLWebHandler)
        handler_post_bearer.command = "POST"
        handler_post_bearer.headers = {"Authorization": "Bearer secret_workbench_key_123"}
        handler_post_bearer.path = "/api/run/create"
        assert handler_post_bearer._is_authenticated() is True
    finally:
        AutoMLWebHandler.require_auth = False
        AutoMLWebHandler.auth_token = None


def test_workbench_payload_size_limit():
    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.headers = {"Content-Length": str(AutoMLWebHandler.MAX_PAYLOAD_SIZE + 100)}
    handler.path = "/api/jobs"
    handler.command = "POST"

    sent_data = {}
    sent_status = None

    def mock_send_json(data, status=HTTPStatus.OK):
        nonlocal sent_data, sent_status
        sent_data = data
        sent_status = status

    handler._send_json = mock_send_json
    handler.require_auth = False
    handler.auth_token = None

    handler.do_POST()
    assert sent_status == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    assert "Payload Too Large" in sent_data.get("error", "")
