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
