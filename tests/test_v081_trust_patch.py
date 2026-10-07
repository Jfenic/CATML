from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.facade import AutoML
from automl.interfaces.web.server import AutoMLWebHandler


def test_trust_patch_leakage_fail_safe(tmp_path):
    """Verifies that AutoML.fit() automatically excludes detected leakage columns."""
    np.random.seed(42)
    n = 100
    target = np.random.choice([0, 1], size=n)
    # Perfect target leakage column
    leakage_col = target.copy()
    valid_feature = np.random.randn(n)

    df = pd.DataFrame({
        "leakage_feature": leakage_col,
        "valid_feature": valid_feature,
        "target": target,
    })

    automl = AutoML(task="classification", random_state=42, models=["logistic_regression"], time_budget=10)
    result = automl.fit(df, target="target")

    # Winning model features must NOT include leakage_feature
    used_features = result.best_model.feature_names
    assert "leakage_feature" not in used_features
    assert "valid_feature" in used_features


def test_trust_patch_random_state_propagation(tmp_path):
    """Verifies that AutoML(random_state=123) propagates into RunConfig.random_seed."""
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "x2": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
        "y": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })

    automl = AutoML(random_state=123, models=["logistic_regression"])
    result = automl.fit(df, target="y")

    # Check run configuration stored in workspace repository
    run = result._workspace.repository.get_run(result.run_id)
    assert run is not None
    assert run.config.random_seed == 123


def test_trust_patch_leaderboard_mae_rmse_sorting():
    """Verifies that AutoMLResult.leaderboard() sorts ascendingly for minimization metrics (MAE/RMSE)."""
    from automl.facade import AutoMLResult
    from unittest.mock import MagicMock

    dummy_model = MagicMock()
    result = AutoMLResult(
        best_model=dummy_model,
        best_score=2.1,
        best_model_id="ridge",
        task_type="regression",
        metric="mae",
        run_id="run_123",
        _leaderboard_data=[
            {"model_id": "bad_model", "score": 15.4, "metric": "mae"},
            {"model_id": "best_model", "score": 2.1, "metric": "mae"},
            {"model_id": "mid_model", "score": 6.8, "metric": "mae"},
        ],
    )

    lb = result.leaderboard()
    assert len(lb) == 3
    # In MAE, lowest score (2.1) must be ranked 1st
    assert lb.iloc[0]["model_id"] == "best_model"
    assert lb.iloc[0]["score"] == 2.1
    assert lb.iloc[0]["rank"] == 1
    # Highest error (15.4) must be ranked last
    assert lb.iloc[2]["model_id"] == "bad_model"
    assert lb.iloc[2]["score"] == 15.4


def test_trust_patch_media_preview_confinement(tmp_path):
    """Verifies that /api/media/preview strictly confines files to the workspace."""
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    inside_img = workspace_dir / "test_inside.png"
    inside_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 30)

    outside_dir = tmp_path.parent / "test_outside_confinement"
    outside_dir.mkdir(exist_ok=True)
    outside_img = outside_dir / "secret.png"
    outside_img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 30)

    class DummyRequest:
        def makefile(self, *args, **kwargs):
            return io.BytesIO(b"")
        def sendall(self, *args, **kwargs):
            pass

    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(workspace_dir)
    handler.client_address = ("127.0.0.1", 12345)
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    handler.headers = {"Host": "127.0.0.1", "Origin": "https://malicious-site.com"}
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET /api/media/preview?path={str(outside_img)} HTTP/1.1"

    # Case 1: Outside absolute path must be forbidden
    handler.path = f"/api/media/preview?path={str(outside_img)}"
    handler.do_GET()
    output = handler.wfile.getvalue().decode("utf-8", errors="ignore")
    assert "403" in output or "Access denied" in output
    # External origin must NOT receive wildcard CORS
    assert "Access-Control-Allow-Origin: *" not in output
    assert "Access-Control-Allow-Origin: https://malicious-site.com" not in output

    # Case 2: Inside file is served
    handler.wfile = io.BytesIO()
    handler.path = f"/api/media/preview?path=test_inside.png"
    handler.requestline = f"GET {handler.path} HTTP/1.1"
    handler.do_GET()
    output_inside = handler.wfile.getvalue().decode("utf-8", errors="ignore")
    assert "200 OK" in output_inside
