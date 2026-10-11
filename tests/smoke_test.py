#!/usr/bin/env python3
"""CATML Release Verification Smoke Test.

Validates an isolated installation of the catml package:
1. Package imports and version metadata.
2. Workbench static assets shipped next to server.py (wheel packaging).
3. Workbench HTTP serves index + CSS/JS with HTTP 200.
4. Training loop via the ergonomic AutoML facade.
5. Model artifact serialization, SHA-256 sidecar checksum generation.
6. Artifact reload and production inference.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_STATIC = (
    "index.html",
    "css/workbench.css",
    "js/app.js",
    "js/views/home.js",
)


def _assert_workbench_static_packaged() -> Path:
    # `automl.interfaces.web` is a namespace package (__file__ is None); resolve via server.py.
    from automl.interfaces.web import server as web_server

    static_dir = Path(web_server.__file__).resolve().parent / "static"
    assert static_dir.is_dir(), f"Workbench static directory missing: {static_dir}"
    missing = [rel for rel in REQUIRED_STATIC if not (static_dir / rel).is_file()]
    assert not missing, f"Workbench static assets missing from installed package: {missing}"
    return static_dir


def _assert_workbench_http_serves_assets(workspace_dir: Path) -> None:
    from automl.interfaces.web.server import AutoMLWebHandler

    AutoMLWebHandler.workspace_dir = str(workspace_dir)
    AutoMLWebHandler.require_auth = False
    AutoMLWebHandler.auth_token = None
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        for path in (
            "/",
            "/static/css/workbench.css",
            "/static/js/app.js",
            "/static/js/views/home.js",
        ):
            url = f"http://127.0.0.1:{port}{path}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                assert resp.status == 200, f"{url} returned {resp.status}"
                body = resp.read()
                assert body, f"{url} returned empty body"
        print(f"      ✓ Workbench HTTP served UI assets on 127.0.0.1:{port}")
    finally:
        server.shutdown()
        server.server_close()
        AutoMLWebHandler.require_auth = False
        AutoMLWebHandler.auth_token = None


def run_smoke_test() -> None:
    print("=== [CATML SMOKE TEST] Starting Release Verification ===")

    # 1. Package Imports & Metadata
    print("[1/6] Checking package import and version metadata...")
    import catml
    import automl
    from catml import AutoML, ModelArtifact

    version = getattr(catml, "__version__", None)
    print(f"      catml version: {version}")
    assert version == automl.__version__, f"Version mismatch: catml has '{version}' but automl has '{automl.__version__}'"
    assert catml.AutoML is automl.AutoML, "Facade parity mismatch between catml and automl"
    print("      ✓ Package metadata verified.")

    # 2. Workbench static assets in installed package
    print("[2/6] Checking Workbench static assets are packaged...")
    static_dir = _assert_workbench_static_packaged()
    print(f"      ✓ Static root: {static_dir}")

    # 3. Workbench HTTP smoke
    print("[3/6] Serving Workbench UI over loopback HTTP...")
    with tempfile.TemporaryDirectory() as tmp_ws:
        _assert_workbench_http_serves_assets(Path(tmp_ws))

    # 4. Synthetic Dataset Creation
    print("[4/6] Creating synthetic verification dataset...")
    np.random.seed(42)
    n_samples = 60
    df = pd.DataFrame({
        "feature_a": np.random.randn(n_samples),
        "feature_b": np.random.uniform(10, 50, n_samples),
        "category": np.random.choice(["alpha", "beta", "gamma"], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    })
    print(f"      Dataset shape: {df.shape}")

    # 5. Fit via AutoML Facade
    print("[5/6] Fitting candidate models via AutoML facade...")
    automl_instance = AutoML(
        task="classification",
        models=["logistic_regression", "random_forest"],
        cv_folds=2,
        time_budget=30,
    )
    result = automl_instance.fit(df, target="target")
    assert result is not None
    assert result.best_model_id is not None
    print(f"      ✓ Training completed. Winning model: '{result.best_model_id}' (score={result.best_score:.4f})")

    # 6. Artifact Export and SHA-256 Integrity Verification + inference
    print("[6/6] Exporting ModelArtifact with SHA-256 checksum sidecar...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        art_path = Path(tmp_dir) / "release_smoke_model.catml"
        checksum_path = Path(str(art_path) + ".sha256")

        result.save_model(str(art_path))
        assert art_path.exists(), f"Artifact file was not created: {art_path}"
        assert checksum_path.exists(), f"Checksum sidecar was not created: {checksum_path}"

        expected_hash = hashlib.sha256(art_path.read_bytes()).hexdigest()
        stored_hash = checksum_path.read_text().strip()
        assert expected_hash == stored_hash, f"Hash mismatch: {expected_hash} != {stored_hash}"
        print(f"      ✓ Cryptographic checksum sidecar verified ({expected_hash[:12]}...).")

        reloaded = ModelArtifact.load(str(art_path), verify_checksum=True)
        assert "classification" in reloaded.task_type, f"Unexpected task_type: {reloaded.task_type}"
        print(f"      ✓ ModelArtifact successfully reloaded with integrity verification ({reloaded.task_type}).")

        print("      Generating predictions with reloaded artifact...")
        X_test = df.drop(columns=["target"])
        preds = reloaded.predict(X_test)
        assert len(preds) == len(df), f"Expected {len(df)} predictions, got {len(preds)}"
        assert not np.isnan(preds).any(), "Predictions contain NaN values"
        print(f"      ✓ Inferred {len(preds)} predictions successfully.")

    print("\n🎉 === [CATML SMOKE TEST] ALL CHECKS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    try:
        run_smoke_test()
    except Exception as exc:
        print(f"\n❌ [CATML SMOKE TEST FAILED]: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
