#!/usr/bin/env python3
"""CATML Release Verification Smoke Test.

Validates an isolated installation of the catml package:
1. Package imports and version metadata.
2. Training loop via the ergonomic AutoML facade.
3. Model artifact serialization, SHA-256 sidecar checksum generation.
4. Cryptographic integrity verification and artifact reloading.
5. Production inference on tabular data.
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd


def run_smoke_test() -> None:
    print("=== [CATML SMOKE TEST] Starting Release Verification ===")

    # 1. Package Imports & Metadata
    print("[1/5] Checking package import and version metadata...")
    import catml
    import automl
    from catml import AutoML, ModelArtifact

    version = getattr(catml, "__version__", None)
    print(f"      catml version: {version}")
    assert version == "0.8.1", f"Expected version '0.8.1', got '{version}'"
    assert catml.AutoML is automl.AutoML, "Facade parity mismatch between catml and automl"
    print("      ✓ Package metadata verified.")

    # 2. Synthetic Dataset Creation
    print("[2/5] Creating synthetic verification dataset...")
    np.random.seed(42)
    n_samples = 60
    df = pd.DataFrame({
        "feature_a": np.random.randn(n_samples),
        "feature_b": np.random.uniform(10, 50, n_samples),
        "category": np.random.choice(["alpha", "beta", "gamma"], n_samples),
        "target": np.random.choice([0, 1], n_samples),
    })
    print(f"      Dataset shape: {df.shape}")

    # 3. Fit via AutoML Facade
    print("[3/5] Fitting candidate models via AutoML facade...")
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

    # 4. Artifact Export and SHA-256 Integrity Verification
    print("[4/5] Exporting ModelArtifact with SHA-256 checksum sidecar...")
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

        # Reload artifact and verify integrity checks
        reloaded = ModelArtifact.load(str(art_path), verify_checksum=True)
        assert "classification" in reloaded.task_type, f"Unexpected task_type: {reloaded.task_type}"
        print(f"      ✓ ModelArtifact successfully reloaded with integrity verification ({reloaded.task_type}).")

        # 5. Production Inference
        print("[5/5] Generating predictions with reloaded artifact...")
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
