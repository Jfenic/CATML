from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from automl.artifacts.model_artifact import ModelArtifact
from automl.engine.training.target_adapter import TargetAdapter


def test_model_artifact_save_load_predict_dataframe(tmp_path: Path):
    X = pd.DataFrame({
        "age": [25, 45, 35, 50, 23],
        "balance": [1000.0, 5000.0, 2500.0, 8000.0, 500.0],
    })
    y = np.array([0, 1, 0, 1, 0])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)),
    ])
    pipe.fit(X, y)

    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["age", "balance"],
        target_name="churn",
        metric="roc_auc",
        score=0.92,
    )

    save_path = tmp_path / "model.pkl"
    saved_file = artifact.save(save_path)
    assert saved_file.exists()

    loaded = ModelArtifact.load(save_path)
    assert loaded.model_id == "logistic_regression"
    assert loaded.score == 0.92
    assert loaded.feature_names == ["age", "balance"]

    # Predict with DataFrame
    preds = loaded.predict(X)
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})

    # Predict proba
    probs = loaded.predict_proba(X)
    assert probs.shape == (5, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_model_artifact_predict_numpy_ndarray(tmp_path: Path):
    X_arr = np.array([
        [25.0, 1000.0],
        [45.0, 5000.0],
        [35.0, 2500.0],
    ])
    y = np.array([0, 1, 0])

    X_df = pd.DataFrame(X_arr, columns=["f1", "f2"])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(random_state=42)),
    ])
    pipe.fit(X_df, y)

    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["f1", "f2"],
        target_name="target",
    )

    # Predict directly with 2D numpy array
    preds = artifact.predict(X_arr)
    assert len(preds) == 3

    # Predict 1D single sample
    single_pred = artifact.predict(np.array([30.0, 2000.0]))
    assert len(single_pred) == 1


def test_model_artifact_with_target_adapter_string_classes(tmp_path: Path):
    X = pd.DataFrame({
        "score": [80.0, 40.0, 90.0, 30.0],
    })
    y_raw = ["Passed", "Failed", "Passed", "Failed"]

    adapter = TargetAdapter()
    y_encoded = adapter.fit_transform(y_raw, "binary_classification")

    pipe = Pipeline([("model", LogisticRegression())])
    pipe.fit(X, y_encoded)

    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["score"],
        target_name="status",
        target_adapter=adapter,
    )

    preds = artifact.predict(X)
    assert list(preds) == ["Passed", "Failed", "Passed", "Failed"]


def test_model_artifact_missing_features_error():
    pipe = Pipeline([("model", LogisticRegression())])
    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["col_a", "col_b"],
    )

    bad_df = pd.DataFrame({"col_a": [1, 2]})
    with pytest.raises(ValueError, match="missing required features"):
        artifact.predict(bad_df)

    bad_arr = np.array([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="Feature count mismatch"):
        artifact.predict(bad_arr)


def test_model_artifact_load_nonexistent_or_invalid(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ModelArtifact.load(tmp_path / "does_not_exist.pkl")

    dummy_file = tmp_path / "dummy.pkl"
    import joblib

    joblib.dump({"not": "a model artifact"}, dummy_file)
    with pytest.raises(TypeError, match="not a ModelArtifact instance"):
        ModelArtifact.load(dummy_file)


def test_model_artifact_provenance_and_describe(tmp_path: Path):
    pipe = Pipeline([("clf", LogisticRegression())])
    pipe.fit(np.array([[1], [2]]), np.array([0, 1]))

    custom_prov = {
        "catml_version": "0.7.0",
        "dataset_hash": "sha256_mock_hash",
        "seed": 42,
    }
    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["f1"],
        provenance=custom_prov,
    )

    desc = artifact.describe()
    assert desc["model_id"] == "logistic_regression"
    assert desc["feature_count"] == 1
    assert desc["features"] == ["f1"]
    assert desc["provenance"]["dataset_hash"] == "sha256_mock_hash"
    assert desc["provenance"]["seed"] == 42

    # Save and reload
    save_file = tmp_path / "prov_model.pkl"
    artifact.save(save_file)
    reloaded = ModelArtifact.load(save_file)
    assert reloaded.provenance["dataset_hash"] == "sha256_mock_hash"
    assert reloaded.describe()["provenance"]["seed"] == 42


def test_model_artifact_checksum_and_security_warning(tmp_path: Path):
    pipe = Pipeline([("model", LogisticRegression())])
    pipe.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))
    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["f1"],
    )
    save_path = tmp_path / "secure_model.pkl"
    artifact.save(save_path)

    # Verify sidecar sha256 file
    sha_file = tmp_path / "secure_model.pkl.sha256"
    assert sha_file.exists()
    assert len(sha_file.read_text().strip()) == 64

    # Verify loading triggers security notice warning
    with pytest.warns(UserWarning, match="Security Notice"):
        loaded = ModelArtifact.load(save_path)
    assert loaded.model_id == "logistic_regression"

    # Verify tampering detection
    save_path.write_bytes(b"tampered content")
    with pytest.raises(ValueError, match="Artifact checksum verification failed"):
        ModelArtifact.load(save_path)

