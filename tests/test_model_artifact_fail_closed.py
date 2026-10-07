from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from automl.artifacts.model_artifact import ModelArtifact


def test_model_artifact_load_fail_closed_corrupted_sidecar(tmp_path: Path):
    pipe = Pipeline([("model", LogisticRegression())])
    pipe.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))
    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["f1"],
    )

    art_file = tmp_path / "model.pkl"
    artifact.save(art_file)

    sha_file = tmp_path / "model.pkl.sha256"
    assert sha_file.exists()

    # Tamper with the sidecar file (corrupt digest)
    sha_file.write_text("corrupted_hash_not_valid", encoding="utf-8")

    with pytest.raises(ValueError, match="Artifact checksum verification failed"):
        ModelArtifact.load(art_file, verify_checksum=True)


def test_model_artifact_load_fail_closed_unreadable_sidecar(tmp_path: Path, monkeypatch):
    pipe = Pipeline([("model", LogisticRegression())])
    pipe.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))
    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["f1"],
    )

    art_file = tmp_path / "model_unreadable.pkl"
    artifact.save(art_file)

    # Force read_text error on the sidecar to verify fail-closed behavior
    original_read_text = Path.read_text

    def mock_read_text(self, *args, **kwargs):
        if str(self).endswith(".sha256"):
            raise OSError("Simulated disk I/O failure")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", mock_read_text)

    with pytest.raises(ValueError, match="unable to read or compute hash integrity"):
        ModelArtifact.load(art_file, verify_checksum=True)


def test_model_artifact_load_fail_closed_tampered_model(tmp_path: Path):
    pipe = Pipeline([("model", LogisticRegression())])
    pipe.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))
    artifact = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["f1"],
    )

    art_file = tmp_path / "model_tampered.pkl"
    artifact.save(art_file)

    # Tamper with model binary content
    art_file.write_bytes(b"tampered binary bytes payload")

    with pytest.raises(ValueError, match="Artifact checksum verification failed"):
        ModelArtifact.load(art_file, verify_checksum=True)
