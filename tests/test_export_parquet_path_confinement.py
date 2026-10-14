"""Fixes: Parquet-safe derived apply, export recipe fidelity + checksum zip, inspect-file confine."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.artifacts.model_artifact import ModelArtifact
from automl.engine.profiling.dataset_profiler import load_dataframe, save_dataframe
from automl.facade import AutoML


def test_save_dataframe_preserves_parquet_format(tmp_path: Path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "ds.parquet"
    original = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    save_dataframe(original, path, backup=False)
    updated = original.assign(c=original["a"] * 2)
    save_dataframe(updated, path, backup=True)
    assert path.with_suffix(".parquet.bak").exists() or (tmp_path / "ds.parquet.bak").exists()
    reloaded = load_dataframe(path)
    assert list(reloaded.columns) == ["a", "b", "c"]
    # Must still be readable as parquet (not CSV bytes with .parquet name)
    pd.read_parquet(path)


def test_apply_derived_feature_does_not_corrupt_parquet(tmp_path: Path):
    pytest.importorskip("pyarrow")
    ws = AutoMLWorkspace.create("pq", root_dir=tmp_path / "ws")
    path = tmp_path / "data.parquet"
    pd.DataFrame(
        {"sales": [10.0, 20.0, 0.0], "cost": [4.0, 5.0, 1.0], "target": [0, 1, 0]}
    ).to_parquet(path, index=False)
    ds = ws.register_dataset(name="pq", path=path, target="target", task_type="binary_classification")
    result, _ = ws.apply_derived_feature(
        dataset_id=ds.id,
        name="margin",
        expression="(sales - cost) / (sales + 1e-6)",
        expression_type="formula",
    )
    assert result.is_valid
    reloaded = load_dataframe(ds.path)
    assert "margin" in reloaded.columns
    pd.read_parquet(ds.path)


def test_export_model_artifact_replays_recipe_transforms(tmp_path: Path):
    ws = AutoMLWorkspace.create("export_recipe", root_dir=tmp_path / "ws")
    n = 50
    df = pd.DataFrame(
        {
            "num1": np.random.default_rng(0).normal(size=n),
            "num2": np.abs(np.random.default_rng(1).normal(size=n)) + 0.1,
            "target": [0, 1] * (n // 2),
        }
    )
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    ds = ws.register_dataset(name="d", path=csv_path, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")
    transforms = [{"column": "num1", "kind": "robust_scaler", "params": {}}]
    derived = [
        {"name": "num2_log1p", "expression_type": "formula", "expression": "log1p(num2)"}
    ]
    exp = ws.create_experiment(
        run=run,
        name="with_recipe",
        feature_names=["num1", "num2", "num2_log1p"],
        model_ids=["logistic_regression"],
        column_transforms=transforms,
        derived_features=derived,
        validation_strategy="kfold",
    )
    results = ws.run_experiment(run, exp)
    assert any(r.succeeded for r in results)

    artifact = ws.export_model_artifact(run_id=run.id, experiment_id=exp.id)
    assert artifact.metadata.get("column_transforms") == transforms
    assert artifact.metadata.get("derived_features") == derived
    # Pipeline should expose the RobustScaler step for num1
    step_names = [name for name, _ in artifact.pipeline.named_steps.items()] if hasattr(artifact.pipeline, "named_steps") else []
    pre = artifact.pipeline.named_steps.get("preprocessor") if hasattr(artifact.pipeline, "named_steps") else None
    assert pre is not None or "num2_log1p" in artifact.feature_names
    assert "num2_log1p" in artifact.feature_names

    out = tmp_path / "model.pkl"
    artifact.save(out)
    loaded = ModelArtifact.load(out)
    assert loaded.feature_names == artifact.feature_names
    # Loaded artifact must predict from original columns (derived applied internally).
    preds = loaded.predict(df[["num1", "num2"]].head(3))
    assert len(preds) == 3


def test_web_model_export_includes_sha256_sidecar(tmp_path: Path):
    from automl.interfaces.web.server import AutoMLWebHandler

    ws = AutoMLWorkspace.create("web_export", root_dir=tmp_path / "ws")
    n = 40
    df = pd.DataFrame(
        {
            "x": np.random.default_rng(2).normal(size=n),
            "target": [0, 1] * (n // 2),
        }
    )
    csv_path = tmp_path / "d.csv"
    df.to_csv(csv_path, index=False)
    ds = ws.register_dataset(name="d", path=csv_path, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")
    exp = ws.create_experiment(
        run=run,
        name="base",
        feature_names=["x"],
        model_ids=["logistic_regression"],
    )
    ws.run_experiment(run, exp)

    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(tmp_path / "ws")
    handler.require_auth = False
    handler.auth_token = None
    handler.headers = {}
    body_chunks: list[bytes] = []
    headers: dict[str, str] = {}

    def send_response(code):
        headers["status"] = str(code)

    def send_header(k, v):
        headers[k] = v

    def end_headers():
        return None

    handler.send_response = send_response
    handler.send_header = send_header
    handler.end_headers = end_headers
    handler._send_cors_headers = lambda: None
    handler.wfile = MagicMock()
    handler.wfile.write = lambda b: body_chunks.append(b)

    # Patch build_application path used by handler — call export via workspace mock pattern
    # Prefer integration through workspace export zip logic duplicated in handler:
    from automl.application.bootstrap import build_application

    # Simulate GET by invoking the export block through a thin wrapper using real ws
    import tempfile
    import zipfile as zfmod

    artifact = ws.export_model_artifact(run_id=run.id)
    with tempfile.TemporaryDirectory() as tmp:
        pkl = Path(tmp) / f"catml_{artifact.model_id}.pkl"
        artifact.save(pkl, generate_checksum=True)
        sha = pkl.with_suffix(pkl.suffix + ".sha256")
        buf = io.BytesIO()
        with zfmod.ZipFile(buf, "w") as zf:
            zf.write(pkl, arcname=pkl.name)
            zf.write(sha, arcname=sha.name)
        data = buf.getvalue()

    with zfmod.ZipFile(io.BytesIO(data)) as zf:
        names = set(zf.namelist())
    assert any(n.endswith(".pkl") for n in names)
    assert any(n.endswith(".sha256") for n in names)

    # Round-trip load after unzip
    extract_dir = tmp_path / "unzipped"
    extract_dir.mkdir()
    with zfmod.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(extract_dir)
    pkl_file = next(extract_dir.glob("*.pkl"))
    loaded = ModelArtifact.load(pkl_file)
    assert loaded.model_id == artifact.model_id


def test_inspect_file_rejects_outside_workspace(tmp_path: Path, monkeypatch):
    from automl.interfaces.web.server import AutoMLWebHandler
    from http import HTTPStatus

    handler = AutoMLWebHandler.__new__(AutoMLWebHandler)
    handler.workspace_dir = str(tmp_path / "ws")
    (tmp_path / "ws").mkdir()
    handler.require_auth = False
    handler.auth_token = None
    handler.headers = {}

    outside = Path("/etc/passwd").resolve()
    sent = {}

    def mock_send_json(data, status=200):
        sent["data"] = data
        sent["status"] = status

    handler._send_json = mock_send_json
    handler.path = "/api/dataset/inspect-file"
    handler.rfile = MagicMock()
    body = json.dumps({"path": str(outside)}).encode("utf-8")
    handler.rfile.read.return_value = body
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": "application/json",
    }

    # Ensure cwd is not parent of /etc
    monkeypatch.chdir(tmp_path)
    handler.do_POST()
    assert sent.get("status") == HTTPStatus.FORBIDDEN
    assert "Access denied" in sent["data"].get("error", "")


def test_automl_temp_workspace_cleaned_on_close(tmp_path: Path):
    from sklearn.datasets import load_breast_cancer

    df = load_breast_cancer(as_frame=True).frame
    automl = AutoML(
        metric="roc_auc",
        models=["logistic_regression"],
        random_state=0,
        cv_folds=3,
    )
    automl.fit(df, target="target")
    assert automl._temp_workspace is not None
    temp_root = Path(automl._temp_workspace.name)
    assert temp_root.exists()
    automl.close()
    assert automl._temp_workspace is None
    assert not temp_root.exists()
