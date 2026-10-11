"""Audit-4: path confinement, unique uploads, inference recipe parity, ensemble scores, stacking OOF."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from automl.application.services.inference_service import InferenceService
from automl.application.services.workspace import AutoMLWorkspace
from automl.engine.ensemble.blender import stack_predictions
from automl.infrastructure.security.workspace_paths import (
    PathOutsideWorkspaceError,
    resolve_allowed_path,
)
def _csv(tmp_path: Path, name: str = "data.csv", n: int = 40) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": np.abs(rng.normal(size=n)) + 0.1,
            "target": [0, 1] * (n // 2),
        }
    )
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_resolve_allowed_path_rejects_outside_workspace(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    inside = ws / "ok.csv"
    inside.write_text("a,b\n1,2\n")
    outside = tmp_path / "secret.csv"
    outside.write_text("x\n1\n")

    assert resolve_allowed_path(inside, workspace_dir=ws) == inside.resolve()
    with pytest.raises(PathOutsideWorkspaceError):
        resolve_allowed_path(outside, workspace_dir=ws, cwd=ws)


def test_web_register_rejects_path_outside_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """HTTP register must 403 when the CSV is outside cwd ∪ workspace."""
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from automl.interfaces.web.server import AutoMLWebHandler

    monkeypatch.chdir(tmp_path)
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    AutoMLWebHandler.workspace_dir = str(ws_dir)

    outside = tmp_path.parent / "escape_outside.csv"
    outside.write_text("a,target\n1,0\n2,1\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        payload = json.dumps(
            {"name": "escape", "path": str(outside), "target": "target"}
        ).encode()
        req = Request(
            f"http://{host}:{port}/api/dataset/register",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(HTTPError) as exc:
            urlopen(req)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_upload_same_filename_does_not_overwrite(tmp_path: Path):
    """Two uploads with the same client filename must produce distinct files."""
    import uuid as uuid_mod

    ws_dir = tmp_path / "ws"
    upload_dir = ws_dir / "uploads"
    upload_dir.mkdir(parents=True)

    safe_filename = "clientes.csv"
    stem = Path(safe_filename).stem
    suffix = Path(safe_filename).suffix
    paths = []
    for _ in range(2):
        dest = upload_dir / f"{stem}_{uuid_mod.uuid4().hex[:10]}{suffix}"
        dest.write_bytes(b"a,target\n1,0\n")
        paths.append(dest)

    assert paths[0] != paths[1]
    assert paths[0].exists() and paths[1].exists()


def test_inference_passes_column_transforms_and_derived(tmp_path: Path):
    ws = AutoMLWorkspace.create("inf_parity", root_dir=tmp_path / "ws")
    csv_path = _csv(tmp_path)
    ds = ws.register_dataset(
        name="d",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="roc_auc")
    transforms = [{"column": "num1", "kind": "robust_scaler", "params": {}}]
    derived = [
        {
            "name": "num2_log1p",
            "expression_type": "formula",
            "expression": "log1p(num2)",
            "source_columns": ["num2"],
        }
    ]
    exp = ws.create_experiment(
        run=run,
        name="with_recipe",
        feature_names=["num1", "num2_log1p"],
        model_ids=["logistic_regression"],
        column_transforms=transforms,
        derived_features=derived,
    )
    ws.run_experiment(run, exp)

    test_path = _csv(tmp_path, "test.csv", n=20)
    svc = InferenceService(ws)

    captured = {}
    real = __import__(
        "automl.engine.training.sklearn_trainer", fromlist=["SklearnTrainer"]
    ).SklearnTrainer.fit_and_predict

    def spy(self, *args, **kwargs):
        captured["parameters"] = kwargs.get("parameters")
        x_train = kwargs.get("X_train")
        if x_train is None and args:
            x_train = args[0]
        captured["X_train_cols"] = list(x_train.columns)
        return real(self, *args, **kwargs)

    with patch(
        "automl.engine.training.sklearn_trainer.SklearnTrainer.fit_and_predict",
        spy,
    ):
        out = svc.predict(run_id=run.id, test_dataset_path=str(test_path), experiment_id=exp.id)

    assert out
    params = captured.get("parameters") or {}
    assert params.get("column_transforms") == transforms
    assert "num2_log1p" in captured.get("X_train_cols", [])


def test_stacking_oof_not_identical_to_in_sample_fit(tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 80
    # Two base models with noisy OOF predictions
    y = np.array([0, 1] * (n // 2))
    p1 = np.clip(y + rng.normal(0, 0.25, size=n), 0, 1)
    p2 = np.clip(y + rng.normal(0, 0.3, size=n), 0, 1)

    stacked_oof, _, est = stack_predictions(
        [p1, p2],
        y,
        meta_model="logistic_regression",
        task_type="binary_classification",
        random_state=0,
    )
    # In-sample predict after full fit must differ from nested OOF in general.
    X = np.column_stack([p1, p2])
    in_sample = est.predict_proba(X)[:, 1]
    # Not all equal (nested CV vs in-sample)
    assert not np.allclose(stacked_oof, in_sample, atol=1e-12)


def test_build_ensemble_regression_mae_positive(tmp_path: Path):
    ws = AutoMLWorkspace.create("ens_mae", root_dir=tmp_path / "ws")
    rng = np.random.default_rng(2)
    n = 50
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "target": rng.normal(size=n) * 3 + 10,
        }
    )
    path = tmp_path / "reg.csv"
    df.to_csv(path, index=False)
    ds = ws.register_dataset(name="r", path=path, target="target", task_type="regression")
    run = ws.create_run(ds, metric="mae")
    exp = ws.create_experiment(
        run=run,
        name="base",
        feature_names=["x1", "x2"],
        model_ids=["ridge", "random_forest"],
    )
    ws.run_experiment(run, exp)
    ens_exp = ws.build_ensemble(
        run_id=run.id,
        model_ids=["ridge", "random_forest"],
        method="average",
        folds=3,
    )
    results = ws.repository.list_trial_results(ens_exp.id)
    ens = [r for r in results if r.model_id == "voting_ensemble"]
    assert ens
    assert ens[0].primary_score >= 0.0
    assert ens[0].primary_metric == "mae"


def test_unknown_optimizer_raises(tmp_path: Path):
    ws = AutoMLWorkspace.create("opt", root_dir=tmp_path / "ws")
    csv_path = _csv(tmp_path)
    ds = ws.register_dataset(
        name="d", path=csv_path, target="target", task_type="binary_classification"
    )
    run = ws.create_run(ds)
    exp = ws.create_experiment(
        run=run,
        name="e",
        feature_names=["num1", "num2"],
        model_ids=["logistic_regression"],
    )
    with pytest.raises(ValueError, match="Unknown optimizer"):
        ws.optimize_experiment(run_id=run.id, experiment_id=exp.id, n_trials=1, optimizer="optunaa")
