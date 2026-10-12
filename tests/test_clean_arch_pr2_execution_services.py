"""Clean Architecture PR2: Workspace facade delegates to execution/optimization services."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.services.experiment_execution_service import ExperimentExecutionService
from automl.application.services.optimization_service import OptimizationService
from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.experiments.trial import ExperimentStatus
from automl.domain.runs.states import RunStatus


def _binary_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "target": [0, 1] * (n // 2),
        }
    )
    path = tmp_path / "binary.csv"
    df.to_csv(path, index=False)
    return path


def _regression_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(1)
    feat1 = rng.normal(size=n)
    feat2 = rng.normal(size=n)
    # Mild signal so profiler does not flag leakage columns.
    df = pd.DataFrame(
        {
            "feat1": feat1,
            "feat2": feat2,
            "target": 0.3 * feat1 + 0.2 * feat2 + rng.normal(scale=1.0, size=n),
        }
    )
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return path


def test_workspace_lazy_services_are_execution_and_optimization(tmp_path: Path):
    ws = AutoMLWorkspace.create("pr2_lazy", root_dir=tmp_path / "ws")
    assert isinstance(ws.experiment_execution_service, ExperimentExecutionService)
    assert isinstance(ws.optimization_service, OptimizationService)
    assert ws.experiment_execution_service is ws.experiment_execution_service
    assert ws.optimization_service is ws.optimization_service
    assert ws.experiment_execution_service.workspace is ws
    assert ws.optimization_service.workspace is ws


def test_run_experiment_facade_binary_classification_smoke(tmp_path: Path):
    ws = AutoMLWorkspace.create("pr2_run_bin", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(
        ds,
        metric="roc_auc",
        validation_strategy="kfold",
        cv_folds=3,
        random_seed=7,
    )
    exp = ws.create_experiment(
        run=run,
        name="smoke_bin",
        feature_names=["a", "b"],
        model_ids=["logistic_regression"],
    )

    results = ws.run_experiment(run, exp)

    assert results
    assert any(r.succeeded for r in results)
    refreshed_exp = ws.repository.get_experiment(exp.id)
    assert refreshed_exp is not None
    assert refreshed_exp.status == ExperimentStatus.COMPLETED
    refreshed_run = ws.repository.get_run(run.id)
    assert refreshed_run is not None
    assert refreshed_run.status == RunStatus.COMPLETED


def test_facade_delegates_run_and_optimize_to_services(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Public Workspace methods are thin delegates (CLI/MCP/Web stay on facade)."""
    ws = AutoMLWorkspace.create("pr2_delegate", root_dir=tmp_path / "ws")
    calls: list[str] = []

    def fake_run(*args, **kwargs):
        calls.append("run")
        return []

    def fake_opt(*args, **kwargs):
        calls.append("opt")
        return {"ok": True}

    monkeypatch.setattr(ws.experiment_execution_service, "run_experiment", fake_run)
    monkeypatch.setattr(ws.optimization_service, "optimize_experiment", fake_opt)

    assert ws.run_experiment(object(), object()) == []  # type: ignore[arg-type]
    assert ws.optimize_experiment(run_id="r", experiment_id="e") == {"ok": True}
    assert calls == ["run", "opt"]


def test_run_experiment_mae_regression_smoke(tmp_path: Path):
    ws = AutoMLWorkspace.create("pr2_mae", root_dir=tmp_path / "ws")
    csv_path = _regression_csv(tmp_path)
    ds = ws.register_dataset(
        name="reg",
        path=csv_path,
        target="target",
        task_type="regression",
    )
    run = ws.create_run(
        ds,
        metric="mae",
        validation_strategy="kfold",
        cv_folds=3,
        random_seed=11,
    )
    exp = ws.create_experiment(
        run=run,
        name="mae_smoke",
        feature_names=["feat1", "feat2"],
        model_ids=["ridge"],
    )

    results = ws.run_experiment(run, exp)

    assert results
    succeeded = [r for r in results if r.succeeded]
    assert succeeded
    assert succeeded[0].primary_metric == "mae"
    assert succeeded[0].primary_score >= 0.0
    assert ws.repository.get_experiment(exp.id).status == ExperimentStatus.COMPLETED


def test_optimize_experiment_facade_delegates_and_returns_shape(tmp_path: Path):
    ws = AutoMLWorkspace.create("pr2_hpo", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path, n=50)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="roc_auc", random_seed=5)
    exp = ws.create_experiment(
        run=run,
        name="hpo",
        feature_names=["a", "b"],
        model_ids=["logistic_regression"],
    )

    out = ws.optimize_experiment(
        run_id=run.id,
        experiment_id=exp.id,
        n_trials=3,
        optimizer="random",
    )

    assert out["experiment_id"] == exp.id
    assert out["model_id"] == "logistic_regression"
    assert out["optimizer"] == "random"
    assert out["trials_executed"] == 3
    assert out["successful_trials"] >= 1
    assert out["best_score"] is not None
    assert out["best_trial_id"]
    assert len(out["trials"]) == 3
    assert ws.repository.get_experiment(exp.id).status == ExperimentStatus.COMPLETED


def test_optimize_unknown_optimizer_fail_closed(tmp_path: Path):
    ws = AutoMLWorkspace.create("pr2_bad_opt", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="roc_auc")
    exp = ws.create_experiment(
        run=run,
        name="bad",
        feature_names=["a", "b"],
        model_ids=["logistic_regression"],
    )

    with pytest.raises(ValueError, match="Unknown optimizer"):
        ws.optimize_experiment(
            run_id=run.id,
            experiment_id=exp.id,
            optimizer="bayesian_nope",
            n_trials=1,
        )
