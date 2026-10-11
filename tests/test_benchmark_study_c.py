"""CATML Benchmark v1 — Study C (robustness & trust) adversarial criteria."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.artifacts.model_artifact import ModelArtifact
from automl.benchmarks.runner import _best_result
from automl.domain.experiments.trial import ExperimentStatus, TrialResult
from automl.domain.runs.states import RunStatus
from automl.engine.profiling.dataset_profiler import detect_is_group_candidate
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_study_c_customer_type_not_group_entity():
    n = 100
    series = pd.Series(["Basic", "Premium"] * (n // 2))
    assert detect_is_group_candidate("Customer Type", series, n, target_column="target") is False


def test_study_c_constant_target_does_not_fake_winner(tmp_path: Path):
    ws = AutoMLWorkspace.create("c_const", root_dir=tmp_path / "ws")
    df = pd.DataFrame({"x": np.random.randn(40), "target": [1] * 40})
    path = tmp_path / "const.csv"
    df.to_csv(path, index=False)
    ds = ws.register_dataset(name="d", path=path, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")
    exp = ws.create_experiment(
        run=run,
        name="const",
        feature_names=["x"],
        model_ids=["logistic_regression"],
    )
    results = ws.run_experiment(run, exp)
    # Constant target should fail trials or mark experiment failed — never invent success.
    succeeded = [r for r in results if r.succeeded]
    refreshed = ws.repository.get_experiment(exp.id)
    if not succeeded:
        assert refreshed.status == ExperimentStatus.FAILED
        assert ws.repository.get_run(run.id).status == RunStatus.FAILED


def test_study_c_hpo_all_fail_no_fake_winner(tmp_path: Path):
    ws = AutoMLWorkspace.create("c_hpo_fail", root_dir=tmp_path / "ws")
    df = pd.DataFrame(
        {
            "num1": np.random.randn(40),
            "num2": np.random.randn(40),
            "target": [0, 1] * 20,
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    ds = ws.register_dataset(name="d", path=path, target="target", task_type="binary_classification")
    run = ws.create_run(ds, metric="roc_auc")
    exp = ws.create_experiment(
        run=run,
        name="broken",
        feature_names=["num1", "num2", "leak_zs"],
        model_ids=["logistic_regression"],
        derived_features=[
            {"name": "leak_zs", "expression_type": "formula", "expression": "zscore(num1)"}
        ],
    )
    out = ws.optimize_experiment(run_id=run.id, experiment_id=exp.id, n_trials=2, optimizer="random")
    assert out["successful_trials"] == 0
    assert out["best_score"] is None
    assert ws.repository.get_experiment(exp.id).status == ExperimentStatus.FAILED
    assert ws.repository.get_run(run.id).status == RunStatus.FAILED


def test_study_c_tampered_artifact_rejected(tmp_path: Path):
    pipe = Pipeline([("model", LogisticRegression())])
    pipe.fit(np.array([[1.0], [2.0]]), np.array([0, 1]))
    art = ModelArtifact(
        pipeline=pipe,
        model_id="logistic_regression",
        task_type="binary_classification",
        feature_names=["f1"],
    )
    path = tmp_path / "m.pkl"
    art.save(path)
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ModelArtifact.load(path)


def test_study_c_benchmark_all_failed_score_absent():
    results = [
        TrialResult(
            trial_id="t1",
            experiment_id="e1",
            model_id="m",
            primary_metric="roc_auc",
            primary_score=0.0,
            failure_reason="boom",
        )
    ]
    score, model = _best_result(results, metric="roc_auc")
    assert score is None
    assert model is None


def test_study_c_time_series_requires_time_column(tmp_path: Path):
    from unittest.mock import MagicMock

    from automl.domain.ports import TrialExecution
    from automl.engine.training.sklearn_trainer import SklearnTrainer

    n = 40
    df = pd.DataFrame({"x": np.random.randn(n), "target": [0, 1] * (n // 2)})
    csv_path = tmp_path / "ts.csv"
    df.to_csv(csv_path, index=False)
    run = MagicMock(id="r1")
    run.config = MagicMock()
    run.config.extra = {}
    result = SklearnTrainer().run(
        TrialExecution(
            trial=MagicMock(model_id="logistic_regression", parameters={}),
            experiment=MagicMock(id="e1"),
            run=run,
            feature_names=["x"],
            dataset_path=str(csv_path),
            target_column="target",
            task_type="binary_classification",
            metric="roc_auc",
            validation_strategy="time_series",
            test_size=0.2,
            cv_folds=3,
            random_seed=42,
        )
    )
    assert result.succeeded is False
    assert "time_column" in (result.failure_reason or "")
