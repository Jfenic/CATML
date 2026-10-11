"""Pre-publish audit: sandbox defaults, CV leakage guards, time_series, run FAILED."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.experiments.trial import ExperimentStatus
from automl.domain.features.derived_feature import DerivedFeatureDefinition, DerivedFeatureType
from automl.domain.ports import TrialExecution
from automl.domain.runs.states import RunStatus
from automl.engine.features.generation.derived_feature_engine import (
    SafeFormulaCalculator,
    SafePythonEvaluator,
    assert_derived_specs_safe_for_cross_validation,
    formula_uses_stateful_aggregates,
)
from automl.engine.training.sklearn_trainer import SklearnTrainer


def test_python_evaluator_disabled_by_default():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    py_eval = SafePythonEvaluator(df)
    with pytest.raises(PermissionError, match="disabled by default"):
        py_eval.evaluate("def compute_feature(df):\n    return df['x'] * 2")


def test_formula_stateful_aggregates_detection():
    assert formula_uses_stateful_aggregates("log1p(x)") is False
    assert formula_uses_stateful_aggregates("zscore(x)") is True
    assert formula_uses_stateful_aggregates("mean(a) + b") is True
    assert formula_uses_stateful_aggregates("a / (b + 1e-6)") is False


def test_assert_derived_specs_reject_python_and_zscore():
    with pytest.raises(ValueError, match="python_code"):
        assert_derived_specs_safe_for_cross_validation(
            [
                {
                    "name": "evil",
                    "expression_type": "python_code",
                    "expression": "def compute_feature(df):\n    return df['x']",
                }
            ]
        )

    with pytest.raises(ValueError, match="stateful aggregate"):
        assert_derived_specs_safe_for_cross_validation(
            [
                DerivedFeatureDefinition(
                    name="zs",
                    expression_type=DerivedFeatureType.FORMULA,
                    expression="zscore(x)",
                )
            ]
        )

    assert_derived_specs_safe_for_cross_validation(
        [
            {
                "name": "safe_log",
                "expression_type": "formula",
                "expression": "log1p(x)",
            }
        ]
    )


def test_sklearn_trainer_rejects_zscore_derived_before_cv(tmp_path: Path):
    n = 40
    df = pd.DataFrame(
        {
            "x": np.random.randn(n),
            "target": [0, 1] * (n // 2),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    trainer = SklearnTrainer()
    trial = MagicMock(model_id="logistic_regression", parameters={"max_iter": 200})
    execution = TrialExecution(
        trial=trial,
        experiment=MagicMock(id="exp_1"),
        run=MagicMock(id="run_1"),
        feature_names=["x", "zs"],
        dataset_path=str(csv_path),
        target_column="target",
        task_type="binary_classification",
        metric="roc_auc",
        validation_strategy="kfold",
        test_size=0.2,
        cv_folds=3,
        random_seed=42,
        derived_features=[
            {
                "name": "zs",
                "expression_type": "formula",
                "expression": "zscore(x)",
            }
        ],
    )
    result = trainer.run(execution)
    assert result.succeeded is False
    assert "stateful aggregate" in (result.failure_reason or "").lower()


def test_sklearn_trainer_rowwise_derived_ok(tmp_path: Path):
    n = 40
    df = pd.DataFrame(
        {
            "x": np.abs(np.random.randn(n)) + 0.1,
            "target": [0, 1] * (n // 2),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    trainer = SklearnTrainer()
    trial = MagicMock(model_id="logistic_regression", parameters={"max_iter": 200})
    execution = TrialExecution(
        trial=trial,
        experiment=MagicMock(id="exp_1"),
        run=MagicMock(id="run_1"),
        feature_names=["x", "lx"],
        dataset_path=str(csv_path),
        target_column="target",
        task_type="binary_classification",
        metric="roc_auc",
        validation_strategy="kfold",
        test_size=0.2,
        cv_folds=3,
        random_seed=42,
        derived_features=[
            {
                "name": "lx",
                "expression_type": "formula",
                "expression": "log1p(x)",
            }
        ],
    )
    result = trainer.run(execution)
    assert result.succeeded is True


def test_time_series_requires_time_column(tmp_path: Path):
    n = 40
    df = pd.DataFrame(
        {
            "x": np.random.randn(n),
            "target": [0, 1] * (n // 2),
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    trainer = SklearnTrainer()
    run = MagicMock(id="run_1")
    run.config = MagicMock()
    run.config.extra = {}
    execution = TrialExecution(
        trial=MagicMock(model_id="logistic_regression", parameters={"max_iter": 200}),
        experiment=MagicMock(id="exp_1"),
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
    result = trainer.run(execution)
    assert result.succeeded is False
    assert "time_column" in (result.failure_reason or "")


def test_time_series_sorts_by_time_column(tmp_path: Path):
    n = 48
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    # Shuffle so unsorted CSV would be wrong without explicit sort
    order = np.random.permutation(n)
    df = pd.DataFrame(
        {
            "x": np.linspace(0, 1, n)[order],
            "event_time": times[order],
            "target": ([0, 1] * (n // 2)),
        }
    )
    # Align target with shuffled rows for a valid binary task
    df["target"] = [0, 1] * (n // 2)
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    trainer = SklearnTrainer()
    run = MagicMock(id="run_1")
    run.config = MagicMock()
    run.config.extra = {"time_column": "event_time"}
    execution = TrialExecution(
        trial=MagicMock(model_id="logistic_regression", parameters={"max_iter": 200}),
        experiment=MagicMock(id="exp_1"),
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
    result = trainer.run(execution)
    assert result.succeeded is True


def test_run_experiment_fails_when_all_trials_fail(tmp_path: Path):
    ws = AutoMLWorkspace.create("fail_all", root_dir=tmp_path / "ws")
    df = pd.DataFrame(
        {
            "x1": np.random.randn(40),
            "x2": np.random.randn(40),
            "target": [0, 1] * 20,
        }
    )
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="roc_auc")
    exp = ws.create_experiment(
        run=run,
        name="broken",
        feature_names=["x1", "x2", "leak_zs"],
        model_ids=["logistic_regression"],
        derived_features=[
            {
                "name": "leak_zs",
                "expression_type": "formula",
                "expression": "zscore(x1)",
            }
        ],
    )

    results = ws.run_experiment(run, exp)
    assert len(results) == 1
    assert results[0].succeeded is False

    refreshed_exp = ws.repository.get_experiment(exp.id)
    refreshed_run = ws.repository.get_run(run.id)
    assert refreshed_exp.status == ExperimentStatus.FAILED
    assert refreshed_run.status == RunStatus.FAILED


def test_safe_formula_preview_allows_stateful_with_warning():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    calc = SafeFormulaCalculator(df, allow_stateful_aggregates=True)
    res, warns = calc.evaluate("zscore(y)")
    assert len(res) == 4
    assert any("Stateful aggregate" in w for w in warns)
