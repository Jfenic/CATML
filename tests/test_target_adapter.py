from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    RunExperimentCommand,
)
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.engine.training.target_adapter import TargetAdapter


def test_target_adapter_binary_string_labels() -> None:
    adapter = TargetAdapter()
    y_raw = pd.Series(["No", "Yes", "No", "Yes", "No"])
    y_enc = adapter.fit_transform(y_raw, task_type="binary_classification")

    assert adapter.is_encoded is True
    assert list(adapter.classes_) == ["No", "Yes"]
    assert np.array_equal(y_enc, np.array([0, 1, 0, 1, 0]))

    # Test inverse transform
    y_dec = adapter.inverse_transform(y_enc)
    assert list(y_dec) == ["No", "Yes", "No", "Yes", "No"]


def test_target_adapter_multiclass_string_labels() -> None:
    adapter = TargetAdapter()
    y_raw = pd.Series(["cat", "dog", "bird", "dog"])
    y_enc = adapter.fit_transform(y_raw, task_type="multiclass_classification")

    assert adapter.is_encoded is True
    assert set(adapter.classes_) == {"bird", "cat", "dog"}

    y_dec = adapter.inverse_transform(y_enc)
    assert list(y_dec) == ["cat", "dog", "bird", "dog"]


def test_target_adapter_regression_passthrough() -> None:
    adapter = TargetAdapter()
    y_raw = pd.Series([10.5, 20.3, 30.1])
    y_out = adapter.fit_transform(y_raw, task_type="regression")

    assert adapter.is_encoded is False
    assert adapter.classes_ is None
    assert np.array_equal(y_out, y_raw.to_numpy())


def test_target_adapter_integer_already_clean() -> None:
    adapter = TargetAdapter()
    y_raw = pd.Series([0, 1, 0, 1])
    y_out = adapter.fit_transform(y_raw, task_type="binary_classification")

    assert adapter.is_encoded is False
    assert np.array_equal(y_out, y_raw.to_numpy())


def test_xgboost_with_string_target_labels(tmp_path: Path) -> None:
    """Verifies resolution of Issue #8: XGBoost must run cleanly with string targets."""
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws_xgb"))

    train_df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cat1": ["A", "B", "A", "B", "A", "B"],
        "target": ["No", "Yes", "No", "Yes", "No", "Yes"],
    })
    train_path = tmp_path / "train.csv"
    train_df.to_csv(train_path, index=False)

    dataset = ws.register_dataset(name="ds_xgb", path=str(train_path), target="target")
    run = ws.create_run(dataset, metric="roc_auc")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="exp_xgb_string",
            feature_names=["num1", "cat1"],
            model_ids=["xgboost"],
        )
    )

    results = cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))
    assert len(results) == 1
    assert results[0].succeeded is True
    assert results[0].primary_score > 0.0

    # Verify fit_and_predict with probability outputs
    trainer = SklearnTrainer(plugin_registry=ws.plugin_registry)
    test_df = pd.DataFrame({"num1": [2.5, 3.5], "cat1": ["A", "B"]})
    probs = trainer.fit_and_predict(
        X_train=train_df[["num1", "cat1"]],
        y_train=train_df["target"],
        X_test=test_df,
        model_id="xgboost",
        task_type="binary_classification",
        predict_proba=True,
    )
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Verify fit_and_predict with label outputs (must decode to original strings)
    labels = trainer.fit_and_predict(
        X_train=train_df[["num1", "cat1"]],
        y_train=train_df["target"],
        X_test=test_df,
        model_id="xgboost",
        task_type="binary_classification",
        predict_proba=False,
    )
    assert len(labels) == 2
    assert set(labels).issubset({"No", "Yes"})
