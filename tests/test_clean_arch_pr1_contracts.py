"""Clean Architecture PR1: MetricSpec + ExecutionSpec contracts."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.experiments.exceptions import InvalidValidationStrategy
from automl.domain.experiments.execution_spec import (
    ExecutionSpec,
    build_execution_spec,
    trial_execution_from_spec,
)
from automl.domain.experiments.validation import require_known_validation_strategy
from automl.domain.metrics.exceptions import InvalidMetric
from automl.domain.metrics.spec import (
    MetricDirection,
    MetricSpec,
    resolve_metric_spec,
)
from automl.domain.ports import TrialExecution
from automl.engine.training.sklearn_trainer import SklearnTrainer


def _binary_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "target": [0, 1] * (n // 2),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_metric_spec_mae_selects_lower_score():
    spec = resolve_metric_spec("mae")
    assert spec.direction is MetricDirection.MINIMIZE
    assert spec.select_best_score([9.0, 4.0, 7.0]) == 4.0
    assert spec.select_best_index([9.0, 4.0, 7.0]) == 1
    assert spec.is_better(4.0, 9.0)
    assert not spec.is_better(9.0, 4.0)


def test_metric_spec_roc_auc_selects_higher_score():
    spec = resolve_metric_spec("roc_auc")
    assert spec.direction is MetricDirection.MAXIMIZE
    assert spec.select_best_score([0.7, 0.9, 0.8]) == 0.9
    assert spec.select_best_index([0.7, 0.9, 0.8]) == 1


def test_resolve_metric_spec_unknown_fails_fast():
    with pytest.raises(InvalidMetric, match="Unknown metric"):
        resolve_metric_spec("not_a_real_metric_xyz")


def test_plugin_metric_greater_is_better_false_respected():
    spec = resolve_metric_spec("custom_cost", greater_is_better=False)
    assert spec.is_minimizing
    assert spec.select_best_score([3.0, 1.0, 2.0]) == 1.0


def test_workspace_resolve_uses_plugin_direction(tmp_path: Path):
    ws = AutoMLWorkspace.create("pr1_plugin_metric", root_dir=tmp_path / "ws")
    plugin = MagicMock()
    plugin.greater_is_better = False
    ws.plugin_registry.get_metric_plugin = MagicMock(return_value=plugin)  # type: ignore[method-assign]

    assert ws._is_minimizing_metric("plugin_loss_metric")
    assert ws._resolve_metric_spec("plugin_loss_metric").select_best_score([5.0, 2.0]) == 2.0


def test_execution_spec_rejects_unknown_validation_strategy():
    with pytest.raises(InvalidValidationStrategy, match="Unknown validation_strategy"):
        require_known_validation_strategy("mystery_cv")

    run = MagicMock()
    run.config.metric = "accuracy"
    run.config.validation_strategy = "mystery_cv"
    run.config.cv_folds = 5
    run.config.random_seed = 42
    run.config.test_size = 0.2
    run.config.task_type = "binary_classification"
    run.config.target = "target"
    run.dataset_id = "ds1"

    experiment = MagicMock()
    experiment.metric = "accuracy"
    experiment.validation_strategy = "mystery_cv"
    experiment.feature_names = ["a"]
    experiment.group_column = None
    experiment.column_transforms = []
    experiment.derived_features = []

    dataset = MagicMock()
    dataset.id = "ds1"
    dataset.target_column = "target"
    dataset.task_type = "binary_classification"
    dataset.path = "/tmp/x.csv"

    with pytest.raises(InvalidValidationStrategy):
        build_execution_spec(run, experiment, dataset, model_id="logistic_regression")


def test_build_execution_spec_normalizes_cv_and_metric():
    run = MagicMock()
    run.config.metric = "roc_auc"
    run.config.validation_strategy = "holdout"
    run.config.cv_folds = 5
    run.config.random_seed = 42
    run.config.test_size = 0.2
    run.config.task_type = "binary_classification"
    run.config.target = "target"
    run.dataset_id = "ds1"

    experiment = MagicMock()
    experiment.metric = "mae"
    experiment.validation_strategy = "k_fold"
    experiment.feature_names = ["a", "b"]
    experiment.group_column = None
    experiment.column_transforms = [{"column": "a", "transform": "standard"}]
    experiment.derived_features = []

    dataset = MagicMock()
    dataset.id = "ds1"
    dataset.target_column = "target"
    dataset.task_type = "binary_classification"
    dataset.path = "/tmp/x.csv"

    spec = build_execution_spec(run, experiment, dataset, model_id="ridge")
    assert isinstance(spec, ExecutionSpec)
    assert spec.validation_strategy == "kfold"
    assert spec.metric == "mae"
    assert spec.metric_spec is not None
    assert spec.metric_spec.is_minimizing
    assert spec.cv_folds == 5
    assert spec.random_seed == 42
    assert spec.model_id == "ridge"
    assert spec.feature_names == ("a", "b")


def test_run_experiment_trial_execution_comes_from_spec(tmp_path: Path):
    ws = AutoMLWorkspace.create("pr1_exec_smoke", root_dir=tmp_path / "ws")
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
        name="smoke",
        feature_names=["a", "b"],
        model_ids=["logistic_regression"],
        validation_strategy="k_fold",
    )

    captured: list[TrialExecution] = []
    real_run = SklearnTrainer.run

    def spy(self, execution: TrialExecution):
        captured.append(execution)
        return real_run(self, execution)

    with patch.object(SklearnTrainer, "run", spy):
        results = ws.run_experiment(run, exp)

    assert results
    assert captured
    assert captured[0].metric == "roc_auc" or captured[0].metric == exp.metric
    assert captured[0].cv_folds == 3
    assert captured[0].random_seed == 7
    assert captured[0].validation_strategy == "kfold"


def test_get_best_trial_and_schedule_agree_on_mae(tmp_path: Path):
    from automl.domain.experiments.trial import TrialResult

    ws = AutoMLWorkspace.create("pr1_best", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="mae")

    fake_results = [
        TrialResult(
            trial_id="t1",
            experiment_id="e",
            model_id="m",
            primary_metric="mae",
            primary_score=9.0,
        ),
        TrialResult(
            trial_id="t2",
            experiment_id="e",
            model_id="m",
            primary_metric="mae",
            primary_score=4.0,
        ),
    ]
    exp = MagicMock()
    exp.id = "exp_mae"
    exp.name = "mae_exp"
    exp.priority = "normal"
    exp.metric = "mae"

    with patch.object(ws, "get_experiment_queue", side_effect=[["c1"], []]):
        with patch.object(ws, "execute_next_experiment", return_value=(exp, fake_results)):
            with patch.object(ws, "_is_run_budget_exhausted", return_value=False):
                out = ws.run_scheduled_experiments(run.id)

    assert out[0]["best_score"] == 4.0

    ws.repository.list_trial_results = MagicMock(return_value=fake_results)  # type: ignore[method-assign]
    ws.repository.get_trial = MagicMock(return_value=MagicMock(parameters={}))  # type: ignore[method-assign]
    best = ws.get_best_trial("e")
    assert best is not None
    assert best["best_score"] == 4.0
    assert best["trial_id"] == "t2"


def test_trial_execution_from_spec_maps_fields():
    run = MagicMock()
    experiment = MagicMock()
    experiment.id = "exp1"
    trial = MagicMock()
    dataset = MagicMock()
    dataset.path = "/data/x.csv"

    spec = ExecutionSpec(
        dataset_id="ds",
        target="y",
        feature_names=("a",),
        model_id="m",
        metric="f1",
        validation_strategy="holdout",
        cv_folds=5,
        random_seed=1,
        test_size=0.25,
        task_type="binary_classification",
        metric_spec=MetricSpec("f1", MetricDirection.MAXIMIZE),
    )
    execution = trial_execution_from_spec(
        trial=trial,
        experiment=experiment,
        run=run,
        dataset=dataset,
        spec=spec,
    )
    assert isinstance(execution, TrialExecution)
    assert execution.metric == "f1"
    assert execution.cv_folds == 5
    assert execution.random_seed == 1
    assert execution.test_size == 0.25
    assert execution.dataset_path == "/data/x.csv"
