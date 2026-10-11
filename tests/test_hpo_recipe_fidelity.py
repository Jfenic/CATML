"""HPO must preserve experiment recipes and fail closed when all trials fail."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.experiments.trial import ExperimentStatus, TrialResult
from automl.domain.ports import TrialExecution
from automl.domain.runs.states import RunStatus
from automl.engine.training.sklearn_trainer import SklearnTrainer


def _binary_csv(tmp_path: Path, n: int = 60) -> Path:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n),
            "num2": np.abs(rng.normal(size=n)) + 0.1,
            "target": [0, 1] * (n // 2),
        }
    )
    path = tmp_path / "hpo_data.csv"
    df.to_csv(path, index=False)
    return path


def test_optimize_experiment_passes_recipe_transforms_to_trainer(tmp_path: Path):
    ws = AutoMLWorkspace.create("hpo_recipe", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    ds = ws.register_dataset(
        name="data",
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
        }
    ]
    exp = ws.create_experiment(
        run=run,
        name="recipe_hpo",
        feature_names=["num1", "num2", "num2_log1p"],
        model_ids=["logistic_regression"],
        column_transforms=transforms,
        derived_features=derived,
    )

    captured: list[TrialExecution] = []
    real_run = SklearnTrainer.run

    def spy_run(self, execution: TrialExecution) -> TrialResult:
        captured.append(execution)
        return real_run(self, execution)

    with patch.object(SklearnTrainer, "run", spy_run):
        out = ws.optimize_experiment(
            run_id=run.id,
            experiment_id=exp.id,
            n_trials=2,
            optimizer="random",
        )

    assert out["successful_trials"] >= 1
    assert len(captured) >= 1
    for execution in captured:
        assert execution.column_transforms == transforms
        assert execution.derived_features == derived
        assert "num2_log1p" in execution.feature_names


def test_optimize_and_train_share_recipe_pipeline_ops(tmp_path: Path):
    """Normal training and HPO must both apply the same recipe ops."""
    ws = AutoMLWorkspace.create("hpo_parity", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="roc_auc")
    transforms = [{"column": "num1", "kind": "yeo_johnson", "params": {}}]
    derived = [
        {
            "name": "num2_log1p",
            "expression_type": "formula",
            "expression": "log1p(num2)",
        }
    ]
    exp = ws.create_experiment(
        run=run,
        name="parity",
        feature_names=["num1", "num2", "num2_log1p"],
        model_ids=["logistic_regression"],
        column_transforms=transforms,
        derived_features=derived,
        validation_strategy="kfold",
    )

    train_execs: list[TrialExecution] = []
    hpo_execs: list[TrialExecution] = []
    real_run = SklearnTrainer.run

    def train_spy(self, execution: TrialExecution) -> TrialResult:
        train_execs.append(execution)
        return real_run(self, execution)

    with patch.object(SklearnTrainer, "run", train_spy):
        ws.run_experiment(run, exp)

    # Fresh experiment for HPO (same recipe) — run may be COMPLETED; create new run
    run2 = ws.create_run(ds, metric="roc_auc")
    exp2 = ws.create_experiment(
        run=run2,
        name="parity_hpo",
        feature_names=["num1", "num2", "num2_log1p"],
        model_ids=["logistic_regression"],
        column_transforms=transforms,
        derived_features=derived,
        validation_strategy="kfold",
    )

    def hpo_spy(self, execution: TrialExecution) -> TrialResult:
        hpo_execs.append(execution)
        return real_run(self, execution)

    with patch.object(SklearnTrainer, "run", hpo_spy):
        ws.optimize_experiment(
            run_id=run2.id,
            experiment_id=exp2.id,
            n_trials=2,
            optimizer="random",
        )

    assert train_execs and hpo_execs
    assert train_execs[0].column_transforms == hpo_execs[0].column_transforms
    assert train_execs[0].derived_features == hpo_execs[0].derived_features


def test_optimize_experiment_all_trials_fail_marks_failed(tmp_path: Path):
    ws = AutoMLWorkspace.create("hpo_fail", root_dir=tmp_path / "ws")
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
        name="broken_hpo",
        feature_names=["num1", "num2", "leak_zs"],
        model_ids=["logistic_regression"],
        derived_features=[
            {
                "name": "leak_zs",
                "expression_type": "formula",
                "expression": "zscore(num1)",
            }
        ],
    )

    out = ws.optimize_experiment(
        run_id=run.id,
        experiment_id=exp.id,
        n_trials=3,
        optimizer="random",
    )

    assert out["trials_executed"] == 3
    assert out["successful_trials"] == 0
    assert out["best_trial_id"] == ""
    assert out["best_score"] is None
    assert out["best_params"] == {}
    assert all(not t["succeeded"] for t in out["trials"])

    refreshed_exp = ws.repository.get_experiment(exp.id)
    refreshed_run = ws.repository.get_run(run.id)
    assert refreshed_exp.status == ExperimentStatus.FAILED
    assert refreshed_run.status == RunStatus.FAILED
