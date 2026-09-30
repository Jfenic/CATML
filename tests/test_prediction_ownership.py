"""Regression tests for Issue #14: prediction run ownership validation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    GenerateSubmissionCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import PredictDatasetQuery


@pytest.fixture
def churn_data(tmp_path: Path) -> tuple[str, str]:
    np.random.seed(42)
    n_train = 60
    n_test = 20

    def make_df(n: int, include_target: bool = True) -> pd.DataFrame:
        customer_ids = [f"CUST_{i:04d}" for i in range(1000, 1000 + n)]
        salary = np.random.uniform(20000, 100000, size=n)
        debt = np.random.uniform(1000, 50000, size=n)
        country = np.random.choice(["Spain", "France", "Germany"], size=n)
        data = {
            "customer_id": customer_ids,
            "salary": salary,
            "debt": debt,
            "country": country,
        }
        if include_target:
            churn = ((salary / 100000) * 0.7 - (debt / 50000) * 0.3 > 0.2).astype(int)
            data["churn"] = churn
        return pd.DataFrame(data)

    train_df = make_df(n_train, include_target=True)
    test_df = make_df(n_test, include_target=False)

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return str(train_path), str(test_path)


def test_predict_rejects_foreign_experiment_id(tmp_path: Path, churn_data: tuple[str, str]) -> None:
    """Verify that predicting with an experiment from another run raises ValueError."""
    train_path, test_path = churn_data
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "workspace"))

    dataset = ws.register_dataset(name="churn_ds", path=train_path, target="churn")
    run_a = ws.create_run(dataset, metric="roc_auc")
    run_b = ws.create_run(dataset, metric="roc_auc")

    # Create and train experiment in run_a
    exp_a = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run_a.id,
            name="exp_run_a",
            feature_names=["salary", "debt"],
            model_ids=["logistic_regression"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run_a.id, experiment_id=exp_a.id))

    # Calling predict for run_b with exp_a.id must raise ValueError
    with pytest.raises(ValueError, match=f"Experiment '{exp_a.id}' does not belong to run '{run_b.id}'"):
        qry.dispatch(
            PredictDatasetQuery(
                run_id=run_b.id,
                test_dataset_path=test_path,
                experiment_id=exp_a.id,
            )
        )

    # Calling workspace.predict directly must also raise ValueError
    with pytest.raises(ValueError, match=f"Experiment '{exp_a.id}' does not belong to run '{run_b.id}'"):
        ws.predict(
            run_id=run_b.id,
            test_dataset_path=test_path,
            experiment_id=exp_a.id,
        )


def test_predict_rejects_foreign_trial_id(tmp_path: Path, churn_data: tuple[str, str]) -> None:
    """Verify that predicting with a trial from another run raises ValueError."""
    train_path, test_path = churn_data
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "workspace"))

    dataset = ws.register_dataset(name="churn_ds", path=train_path, target="churn")
    run_a = ws.create_run(dataset, metric="roc_auc")
    run_b = ws.create_run(dataset, metric="roc_auc")

    exp_a = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run_a.id,
            name="exp_run_a",
            feature_names=["salary", "debt"],
            model_ids=["logistic_regression"],
        )
    )
    trials = cmd.dispatch(RunExperimentCommand(run_id=run_a.id, experiment_id=exp_a.id))
    trial_a = trials[0]

    # Calling predict for run_b with trial_a.trial_id must raise ValueError
    with pytest.raises(ValueError, match=f"Trial '{trial_a.trial_id}' .* does not belong to run '{run_b.id}'"):
        qry.dispatch(
            PredictDatasetQuery(
                run_id=run_b.id,
                test_dataset_path=test_path,
                trial_id=trial_a.trial_id,
            )
        )

    with pytest.raises(ValueError, match=f"Trial '{trial_a.trial_id}' .* does not belong to run '{run_b.id}'"):
        ws.predict(
            run_id=run_b.id,
            test_dataset_path=test_path,
            trial_id=trial_a.trial_id,
        )


def test_generate_submission_rejects_foreign_ids(tmp_path: Path, churn_data: tuple[str, str]) -> None:
    """Verify GenerateSubmissionCommand rejects foreign experiment and trial IDs."""
    train_path, test_path = churn_data
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "workspace"))

    dataset = ws.register_dataset(name="churn_ds", path=train_path, target="churn")
    run_a = ws.create_run(dataset, metric="roc_auc")
    run_b = ws.create_run(dataset, metric="roc_auc")

    exp_a = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run_a.id,
            name="exp_run_a",
            feature_names=["salary", "debt"],
            model_ids=["logistic_regression"],
        )
    )
    trials = cmd.dispatch(RunExperimentCommand(run_id=run_a.id, experiment_id=exp_a.id))
    trial_a = trials[0]

    output_sub = str(tmp_path / "sub.csv")

    with pytest.raises(ValueError, match=f"Experiment '{exp_a.id}' does not belong to run '{run_b.id}'"):
        cmd.dispatch(
            GenerateSubmissionCommand(
                run_id=run_b.id,
                test_dataset_path=test_path,
                output_path=output_sub,
                experiment_id=exp_a.id,
            )
        )

    with pytest.raises(ValueError, match=f"Trial '{trial_a.trial_id}' .* does not belong to run '{run_b.id}'"):
        cmd.dispatch(
            GenerateSubmissionCommand(
                run_id=run_b.id,
                test_dataset_path=test_path,
                output_path=output_sub,
                trial_id=trial_a.trial_id,
            )
        )


def test_predict_succeeds_with_owned_experiment_and_trial(tmp_path: Path, churn_data: tuple[str, str]) -> None:
    """Verify that predictions within the same run continue to work as expected."""
    train_path, test_path = churn_data
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "workspace"))

    dataset = ws.register_dataset(name="churn_ds", path=train_path, target="churn")
    run_a = ws.create_run(dataset, metric="roc_auc")

    exp_a = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run_a.id,
            name="exp_run_a",
            feature_names=["salary", "debt"],
            model_ids=["logistic_regression"],
        )
    )
    trials = cmd.dispatch(RunExperimentCommand(run_id=run_a.id, experiment_id=exp_a.id))
    trial_a = trials[0]

    # Predict with owned experiment_id
    preds_exp = qry.dispatch(
        PredictDatasetQuery(
            run_id=run_a.id,
            test_dataset_path=test_path,
            experiment_id=exp_a.id,
        )
    )
    assert len(preds_exp) == 20

    # Predict with owned trial_id
    preds_trial = qry.dispatch(
        PredictDatasetQuery(
            run_id=run_a.id,
            test_dataset_path=test_path,
            trial_id=trial_a.trial_id,
        )
    )
    assert len(preds_trial) == 20
