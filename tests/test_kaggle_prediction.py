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
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.interfaces.cli.main import main as cli_main


@pytest.fixture
def kaggle_data(tmp_path: Path) -> tuple[str, str]:
    np.random.seed(42)
    n_train = 120
    n_test = 30

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


def test_trainer_fit_and_predict() -> None:
    trainer = SklearnTrainer()
    X_train = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cat": ["A", "B", "A", "B", "A", "B"],
    })
    y_train = pd.Series([0, 0, 0, 1, 1, 1])
    X_test = pd.DataFrame({
        "num": [1.5, 5.5],
        "cat": ["A", "C"],  # unseen category 'C'
    })

    # Test label predictions
    preds = trainer.fit_and_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model_id="logistic_regression",
        predict_proba=False,
    )
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})

    # Test probability predictions
    probs = trainer.fit_and_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model_id="logistic_regression",
        predict_proba=True,
    )
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_workspace_predict_and_submission(kaggle_data: tuple[str, str], tmp_path: Path) -> None:
    train_path, test_path = kaggle_data
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))

    dataset = ws.register_dataset(
        name="kaggle_churn",
        path=train_path,
        target="churn",
    )
    run = ws.create_run(dataset, metric="roc_auc")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="churn_baseline",
            feature_names=["salary", "debt", "country"],
            model_ids=["random_forest", "lightgbm"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    # Test predict query (probabilities)
    probs = qry.dispatch(
        PredictDatasetQuery(
            run_id=run.id,
            test_dataset_path=test_path,
            predict_proba=True,
        )
    )
    assert len(probs) == 30
    assert all(0.0 <= p <= 1.0 for p in probs)

    # Test generate submission command
    submission_path = tmp_path / "submission.csv"
    res = cmd.dispatch(
        GenerateSubmissionCommand(
            run_id=run.id,
            test_dataset_path=test_path,
            output_path=str(submission_path),
            id_column="customer_id",
            predict_proba=True,
        )
    )

    assert submission_path.exists()
    assert res["row_count"] == 30
    assert res["id_column"] == "customer_id"

    sub_df = pd.read_csv(submission_path)
    assert list(sub_df.columns) == ["customer_id", "churn"]
    assert len(sub_df) == 30
    assert sub_df["customer_id"].iloc[0] == "CUST_1000"
    assert sub_df["churn"].dtype == float
    assert not sub_df["churn"].isna().any()


def test_cli_predict_command(kaggle_data: tuple[str, str], tmp_path: Path) -> None:
    train_path, test_path = kaggle_data
    ws_dir = str(tmp_path / "cli_ws")
    ws, cmd, _ = build_application(root_dir=ws_dir)

    dataset = ws.register_dataset(
        name="kaggle_churn_cli",
        path=train_path,
        target="churn",
    )
    run = ws.create_run(dataset, metric="roc_auc")
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="cli_exp",
            feature_names=["salary", "debt"],
            model_ids=["logistic_regression"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    output_csv = str(tmp_path / "cli_sub.csv")

    # CLI test
    rc = cli_main([
        "predict",
        "--workspace", ws_dir,
        "--run-id", run.id,
        "--test-dataset", test_path,
        "--output", output_csv,
        "--id-column", "customer_id",
        "--proba",
    ])
    assert rc == 0
    assert Path(output_csv).exists()

    # CLI json test
    rc_json = cli_main([
        "predict",
        "--workspace", ws_dir,
        "--run-id", run.id,
        "--test-dataset", test_path,
        "--output", str(tmp_path / "cli_sub_json.csv"),
        "--json",
    ])
    assert rc_json == 0
