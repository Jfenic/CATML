from __future__ import annotations

import json
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
from automl.interfaces.cli.main import main as cli_main


@pytest.fixture
def mock_dataset_and_template(tmp_path: Path) -> tuple[str, str, str]:
    # 50 training samples
    np.random.seed(42)
    n_train = 50
    train_ids = [f"ID_{i:03d}" for i in range(n_train)]
    x1 = np.random.uniform(10, 100, size=n_train)
    x2 = np.random.uniform(0, 1, size=n_train)
    y = (x1 * 0.05 + x2 > 2.5).astype(int)

    train_df = pd.DataFrame({
        "id": train_ids,
        "feature1": x1,
        "feature2": x2,
        "churn": y,
    })

    # 10 test samples in sequential order: ID_100 to ID_109
    test_ids = [f"ID_{i:03d}" for i in range(100, 110)]
    test_x1 = np.linspace(10, 100, 10)
    test_x2 = np.linspace(0, 1, 10)
    test_df = pd.DataFrame({
        "id": test_ids,
        "feature1": test_x1,
        "feature2": test_x2,
    })

    # Template with shuffled IDs and different target column name ("Probability")
    shuffled_ids = [
        "ID_108", "ID_102", "ID_105", "ID_100", "ID_109",
        "ID_101", "ID_107", "ID_103", "ID_106", "ID_104"
    ]
    template_df = pd.DataFrame({
        "id": shuffled_ids,
        "Probability": [0.5] * len(shuffled_ids),
    })

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    template_path = tmp_path / "sample_submission.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    template_df.to_csv(template_path, index=False)

    return str(train_path), str(test_path), str(template_path)


def test_submission_matches_template_columns_and_row_order(
    mock_dataset_and_template: tuple[str, str, str], tmp_path: Path
) -> None:
    train_path, test_path, template_path = mock_dataset_and_template
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "app"))

    dataset = ws.register_dataset(name="churn_data", path=train_path, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="exp_test",
            feature_names=["feature1", "feature2"],
            model_ids=["logistic_regression"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    # Standard unaligned prediction for reference
    raw_preds = qry.dispatch(
        PredictDatasetQuery(
            run_id=run.id,
            test_dataset_path=test_path,
            predict_proba=True,
        )
    )
    test_df = pd.read_csv(test_path)
    id_to_pred = dict(zip(test_df["id"], raw_preds))

    # Generate submission using template
    out_csv = tmp_path / "submission_aligned.csv"
    res = cmd.dispatch(
        GenerateSubmissionCommand(
            run_id=run.id,
            test_dataset_path=test_path,
            output_path=str(out_csv),
            template_path=template_path,
            predict_proba=True,
        )
    )

    assert out_csv.exists()
    assert res["row_count"] == 10
    assert res["id_column"] == "id"
    assert res["target_column"] == "Probability"
    assert res["template_used"] is not None

    sub_df = pd.read_csv(out_csv)
    template_df = pd.read_csv(template_path)

    # 1. Exact columns and order match template
    assert list(sub_df.columns) == ["id", "Probability"]

    # 2. Exact row order and IDs match template
    assert list(sub_df["id"]) == list(template_df["id"])

    # 3. Predictions align accurately with each ID
    for _, row in sub_df.iterrows():
        expected_prob = id_to_pred[row["id"]]
        assert pytest.approx(row["Probability"], abs=1e-5) == expected_prob


def test_predict_query_with_template_alignment(
    mock_dataset_and_template: tuple[str, str, str], tmp_path: Path
) -> None:
    train_path, test_path, template_path = mock_dataset_and_template
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "app_qry"))

    dataset = ws.register_dataset(name="churn_data", path=train_path, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="exp_test",
            feature_names=["feature1", "feature2"],
            model_ids=["logistic_regression"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    test_df = pd.read_csv(test_path)
    template_df = pd.read_csv(template_path)

    raw_preds = qry.dispatch(
        PredictDatasetQuery(
            run_id=run.id,
            test_dataset_path=test_path,
            predict_proba=True,
        )
    )
    id_to_pred = dict(zip(test_df["id"], raw_preds))

    aligned_preds = qry.dispatch(
        PredictDatasetQuery(
            run_id=run.id,
            test_dataset_path=test_path,
            predict_proba=True,
            template_path=template_path,
        )
    )

    assert len(aligned_preds) == len(template_df)
    for tid, pred in zip(template_df["id"], aligned_preds):
        assert pytest.approx(pred, abs=1e-5) == id_to_pred[tid]


def test_template_with_custom_id_column(tmp_path: Path) -> None:
    # Test Kaggle competition pattern like PassengerId and Survived
    train_df = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4],
        "Age": [22.0, 38.0, 26.0, 35.0],
        "Survived": [0, 1, 1, 0],
    })
    test_df = pd.DataFrame({
        "PassengerId": [5, 6, 7],
        "Age": [54.0, 2.0, 27.0],
    })
    template_df = pd.DataFrame({
        "PassengerId": [7, 5, 6],
        "Survived": [0, 0, 0],
    })

    train_path = tmp_path / "titanic_train.csv"
    test_path = tmp_path / "titanic_test.csv"
    template_path = tmp_path / "titanic_sub.csv"
    out_path = tmp_path / "submission.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    template_df.to_csv(template_path, index=False)

    ws, cmd, _ = build_application(root_dir=str(tmp_path / "app_titanic"))
    dataset = ws.register_dataset(name="titanic", path=train_path, target="Survived")
    run = ws.create_run(dataset, metric="accuracy")
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="tit_exp",
            feature_names=["Age"],
            model_ids=["logistic_regression"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    # Auto-detect PassengerId
    res = cmd.dispatch(
        GenerateSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_path),
            output_path=str(out_path),
            template_path=str(template_path),
            predict_proba=False,
        )
    )

    assert res["id_column"] == "PassengerId"
    assert res["target_column"] == "Survived"
    sub_df = pd.read_csv(out_path)
    assert list(sub_df.columns) == ["PassengerId", "Survived"]
    assert list(sub_df["PassengerId"]) == [7, 5, 6]


def test_template_error_handling(
    mock_dataset_and_template: tuple[str, str, str], tmp_path: Path
) -> None:
    train_path, test_path, _ = mock_dataset_and_template
    ws, cmd, _ = build_application(root_dir=str(tmp_path / "app_err"))

    dataset = ws.register_dataset(name="err_data", path=train_path, target="churn")
    run = ws.create_run(dataset)
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="exp",
            feature_names=["feature1"],
            model_ids=["logistic_regression"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    # 1. Missing IDs in test predictions
    bad_template_df = pd.DataFrame({
        "id": ["ID_999", "ID_100"],  # ID_999 does not exist in test set
        "churn": [0, 0],
    })
    bad_template_path = tmp_path / "bad_template.csv"
    bad_template_df.to_csv(bad_template_path, index=False)

    with pytest.raises(ValueError, match="Template contains 1 IDs not found"):
        cmd.dispatch(
            GenerateSubmissionCommand(
                run_id=run.id,
                test_dataset_path=test_path,
                output_path=str(tmp_path / "out.csv"),
                template_path=str(bad_template_path),
            )
        )

    # 2. Empty template
    empty_template_path = tmp_path / "empty_template.csv"
    pd.DataFrame().to_csv(empty_template_path, index=False)

    with pytest.raises(ValueError, match="Template CSV is empty"):
        cmd.dispatch(
            GenerateSubmissionCommand(
                run_id=run.id,
                test_dataset_path=test_path,
                output_path=str(tmp_path / "out.csv"),
                template_path=str(empty_template_path),
            )
        )

    # 3. Specified id_column not in template
    with pytest.raises(ValueError, match="Specified ID column 'nonexistent' not found"):
        cmd.dispatch(
            GenerateSubmissionCommand(
                run_id=run.id,
                test_dataset_path=test_path,
                output_path=str(tmp_path / "out.csv"),
                template_path=str(bad_template_path),
                id_column="nonexistent",
            )
        )

    # 4. Template without target column
    single_col_path = tmp_path / "single_col.csv"
    pd.DataFrame({"id": ["ID_100", "ID_101"]}).to_csv(single_col_path, index=False)

    with pytest.raises(ValueError, match="Template CSV must contain at least one target column"):
        cmd.dispatch(
            GenerateSubmissionCommand(
                run_id=run.id,
                test_dataset_path=test_path,
                output_path=str(tmp_path / "out.csv"),
                template_path=str(single_col_path),
            )
        )


def test_cli_predict_with_template(
    mock_dataset_and_template: tuple[str, str, str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train_path, test_path, template_path = mock_dataset_and_template
    ws_dir = str(tmp_path / "cli_ws")
    ws, cmd, _ = build_application(root_dir=ws_dir)

    dataset = ws.register_dataset(name="cli_data", path=train_path, target="churn")
    run = ws.create_run(dataset)
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="cli_exp",
            feature_names=["feature1"],
            model_ids=["logistic_regression"],
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    out_csv = str(tmp_path / "cli_submission.csv")

    rc = cli_main([
        "predict",
        "--workspace", ws_dir,
        "--run-id", run.id,
        "--test-dataset", test_path,
        "--output", out_csv,
        "--template", template_path,
        "--proba",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Template:" in captured.out
    assert Path(out_csv).exists()

    sub_df = pd.read_csv(out_csv)
    template_df = pd.read_csv(template_path)
    assert list(sub_df["id"]) == list(template_df["id"])
    assert list(sub_df.columns) == list(template_df.columns)
