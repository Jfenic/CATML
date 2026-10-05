from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    BuildEnsembleCommand,
    CreateExperimentCommand,
    RunExperimentCommand,
)
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.trial import ExperimentStatus
from automl.domain.ports import TrialExecution
from automl.engine.training.sklearn_trainer import SklearnTrainer


@pytest.fixture
def binary_data():
    np.random.seed(42)
    n = 60
    X = pd.DataFrame({
        "num_1": np.random.randn(n),
        "num_2": np.random.randn(n),
        "cat_1": ["A", "B", "C"] * (n // 3),
    })
    y = pd.Series([0, 1] * (n // 2), name="target")
    return X, y


def test_sklearn_trainer_validation_strategies(tmp_path, binary_data):
    X, y = binary_data
    csv_path = tmp_path / "data.csv"
    pd.concat([X, y], axis=1).to_csv(csv_path, index=False)
    trainer = SklearnTrainer()

    trial = MagicMock(model_id="logistic_regression", parameters={"max_iter": 200})
    exp = MagicMock(id="exp_1")
    run = MagicMock(id="run_1")

    # Stratified K-Fold
    exec_stratified = TrialExecution(
        trial=trial,
        experiment=exp,
        run=run,
        feature_names=["num_1", "num_2", "cat_1"],
        dataset_path=str(csv_path),
        target_column="target",
        task_type="binary_classification",
        metric="roc_auc",
        validation_strategy="stratified_kfold",
        test_size=0.2,
        cv_folds=3,
        random_seed=42,
    )
    res_strat = trainer.run(exec_stratified)
    assert res_strat.succeeded is True
    assert 0.0 <= res_strat.primary_score <= 1.0
    assert "cv_std" in res_strat.secondary_metrics

    # Standard K-Fold
    exec_kfold = TrialExecution(
        trial=trial,
        experiment=exp,
        run=run,
        feature_names=["num_1", "num_2", "cat_1"],
        dataset_path=str(csv_path),
        target_column="target",
        task_type="binary_classification",
        metric="roc_auc",
        validation_strategy="kfold",
        test_size=0.2,
        cv_folds=3,
        random_seed=42,
    )
    res_kfold = trainer.run(exec_kfold)
    assert res_kfold.succeeded is True
    assert "cv_std" in res_kfold.secondary_metrics

    # Time-Series Split
    exec_ts = TrialExecution(
        trial=trial,
        experiment=exp,
        run=run,
        feature_names=["num_1", "num_2", "cat_1"],
        dataset_path=str(csv_path),
        target_column="target",
        task_type="binary_classification",
        metric="roc_auc",
        validation_strategy="time_series",
        test_size=0.2,
        cv_folds=3,
        random_seed=42,
    )
    res_ts = trainer.run(exec_ts)
    assert res_ts.succeeded is True
    assert "cv_std" in res_ts.secondary_metrics


def test_build_ensemble_workspace_and_command(tmp_path, binary_data):
    X, y = binary_data
    csv_path = tmp_path / "train.csv"
    pd.concat([X, y], axis=1).to_csv(csv_path, index=False)

    ws, cmd, qry = build_application(root_dir=str(tmp_path / "workspace"))
    ds = ws.register_dataset(name="churn", path=str(csv_path), target="target")
    run = ws.create_run(ds, metric="roc_auc")

    # Train baseline experiment
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="baseline",
            feature_names=["num_1", "num_2", "cat_1"],
            model_ids=["logistic_regression", "random_forest"],
            validation_strategy="stratified_kfold",
        )
    )
    cmd.dispatch(RunExperimentCommand(run.id, exp.id))

    # Build Ensemble: Average
    ens_avg_id = cmd.dispatch(
        BuildEnsembleCommand(
            run_id=run.id,
            model_ids=["logistic_regression", "random_forest"],
            method="average",
            folds=3,
        )
    )
    ens_avg = ws.repository.get_experiment(ens_avg_id)
    assert ens_avg.status == ExperimentStatus.COMPLETED
    trial_avg = ws.repository.list_trial_results(ens_avg_id)[0]
    assert 0.0 <= trial_avg.primary_score <= 1.0

    # Build Ensemble: Rank
    ens_rank_id = cmd.dispatch(
        BuildEnsembleCommand(
            run_id=run.id,
            model_ids=["logistic_regression", "random_forest"],
            method="rank",
            folds=3,
        )
    )
    ens_rank = ws.repository.get_experiment(ens_rank_id)
    assert ens_rank.status == ExperimentStatus.COMPLETED
    trial_rank = ws.repository.list_trial_results(ens_rank_id)[0]
    assert 0.0 <= trial_rank.primary_score <= 1.0

    # Build Ensemble: Simplex
    ens_simplex_id = cmd.dispatch(
        BuildEnsembleCommand(
            run_id=run.id,
            model_ids=["logistic_regression", "random_forest"],
            method="simplex",
            folds=3,
        )
    )
    ens_simplex = ws.repository.get_experiment(ens_simplex_id)
    assert ens_simplex.status == ExperimentStatus.COMPLETED
    trial_simplex = ws.repository.list_trial_results(ens_simplex_id)[0]
    assert 0.0 <= trial_simplex.primary_score <= 1.0

    # Build Ensemble: Stacked
    ens_stack_id = cmd.dispatch(
        BuildEnsembleCommand(
            run_id=run.id,
            model_ids=["logistic_regression", "random_forest"],
            method="stacked",
            meta_model="ridge",
            folds=3,
        )
    )
    ens_stack = ws.repository.get_experiment(ens_stack_id)
    assert ens_stack.status == ExperimentStatus.COMPLETED
    trial_stack = ws.repository.list_trial_results(ens_stack_id)[0]
    assert 0.0 <= trial_stack.primary_score <= 1.0

    # Check error conditions
    with pytest.raises(ValueError, match="at least 2 distinct"):
        cmd.dispatch(BuildEnsembleCommand(run_id=run.id, model_ids=["logistic_regression"]))

    with pytest.raises(ValueError, match="Unknown ensemble method"):
        cmd.dispatch(BuildEnsembleCommand(run_id=run.id, model_ids=["logistic_regression", "random_forest"], method="invalid"))


def test_web_endpoints_ensemble_and_kaggle(tmp_path, binary_data, monkeypatch):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.request import Request, urlopen
    from automl.interfaces.web.server import AutoMLWebHandler

    X, y = binary_data
    csv_path = tmp_path / "train.csv"
    pd.concat([X, y], axis=1).to_csv(csv_path, index=False)

    ws, cmd, qry = build_application(root_dir=str(tmp_path / "workspace"))
    ds = ws.register_dataset(name="churn", path=str(csv_path), target="target")
    run = ws.create_run(ds, metric="roc_auc")

    monkeypatch.setattr(AutoMLWebHandler, "workspace_dir", str(ws.root_dir))
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        # Test 1: POST /api/kaggle/upload-template
        template_csv = "id,Probability\n101,0.0\n102,0.0\n103,0.0\n"
        req_up = Request(
            f"{base_url}/api/kaggle/upload-template",
            data=json.dumps({"filename": "sample_submission.csv", "content": template_csv}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req_up) as resp:
            up_res = json.load(resp)
        assert up_res["status"] == "success"
        assert up_res["id_column"] == "id"
        assert up_res["target_column"] == "Probability"
        assert up_res["row_count"] == 3
        template_file = Path(up_res["template_path"])
        assert template_file.exists()

        # Test 2: GET /api/kaggle/download
        req_dl = Request(f"{base_url}/api/kaggle/download?file={template_file}")
        with urlopen(req_dl) as resp:
            content = resp.read().decode("utf-8")
            disposition = resp.headers.get("Content-Disposition")
        assert content == template_csv
        assert "attachment" in disposition

        # Test 3: POST /api/ensemble/build
        req_ens = Request(
            f"{base_url}/api/ensemble/build",
            data=json.dumps({
                "run_id": run.id,
                "models": ["logistic_regression", "random_forest"],
                "method": "simplex",
                "folds": 3,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(req_ens) as resp:
            ens_res = json.load(resp)
        assert ens_res["status"] == "success"
        assert ens_res["experiment_id"]
        assert ens_res["score"] is not None
        assert ens_res["method"] == "simplex"

    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
