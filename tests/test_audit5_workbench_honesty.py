"""Audit-5: Workbench launch contract, honest metrics UI, multiclass ROC-AUC, model backend."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.job_commands import SubmitJobCommand
from automl.application.queries.job_queries import GetJobQuery
from automl.application.services.job_executor import JobExecutor
from automl.domain.experiments.budgets import (
    EXPERIMENT_BUDGET_SECONDS,
    resolve_time_budget_seconds,
)
from automl.domain.experiments.trial import Experiment, ExperimentStatus, Trial
from automl.domain.jobs.job import Job, JobStatus
from automl.domain.ports import TrialExecution
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.engine.training.sklearn_trainer import SklearnTrainer, _sklearn_scoring


def test_budget_labels_are_canonical_and_match_ui_contract():
    assert EXPERIMENT_BUDGET_SECONDS == {
        "quick": 60.0,
        "balanced": 300.0,
        "thorough": 1200.0,
    }
    assert resolve_time_budget_seconds("quick") == 60.0
    assert resolve_time_budget_seconds("balanced", 999) == 999.0
    assert resolve_time_budget_seconds(None, 42) == 42.0

    new_exp = Path("src/automl/interfaces/web/static/js/views/new_experiment.js").read_text(
        encoding="utf-8"
    )
    api_js = Path("src/automl/interfaces/web/static/js/api.js").read_text(encoding="utf-8")
    assert "quick: 60" in new_exp
    assert "balanced: 300" in new_exp
    assert "thorough: 1200" in new_exp
    assert "jobPayload.time_budget_seconds" in api_js
    assert 'mode === "guided" || this.mode === "manual"' in new_exp
    assert "escapeHtml(r.name || r.id)" in new_exp
    assert "escapeHtml(" in new_exp
    # Dataset identity fields must be escaped (XSS audit)
    assert "escapeHtml(activeDs" in new_exp


def test_job_service_accepts_time_budget_seconds(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"x": rng.normal(size=20), "target": [0, 1] * 10}).to_csv(path, index=False)
    workspace, commands, queries = build_application(str(tmp_path / "ws"))
    dataset = workspace.register_dataset("tiny", path, "target")
    run = workspace.create_run(dataset, metric="roc_auc")
    job_id = commands.dispatch(
        SubmitJobCommand(
            "experiment",
            run.id,
            {
                "name": "budgeted",
                "model_ids": ["logistic_regression"],
                "feature_names": ["x"],
                "budget": "quick",
                "time_budget_seconds": 60,
                "mode": "manual",
                "validation_strategy": "holdout",
            },
            "audit5-budget",
        )
    )
    job = queries.dispatch(GetJobQuery(job_id))
    assert job["payload"]["time_budget_seconds"] == 60
    assert job["payload"]["budget"] == "quick"


def test_optimize_api_does_not_fabricate_param_importance():
    src = Path("src/automl/interfaces/web/api/experiments.py").read_text(encoding="utf-8")
    assert '"importance": 0.34' not in src
    assert "param_importance_available" in src
    assert '"zero_missing_values": True' not in src
    assert "id_column_present" in src


def test_sklearn_scoring_multiclass_roc_auc_uses_ovr():
    assert _sklearn_scoring("roc_auc", "binary_classification") == "roc_auc"
    assert _sklearn_scoring("roc_auc", "multiclass_classification") == "roc_auc_ovr"


def test_multiclass_holdout_computes_roc_auc(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    path = tmp_path / "multi.csv"
    pd.DataFrame({
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "target": rng.integers(0, 3, size=n),
    }).to_csv(path, index=False)

    run = AutoMLRun(
        id="run_multi",
        workspace_id="ws",
        dataset_id="ds_multi",
        config=RunConfig(
            task_type="multiclass_classification",
            target="target",
            metric="roc_auc",
            validation_strategy="holdout",
        ),
    )
    exp = Experiment(
        id="exp_multi",
        run_id=run.id,
        name="multi-roc",
        hypothesis="multiclass roc_auc",
        feature_names=["f1", "f2"],
        model_ids=["logistic_regression"],
        metric="roc_auc",
        validation_strategy="holdout",
        status=ExperimentStatus.CREATED,
    )
    trial = Trial(id="tr_multi", experiment_id=exp.id, model_id="logistic_regression")
    execution = TrialExecution(
        trial=trial,
        experiment=exp,
        run=run,
        feature_names=["f1", "f2"],
        dataset_path=str(path),
        target_column="target",
        task_type="multiclass_classification",
        metric="roc_auc",
        validation_strategy="holdout",
        test_size=0.3,
        cv_folds=3,
        random_seed=42,
    )
    result = SklearnTrainer().run(execution)
    assert result.succeeded, result.failure_reason
    assert result.primary_metric == "roc_auc"
    assert result.primary_score > 0.0
    assert result.artifacts.get("estimator_class")
    assert result.artifacts.get("backend") == "native"


def test_job_executor_prefers_explicit_time_budget_seconds():
    workspace = MagicMock()
    commands = MagicMock()
    repo = MagicMock()
    mock_run = AutoMLRun(
        id="run-1",
        workspace_id="ws-1",
        dataset_id="ds-1",
        config=RunConfig(task_type="classification", target="target"),
    )
    mock_dataset = MagicMock()
    mock_dataset.id = "ds-1"
    mock_dataset.target_column = "target"
    profile = MagicMock()
    profile.columns = []
    repo.get_dataset_profile.return_value = profile
    workspace._get_run.return_value = mock_run
    workspace._get_dataset.return_value = mock_dataset
    workspace.repository = repo
    mock_experiment = MagicMock()
    mock_experiment.id = "exp-1"
    mock_experiment.run_id = "run-1"
    commands.dispatch.return_value = mock_experiment

    executor = JobExecutor(
        workspace=workspace, command_bus=commands, query_bus=MagicMock(), repository=repo
    )
    job = Job(
        id="job-1",
        run_id="run-1",
        operation="experiment",
        status=JobStatus.RUNNING,
        payload={
            "budget": "thorough",
            "time_budget_seconds": 90,
            "name": "Explicit",
            "model_ids": ["logistic_regression"],
        },
    )
    executor.workspace.execution_check = MagicMock()
    executor.workspace.run_experiment = MagicMock()
    repo.get.return_value = job
    repo.get_checkpoint.return_value = None
    repo.list_trial_results.return_value = [MagicMock(succeeded=True)]
    executor.execute(job)
    assert mock_run.config.time_budget_seconds == 90.0


def test_static_guards_manual_mode_and_studio_fallback_badge():
    new_exp = Path("src/automl/interfaces/web/static/js/views/new_experiment.js").read_text(
        encoding="utf-8"
    )
    studio = Path("src/automl/interfaces/web/static/js/views/studio.js").read_text(encoding="utf-8")
    assert "showModelPicker" in new_exp
    assert 'backend === "fallback"' in studio
    assert "_modelLabel" in new_exp
