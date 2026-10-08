import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import CreateExperimentCommand
from automl.domain.models.registry import ModelRegistry
from automl.engine.planning.experiment_planner import RuleBasedExperimentPlanner
from automl.facade import AutoML


def test_automl_fit_fails_closed_when_all_features_have_leakage(tmp_path):
    """Verify AutoML.fit() stops with explicit ValueError when all candidate features are leakage."""
    # Construct a dataset where all non-target columns have extreme target leakage (|r| = 1.0) or are IDs
    n_samples = 50
    target_values = np.random.RandomState(42).randn(n_samples)
    
    df = pd.DataFrame({
        "id": np.arange(n_samples),  # Identifier column
        "leak_feature_1": target_values,  # Perfect correlation (r = 1.0)
        "leak_feature_2": -target_values,  # Perfect correlation (r = -1.0)
        "target": target_values,
    })

    automl = AutoML(
        time_budget=30,
        random_state=42,
    )

    with pytest.raises(ValueError) as excinfo:
        automl.fit(df, target="target")

    assert "No safe feature candidates available for training" in str(excinfo.value)
    assert "leak_feature_1" in str(excinfo.value) or "leak_feature_2" in str(excinfo.value)


def test_create_experiment_rejects_leakage_features(tmp_path):
    """Verify AutoMLWorkspace.create_experiment rejects explicit requests with leakage features."""
    workspace, command_bus, query_bus = build_application(tmp_path)

    n_samples = 40
    y = np.linspace(10, 100, n_samples)
    df = pd.DataFrame({
        "safe_feature": np.random.randn(n_samples),
        "leak_col": y,  # Exact duplicate of target
        "target": y,
    })

    dataset = workspace.register_dataset(
        name="Leakage Test Dataset",
        path=df,
        target="target",
    )

    run = workspace.create_run(dataset, metric="r2")

    # 1. Attempting to manually create an experiment requesting the leakage column must fail
    with pytest.raises(ValueError) as excinfo:
        workspace.create_experiment(
            run=run,
            name="unsafe_experiment",
            feature_names=["safe_feature", "leak_col"],
        )
    assert "Features contain confirmed data leakage columns" in str(excinfo.value)
    assert "leak_col" in str(excinfo.value)

    # 2. Automated resolution (feature_names=None) must safely pick ONLY safe_feature
    exp = workspace.create_experiment(
        run=run,
        name="safe_experiment",
        feature_names=None,
    )
    assert exp.feature_names == ["safe_feature"]
    assert "leak_col" not in exp.feature_names


def test_experiment_planner_proposes_no_candidates_when_only_leakage_exists(tmp_path):
    """Verify ExperimentPlanner does not fall back to proposing leakage features."""
    workspace, command_bus, query_bus = build_application(tmp_path)

    n_samples = 30
    y = np.arange(n_samples, dtype=float)
    df = pd.DataFrame({
        "id_col": np.arange(n_samples),
        "leak_col": y,
        "target": y,
    })

    dataset = workspace.register_dataset(
        name="All Leakage Dataset",
        path=df,
        target="target",
    )

    run = workspace.create_run(dataset, metric="r2")
    profile = workspace.repository.get_dataset_profile(dataset.id)
    feature_reg = workspace.get_feature_registry(dataset.id)
    model_reg = ModelRegistry()

    planner = RuleBasedExperimentPlanner()
    candidates = planner.propose(
        run=run,
        profile=profile,
        feature_registry=feature_reg,
        model_registry=model_reg,
    )

    # Must return empty candidates list, NEVER candidates containing leak_col or id_col
    assert candidates == []


def test_job_executor_automatically_excludes_leakage_features(tmp_path):
    """Verify JobExecutor excludes leakage features when generating default feature names."""
    from automl.application.services.job_executor import JobExecutor
    from automl.domain.jobs.job import Job, JobStatus
    from automl.infrastructure.database.sqlite_jobs import SQLiteJobRepository

    workspace, command_bus, query_bus = build_application(tmp_path)
    job_repo = SQLiteJobRepository(workspace.repository.db_path)

    n_samples = 40
    y = np.linspace(1, 10, n_samples)
    df = pd.DataFrame({
        "good_feature_1": np.random.randn(n_samples),
        "good_feature_2": np.random.randn(n_samples),
        "leakage_feature": y,
        "target": y,
    })

    dataset = workspace.register_dataset(
        name="Job Executor Test Dataset",
        path=df,
        target="target",
    )
    run = workspace.create_run(dataset, metric="r2")

    job = Job(
        id="job_bg_test",
        operation="experiment",
        run_id=run.id,
        payload={"model_ids": ["ridge"]},  # feature_names omitted
        idempotency_key="job_key_1",
        status=JobStatus.QUEUED,
    )
    job_repo.enqueue(job)
    claimed_job = job_repo.claim()
    assert claimed_job is not None

    executor = JobExecutor(workspace, command_bus, query_bus, job_repo)
    result = executor.execute(claimed_job)
    assert result["status"] == "success"
    exp = workspace.repository.get_experiment(result["experiment_id"])
    assert "leakage_feature" not in exp.feature_names
    assert set(exp.feature_names) == {"good_feature_1", "good_feature_2"}


def test_create_experiment_allows_leakage_when_explicitly_requested(tmp_path):
    """Verify allow_leakage=True allows explicitly requested features even if flagged."""
    workspace, command_bus, query_bus = build_application(tmp_path)

    n_samples = 40
    y = np.linspace(10, 100, n_samples)
    df = pd.DataFrame({
        "safe_feature": np.random.randn(n_samples),
        "leak_col": y,
        "target": y,
    })

    dataset = workspace.register_dataset(
        name="Leakage Dataset",
        path=df,
        target="target",
    )
    run = workspace.create_run(dataset, metric="r2")

    exp = workspace.create_experiment(
        run=run,
        name="allowed_leakage_exp",
        feature_names=["safe_feature", "leak_col"],
        allow_leakage=True,
    )
    assert set(exp.feature_names) == {"safe_feature", "leak_col"}


def test_clustering_task_does_not_filter_target_leakage(tmp_path):
    """Verify clustering task does not reject features correlated with target column."""
    workspace, command_bus, query_bus = build_application(tmp_path)

    n_samples = 50
    y = np.linspace(1, 10, n_samples)
    df = pd.DataFrame({
        "f1": y,
        "f2": np.random.randn(n_samples),
        "cluster_target": y,
    })

    dataset = workspace.register_dataset(
        name="Clustering Data",
        path=df,
        target="cluster_target",
        task_type="clustering",
    )
    run = workspace.create_run(dataset)

    exp = workspace.create_experiment(
        run=run,
        name="clustering_exp",
        feature_names=["f1", "f2"],
        model_ids=["kmeans"],
    )
    assert "f1" in exp.feature_names
    assert "f2" in exp.feature_names

