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


def test_experiment_planner_proposes_candidates_on_mixed_clean_and_leakage_dataset(tmp_path):
    """Verify RuleBasedExperimentPlanner filters out leakage and ID columns but continues with safe columns."""
    workspace, command_bus, query_bus = build_application(tmp_path)

    n_samples = 40
    y = np.linspace(1, 10, n_samples)
    df = pd.DataFrame({
        "customer_id": np.arange(n_samples),  # identifier
        "clean_f1": np.random.randn(n_samples),  # safe
        "clean_f2": np.random.randn(n_samples),  # safe
        "leakage_col": y,  # target leakage
        "target": y,
    })

    dataset = workspace.register_dataset(
        name="Mixed Dataset",
        path=df,
        target="target",
    )
    run = workspace.create_run(dataset, metric="r2")
    profile = workspace.repository.get_dataset_profile(dataset.id)
    feature_reg = workspace.get_feature_registry(dataset.id)
    model_reg = workspace.model_registry

    planner = RuleBasedExperimentPlanner()
    candidates = planner.propose(
        run=run,
        profile=profile,
        feature_registry=feature_reg,
        model_registry=model_reg,
    )

    # Must propose candidates using ONLY the safe features
    assert len(candidates) > 0
    for cand in candidates:
        assert set(cand.feature_names) == {"clean_f1", "clean_f2"}
        assert "leakage_col" not in cand.feature_names
        assert "customer_id" not in cand.feature_names
        assert "target" not in cand.feature_names


def test_resolve_safe_feature_names_strict_vs_filtering_mode(tmp_path):
    """Verify resolve_safe_feature_names strict raises on leakage, while strict=False filters safely."""
    workspace, _, _ = build_application(tmp_path)

    n_samples = 40
    y = np.linspace(1, 10, n_samples)
    df = pd.DataFrame({
        "clean_feature": np.random.randn(n_samples),
        "leak_col": y,
        "target": y,
    })
    dataset = workspace.register_dataset(name="Strict vs Filter Test", path=df, target="target")
    profile = workspace.repository.get_dataset_profile(dataset.id)

    # 1. Strict mode (default) raises ValueError when requested_features contains leakage
    with pytest.raises(ValueError) as excinfo:
        profile.resolve_safe_feature_names(
            requested_features=["clean_feature", "leak_col"],
            strict=True,
        )
    assert "Features contain confirmed data leakage columns" in str(excinfo.value)

    # 2. Filtering mode (strict=False) safely excludes leak_col and returns clean_feature
    safe = profile.resolve_safe_feature_names(
        requested_features=["clean_feature", "leak_col"],
        strict=False,
    )
    assert safe == ["clean_feature"]

    # 3. Filtering mode when ALL features are leakage raises ValueError ("No safe feature candidates...")
    with pytest.raises(ValueError) as excinfo2:
        profile.resolve_safe_feature_names(
            requested_features=["leak_col"],
            strict=False,
        )
    assert "No safe feature candidates available for training" in str(excinfo2.value)


