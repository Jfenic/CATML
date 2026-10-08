from pathlib import Path
import tempfile
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.domain.datasets.profile import Dataset, DatasetProfile, ColumnProfile
from automl.domain.jobs.job import Job, JobStatus
from automl.domain.tasks.task_type import TaskType
from automl.engine.profiling.dataset_profiler import profile_dataset
from automl.infrastructure.database.sqlite_jobs import SQLiteJobRepository


def test_dataset_profile_without_target_column():
    """Verify Dataset and profile_dataset work properly when target_column is None."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature_a": [1.0, 2.5, 3.2, 4.1, 5.0, 6.2, 7.1, 8.5, 9.0, 10.2],
        "feature_b": ["alpha", "beta", "alpha", "beta", "gamma", "alpha", "beta", "gamma", "alpha", "beta"],
        "feature_c": [10.0, 20.0, 15.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0],
    })

    dataset = Dataset(
        id="ds_test_unsupervised",
        workspace_id="ws_1",
        name="Unsupervised Test Data",
        path="/dummy/path.csv",
        target_column=None,
        task_type=None,
    )

    profile = profile_dataset(dataset, df=df)

    assert profile.target_column is None
    assert profile.task_type is None
    assert profile.row_count == 10
    assert profile.column_count == 4

    # Ensure to_dict works with None target
    profile_dict = profile.to_dict()
    assert profile_dict["target_column"] is None
    assert profile_dict["task_type"] is None
    assert len(profile_dict["columns"]) == 4

    # Check recommended features includes non-id columns
    recommended = profile.recommended_feature_names
    assert "feature_a" in recommended
    assert "feature_b" in recommended
    assert "feature_c" in recommended


def test_workspace_register_dataset_without_target(tmp_path: Path):
    """Test AutoMLWorkspace registration of dataset with target=None."""
    workspace, command_bus, query_bus = build_application(tmp_path)

    df = pd.DataFrame({
        "numeric_1": [10, 20, 30, 40, 50],
        "numeric_2": [1.5, 2.5, 3.5, 4.5, 5.5],
        "category_1": ["cat", "dog", "cat", "dog", "cat"],
    })

    dataset = workspace.register_dataset(
        name="Clustering Data",
        path=df,
        target=None,
    )

    assert dataset.target_column is None
    assert dataset.task_type is None

    # Retrieve from repository and check persistence
    retrieved = workspace.repository.get_dataset(dataset.id)
    assert retrieved is not None
    assert retrieved.name == "Clustering Data"
    assert retrieved.target_column is None
    assert retrieved.task_type is None

    # Profile retrieval
    retrieved_profile = workspace.get_dataset_profile(dataset.id)
    assert retrieved_profile is not None
    assert retrieved_profile.target_column is None
    assert retrieved_profile.row_count == 5

    # Feature registry
    feature_reg = workspace.get_feature_registry(dataset.id)
    active_features = feature_reg.active_feature_names()
    assert set(active_features) == {"numeric_1", "numeric_2", "category_1"}


def test_sqlite_jobs_with_none_run_id(tmp_path: Path):
    """Test SQLiteJobRepository supports jobs without a run_id (run_id=None)."""
    db_file = tmp_path / "test_jobs.db"
    repo = SQLiteJobRepository(db_file)

    job = Job(
        id="job_explore_101",
        operation="statistical_study",
        run_id=None,
        payload={"study_type": "correlation_analysis", "dataset_id": "ds_123"},
        idempotency_key="study_run_101",
        status=JobStatus.QUEUED,
        message="Queued for analysis worker",
    )

    job_id = repo.enqueue(job)
    assert job_id == "job_explore_101"

    # Retrieve job
    fetched = repo.get("job_explore_101")
    assert fetched.id == "job_explore_101"
    assert fetched.run_id is None
    assert fetched.operation == "statistical_study"
    assert fetched.payload["study_type"] == "correlation_analysis"

    # List all jobs
    all_jobs = repo.list()
    assert len(all_jobs) == 1
    assert all_jobs[0].run_id is None

    # List filtering by an AutoML run_id should NOT include this decoupled job
    automl_jobs = repo.list(run_id="run_automl_xyz")
    assert len(automl_jobs) == 0

    # Test state transition
    repo.change("job_explore_101", {JobStatus.QUEUED}, status=JobStatus.RUNNING, message="Analyzing")
    updated = repo.get("job_explore_101")
    assert updated.status == JobStatus.RUNNING
    assert updated.message == "Analyzing"


def test_job_service_control_without_run_id(tmp_path: Path):
    """Test JobService controls work gracefully when job.run_id is None."""
    from automl.application.commands.job_commands import ControlJobCommand
    from automl.application.services.jobs import JobService

    workspace, command_bus, query_bus = build_application(tmp_path)
    job_repo = SQLiteJobRepository(workspace.repository.db_path)
    job_service = JobService(workspace, job_repo)

    job = Job(
        id="job_study_decoupled",
        operation="analysis",
        run_id=None,
        payload={"task": "bivariate_tests"},
        idempotency_key="idemp_analysis_1",
        status=JobStatus.QUEUED,
    )
    job_repo.enqueue(job)

    # Pause job
    job_service.control(ControlJobCommand(job_id="job_study_decoupled", action="pause"))
    paused_job = job_repo.get("job_study_decoupled")
    assert paused_job.status == JobStatus.PAUSED

    # Resume job (should not raise an error even with job.run_id is None)
    job_service.control(ControlJobCommand(job_id="job_study_decoupled", action="resume"))
    resumed_job = job_repo.get("job_study_decoupled")
    assert resumed_job.status == JobStatus.QUEUED
