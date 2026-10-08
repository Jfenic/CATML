import json
from pathlib import Path
import sqlite3
import pytest

from automl.domain.datasets.profile import Dataset
from automl.domain.jobs.job import Job, JobStatus
from automl.infrastructure.database.sqlite_jobs import SQLiteJobRepository
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository


def test_migrate_legacy_datasets_table_preserves_data_and_allows_nullable(tmp_path: Path):
    """Test migration of a legacy database where datasets table has NOT NULL constraints."""
    db_path = tmp_path / "legacy_workspace.db"

    # 1. Create a legacy database mimicking v0.8.0 / v0.8.1 schema
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE datasets (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                target_column TEXT NOT NULL,
                task_type TEXT NOT NULL
            );
            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                current_phase TEXT NOT NULL
            );
            CREATE TABLE dataset_profiles (
                dataset_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL
            );
            CREATE TABLE features (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                name TEXT NOT NULL,
                physical_dtype TEXT,
                status TEXT NOT NULL,
                user_priority REAL DEFAULT 0,
                semantic_type TEXT DEFAULT 'unknown'
            );
            CREATE TABLE feature_sets (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                feature_names_json TEXT NOT NULL,
                created_by TEXT,
                lineage TEXT
            );
            CREATE TABLE experiments (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                hypothesis TEXT,
                feature_set_id TEXT,
                feature_names_json TEXT NOT NULL,
                model_ids_json TEXT NOT NULL,
                metric TEXT NOT NULL,
                validation_strategy TEXT NOT NULL,
                status TEXT NOT NULL,
                created_by TEXT,
                priority TEXT,
                group_column TEXT
            );
            """
        )
        # Seed pre-existing legacy record
        conn.execute(
            """
            INSERT INTO datasets (id, workspace_id, name, path, target_column, task_type)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("ds_legacy_1", "ws_1", "Old Supervised Dataset", "/path/to/old.csv", "price", "regression"),
        )

    # Verify that legacy table had NOT NULL constraints
    with sqlite3.connect(db_path) as conn:
        cols = conn.execute("PRAGMA table_info(datasets)").fetchall()
        col_dict = {c[1]: c[3] for c in cols}  # name: notnull
        assert col_dict["target_column"] == 1
        assert col_dict["task_type"] == 1

        # Trying to insert NULL directly in this legacy state would fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?)",
                ("ds_fail", "ws_1", "Fail Data", "/p.csv", None, None),
            )

    # 2. Instantiate repository on the legacy database (triggers migration)
    repo = SQLiteExperimentRepository(db_path)

    # 3. Verify schema is now nullable
    with sqlite3.connect(db_path) as conn:
        cols = conn.execute("PRAGMA table_info(datasets)").fetchall()
        col_dict = {c[1]: c[3] for c in cols}
        assert col_dict["target_column"] == 0
        assert col_dict["task_type"] == 0

    # 4. Verify existing legacy data was 100% preserved
    legacy_ds = repo.get_dataset("ds_legacy_1")
    assert legacy_ds is not None
    assert legacy_ds.id == "ds_legacy_1"
    assert legacy_ds.name == "Old Supervised Dataset"
    assert legacy_ds.target_column == "price"
    assert legacy_ds.task_type == "regression"

    # 5. Verify new datasets with target_column=None and task_type=None can be saved without error
    unsupervised_ds = Dataset(
        id="ds_unsupervised_new",
        workspace_id="ws_1",
        name="Explore Dataset",
        path="/path/to/explore.parquet",
        target_column=None,
        task_type=None,
    )
    repo.save_dataset(unsupervised_ds)

    retrieved = repo.get_dataset("ds_unsupervised_new")
    assert retrieved is not None
    assert retrieved.name == "Explore Dataset"
    assert retrieved.target_column is None
    assert retrieved.task_type is None

    # 6. Verify listing both legacy and new datasets
    all_datasets = repo.list_datasets()
    assert len(all_datasets) == 2
    names = {d.name for d in all_datasets}
    assert names == {"Old Supervised Dataset", "Explore Dataset"}


def test_migrate_legacy_jobs_table_preserves_data_and_allows_nullable_run_id(tmp_path: Path):
    """Test migration of legacy jobs table with run_id TEXT NOT NULL constraint."""
    db_path = tmp_path / "legacy_jobs.db"

    # 1. Create a legacy jobs table mimicking v0.8.0 / v0.8.1
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, "
            "run_id TEXT NOT NULL, status TEXT NOT NULL, data TEXT NOT NULL)"
        )
        conn.execute("CREATE INDEX jobs_status ON jobs(status)")

        # Seed pre-existing legacy job
        legacy_job = Job(
            id="job_legacy_1",
            operation="experiment",
            run_id="run_old_123",
            payload={"model_ids": ["rf"]},
            idempotency_key="idemp_old_1",
            status=JobStatus.COMPLETED,
            message="Completed legacy experiment",
        )
        conn.execute(
            "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
            ("job_legacy_1", "idemp_old_1", "run_old_123", "completed", json.dumps(legacy_job.to_dict())),
        )

    # Verify that legacy table had NOT NULL constraint on run_id
    with sqlite3.connect(db_path) as conn:
        cols = conn.execute("PRAGMA table_info(jobs)").fetchall()
        col_dict = {c[1]: c[3] for c in cols}
        assert col_dict["run_id"] == 1

        # Trying to insert NULL run_id directly in legacy state would fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
                ("job_fail", "idemp_fail", None, "queued", "{}"),
            )

    # 2. Instantiate SQLiteJobRepository (triggers migration)
    job_repo = SQLiteJobRepository(db_path)

    # 3. Verify schema is now nullable for run_id
    with sqlite3.connect(db_path) as conn:
        cols = conn.execute("PRAGMA table_info(jobs)").fetchall()
        col_dict = {c[1]: c[3] for c in cols}
        assert col_dict["run_id"] == 0

    # 4. Verify existing legacy job was 100% preserved
    old_job = job_repo.get("job_legacy_1")
    assert old_job is not None
    assert old_job.id == "job_legacy_1"
    assert old_job.run_id == "run_old_123"
    assert old_job.operation == "experiment"
    assert old_job.status == JobStatus.COMPLETED

    # 5. Verify new job with run_id=None can be enqueued and retrieved
    new_job = Job(
        id="job_explore_decoupled",
        operation="statistical_analysis",
        run_id=None,
        payload={"study_id": "study_456"},
        idempotency_key="idemp_study_1",
        status=JobStatus.QUEUED,
        message="Queued for analysis",
    )
    job_repo.enqueue(new_job)

    fetched_new = job_repo.get("job_explore_decoupled")
    assert fetched_new is not None
    assert fetched_new.run_id is None
    assert fetched_new.operation == "statistical_analysis"

    # 6. Verify list behavior
    all_jobs = job_repo.list()
    assert len(all_jobs) == 2
    assert {j.id for j in all_jobs} == {"job_legacy_1", "job_explore_decoupled"}
