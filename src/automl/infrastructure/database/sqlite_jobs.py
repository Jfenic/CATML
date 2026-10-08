"""Transactional job queue sharing the workspace SQLite database."""
from datetime import datetime, timezone
from contextlib import closing, contextmanager
import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

from automl.domain.jobs.job import Job, JobStatus, ACTIVE_STATUSES


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteJobRepository:
    def __init__(self, db_path: str | Path):
        self.db_path = db_path
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, "
                               "idempotency_key TEXT UNIQUE NOT NULL, run_id TEXT, "
                               "status TEXT NOT NULL, data TEXT NOT NULL)")
            connection.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status)")
            try:
                cols = connection.execute("PRAGMA table_info(jobs)").fetchall()
                run_id_col = next((c for c in cols if (c["name"] if hasattr(c, "keys") else c[1]) == "run_id"), None)
                run_id_notnull = (run_id_col["notnull"] if hasattr(run_id_col, "keys") else run_id_col[3]) if run_id_col else 0
                if run_id_notnull == 1:
                    logger.info("Migrating legacy jobs table to support nullable run_id")
                    pre_count_row = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()
                    pre_count = pre_count_row[0] if pre_count_row else 0

                    old_isolation = connection.isolation_level
                    try:
                        connection.isolation_level = None
                        connection.execute("BEGIN IMMEDIATE")
                        connection.execute("CREATE TABLE jobs_dg_tmp (id TEXT PRIMARY KEY, "
                                           "idempotency_key TEXT UNIQUE NOT NULL, run_id TEXT, "
                                           "status TEXT NOT NULL, data TEXT NOT NULL)")
                        connection.execute("INSERT INTO jobs_dg_tmp SELECT id, idempotency_key, run_id, status, data FROM jobs")
                        connection.execute("DROP TABLE jobs")
                        connection.execute("ALTER TABLE jobs_dg_tmp RENAME TO jobs")
                        connection.execute("CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status)")

                        post_count_row = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()
                        post_count = post_count_row[0] if post_count_row else 0
                        if post_count != pre_count:
                            raise RuntimeError(f"Jobs migration data loss detected: expected {pre_count}, found {post_count}")

                        connection.execute("COMMIT")
                    except Exception as exc:
                        try:
                            connection.execute("ROLLBACK")
                        except Exception:
                            pass
                        logger.error("Failed to migrate jobs table schema: %s", exc)
                        raise RuntimeError(f"Failed to migrate jobs table schema: {exc}") from exc
                    finally:
                        connection.isolation_level = old_isolation
            except Exception as exc:
                if not isinstance(exc, RuntimeError):
                    logger.error("Failed to migrate jobs table schema: %s", exc)
                    raise RuntimeError(f"Failed to migrate jobs table schema: {exc}") from exc
                raise

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.db_path, timeout=30)) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection

    def _write(self, connection, job):
        job.updated_at = timestamp()
        connection.execute("UPDATE jobs SET status=?, data=? WHERE id=?",
                           (job.status.value, json.dumps(job.to_dict()), job.id))

    def enqueue(self, job: Job) -> str:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT data FROM jobs WHERE idempotency_key=?",
                                     (job.idempotency_key,)).fetchone()
            if row:
                existing = Job.from_dict(json.loads(row["data"]))
                if (existing.operation, existing.run_id, existing.payload) != (job.operation, job.run_id, job.payload):
                    raise ValueError("Idempotency key already used with different arguments")
                return existing.id
            job.created_at = job.updated_at = timestamp()
            connection.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?)",
                               (job.id, job.idempotency_key, job.run_id, job.status.value, json.dumps(job.to_dict())))
            return job.id

    def get(self, job_id: str) -> Job:
        with self._connect() as connection:
            row = connection.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Job not found: {job_id}")
        return Job.from_dict(json.loads(row["data"]))

    def list(self, run_id: str | None = None) -> list[Job]:
        with self._connect() as connection:
            if run_id is None:
                rows = connection.execute("SELECT data FROM jobs ORDER BY rowid DESC").fetchall()
            else:
                rows = connection.execute("SELECT data FROM jobs WHERE run_id=? ORDER BY rowid DESC", (run_id,)).fetchall()
        return [Job.from_dict(json.loads(row["data"])) for row in rows]

    def change(self, job_id: str, allowed: set[JobStatus], **changes) -> Job:
        return self._change(job_id, allowed, lambda job: changes)

    def finish(self, job_id: str, status: JobStatus, **changes) -> Job:
        return self._change(job_id, ACTIVE_STATUSES,
                            lambda job: {**changes, "status": job.final_status(status)})

    def _change(self, job_id, allowed, changes_for) -> Job:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT data FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(f"Job not found: {job_id}")
            job = Job.from_dict(json.loads(row["data"]))
            if job.status not in allowed:
                raise ValueError(f"Job is {job.status.value}; action is not allowed")
            for key, value in changes_for(job).items():
                setattr(job, key, value)
            self._write(connection, job)
            return job

    def claim(self) -> Job | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM jobs WHERE status IN (?, ?, ?) LIMIT 1",
                                  tuple(s.value for s in ACTIVE_STATUSES)).fetchone():
                return None
            row = connection.execute("SELECT data FROM jobs WHERE status=? ORDER BY rowid LIMIT 1",
                                     (JobStatus.QUEUED.value,)).fetchone()
            if row is None:
                return None
            job = Job.from_dict(json.loads(row["data"]))
            job.status = JobStatus.RUNNING
            job.attempt += 1
            job.error = None
            job.message = "Starting"
            self._write(connection, job)
            return job

    def recover(self) -> None:
        # Called only after acquiring the exclusive workspace worker lock.
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for row in connection.execute("SELECT data FROM jobs WHERE status IN (?, ?, ?)",
                                          tuple(s.value for s in ACTIVE_STATUSES)).fetchall():
                job = Job.from_dict(json.loads(row["data"]))
                job.status = JobStatus.CANCELLED if job.status == JobStatus.CANCEL_REQUESTED else JobStatus.INTERRUPTED
                job.message = "Worker stopped; explicit retry required"
                self._write(connection, job)
