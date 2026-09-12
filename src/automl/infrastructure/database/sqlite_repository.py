from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from automl.domain.datasets.profile import ColumnProfile, Dataset, DatasetProfile
from automl.domain.experiments.trial import (
    Experiment,
    ExperimentStatus,
    Trial,
    TrialResult,
    TrialStatus,
)
from automl.domain.features.feature import Feature, FeatureStatus
from automl.domain.features.feature_set import FeatureSet
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.domain.runs.states import RunPhase, RunStatus


class SQLiteExperimentRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    path TEXT NOT NULL,
                    target_column TEXT NOT NULL,
                    task_type TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_phase TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_profiles (
                    dataset_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS features (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    physical_dtype TEXT,
                    status TEXT NOT NULL,
                    user_priority REAL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS feature_sets (
                    id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    feature_names_json TEXT NOT NULL,
                    created_by TEXT,
                    lineage TEXT
                );
                CREATE TABLE IF NOT EXISTS experiments (
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
                    priority TEXT
                );
                CREATE TABLE IF NOT EXISTS trials (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    parameters_json TEXT,
                    seed INTEGER,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trial_results (
                    trial_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    primary_metric TEXT NOT NULL,
                    primary_score REAL NOT NULL,
                    secondary_metrics_json TEXT,
                    training_time_seconds REAL,
                    failure_reason TEXT,
                    artifacts_json TEXT
                );
                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    run_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    next_model_index INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    run_id TEXT,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS benchmark_results (
                    id TEXT PRIMARY KEY,
                    platform_version TEXT NOT NULL,
                    scenario_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    best_score REAL NOT NULL,
                    best_model TEXT,
                    total_time_s REAL,
                    details_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS problem_definitions (
                    dataset_id TEXT PRIMARY KEY,
                    definition_json TEXT NOT NULL
                );
                """
            )
            self._migrate(conn)

    def save_problem_definition(self, definition) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO problem_definitions (dataset_id, definition_json)
                VALUES (?, ?)
                """,
                (definition.dataset_id, json.dumps(definition.to_dict())),
            )

    def get_problem_definition(self, dataset_id: str):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT definition_json FROM problem_definitions WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return None
        from automl.domain.tasks.problem_definition import ProblemDefinition
        from automl.domain.tasks.task_type import TaskType

        data = json.loads(row["definition_json"])
        return ProblemDefinition(
            dataset_id=data["dataset_id"],
            task_type=TaskType.parse(data["task_type"]),
            target_column=data.get("target_column"),
            default_metric=data["default_metric"],
            available_metrics=list(data.get("available_metrics", [])),
            recommended_models=list(data.get("recommended_models", [])),
            inferred=bool(data.get("inferred", True)),
            reasoning=data.get("reasoning", ""),
        )

    def _migrate(self, conn: sqlite3.Connection) -> None:
        experiment_cols = {row[1] for row in conn.execute("PRAGMA table_info(experiments)").fetchall()}
        if "feature_set_id" not in experiment_cols:
            conn.execute("ALTER TABLE experiments ADD COLUMN feature_set_id TEXT")

    # --- datasets ---

    def save_dataset(self, dataset: Dataset) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets
                (id, workspace_id, name, path, target_column, task_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset.id,
                    dataset.workspace_id,
                    dataset.name,
                    dataset.path,
                    dataset.target_column,
                    dataset.task_type,
                ),
            )

    def get_dataset(self, dataset_id: str) -> Dataset | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            return None
        return Dataset(
            id=row["id"],
            workspace_id=row["workspace_id"],
            name=row["name"],
            path=row["path"],
            target_column=row["target_column"],
            task_type=row["task_type"],
        )

    def list_datasets(self, workspace_id: str | None = None) -> list[Dataset]:
        with self._connect() as conn:
            if workspace_id:
                rows = conn.execute(
                    "SELECT * FROM datasets WHERE workspace_id = ? ORDER BY name",
                    (workspace_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM datasets ORDER BY name").fetchall()
        return [
            Dataset(
                id=r["id"],
                workspace_id=r["workspace_id"],
                name=r["name"],
                path=r["path"],
                target_column=r["target_column"],
                task_type=r["task_type"],
            )
            for r in rows
        ]

    # --- runs ---

    def save_run(self, run: AutoMLRun) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (id, workspace_id, dataset_id, config_json, status, current_phase)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.workspace_id,
                    run.dataset_id,
                    json.dumps(run.config.to_dict()),
                    run.status.value,
                    run.current_phase.value,
                ),
            )

    def get_run(self, run_id: str) -> AutoMLRun | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return AutoMLRun(
            id=row["id"],
            workspace_id=row["workspace_id"],
            dataset_id=row["dataset_id"],
            config=RunConfig.from_dict(json.loads(row["config_json"])),
            status=RunStatus(row["status"]),
            current_phase=RunPhase(row["current_phase"]),
        )

    def list_runs(self, workspace_id: str | None = None) -> list[AutoMLRun]:
        with self._connect() as conn:
            if workspace_id:
                rows = conn.execute(
                    "SELECT * FROM runs WHERE workspace_id = ? ORDER BY id",
                    (workspace_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
        return [
            AutoMLRun(
                id=r["id"],
                workspace_id=r["workspace_id"],
                dataset_id=r["dataset_id"],
                config=RunConfig.from_dict(json.loads(r["config_json"])),
                status=RunStatus(r["status"]),
                current_phase=RunPhase(r["current_phase"]),
            )
            for r in rows
        ]

    # --- profiles ---

    def save_dataset_profile(self, profile: DatasetProfile) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO dataset_profiles (dataset_id, profile_json)
                VALUES (?, ?)
                """,
                (profile.dataset_id, json.dumps(profile.to_dict())),
            )

    def get_dataset_profile(self, dataset_id: str) -> DatasetProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM dataset_profiles WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            return None
        data = json.loads(row["profile_json"])
        columns = [ColumnProfile(**c) for c in data["columns"]]
        return DatasetProfile(
            dataset_id=data["dataset_id"],
            row_count=data["row_count"],
            column_count=data["column_count"],
            target_column=data["target_column"],
            task_type=data["task_type"],
            columns=columns,
        )

    # --- features ---

    def save_feature(self, feature: Feature) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO features
                (id, dataset_id, name, physical_dtype, status, user_priority)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    feature.id,
                    feature.dataset_id,
                    feature.name,
                    feature.physical_dtype,
                    feature.status.value,
                    feature.user_priority,
                ),
            )

    def list_features(self, dataset_id: str) -> list[Feature]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM features WHERE dataset_id = ? ORDER BY name",
                (dataset_id,),
            ).fetchall()
        return [_row_to_feature(r) for r in rows]

    # --- feature sets ---

    def save_feature_set(self, feature_set: FeatureSet) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_sets
                (id, dataset_id, name, version, feature_names_json, created_by, lineage)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    feature_set.id,
                    feature_set.dataset_id,
                    feature_set.name,
                    feature_set.version,
                    json.dumps(feature_set.feature_names),
                    feature_set.created_by,
                    feature_set.lineage,
                ),
            )

    def get_feature_set(self, feature_set_id: str) -> FeatureSet | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM feature_sets WHERE id = ?", (feature_set_id,)).fetchone()
        if row is None:
            return None
        return FeatureSet(
            id=row["id"],
            dataset_id=row["dataset_id"],
            name=row["name"],
            version=row["version"],
            feature_names=json.loads(row["feature_names_json"]),
            created_by=row["created_by"] or "user",
            lineage=row["lineage"] or "",
        )

    def list_feature_sets(self, dataset_id: str) -> list[FeatureSet]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feature_sets WHERE dataset_id = ? ORDER BY name, version",
                (dataset_id,),
            ).fetchall()
        return [
            FeatureSet(
                id=r["id"],
                dataset_id=r["dataset_id"],
                name=r["name"],
                version=r["version"],
                feature_names=json.loads(r["feature_names_json"]),
                created_by=r["created_by"] or "user",
                lineage=r["lineage"] or "",
            )
            for r in rows
        ]

    # --- experiments ---

    def save_experiment(self, experiment: Experiment) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (id, run_id, name, hypothesis, feature_set_id, feature_names_json, model_ids_json,
                 metric, validation_strategy, status, created_by, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.run_id,
                    experiment.name,
                    experiment.hypothesis,
                    experiment.feature_set_id,
                    json.dumps(experiment.feature_names),
                    json.dumps(experiment.model_ids),
                    experiment.metric,
                    experiment.validation_strategy,
                    experiment.status.value,
                    experiment.created_by,
                    experiment.priority,
                ),
            )

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            return None
        return _row_to_experiment(row)

    def list_experiments(self, run_id: str) -> list[Experiment]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments WHERE run_id = ? ORDER BY name",
                (run_id,),
            ).fetchall()
        return [_row_to_experiment(r) for r in rows]

    # --- trials ---

    def save_trial(self, trial: Trial) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trials
                (id, experiment_id, model_id, parameters_json, seed, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.id,
                    trial.experiment_id,
                    trial.model_id,
                    json.dumps(trial.parameters),
                    trial.seed,
                    trial.status.value,
                ),
            )

    def get_trial(self, trial_id: str) -> Trial | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM trials WHERE id = ?", (trial_id,)).fetchone()
        if row is None:
            return None
        return Trial(
            id=row["id"],
            experiment_id=row["experiment_id"],
            model_id=row["model_id"],
            parameters=json.loads(row["parameters_json"] or "{}"),
            seed=row["seed"],
            status=TrialStatus(row["status"]),
        )

    def save_trial_result(self, result: TrialResult) -> None:

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trial_results
                (trial_id, experiment_id, model_id, primary_metric, primary_score,
                 secondary_metrics_json, training_time_seconds, failure_reason, artifacts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.trial_id,
                    result.experiment_id,
                    result.model_id,
                    result.primary_metric,
                    result.primary_score,
                    json.dumps(result.secondary_metrics),
                    result.training_time_seconds,
                    result.failure_reason,
                    json.dumps(result.artifacts),
                ),
            )

    def list_trial_results(self, experiment_id: str) -> list[TrialResult]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM trial_results WHERE experiment_id = ? ORDER BY primary_score DESC",
                (experiment_id,),
            ).fetchall()
        return [_row_to_result(row) for row in rows]

    def get_leaderboard(self, run_id: str) -> list[TrialResult]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tr.* FROM trial_results tr
                JOIN experiments e ON e.id = tr.experiment_id
                WHERE e.run_id = ?
                ORDER BY tr.primary_score DESC
                """,
                (run_id,),
            ).fetchall()
        return [_row_to_result(row) for row in rows]

    def compare_experiments(self, experiment_ids: list[str]) -> list[dict]:
        if not experiment_ids:
            return []
        placeholders = ",".join("?" for _ in experiment_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT e.id AS experiment_id, e.name, e.priority, e.status,
                       tr.trial_id, tr.model_id, tr.primary_metric, tr.primary_score,
                       tr.training_time_seconds, tr.failure_reason
                FROM experiments e
                LEFT JOIN trial_results tr ON tr.experiment_id = e.id
                WHERE e.id IN ({placeholders})
                ORDER BY e.name, tr.primary_score DESC
                """,
                experiment_ids,
            ).fetchall()
        return [dict(row) for row in rows]

    # --- checkpoints ---

    def save_checkpoint(self, run_id: str, experiment_id: str, next_model_index: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_checkpoints (run_id, experiment_id, next_model_index)
                VALUES (?, ?, ?)
                """,
                (run_id, experiment_id, next_model_index),
            )

    def get_checkpoint(self, run_id: str) -> tuple[str, int] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT experiment_id, next_model_index FROM run_checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return row["experiment_id"], int(row["next_model_index"])

    def clear_checkpoint(self, run_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM run_checkpoints WHERE run_id = ?", (run_id,))

    # --- events ---

    def append_event(self, event_type: str, payload: dict, run_id: str | None = None) -> str:
        event_id = f"evt_{uuid.uuid4().hex[:10]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (id, timestamp, run_id, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    datetime.now(timezone.utc).isoformat(),
                    run_id,
                    event_type,
                    json.dumps(payload),
                ),
            )
        return event_id

    def list_events(self, run_id: str | None = None, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            if run_id:
                rows = conn.execute(
                    """
                    SELECT * FROM events WHERE run_id = ?
                    ORDER BY timestamp DESC LIMIT ?
                    """,
                    (run_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "run_id": r["run_id"],
                "event_type": r["event_type"],
                "payload": json.loads(r["payload_json"]),
            }
            for r in rows
        ]

    # --- benchmark ---

    def save_benchmark_result(
        self,
        platform_version: str,
        scenario_id: str,
        metric: str,
        best_score: float,
        best_model: str | None,
        total_time_s: float,
        details: dict,
    ) -> str:
        result_id = f"bench_{uuid.uuid4().hex[:10]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO benchmark_results
                (id, platform_version, scenario_id, timestamp, metric, best_score,
                 best_model, total_time_s, details_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result_id,
                    platform_version,
                    scenario_id,
                    datetime.now(timezone.utc).isoformat(),
                    metric,
                    best_score,
                    best_model,
                    total_time_s,
                    json.dumps(details),
                ),
            )
        return result_id

    def list_benchmark_results(
        self,
        platform_version: str | None = None,
        scenario_id: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM benchmark_results WHERE 1=1"
        params: list = []
        if platform_version:
            query += " AND platform_version = ?"
            params.append(platform_version)
        if scenario_id:
            query += " AND scenario_id = ?"
            params.append(scenario_id)
        query += " ORDER BY timestamp DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "id": r["id"],
                "platform_version": r["platform_version"],
                "scenario_id": r["scenario_id"],
                "timestamp": r["timestamp"],
                "metric": r["metric"],
                "best_score": r["best_score"],
                "best_model": r["best_model"],
                "total_time_s": r["total_time_s"],
                "details": json.loads(r["details_json"]),
            }
            for r in rows
        ]


def _row_to_feature(row: sqlite3.Row) -> Feature:
    return Feature(
        id=row["id"],
        dataset_id=row["dataset_id"],
        name=row["name"],
        physical_dtype=row["physical_dtype"] or "unknown",
        status=FeatureStatus(row["status"]),
        user_priority=float(row["user_priority"] or 0),
    )


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    return Experiment(
        id=row["id"],
        run_id=row["run_id"],
        name=row["name"],
        hypothesis=row["hypothesis"] or "",
        feature_names=json.loads(row["feature_names_json"]),
        model_ids=json.loads(row["model_ids_json"]),
        metric=row["metric"],
        validation_strategy=row["validation_strategy"],
        status=ExperimentStatus(row["status"]),
        created_by=row["created_by"] or "user",
        priority=row["priority"] or "normal",
        feature_set_id=row["feature_set_id"],
    )


def _row_to_result(row: sqlite3.Row) -> TrialResult:
    return TrialResult(
        trial_id=row["trial_id"],
        experiment_id=row["experiment_id"],
        model_id=row["model_id"],
        primary_metric=row["primary_metric"],
        primary_score=row["primary_score"],
        secondary_metrics=json.loads(row["secondary_metrics_json"] or "{}"),
        training_time_seconds=row["training_time_seconds"] or 0.0,
        failure_reason=row["failure_reason"],
        artifacts=json.loads(row["artifacts_json"] or "{}"),
    )
