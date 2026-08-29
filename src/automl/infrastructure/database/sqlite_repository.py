from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from automl.domain.datasets.profile import ColumnProfile, DatasetProfile
from automl.domain.experiments.trial import Experiment, ExperimentStatus, Trial, TrialResult, TrialStatus
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
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    hypothesis TEXT,
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
                """
            )

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

    def save_experiment(self, experiment: Experiment) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (id, run_id, name, hypothesis, feature_names_json, model_ids_json,
                 metric, validation_strategy, status, created_by, priority)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment.id,
                    experiment.run_id,
                    experiment.name,
                    experiment.hypothesis,
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
        return [_row_to_experiment(row) for row in rows]

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
