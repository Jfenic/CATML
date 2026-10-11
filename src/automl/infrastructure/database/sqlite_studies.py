"""Additive relational SQLite persistence for CATML Explore studies and findings."""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sqlite3
from typing import Generator

from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    EvidenceLink,
    StatisticalFinding,
    StudySpec,
    StudyStatus,
    VisualizationSpec,
)
from automl.domain.analysis.ports import StudyRepositoryPort

logger = logging.getLogger(__name__)


class SQLiteStudyRepository(StudyRepositoryPort):
    """SQLite implementation of StudyRepositoryPort."""

    def __init__(self, db_path: str | Path):
        self.is_memory = str(db_path) == ":memory:"
        self.db_path = Path(db_path) if not self.is_memory else ":memory:"
        if not self.is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._mem_conn = None
        else:
            self._mem_conn = sqlite3.connect(":memory:", timeout=30.0)
            self._mem_conn.row_factory = sqlite3.Row
        self._init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        if self.is_memory and self._mem_conn is not None:
            try:
                yield self._mem_conn
                self._mem_conn.commit()
            except Exception:
                self._mem_conn.rollback()
                raise
            return

        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS analysis_studies (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    target_column TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_studies_ws ON analysis_studies(workspace_id, dataset_id);

                CREATE TABLE IF NOT EXISTS analysis_runs (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    findings_count INTEGER DEFAULT 0,
                    visualizations_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    data TEXT NOT NULL,
                    FOREIGN KEY(study_id) REFERENCES analysis_studies(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_runs_study ON analysis_runs(study_id);

                CREATE TABLE IF NOT EXISTS analysis_findings (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    analysis_run_id TEXT NOT NULL,
                    finding_type TEXT NOT NULL,
                    column_name TEXT,
                    secondary_column TEXT,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL,
                    FOREIGN KEY(study_id) REFERENCES analysis_studies(id) ON DELETE CASCADE,
                    FOREIGN KEY(analysis_run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_findings_study ON analysis_findings(study_id, analysis_run_id);

                CREATE TABLE IF NOT EXISTS analysis_visualizations (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    analysis_run_id TEXT NOT NULL,
                    chart_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL,
                    FOREIGN KEY(study_id) REFERENCES analysis_studies(id) ON DELETE CASCADE,
                    FOREIGN KEY(analysis_run_id) REFERENCES analysis_runs(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_viz_study ON analysis_visualizations(study_id, analysis_run_id);

                CREATE TABLE IF NOT EXISTS analysis_hypotheses (
                    id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    finding_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL,
                    FOREIGN KEY(study_id) REFERENCES analysis_studies(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_hyp_study ON analysis_hypotheses(study_id);

                CREATE TABLE IF NOT EXISTS evidence_links (
                    id TEXT PRIMARY KEY,
                    hypothesis_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_hyp ON evidence_links(hypothesis_id);
            """)

    def save_study(self, study: StudySpec) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_studies (id, workspace_id, dataset_id, name, target_column, status, created_at, data)
                VALUES (:id, :workspace_id, :dataset_id, :name, :target_column, :status, :created_at, :data)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    target_column = excluded.target_column,
                    status = excluded.status,
                    data = excluded.data
                """,
                {
                    "id": study.id,
                    "workspace_id": study.workspace_id,
                    "dataset_id": study.data_source.dataset_id,
                    "name": study.name,
                    "target_column": study.target_column,
                    "status": study.status.value,
                    "created_at": study.created_at,
                    "data": json.dumps(study.to_dict()),
                },
            )

    def get_study(self, study_id: str) -> StudySpec | None:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM analysis_studies WHERE id = ?", (study_id,)).fetchone()
            if not row:
                return None
            return StudySpec.from_dict(json.loads(row["data"]))

    def list_studies(self, workspace_id: str, dataset_id: str | None = None) -> list[StudySpec]:
        with self._connect() as conn:
            if dataset_id:
                cursor = conn.execute(
                    "SELECT data FROM analysis_studies WHERE workspace_id = ? AND dataset_id = ? ORDER BY created_at DESC",
                    (workspace_id, dataset_id),
                )
            else:
                cursor = conn.execute(
                    "SELECT data FROM analysis_studies WHERE workspace_id = ? ORDER BY created_at DESC",
                    (workspace_id,),
                )
            return [StudySpec.from_dict(json.loads(r["data"])) for r in cursor.fetchall()]

    def archive_study(self, study_id: str) -> None:
        study = self.get_study(study_id)
        if study:
            study.status = StudyStatus.ARCHIVED
            self.save_study(study)

    def save_analysis_run(self, run: AnalysisRun) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_runs (id, study_id, status, started_at, completed_at, findings_count, visualizations_count, error_message, data)
                VALUES (:id, :study_id, :status, :started_at, :completed_at, :findings_count, :visualizations_count, :error_message, :data)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    completed_at = excluded.completed_at,
                    findings_count = excluded.findings_count,
                    visualizations_count = excluded.visualizations_count,
                    error_message = excluded.error_message,
                    data = excluded.data
                """,
                {
                    "id": run.id,
                    "study_id": run.study_id,
                    "status": run.status,
                    "started_at": run.started_at,
                    "completed_at": run.completed_at,
                    "findings_count": run.findings_count,
                    "visualizations_count": run.visualizations_count,
                    "error_message": run.error_message,
                    "data": json.dumps(run.to_dict()),
                },
            )

    def get_analysis_run(self, run_id: str) -> AnalysisRun | None:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM analysis_runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            return AnalysisRun.from_dict(json.loads(row["data"]))

    def list_analysis_runs(self, study_id: str) -> list[AnalysisRun]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT data FROM analysis_runs WHERE study_id = ? ORDER BY started_at DESC",
                (study_id,),
            )
            return [AnalysisRun.from_dict(json.loads(r["data"])) for r in cursor.fetchall()]

    def save_finding(self, finding: StatisticalFinding) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_findings (id, study_id, analysis_run_id, finding_type, column_name, secondary_column, created_at, data)
                VALUES (:id, :study_id, :analysis_run_id, :finding_type, :column_name, :secondary_column, :created_at, :data)
                ON CONFLICT(id) DO UPDATE SET
                    finding_type = excluded.finding_type,
                    column_name = excluded.column_name,
                    secondary_column = excluded.secondary_column,
                    data = excluded.data
                """,
                {
                    "id": finding.id,
                    "study_id": finding.study_id,
                    "analysis_run_id": finding.analysis_run_id,
                    "finding_type": finding.finding_type,
                    "column_name": finding.column_name,
                    "secondary_column": finding.secondary_column,
                    "created_at": finding.created_at,
                    "data": json.dumps(finding.to_dict()),
                },
            )

    def list_findings(self, study_id: str, run_id: str | None = None) -> list[StatisticalFinding]:
        with self._connect() as conn:
            if run_id:
                cursor = conn.execute(
                    "SELECT data FROM analysis_findings WHERE study_id = ? AND analysis_run_id = ? ORDER BY created_at ASC",
                    (study_id, run_id),
                )
            else:
                cursor = conn.execute(
                    "SELECT data FROM analysis_findings WHERE study_id = ? ORDER BY created_at ASC",
                    (study_id,),
                )
            return [StatisticalFinding.from_dict(json.loads(r["data"])) for r in cursor.fetchall()]

    def save_visualization_spec(self, spec: VisualizationSpec) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_visualizations (id, study_id, analysis_run_id, chart_type, created_at, data)
                VALUES (:id, :study_id, :analysis_run_id, :chart_type, :created_at, :data)
                ON CONFLICT(id) DO UPDATE SET
                    chart_type = excluded.chart_type,
                    data = excluded.data
                """,
                {
                    "id": spec.id,
                    "study_id": spec.study_id,
                    "analysis_run_id": spec.analysis_run_id,
                    "chart_type": spec.chart_type,
                    "created_at": spec.created_at,
                    "data": json.dumps(spec.to_dict()),
                },
            )

    def list_visualization_specs(self, study_id: str, run_id: str | None = None) -> list[VisualizationSpec]:
        with self._connect() as conn:
            if run_id:
                cursor = conn.execute(
                    "SELECT data FROM analysis_visualizations WHERE study_id = ? AND analysis_run_id = ? ORDER BY created_at ASC",
                    (study_id, run_id),
                )
            else:
                cursor = conn.execute(
                    "SELECT data FROM analysis_visualizations WHERE study_id = ? ORDER BY created_at ASC",
                    (study_id,),
                )
            return [VisualizationSpec.from_dict(json.loads(r["data"])) for r in cursor.fetchall()]

    def save_hypothesis(self, hypothesis: AnalysisHypothesis) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analysis_hypotheses (id, study_id, finding_id, status, created_at, data)
                VALUES (:id, :study_id, :finding_id, :status, :created_at, :data)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    data = excluded.data
                """,
                {
                    "id": hypothesis.id,
                    "study_id": hypothesis.study_id,
                    "finding_id": hypothesis.finding_id,
                    "status": hypothesis.status,
                    "created_at": hypothesis.created_at,
                    "data": json.dumps(hypothesis.to_dict()),
                },
            )

    def get_hypothesis(self, hypothesis_id: str) -> AnalysisHypothesis | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT data FROM analysis_hypotheses WHERE id = ?",
                (hypothesis_id,),
            ).fetchone()
            if not row:
                return None
            return AnalysisHypothesis.from_dict(json.loads(row["data"]))

    def list_hypotheses(self, study_id: str) -> list[AnalysisHypothesis]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT data FROM analysis_hypotheses WHERE study_id = ? ORDER BY created_at ASC",
                (study_id,),
            )
            return [AnalysisHypothesis.from_dict(json.loads(r["data"])) for r in cursor.fetchall()]

    def save_evidence_link(self, link: EvidenceLink) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO evidence_links (id, hypothesis_id, experiment_id, accepted, created_at, data)
                VALUES (:id, :hypothesis_id, :experiment_id, :accepted, :created_at, :data)
                ON CONFLICT(id) DO UPDATE SET
                    accepted = excluded.accepted,
                    data = excluded.data
                """,
                {
                    "id": link.id,
                    "hypothesis_id": link.hypothesis_id,
                    "experiment_id": link.experiment_id,
                    "accepted": 1 if link.accepted else 0,
                    "created_at": link.created_at,
                    "data": json.dumps(link.to_dict()),
                },
            )

    def get_evidence_link(self, link_id: str) -> EvidenceLink | None:
        with self._connect() as conn:
            row = conn.execute("SELECT data FROM evidence_links WHERE id = ?", (link_id,)).fetchone()
            if not row:
                return None
            return EvidenceLink.from_dict(json.loads(row["data"]))

    def list_evidence_links(self, hypothesis_id: str | None = None) -> list[EvidenceLink]:
        with self._connect() as conn:
            if hypothesis_id:
                cursor = conn.execute(
                    "SELECT data FROM evidence_links WHERE hypothesis_id = ? ORDER BY created_at ASC",
                    (hypothesis_id,),
                )
            else:
                cursor = conn.execute(
                    "SELECT data FROM evidence_links ORDER BY created_at ASC"
                )
            return [EvidenceLink.from_dict(json.loads(r["data"])) for r in cursor.fetchall()]
