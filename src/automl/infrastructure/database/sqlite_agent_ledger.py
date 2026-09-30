"""SQLite transactional repository for agent audit ledger, approvals, hypotheses, and sessions."""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any

from automl.domain.agents.entities import ApprovalStatus, Hypothesis, OperationStatus
from automl.application.agents.contracts import (
    AgentSessionState,
    ApprovalRequest,
    OperationRecord,
)
from automl.application.agents.ports import AgentLedgerPort


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteAgentLedger(AgentLedgerPort):
    """Transactional SQLite implementation of the AgentLedgerPort."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        # Ensure parent directory exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_operations (
                    operation_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    arguments_hash TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reserved_budget_json TEXT NOT NULL,
                    consumed_budget_json TEXT NOT NULL,
                    result_ref TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_op_idempotency
                ON agent_operations (run_id, action, idempotency_key)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_approvals (
                    approval_id TEXT PRIMARY KEY,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    arguments_hash TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    max_cost_json TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reviewer TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_appr_run
                ON agent_approvals (run_id, status)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    reasoning TEXT NOT NULL,
                    candidate_config_json TEXT NOT NULL,
                    target_metric TEXT NOT NULL,
                    metric_direction TEXT NOT NULL,
                    baseline_metric REAL,
                    verification_criteria_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_hyp_run
                ON agent_hypotheses (run_id)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.db_path, timeout=30)) as connection:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection

    def record_operation(self, op: OperationRecord) -> None:
        now = _utc_now()
        if not op.created_at:
            op.created_at = now
        if not op.updated_at:
            op.updated_at = now

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT arguments_hash FROM agent_operations
                WHERE run_id = ? AND action = ? AND idempotency_key = ?
                """,
                (op.run_id, op.action, op.idempotency_key),
            ).fetchone()

            if row:
                existing_hash = row["arguments_hash"]
                if existing_hash != op.arguments_hash:
                    raise ValueError(
                        f"Idempotency conflict: key '{op.idempotency_key}' already used with different arguments hash"
                    )
                # Idempotent match: record already registered
                return

            connection.execute(
                """
                INSERT INTO agent_operations (
                    operation_id, run_id, actor, idempotency_key, action,
                    arguments_hash, arguments_json, status, reserved_budget_json,
                    consumed_budget_json, result_ref, error_code, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    op.operation_id,
                    op.run_id,
                    op.actor,
                    op.idempotency_key,
                    op.action,
                    op.arguments_hash,
                    json.dumps(op.arguments, sort_keys=True),
                    op.status.value,
                    json.dumps(op.reserved_budget, sort_keys=True),
                    json.dumps(op.consumed_budget, sort_keys=True),
                    op.result_ref,
                    op.error_code,
                    op.error_message,
                    op.created_at,
                    op.updated_at,
                ),
            )

    def get_operation(self, operation_id: str) -> OperationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_operation(row)

    def get_operation_by_idempotency_key(
        self, run_id: str, action: str, idempotency_key: str
    ) -> OperationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM agent_operations
                WHERE run_id = ? AND action = ? AND idempotency_key = ?
                """,
                (run_id, action, idempotency_key),
            ).fetchone()
            if not row:
                return None
            return self._row_to_operation(row)

    def update_operation_status(
        self,
        operation_id: str,
        status: OperationStatus,
        consumed: dict[str, Any] | None = None,
        result_ref: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> OperationRecord:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM agent_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Operation not found: {operation_id}")

            current_consumed = json.loads(row["consumed_budget_json"])
            if consumed:
                current_consumed.update(consumed)

            new_result_ref = result_ref if result_ref is not None else row["result_ref"]
            new_error_code = error_code if error_code is not None else row["error_code"]
            new_error_msg = error_message if error_message is not None else row["error_message"]

            connection.execute(
                """
                UPDATE agent_operations
                SET status = ?, consumed_budget_json = ?, result_ref = ?,
                    error_code = ?, error_message = ?, updated_at = ?
                WHERE operation_id = ?
                """,
                (
                    status.value,
                    json.dumps(current_consumed, sort_keys=True),
                    new_result_ref,
                    new_error_code,
                    new_error_msg,
                    now,
                    operation_id,
                ),
            )
            updated_row = connection.execute(
                "SELECT * FROM agent_operations WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
            return self._row_to_operation(updated_row)

    def save_approval(self, req: ApprovalRequest) -> None:
        now = _utc_now()
        if not req.created_at:
            req.created_at = now
        if not req.updated_at:
            req.updated_at = now

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_approvals (
                    approval_id, action, actor, run_id, arguments_hash,
                    arguments_json, policy_version, max_cost_json, expires_at,
                    status, reviewer, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    req.approval_id,
                    req.action,
                    req.actor,
                    req.run_id,
                    req.arguments_hash,
                    json.dumps(req.arguments, sort_keys=True),
                    req.policy_version,
                    json.dumps(req.max_cost, sort_keys=True),
                    req.expires_at,
                    req.status.value,
                    req.reviewer,
                    req.created_at,
                    req.updated_at,
                ),
            )

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_approval(row)

    def update_approval_status(
        self, approval_id: str, status: ApprovalStatus, reviewer: str | None = None
    ) -> ApprovalRequest:
        now = _utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM agent_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Approval request not found: {approval_id}")

            current_reviewer = reviewer if reviewer is not None else row["reviewer"]
            connection.execute(
                """
                UPDATE agent_approvals
                SET status = ?, reviewer = ?, updated_at = ?
                WHERE approval_id = ?
                """,
                (status.value, current_reviewer, now, approval_id),
            )
            updated_row = connection.execute(
                "SELECT * FROM agent_approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            return self._row_to_approval(updated_row)

    def save_hypothesis(self, hyp: Hypothesis) -> None:
        now = _utc_now()
        if not hyp.created_at:
            hyp.created_at = now

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_hypotheses (
                    hypothesis_id, run_id, reasoning, candidate_config_json,
                    target_metric, metric_direction, baseline_metric,
                    verification_criteria_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hyp.hypothesis_id,
                    hyp.run_id,
                    hyp.reasoning,
                    json.dumps(hyp.candidate_config, sort_keys=True),
                    hyp.target_metric,
                    hyp.metric_direction,
                    hyp.baseline_metric,
                    json.dumps(hyp.verification_criteria, sort_keys=True),
                    hyp.status,
                    hyp.created_at,
                ),
            )

    def get_hypothesis(self, hypothesis_id: str) -> Hypothesis | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_hypothesis(row)

    def list_hypotheses(self, run_id: str) -> list[Hypothesis]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM agent_hypotheses WHERE run_id = ? ORDER BY created_at ASC",
                (run_id,),
            ).fetchall()
            return [self._row_to_hypothesis(r) for r in rows]

    def save_session_state(self, state: AgentSessionState) -> None:
        now = _utc_now()
        if not state.created_at:
            state.created_at = now
        state.updated_at = now

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO agent_sessions (
                    session_id, run_id, goal, version, status,
                    state_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.session_id,
                    state.run_id,
                    state.goal,
                    state.version,
                    state.status,
                    json.dumps(state.to_dict(), sort_keys=True),
                    state.created_at,
                    state.updated_at,
                ),
            )

    def get_session_state(self, session_id: str) -> AgentSessionState | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM agent_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            data = json.loads(row["state_json"])
            return AgentSessionState.from_dict(data)

    def _row_to_operation(self, row: sqlite3.Row) -> OperationRecord:
        return OperationRecord(
            operation_id=row["operation_id"],
            run_id=row["run_id"],
            actor=row["actor"],
            idempotency_key=row["idempotency_key"],
            action=row["action"],
            arguments_hash=row["arguments_hash"],
            arguments=json.loads(row["arguments_json"]),
            status=OperationStatus(row["status"]),
            reserved_budget=json.loads(row["reserved_budget_json"]),
            consumed_budget=json.loads(row["consumed_budget_json"]),
            result_ref=row["result_ref"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_approval(self, row: sqlite3.Row) -> ApprovalRequest:
        return ApprovalRequest(
            approval_id=row["approval_id"],
            action=row["action"],
            actor=row["actor"],
            run_id=row["run_id"],
            arguments_hash=row["arguments_hash"],
            arguments=json.loads(row["arguments_json"]),
            policy_version=row["policy_version"],
            max_cost=json.loads(row["max_cost_json"]),
            expires_at=row["expires_at"],
            status=ApprovalStatus(row["status"]),
            reviewer=row["reviewer"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _row_to_hypothesis(self, row: sqlite3.Row) -> Hypothesis:
        return Hypothesis(
            hypothesis_id=row["hypothesis_id"],
            run_id=row["run_id"],
            reasoning=row["reasoning"],
            candidate_config=json.loads(row["candidate_config_json"]),
            target_metric=row["target_metric"],
            metric_direction=row["metric_direction"],
            baseline_metric=row["baseline_metric"],
            verification_criteria=json.loads(row["verification_criteria_json"]),
            status=row["status"],
            created_at=row["created_at"],
        )
