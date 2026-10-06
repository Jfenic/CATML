"""Durable SQLite checkpoint saver adapter for LangGraph agent workflows."""
from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, Optional

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover
    SqliteSaver = None  # type: ignore[assignment, misc]
    BaseCheckpointSaver = Any  # type: ignore[assignment, misc]
    CheckpointTuple = Any  # type: ignore[assignment, misc]
    _LANGGRAPH_AVAILABLE = False


class SqliteCheckpointSaver:
    """Manages durable SQLite persistence for LangGraph execution checkpoints and state tuples."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        if not _LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph checkpoint dependencies not found. "
                "Install them via `pip install 'catml[agents]'`."
            )

        self._db_path = Path(db_path) if db_path and str(db_path) != ":memory:" else None
        self._is_memory = str(db_path) == ":memory:" or (db_path is None and connection is None)

        if connection is not None:
            self._connection = connection
            self._owns_connection = False
        elif self._is_memory:
            self._connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._owns_connection = True
        else:
            assert self._db_path is not None
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=30.0,
            )
            self._owns_connection = True

        self._saver = SqliteSaver(self._connection)
        self._saver.setup()

    @property
    def saver(self) -> BaseCheckpointSaver:
        """Return the underlying LangGraph BaseCheckpointSaver."""
        return self._saver

    @property
    def connection(self) -> sqlite3.Connection:
        """Return the underlying sqlite3 Connection."""
        return self._connection

    @property
    def db_path(self) -> Optional[Path]:
        """Return path to database file if persistent."""
        return self._db_path

    def get_tuple(
        self,
        thread_id: str,
        checkpoint_ns: str = "",
        checkpoint_id: Optional[str] = None,
    ) -> Optional[CheckpointTuple]:
        """Retrieve a specific checkpoint tuple or latest tuple for thread_id."""
        config: dict[str, Any] = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        return self._saver.get_tuple(config)

    def get_latest_state(
        self,
        thread_id: str,
        checkpoint_ns: str = "",
    ) -> Optional[dict[str, Any]]:
        """Retrieve channel values from the latest checkpoint for thread_id."""
        tup = self.get_tuple(thread_id=thread_id, checkpoint_ns=checkpoint_ns)
        if tup is None or not tup.checkpoint:
            return None
        return dict(tup.checkpoint.get("channel_values", {}))

    def list_checkpoints(
        self,
        thread_id: str,
        checkpoint_ns: str = "",
        limit: Optional[int] = None,
    ) -> list[CheckpointTuple]:
        """List checkpoint history for a specific thread."""
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
            }
        }
        return list(self._saver.list(config, limit=limit))

    def close(self) -> None:
        """Close connection if owned."""
        if self._owns_connection and self._connection:
            try:
                self._connection.close()
            except Exception:
                pass

    def __enter__(self) -> SqliteCheckpointSaver:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
