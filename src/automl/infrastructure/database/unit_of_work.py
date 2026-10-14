"""Unit of Work over a shared SQLite connection (Clean Arch PR4)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from automl.infrastructure.database.connection import open_sqlite


class SqliteUnitOfWork:
    """Owns one connection for a multi-step persistence transaction.

    Usage::

        with SqliteUnitOfWork(db_path) as uow:
            repo_ops_using(uow.connection)
            uow.commit()
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.connection: Any = None
        self._committed = False

    def __enter__(self) -> SqliteUnitOfWork:
        self.connection = open_sqlite(self.db_path)
        self._committed = False
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            if self.connection is None:
                return
            if exc_type is not None or not self._committed:
                self.connection.rollback()
            else:
                self.connection.commit()
        finally:
            if self.connection is not None:
                self.connection.close()
                self.connection = None

    def commit(self) -> None:
        if self.connection is None:
            raise RuntimeError("Unit of Work is not active")
        self.connection.commit()
        self._committed = True

    def rollback(self) -> None:
        if self.connection is None:
            return
        self.connection.rollback()
        self._committed = False
