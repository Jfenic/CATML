"""Shared SQLite connection helpers for workspace persistence."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


def open_sqlite(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def connect(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    """Open a short-lived connection that always closes (sqlite3 CM does not)."""
    conn = open_sqlite(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
