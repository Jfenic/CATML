"""SQLite persistence stores by aggregate (Clean Arch PR4).

The public facade remains ``SQLiteExperimentRepository``. These modules document
aggregate boundaries and host shared helpers as the god-repository is split further.
"""

from automl.infrastructure.database.connection import connect, open_sqlite
from automl.infrastructure.database.unit_of_work import SqliteUnitOfWork

__all__ = [
    "connect",
    "open_sqlite",
    "SqliteUnitOfWork",
]
