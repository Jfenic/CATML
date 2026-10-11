"""Bounded stress check: workspace SQLite connections must not leak FDs."""
from __future__ import annotations

import gc
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository


def _open_fd_count() -> int:
    fd_dir = Path("/proc/self/fd")
    if fd_dir.is_dir():
        return len(list(fd_dir.iterdir()))
    return -1


def test_repository_context_closes_connection(tmp_path: Path):
    repo = SQLiteExperimentRepository(tmp_path / "repo.db")
    # Flush finalizers from prior tests so their ResourceWarnings are not attributed here.
    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for _ in range(50):
            with repo._connect() as conn:
                conn.execute("SELECT 1").fetchone()
        gc.collect()
    resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not resource_warnings, f"Unexpected ResourceWarnings: {resource_warnings[:5]}"


def test_workspace_create_dispose_stress_no_fd_growth(tmp_path: Path):
    csv_path = tmp_path / "tiny.csv"
    pd.DataFrame(
        {
            "x": np.random.randn(20),
            "y": [0, 1] * 10,
        }
    ).to_csv(csv_path, index=False)

    baseline = _open_fd_count()
    iterations = 80
    for i in range(iterations):
        ws = AutoMLWorkspace.create(f"stress_{i}", root_dir=tmp_path / f"ws_{i}")
        ds = ws.register_dataset(
            name="d",
            path=csv_path,
            target="y",
            task_type="binary_classification",
        )
        assert ds.id
        del ws
        gc.collect()

    after = _open_fd_count()
    if baseline < 0 or after < 0:
        pytest.skip("FD counting unavailable on this platform")
    # Allow modest slack for interpreter/cache; fail on sustained leak (~1 FD/iter).
    assert after - baseline < iterations // 2, (
        f"Open FDs grew too much: baseline={baseline}, after={after}, delta={after - baseline}"
    )
