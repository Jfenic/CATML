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


def test_in_memory_repos_context_and_gc_no_resource_warnings():
    """Verify in-memory SQLite repos release connection cleanly under ResourceWarning scrutiny."""
    from automl.domain.analysis.models import DataSourceRef, StudySpec
    from automl.domain.recipes.recipe import ExperimentRecipe, RecipeStatus
    from automl.infrastructure.database.sqlite_recipes import SQLiteRecipeRepository
    from automl.infrastructure.database.sqlite_studies import SQLiteStudyRepository

    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)

        # 1. Context manager usage
        for i in range(25):
            with SQLiteRecipeRepository(":memory:") as r_repo:
                rec = ExperimentRecipe.create(f"r_{i}", f"ds_{i}", ["f1"], status=RecipeStatus.DRAFT)
                r_repo.save(rec)
                assert r_repo.get(rec.id) is not None

            with SQLiteStudyRepository(":memory:") as s_repo:
                ds = DataSourceRef(dataset_id=f"ds_{i}", path=f"/tmp/ds_{i}.csv")
                study = StudySpec.create(f"ws_{i}", ds, f"Study {i}")
                s_repo.save_study(study)
                assert s_repo.get_study(study.id) is not None

        # 2. GC / finalizer path (without explicit close)
        for i in range(25):
            r_repo = SQLiteRecipeRepository(":memory:")
            rec = ExperimentRecipe.create(f"r_gc_{i}", f"ds_{i}", ["f1"], status=RecipeStatus.DRAFT)
            r_repo.save(rec)
            del r_repo

            s_repo = SQLiteStudyRepository(":memory:")
            ds = DataSourceRef(dataset_id=f"ds_gc_{i}", path=f"/tmp/ds_gc_{i}.csv")
            study = StudySpec.create(f"ws_gc_{i}", ds, f"Study GC {i}")
            s_repo.save_study(study)
            del s_repo

        gc.collect()

    resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not resource_warnings, f"Unexpected ResourceWarnings: {resource_warnings[:5]}"


def test_workspace_soak_study_and_recipe_ops_no_warnings_and_no_fd_leak(tmp_path: Path):
    """Issue #94 soak test: repeated workspace ops with studies/recipes and dispose()."""
    from automl.domain.analysis.models import DataSourceRef, StudySpec
    from automl.domain.recipes.recipe import ExperimentRecipe, RecipeStatus

    csv_path = tmp_path / "soak_data.csv"
    pd.DataFrame(
        {
            "x1": np.random.randn(25),
            "x2": np.random.randn(25),
            "target": [0, 1] * 12 + [0],
        }
    ).to_csv(csv_path, index=False)

    gc.collect()
    baseline = _open_fd_count()
    iterations = 50

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)

        for i in range(iterations):
            ws_dir = tmp_path / f"soak_ws_{i}"
            with AutoMLWorkspace.create(f"soak_{i}", root_dir=ws_dir) as ws:
                ds = ws.register_dataset(
                    name=f"d_{i}",
                    path=csv_path,
                    target="target",
                    task_type="binary_classification",
                )
                assert ds.id

                # Study repo ops through workspace
                s_repo = ws.get_study_repository()
                ref = DataSourceRef(dataset_id=ds.id, path=str(csv_path))
                study = StudySpec.create(ws.id, ref, f"Study {i}")
                s_repo.save_study(study)
                fetched_s = s_repo.get_study(study.id)
                assert fetched_s is not None

                # Recipe repo ops through workspace
                r_repo = ws.get_recipe_repository()
                rec = ExperimentRecipe.create(f"Recipe {i}", ds.id, ["x1", "x2"], status=RecipeStatus.DRAFT)
                r_repo.save(rec)
                fetched_r = r_repo.get(rec.id)
                assert fetched_r is not None

            # Context manager exited, dispose() called
            gc.collect()

    resource_warnings = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not resource_warnings, f"Unexpected ResourceWarnings in soak test: {resource_warnings[:5]}"

    after = _open_fd_count()
    if baseline >= 0 and after >= 0:
        assert after - baseline < iterations // 2, (
            f"Open FDs grew too much in soak: baseline={baseline}, after={after}, delta={after - baseline}"
        )
