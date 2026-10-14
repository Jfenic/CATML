"""Additive relational SQLite persistence for CATML Experiment Recipes."""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import sqlite3
from typing import Any, Generator

from automl.domain.recipes.ports import ExperimentRecipeRepositoryPort
from automl.domain.recipes.recipe import ExperimentRecipe, RecipeStatus

logger = logging.getLogger(__name__)


class SQLiteRecipeRepository(ExperimentRecipeRepositoryPort):
    """SQLite implementation of ExperimentRecipeRepositoryPort."""

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

    def close(self) -> None:
        """Release the long-lived in-memory connection, if any."""
        if self._mem_conn is not None:
            try:
                self._mem_conn.close()
            except Exception:
                pass
            self._mem_conn = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> SQLiteRecipeRepository:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiment_recipes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dataset_content_hash TEXT,
                    feature_names_json TEXT NOT NULL,
                    derived_features_json TEXT NOT NULL,
                    column_transforms_json TEXT NOT NULL,
                    drop_features_json TEXT NOT NULL,
                    model_ids_json TEXT NOT NULL,
                    model_params_json TEXT NOT NULL,
                    validation_strategy TEXT NOT NULL,
                    cv_folds INTEGER NOT NULL,
                    metric TEXT NOT NULL,
                    random_seed INTEGER NOT NULL,
                    evidence_link_id TEXT,
                    hypothesis_id TEXT,
                    study_id TEXT,
                    lineage_json TEXT NOT NULL,
                    experiment_ids_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_recipes_dataset_id ON experiment_recipes(dataset_id);
                CREATE INDEX IF NOT EXISTS idx_recipes_status ON experiment_recipes(status);
                """
            )

    def save(self, recipe: ExperimentRecipe) -> None:
        status_val = recipe.status.value if isinstance(recipe.status, RecipeStatus) else str(recipe.status).lower()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiment_recipes (
                    id, name, dataset_id, status, dataset_content_hash,
                    feature_names_json, derived_features_json, column_transforms_json,
                    drop_features_json, model_ids_json, model_params_json,
                    validation_strategy, cv_folds, metric, random_seed,
                    evidence_link_id, hypothesis_id, study_id,
                    lineage_json, experiment_ids_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    dataset_id = excluded.dataset_id,
                    status = excluded.status,
                    dataset_content_hash = excluded.dataset_content_hash,
                    feature_names_json = excluded.feature_names_json,
                    derived_features_json = excluded.derived_features_json,
                    column_transforms_json = excluded.column_transforms_json,
                    drop_features_json = excluded.drop_features_json,
                    model_ids_json = excluded.model_ids_json,
                    model_params_json = excluded.model_params_json,
                    validation_strategy = excluded.validation_strategy,
                    cv_folds = excluded.cv_folds,
                    metric = excluded.metric,
                    random_seed = excluded.random_seed,
                    evidence_link_id = excluded.evidence_link_id,
                    hypothesis_id = excluded.hypothesis_id,
                    study_id = excluded.study_id,
                    lineage_json = excluded.lineage_json,
                    experiment_ids_json = excluded.experiment_ids_json,
                    updated_at = excluded.updated_at
                """,
                (
                    recipe.id,
                    recipe.name,
                    recipe.dataset_id,
                    status_val,
                    recipe.dataset_content_hash,
                    json.dumps(recipe.feature_names),
                    json.dumps(recipe.derived_features),
                    json.dumps(recipe.column_transforms),
                    json.dumps(recipe.drop_features),
                    json.dumps(recipe.model_ids),
                    json.dumps(recipe.model_params),
                    recipe.validation_strategy,
                    recipe.cv_folds,
                    recipe.metric,
                    recipe.random_seed,
                    recipe.evidence_link_id,
                    recipe.hypothesis_id,
                    recipe.study_id,
                    json.dumps(recipe.lineage),
                    json.dumps(recipe.experiment_ids),
                    recipe.created_at,
                    recipe.updated_at,
                ),
            )

    def get(self, recipe_id: str) -> ExperimentRecipe | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiment_recipes WHERE id = ?",
                (recipe_id,),
            ).fetchone()
            if not row:
                return None
            return self._row_to_recipe(row)

    def list(
        self,
        dataset_id: str | None = None,
        status: RecipeStatus | str | None = None,
    ) -> list[ExperimentRecipe]:
        query = "SELECT * FROM experiment_recipes"
        conditions: list[str] = []
        params: list[Any] = []

        if dataset_id is not None:
            conditions.append("dataset_id = ?")
            params.append(dataset_id)
        if status is not None:
            status_val = status.value if isinstance(status, RecipeStatus) else str(status).lower()
            conditions.append("status = ?")
            params.append(status_val)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_recipe(r) for r in rows]

    def delete(self, recipe_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM experiment_recipes WHERE id = ?", (recipe_id,))

    def _row_to_recipe(self, row: sqlite3.Row) -> ExperimentRecipe:
        return ExperimentRecipe(
            id=row["id"],
            name=row["name"],
            dataset_id=row["dataset_id"],
            status=RecipeStatus(row["status"]),
            dataset_content_hash=row["dataset_content_hash"],
            feature_names=json.loads(row["feature_names_json"]),
            derived_features=json.loads(row["derived_features_json"]),
            column_transforms=json.loads(row["column_transforms_json"]),
            drop_features=json.loads(row["drop_features_json"]),
            model_ids=json.loads(row["model_ids_json"]),
            model_params=json.loads(row["model_params_json"]),
            validation_strategy=row["validation_strategy"],
            cv_folds=row["cv_folds"],
            metric=row["metric"],
            random_seed=row["random_seed"],
            evidence_link_id=row["evidence_link_id"],
            hypothesis_id=row["hypothesis_id"],
            study_id=row["study_id"],
            lineage=json.loads(row["lineage_json"]),
            experiment_ids=json.loads(row["experiment_ids_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
