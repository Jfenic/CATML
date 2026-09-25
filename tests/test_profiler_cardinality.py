from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.domain.datasets.profile import ColumnProfile, Dataset, DatasetProfile
from automl.engine.planning.experiment_planner import RuleBasedExperimentPlanner
from automl.engine.profiling.dataset_profiler import (
    detect_column_cardinality_and_role,
    profile_dataset,
)


def test_detect_column_cardinality_name_heuristics():
    # Suffixes, exact names, and prefixes with sufficient uniqueness
    series_ids = pd.Series([f"C_{i}" for i in range(100)])
    
    for id_col in ["id", "customer_id", "CustomerId", "PassengerId", "guid", "user_uuid", "id_client", "record_pk"]:
        card_ratio, is_id, is_high_card = detect_column_cardinality_and_role(
            name=id_col,
            series=series_ids,
            row_count=100,
        )
        assert is_id is True, f"Expected {id_col} to be detected as identifier"
        assert card_ratio == 1.0


def test_detect_column_cardinality_non_id_words_not_flagged():
    # Words ending in 'id' that are not identifiers
    series_numeric = pd.Series([10.5 + i * 2.3 for i in range(100)])
    for non_id in ["valid", "solid", "liquid", "grid", "hybrid", "acid"]:
        _, is_id, _ = detect_column_cardinality_and_role(
            name=non_id,
            series=series_numeric,
            row_count=100,
        )
        assert is_id is False, f"Did not expect {non_id} to be flagged as identifier"


def test_continuous_numerical_features_not_flagged_as_identifiers():
    # Salary has high uniqueness ratio in real life (e.g. 98%), but is a numeric feature
    salary_series = pd.Series([20000 + (i * 357) % 50000 for i in range(100)], dtype=float)
    card_ratio, is_id, is_high_card = detect_column_cardinality_and_role(
        name="salary",
        series=salary_series,
        row_count=100,
    )
    assert card_ratio > 0.70
    assert is_id is False
    assert is_high_card is False


def test_low_cardinality_code_not_flagged():
    # Column named status_id with only 2 unique values out of 100 rows
    status_series = pd.Series([i % 2 for i in range(100)])
    card_ratio, is_id, _ = detect_column_cardinality_and_role(
        name="status_id",
        series=status_series,
        row_count=100,
    )
    assert card_ratio == 0.02
    assert is_id is False


def test_monotonic_sequential_integer_index():
    # Synthetic index columns without 'id' in name (e.g. 0, 1, 2, ... 99)
    index_series = pd.Series(list(range(100)))
    card_ratio, is_id, _ = detect_column_cardinality_and_role(
        name="row_num",
        series=index_series,
        row_count=100,
    )
    assert card_ratio == 1.0
    assert is_id is True


def test_high_cardinality_text_columns():
    # Names or hashes with >70% uniqueness in >=50 rows
    text_series = pd.Series([f"Description_text_{i}" for i in range(80)] + ["Repeated"] * 20)
    card_ratio, is_id, is_high_card = detect_column_cardinality_and_role(
        name="description",
        series=text_series,
        row_count=100,
    )
    assert card_ratio == 0.81
    assert is_high_card is True
    assert is_id is True  # Near-unique strings act as row identifiers


def test_target_column_never_marked_as_identifier():
    series = pd.Series(list(range(100)))
    card_ratio, is_id, is_high_card = detect_column_cardinality_and_role(
        name="id_target",
        series=series,
        row_count=100,
        target_column="id_target",
    )
    assert is_id is False
    assert is_high_card is False


def test_profile_dataset_and_properties(tmp_path: Path):
    csv_file = tmp_path / "cardinality_sample.csv"
    df = pd.DataFrame(
        {
            "customer_id": [f"ID_{i:04d}" for i in range(100)],
            "salary": [30000 + i * 500 for i in range(100)],
            "country": ["ES", "FR", "DE", "IT"] * 25,
            "surname": [f"Surname_{i}" for i in range(85)] + ["Smith"] * 15,
            "churn": [i % 2 for i in range(100)],
        }
    )
    df.to_csv(csv_file, index=False)

    dataset = Dataset(
        id="ds_card",
        workspace_id="ws_1",
        name="card_test",
        path=str(csv_file),
        target_column="churn",
        task_type="binary_classification",
    )

    profile = profile_dataset(dataset)
    assert profile.row_count == 100
    assert profile.column_count == 5

    # Check identifier columns detection
    assert "customer_id" in profile.identifier_column_names
    assert "surname" in profile.identifier_column_names
    assert "salary" not in profile.identifier_column_names
    assert "country" not in profile.identifier_column_names
    assert "churn" not in profile.identifier_column_names

    # Check recommended features excludes identifiers and target
    recommended = profile.recommended_feature_names
    assert "salary" in recommended
    assert "country" in recommended
    assert "customer_id" not in recommended
    assert "surname" not in recommended
    assert "churn" not in recommended

    # Serialization to dict
    d = profile.to_dict()
    assert any(c["name"] == "customer_id" and c["is_identifier"] is True for c in d["columns"])
    assert any(c["name"] == "salary" and c["is_identifier"] is False for c in d["columns"])


def test_workspace_exclude_identifiers_and_planner_candidate_exclusion(tmp_path: Path):
    csv_file = tmp_path / "bank_sample.csv"
    df = pd.DataFrame(
        {
            "CustomerId": [1000 + i for i in range(100)],
            "salary": [50000 + i * 100 for i in range(100)],
            "debt": [2000 + (i % 5) * 500 for i in range(100)],
            "churn": [i % 2 for i in range(100)],
        }
    )
    df.to_csv(csv_file, index=False)

    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws"))
    dataset = ws.register_dataset("bank", str(csv_file), target="churn")
    run = ws.create_run(dataset, metric="roc_auc")

    # Feature registry records semantic type
    feat_cust = ws.get_feature_registry(dataset.id).get("CustomerId")
    assert feat_cust is not None
    assert feat_cust.semantic_type == "identifier"

    # RuleBasedExperimentPlanner automatically excludes CustomerId from candidates
    candidates = ws.planner.propose(
        run=run,
        profile=ws.repository.get_dataset_profile(dataset.id),
        feature_registry=ws.get_feature_registry(dataset.id),
        model_registry=ws.model_registry,
    )
    assert len(candidates) > 0
    for cand in candidates:
        assert "CustomerId" not in cand.feature_names
        assert "salary" in cand.feature_names

    # Test workspace.exclude_identifiers explicitly
    excluded = ws.exclude_identifiers(dataset.id, run_id=run.id)
    assert "CustomerId" in excluded
    assert "CustomerId" in run.config.features_excluded


def test_semantic_type_persists_in_sqlite_and_survives_workspace_reload(tmp_path: Path):
    from automl.application.services.workspace import AutoMLWorkspace

    csv_file = tmp_path / "reload_sample.csv"
    df = pd.DataFrame(
        {
            "user_id": [f"U_{i}" for i in range(100)],
            "score": [float(i * 1.5) for i in range(100)],
            "target": [i % 2 for i in range(100)],
        }
    )
    df.to_csv(csv_file, index=False)

    ws_dir = tmp_path / "ws_persist"
    ws = AutoMLWorkspace.create("persist_test", root_dir=ws_dir)
    dataset = ws.register_dataset("users", str(csv_file), target="target")

    # Verify in memory
    feature_before = ws.get_feature_registry(dataset.id).get("user_id")
    assert feature_before is not None
    assert feature_before.semantic_type == "identifier"

    # Reload from disk
    ws_reloaded = AutoMLWorkspace.load(ws_dir)
    feature_after = ws_reloaded.get_feature_registry(dataset.id).get("user_id")
    assert feature_after is not None
    # Crucial test: semantic_type must not revert to "unknown"
    assert feature_after.semantic_type == "identifier"


def test_sqlite_migration_adds_semantic_type_column(tmp_path: Path):
    import sqlite3
    from automl.domain.features.feature import Feature, FeatureStatus
    from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository

    db_path = tmp_path / "legacy.db"

    # Create a legacy features table without semantic_type column
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE features (
                id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                name TEXT NOT NULL,
                physical_dtype TEXT,
                status TEXT NOT NULL,
                user_priority REAL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO features (id, dataset_id, name, physical_dtype, status, user_priority)
            VALUES ('feat_legacy', 'ds_1', 'legacy_col', 'int64', 'ACTIVE', 0)
            """
        )

    # Initializing repository should automatically run _migrate() and add semantic_type
    repo = SQLiteExperimentRepository(db_path)

    # Load legacy feature - should default to "unknown" without error
    features = repo.list_features("ds_1")
    assert len(features) == 1
    assert features[0].name == "legacy_col"
    assert features[0].semantic_type == "unknown"

    # Save a feature with semantic_type="identifier"
    new_feat = Feature(
        id="feat_new",
        dataset_id="ds_1",
        name="new_id_col",
        semantic_type="identifier",
        status=FeatureStatus.ACTIVE,
    )
    repo.save_feature(new_feat)

    reloaded_feat = next(f for f in repo.list_features("ds_1") if f.name == "new_id_col")
    assert reloaded_feat.semantic_type == "identifier"

