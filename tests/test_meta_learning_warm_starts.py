"""
Unit and Integration Tests for Phase 1: Meta-Learning Warm Starts and Dataset Fingerprinting.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.queries.workspace_queries import GetMetaKnowledgeQuery
from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.datasets.profile import ColumnProfile, Dataset, DatasetProfile
from automl.domain.experiments.trial import TrialResult
from automl.domain.meta_learning.fingerprint import (
    DatasetFingerprint,
    HistoricalModelRanking,
    MetaLearningKnowledge,
    SimilarDatasetMatch,
    WarmStartRecommendation,
)
from automl.domain.optimization.search_space import ParameterSpec, SearchSpace
from automl.engine.meta_learning.extractor import extract_fingerprint
from automl.engine.meta_learning.knowledge_base import MetaKnowledgeBase, cosine_similarity
from automl.engine.optimization.random_search import RandomSearchOptimizer
from automl.plugins.optimizers.optuna_optimizer import OptunaOptimizer


def test_dataset_fingerprint_entity():
    """Verify DatasetFingerprint calculations and vector normalization."""
    fp = DatasetFingerprint(
        dataset_id="ds_1",
        dataset_name="Test DS",
        task_type="binary_classification",
        row_count=1000,
        feature_count=20,
        numerical_feature_count=15,
        categorical_feature_count=5,
        missing_cells_ratio=0.02,
        target_entropy=0.85,
    )

    assert fp.numerical_ratio == 0.75
    assert fp.categorical_ratio == 0.25
    assert fp.feature_to_row_ratio == 0.02

    vec = fp.to_vector()
    assert len(vec) == 6
    for val in vec:
        assert 0.0 <= val <= 1.0

    d = fp.to_dict()
    assert d["rows"] == 1000
    assert d["features"] == 20
    assert d["numerical_ratio"] == 0.75
    assert d["categorical_ratio"] == 0.25


def test_extract_fingerprint_classification(tmp_path: Path):
    """Test extract_fingerprint with a synthetic binary classification dataset."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num_1": np.random.randn(n),
        "num_2": np.random.uniform(10, 50, n),
        "cat_1": np.random.choice(["A", "B", "C"], n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    })
    csv_file = tmp_path / "clf_data.csv"
    df.to_csv(csv_file, index=False)

    ds = Dataset(
        id="ds_clf",
        workspace_id="ws_test",
        name="Clf Data",
        path=str(csv_file),
        target_column="target",
        task_type="binary_classification",
    )

    from automl.engine.profiling.dataset_profiler import profile_dataset
    profile = profile_dataset(ds, df=df)

    fp = extract_fingerprint(ds, profile, df=df)
    assert fp.dataset_id == "ds_clf"
    assert fp.row_count == 100
    assert fp.feature_count == 3
    assert fp.numerical_feature_count == 2
    assert fp.categorical_feature_count == 1
    assert 0.0 < fp.target_entropy <= 1.0


def test_cosine_similarity():
    """Verify cosine similarity mathematical bounds."""
    v1 = [1.0, 0.0, 0.5]
    v2 = [1.0, 0.0, 0.5]
    assert cosine_similarity(v1, v2) == pytest.approx(1.0, rel=1e-4)

    v_orth1 = [1.0, 0.0]
    v_orth2 = [0.0, 1.0]
    assert cosine_similarity(v_orth1, v_orth2) == 0.0

    # Empty or zero vectors
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_meta_knowledge_base_matching_and_recommendations():
    """Verify MetaKnowledgeBase benchmark matching and warm start rules."""
    kb = MetaKnowledgeBase()

    # 1. Categorical heavy dataset -> should recommend CatBoost
    cat_fp = DatasetFingerprint(
        dataset_id="ds_cat",
        dataset_name="Categorical Data",
        task_type="binary_classification",
        row_count=10000,
        feature_count=20,
        numerical_feature_count=6,
        categorical_feature_count=14,  # 70% categorical
        missing_cells_ratio=0.01,
        target_entropy=0.75,
    )
    similar = kb.find_similar(cat_fp, top_k=2)
    assert len(similar) == 2
    assert any("Categorical" in s.name or "Churn" in s.name for s in similar)

    warm_cat = kb.recommend_warm_start(cat_fp)
    assert warm_cat.recommended_model == "CatBoost"
    assert "depth" in warm_cat.params

    # 2. Dense large numerical dataset -> should recommend LightGBM
    num_fp = DatasetFingerprint(
        dataset_id="ds_num",
        dataset_name="Numerical Data",
        task_type="binary_classification",
        row_count=12000,
        feature_count=50,
        numerical_feature_count=45,  # 90% numerical
        categorical_feature_count=5,
        missing_cells_ratio=0.0,
        target_entropy=0.8,
    )
    warm_num = kb.recommend_warm_start(num_fp)
    assert warm_num.recommended_model == "LightGBM"
    assert "num_leaves" in warm_num.params

    # 3. Small dataset -> should recommend RandomForest or Ridge
    small_fp = DatasetFingerprint(
        dataset_id="ds_small",
        dataset_name="Small Data",
        task_type="binary_classification",
        row_count=300,
        feature_count=8,
        numerical_feature_count=5,
        categorical_feature_count=3,
        missing_cells_ratio=0.0,
        target_entropy=0.5,
    )
    warm_small = kb.recommend_warm_start(small_fp)
    assert warm_small.recommended_model == "RandomForest"


def test_meta_knowledge_base_dynamic_empirical_rankings():
    """Verify that active workspace trials override or enhance benchmark rankings."""
    kb = MetaKnowledgeBase()
    fp = DatasetFingerprint(
        dataset_id="ds_1",
        dataset_name="DS 1",
        task_type="binary_classification",
        row_count=2000,
        feature_count=15,
        numerical_feature_count=10,
        categorical_feature_count=5,
        missing_cells_ratio=0.0,
        target_entropy=0.7,
    )

    trials = [
        TrialResult(
            trial_id="t1",
            experiment_id="exp_1",
            model_id="xgboost",
            primary_metric="roc_auc",
            primary_score=0.92,
        ),
        TrialResult(
            trial_id="t2",
            experiment_id="exp_1",
            model_id="lightgbm",
            primary_metric="roc_auc",
            primary_score=0.85,
        ),
    ]

    rankings = kb.compute_rankings(fp, workspace_trials=trials)
    assert len(rankings) == 2
    assert rankings[0].model == "xgboost"
    assert rankings[0].mean_rank == 1.0


def test_random_search_optimizer_warm_start():
    """Verify that RandomSearchOptimizer yields warm-start parameters on trial 0."""
    space = SearchSpace()
    space.add(ParameterSpec.int("n_estimators", 10, 200, default=100))
    space.add(ParameterSpec.float("learning_rate", 0.01, 0.3, default=0.1))

    warm_params = {"n_estimators": 150, "learning_rate": 0.05}
    opt = RandomSearchOptimizer(seed=42, warm_start_params=warm_params)

    suggested_0 = opt.suggest(0, space)
    assert suggested_0["n_estimators"] == 150
    assert suggested_0["learning_rate"] == 0.05

    # Trial 1 should be randomized
    suggested_1 = opt.suggest(1, space)
    assert "n_estimators" in suggested_1
    assert "learning_rate" in suggested_1


def test_optuna_optimizer_warm_start():
    """Verify that OptunaOptimizer enqueues and suggests warm-start parameters."""
    space = SearchSpace()
    space.add(ParameterSpec.int("n_estimators", 10, 200, default=100))
    space.add(ParameterSpec.float("learning_rate", 0.01, 0.3, default=0.1))

    warm_params = {"n_estimators": 120, "learning_rate": 0.08}
    opt = OptunaOptimizer(seed=42, warm_start_params=warm_params)

    suggested_0 = opt.suggest(0, space)
    assert suggested_0["n_estimators"] == 120
    assert suggested_0["learning_rate"] == pytest.approx(0.08, rel=1e-3)


def test_workspace_and_query_parity(tmp_path: Path):
    """Verify full end-to-end integration with Workspace and QueryBus."""
    ws_dir = tmp_path / "ws_meta"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    # Register simple dataset
    csv_path = tmp_path / "sample.csv"
    np.random.seed(42)
    df = pd.DataFrame({
        "x1": np.random.randn(80),
        "x2": np.random.randn(80),
        "cat": np.random.choice(["X", "Y"], 80),
        "target": np.random.choice([0, 1], 80),
    })
    df.to_csv(csv_path, index=False)

    ds = ws.register_dataset(
        name="sample_meta",
        path=csv_path,
        target="target",
    )

    # 1. Direct workspace call
    knowledge_ws = ws.get_meta_knowledge(ds.id)
    assert isinstance(knowledge_ws, MetaLearningKnowledge)
    assert knowledge_ws.dataset_name == "sample_meta"
    assert knowledge_ws.version == "0.8.0"
    assert len(knowledge_ws.similar_datasets) > 0
    assert len(knowledge_ws.historical_rankings) > 0
    assert knowledge_ws.warm_start is not None

    # 2. QueryBus parity
    knowledge_qb = qry_bus.dispatch(GetMetaKnowledgeQuery(dataset_id=ds.id))
    assert knowledge_qb.dataset_name == knowledge_ws.dataset_name
    assert knowledge_qb.warm_start.recommended_model == knowledge_ws.warm_start.recommended_model

    # 3. Serialized dictionary format checks
    d = knowledge_ws.to_dict()
    assert "fingerprint" in d
    assert "current_fingerprint" in d
    assert "similar_datasets" in d
    assert "historical_rankings" in d
    assert "warm_start" in d
