"""
Tests for Anti-Leakage Guardian and Plugin Observability (Phase 4).
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from automl.domain.datasets.profile import Dataset, DatasetProfile
from automl.engine.profiling.dataset_profiler import profile_dataset
from automl.plugins.models.gradient_boosting import LightGBMPlugin, XGBoostPlugin
from automl.plugins.models.catboost_plugin import CatBoostPlugin
from automl.application.services.workspace import AutoMLWorkspace


def test_anti_leakage_target_correlation(tmp_path):
    """Test that a feature with |r| >= 0.999 to target is identified as leakage and excluded."""
    csv_path = tmp_path / "leaky_data.csv"
    np.random.seed(42)
    n = 100
    target = np.random.randn(n)
    leaky_feature = target + np.random.normal(0, 1e-5, n)  # r ~ 0.99999
    normal_feature = np.random.randn(n)

    df = pd.DataFrame({
        "feat_clean": normal_feature,
        "feat_leaky": leaky_feature,
        "target": target,
    })
    df.to_csv(csv_path, index=False)

    ds = Dataset(
        id="ds_leak_test",
        workspace_id="ws_test",
        name="Leak Test",
        path=str(csv_path),
        target_column="target",
        task_type="regression",
    )

    profile = profile_dataset(ds, df=df)

    assert profile.has_leakage is True
    assert "feat_leaky" in profile.leakage_column_names
    assert "feat_clean" not in profile.leakage_column_names
    assert "target" not in profile.leakage_column_names

    # Recommended features must exclude the leaky feature
    assert "feat_clean" in profile.recommended_feature_names
    assert "feat_leaky" not in profile.recommended_feature_names

    # Check to_dict serialization
    d = profile.to_dict()
    assert d["has_leakage"] is True
    assert "feat_leaky" in d["leakage_columns"]

    # Verify recommendation details
    leak_recs = [r for r in profile.recommendations if r.get("type") == "leakage" and r.get("column") == "feat_leaky"]
    assert len(leak_recs) == 1
    assert leak_recs[0]["badge"] == "Target Leakage"
    assert leak_recs[0]["severity"] == "danger"
    assert leak_recs[0]["action"] == "exclude"


def test_anti_leakage_clean_dataset(tmp_path):
    """Test that a clean dataset has has_leakage == False and empty leakage_column_names."""
    csv_path = tmp_path / "clean_data.csv"
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "f2": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n),
    })
    df.to_csv(csv_path, index=False)

    ds = Dataset(
        id="ds_clean_test",
        workspace_id="ws_test",
        name="Clean Test",
        path=str(csv_path),
        target_column="target",
        task_type="binary_classification",
    )

    profile = profile_dataset(ds, df=df)
    assert profile.has_leakage is False
    assert profile.leakage_column_names == []
    assert set(profile.recommended_feature_names) == {"f1", "f2"}
    assert profile.to_dict()["has_leakage"] is False
    assert profile.to_dict()["leakage_columns"] == []


def test_anti_leakage_sequential_row_order(tmp_path):
    """Test detection of sequential/ordering leakage where target is sorted by row index."""
    csv_path = tmp_path / "sorted_target.csv"
    n = 50
    # Strictly increasing target sorted with rows
    df = pd.DataFrame({
        "f1": np.random.randn(n),
        "target": np.arange(n, dtype=float),
    })
    df.to_csv(csv_path, index=False)

    ds = Dataset(
        id="ds_seq_test",
        workspace_id="ws_test",
        name="Seq Test",
        path=str(csv_path),
        target_column="target",
        task_type="regression",
    )

    profile = profile_dataset(ds, df=df)
    assert profile.has_leakage is True
    # The target itself should have a sequential leakage recommendation
    seq_recs = [r for r in profile.recommendations if r.get("badge") == "Sequential Leakage"]
    assert len(seq_recs) == 1
    assert seq_recs[0]["column"] == "target"
    assert seq_recs[0]["action"] == "warn"
    # Target should not be in leakage_column_names (which lists features to exclude)
    assert "target" not in profile.leakage_column_names


def test_plugin_observability_properties():
    """Verify fallback_backend and is_native attributes across gradient boosting plugins."""
    lgb_plugin = LightGBMPlugin()
    assert lgb_plugin.fallback_backend == "HistGradientBoosting"
    assert isinstance(lgb_plugin.is_native, bool)
    assert lgb_plugin.is_native == lgb_plugin.is_available

    xgb_plugin = XGBoostPlugin()
    assert xgb_plugin.fallback_backend == "GradientBoosting"
    assert isinstance(xgb_plugin.is_native, bool)
    assert xgb_plugin.is_native == xgb_plugin.is_available

    cat_plugin = CatBoostPlugin()
    assert cat_plugin.fallback_backend == "HistGradientBoosting"
    assert isinstance(cat_plugin.is_native, bool)
    assert cat_plugin.is_native == cat_plugin.is_available


def test_plugin_observability_mocked_unavailable():
    """Verify is_native behaves correctly when native libraries are mocked as unavailable."""
    with patch.object(LightGBMPlugin, "is_available", new=False):
        lgb_plugin = LightGBMPlugin()
        assert lgb_plugin.is_native is False

    with patch.object(XGBoostPlugin, "is_available", new=False):
        xgb_plugin = XGBoostPlugin()
        assert xgb_plugin.is_native is False


def test_workspace_list_plugins_observability(tmp_path):
    """Verify workspace.list_plugins includes is_native, backend_status, and fallback_backend."""
    ws = AutoMLWorkspace.create("test_ws", root_dir=tmp_path)

    plugins = ws.list_plugins()
    assert len(plugins) > 0

    plugin_map = {p["plugin_id"]: p for p in plugins}

    # Verify standard keys exist
    for p in plugins:
        assert "is_native" in p
        assert isinstance(p["is_native"], bool)
        assert "backend_status" in p
        assert isinstance(p["backend_status"], str)

    # CatBoost is typically fallback in current env
    if "catboost" in plugin_map:
        cat = plugin_map["catboost"]
        if not cat["is_native"]:
            assert "fallback" in cat["backend_status"]
            assert cat["fallback_backend"] == "HistGradientBoosting"
        else:
            assert cat["backend_status"] == "native"
