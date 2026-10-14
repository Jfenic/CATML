"""Audit-4 mediums: target cleaning, tree vs SHAP evidence, exploratory selection, nested simplex."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.features.evidence import FeatureEvidence
from automl.domain.features.selection_strategy import FeatureSelectionStrategy
from automl.engine.ensemble.blender import blend_predictions, optimize_ensemble_weights
from automl.engine.features.common import prepare_feature_matrix


def test_prepare_feature_matrix_regression_drops_missing_targets():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "target": [1.5, np.nan, 3.5, "bad"],
        }
    )
    X, y = prepare_feature_matrix(df, ["x"], target_column="target", task_type="regression")
    assert len(X) == 2
    assert y is not None
    assert list(y) == pytest.approx([1.5, 3.5])
    assert 0.0 not in list(y)  # never fabricate zeros for missing targets


def test_prepare_feature_matrix_classification_does_not_create_nan_class():
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "target": ["a", "b", np.nan, "a"],
        }
    )
    X, y = prepare_feature_matrix(
        df, ["x"], target_column="target", task_type="binary_classification"
    )
    assert len(X) == 3
    assert y is not None
    assert set(np.unique(y)) == {0, 1}


def test_prepare_feature_matrix_rejects_all_missing_regression_targets():
    df = pd.DataFrame({"x": [1.0, 2.0], "target": [np.nan, np.nan]})
    with pytest.raises(ValueError, match="no valid numeric"):
        prepare_feature_matrix(df, ["x"], target_column="target", task_type="regression")


def test_select_features_stores_tree_importance_not_shap(tmp_path: Path):
    ws = AutoMLWorkspace.create("feat_ev", root_dir=tmp_path / "ws")
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "target": [0, 1] * (n // 2),
        }
    )
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    ds = ws.register_dataset(
        name="d", path=path, target="target", task_type="binary_classification"
    )
    run = ws.create_run(ds)
    candidates = ws.select_features(
        run.id,
        strategy=FeatureSelectionStrategy(methods=["importance", "mutual_information"], top_k=[2]),
    )
    assert candidates
    assert candidates[0].metadata.get("evaluation_mode") == "nested_in_fold"
    assert candidates[0].metadata.get("supervised_selection_leakage_risk") is False
    assert candidates[0].metadata.get("column_transforms")

    evidence_rows = ws.repository.list_feature_evidence(run.id)
    assert evidence_rows
    assert any(ev.tree_importance is not None for ev in evidence_rows)
    assert all(ev.shap_importance is None for ev in evidence_rows)


def test_feature_evidence_roundtrip_tree_importance(tmp_path: Path):
    from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository

    repo = SQLiteExperimentRepository(tmp_path / "r.db")
    ev = FeatureEvidence(feature_id="a", tree_importance=0.42, mutual_information=0.1)
    repo.save_feature_evidence("run1", ev)
    loaded = repo.get_feature_evidence("run1", "a")
    assert loaded is not None
    assert loaded.tree_importance == pytest.approx(0.42)
    assert loaded.shap_importance is None


def test_simplex_nested_score_not_identical_to_in_sample_weight_fit():
    rng = np.random.default_rng(3)
    n = 80
    y = np.array([0, 1] * (n // 2))
    p1 = np.clip(y + rng.normal(0, 0.2, size=n), 0, 1)
    p2 = np.clip(y + rng.normal(0, 0.35, size=n), 0, 1)

    # Build synthetic OOF via evaluate_oof with tiny models would be heavy; exercise nested path
    # by comparing full-fit blend score vs nested fold blend constructed like oof.py.
    from sklearn.model_selection import StratifiedKFold

    folds = 4
    fold_ids = np.zeros(n, dtype=int)
    for i, (_, valid_idx) in enumerate(
        StratifiedKFold(n_splits=folds, shuffle=True, random_state=0).split(np.zeros(n), y)
    ):
        fold_ids[valid_idx] = i

    oof_list = [p1, p2]
    nested = np.zeros(n, dtype=float)
    for fold in range(folds):
        train_mask = fold_ids != fold
        hold_mask = fold_ids == fold
        w, _ = optimize_ensemble_weights(
            [p[train_mask] for p in oof_list],
            y[train_mask],
            metric="roc_auc",
            task_type="binary_classification",
            method="average",
        )
        nested[hold_mask] = blend_predictions(
            [p[hold_mask] for p in oof_list], weights=w, method="average"
        )

    full_w, _ = optimize_ensemble_weights(
        oof_list, y, metric="roc_auc", task_type="binary_classification", method="average"
    )
    in_sample = blend_predictions(oof_list, weights=full_w, method="average")
    assert not np.allclose(nested, in_sample, atol=1e-12)
