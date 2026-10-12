"""Audit-4 remaining: paired hypothesis baseline, PARTIAL budget, PCA recipe, image cache."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.analysis.commands import CreateStudyCommand, VerifyHypothesisCommand
from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.analysis.models import AnalysisHypothesis
from automl.domain.experiments.trial import ExperimentStatus
from automl.domain.features.selection_strategy import FeatureSelectionStrategy
from automl.engine.training.sklearn_trainer import _build_pipeline
from automl.engine.vision.image_encoder import ImageEncoderNode
from sklearn.linear_model import LogisticRegression


def test_verify_hypothesis_uses_paired_baseline_not_leaderboard(tmp_path: Path):
    ws, cb, _ = build_application(root_dir=str(tmp_path / "ws"))
    rng = np.random.default_rng(0)
    n = 80
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
    run = ws.create_run(ds, metric="roc_auc", validation_strategy="kfold", cv_folds=3)
    # Unrelated prior experiment that must NOT become the verification baseline.
    prior = ws.create_experiment(
        run=run,
        name="unrelated_strong",
        feature_names=["a", "b"],
        model_ids=["random_forest"],
    )
    ws.run_experiment(run, prior)

    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="S"))
    hyp = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id="f1",
        description="drop b",
        proposed_action="resolve_collinearity",
        experiment_delta={"collinear_pair": ["a", "b"]},
    )
    ws.get_study_repository().save_hypothesis(hyp)

    link = cb.dispatch(VerifyHypothesisCommand(hypothesis_id=hyp.id, run_id=run.id, min_improvement=0.0))
    assert link["paired_baseline_experiment_id"] != prior.id
    assert "improved_in_experiment" in link
    assert "validated" in link
    baseline_exp = ws.repository.get_experiment(link["paired_baseline_experiment_id"])
    assert baseline_exp is not None
    assert set(baseline_exp.feature_names) == {"a", "b"}
    cand = ws.repository.get_experiment(link["experiment_id"])
    assert cand is not None
    assert "b" not in cand.feature_names


def test_budget_exhaustion_marks_experiment_partial(tmp_path: Path):
    ws = AutoMLWorkspace.create("budget_partial", root_dir=tmp_path / "ws")
    df = pd.DataFrame(
        {"x1": np.random.randn(60), "x2": np.random.randn(60), "target": [0, 1] * 30}
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    ds = ws.register_dataset(
        name="data", path=path, target="target", task_type="binary_classification"
    )
    run = ws.create_run(ds, time_budget_seconds=30.0)
    exp = ws.create_experiment(
        run=run,
        name="multi",
        feature_names=["x1", "x2"],
        model_ids=["logistic_regression", "random_forest", "extra_trees"],
    )

    # Let the first model finish, then expire the budget before the rest.
    original_is_exhausted = ws._is_run_budget_exhausted
    calls = {"n": 0}

    def _flaky_budget(r):
        calls["n"] += 1
        # After first trial scheduling checks, force exhaustion.
        return calls["n"] > 2

    ws._is_run_budget_exhausted = _flaky_budget  # type: ignore[method-assign]
    try:
        results = ws.run_experiment(run, exp)
    finally:
        ws._is_run_budget_exhausted = original_is_exhausted  # type: ignore[method-assign]

    refreshed_exp = ws.repository.get_experiment(exp.id)
    refreshed_run = ws.repository.get_run(run.id)
    assert refreshed_exp is not None
    assert len([r for r in results if r.succeeded]) >= 1
    assert refreshed_exp.status == ExperimentStatus.PARTIAL
    assert refreshed_run.config.extra.get("completion_reason") == "BUDGET_EXHAUSTED"
    assert refreshed_run.config.extra.get("time_budget_exhausted") is True


def test_pca_candidates_use_source_features_not_pc_names(tmp_path: Path):
    ws = AutoMLWorkspace.create("pca", root_dir=tmp_path / "ws")
    rng = np.random.default_rng(1)
    n = 50
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "c": rng.normal(size=n),
            "target": [0, 1] * (n // 2),
        }
    )
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    ds = ws.register_dataset(
        name="d", path=path, target="target", task_type="binary_classification"
    )
    run = ws.create_run(ds)
    cands = ws.select_features(
        run.id,
        strategy=FeatureSelectionStrategy(
            methods=["variance"],
            top_k=[2],
            include_reduction=True,
            reduction_variances=[0.95],
        ),
    )
    pca_cands = [c for c in cands if c.method == "pca"]
    assert pca_cands
    for c in pca_cands:
        assert all(not str(f).startswith("PC") for f in c.feature_names)
        assert c.metadata.get("reduction", {}).get("kind") == "pca"
        transforms = c.metadata.get("column_transforms") or []
        assert transforms and transforms[0]["kind"] == "pca"

    # Synthetic PC names remain rejected by create_experiment.
    with pytest.raises(ValueError, match="PCA component names"):
        ws.create_experiment(
            run=run,
            name="bad_pca",
            feature_names=["PC1", "PC2"],
            model_ids=["logistic_regression"],
        )


def test_build_pipeline_applies_pca_reduction_transform():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    pipe = _build_pipeline(
        X,
        LogisticRegression(max_iter=200),
        column_transforms=[{"column": "*", "kind": "pca", "params": {"n_components": 1}}],
    )
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["preprocessor", "pca", "model"]


def test_image_cache_invalidates_when_file_replaced(tmp_path: Path):
    img = tmp_path / "frame.png"
    # Minimal PNG header bytes; encoder may fall back depending on backend.
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    enc = ImageEncoderNode(
        node_id="t",
        model_name="deterministic",
        output_dim=16,
        use_cache=True,
        cache_dir=tmp_path / "cache",
        allow_fallback=True,
    )
    key1 = enc._get_cache_key(img)
    time.sleep(0.02)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\xff" * 48)
    key2 = enc._get_cache_key(img)
    assert key1 != key2
