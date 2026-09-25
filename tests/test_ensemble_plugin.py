from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge

from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.engine.plugins.validator import CompatibilityValidator
from automl.plugins.models.ensemble import (
    VotingEnsembleEstimator,
    VotingEnsemblePlugin,
    blend_predictions,
)


def test_blend_predictions_validation() -> None:
    with pytest.raises(ValueError, match="Cannot blend empty predictions"):
        blend_predictions([])

    with pytest.raises(ValueError, match="shape mismatch"):
        blend_predictions([np.array([1, 2]), np.array([1, 2, 3])])

    with pytest.raises(ValueError, match="Weights length .* does not match"):
        blend_predictions([np.array([1, 2]), np.array([3, 4])], weights=[1.0])

    with pytest.raises(ValueError, match="Weights must be non-negative"):
        blend_predictions([np.array([1, 2]), np.array([3, 4])], weights=[-1.0, 1.0])

    with pytest.raises(ValueError, match="Sum of weights must be strictly positive"):
        blend_predictions([np.array([1, 2]), np.array([3, 4])], weights=[0.0, 0.0])


def test_blend_predictions_computation() -> None:
    p1 = np.array([0.2, 0.8])
    p2 = np.array([0.4, 0.6])

    # Equal weights
    blended = blend_predictions([p1, p2], task_type="regression")
    np.testing.assert_allclose(blended, [0.3, 0.7])

    # Weighted: 3 to 1
    blended_weighted = blend_predictions([p1, p2], weights=[3.0, 1.0], task_type="regression")
    np.testing.assert_allclose(blended_weighted, [0.25, 0.75])

    # 2D probabilities normalization
    prob1 = np.array([[0.3, 0.7], [0.6, 0.4]])
    prob2 = np.array([[0.5, 0.5], [0.8, 0.2]])
    blended_prob = blend_predictions([prob1, prob2], task_type="binary_classification")
    assert blended_prob.shape == (2, 2)
    np.testing.assert_allclose(blended_prob.sum(axis=1), [1.0, 1.0])


def test_voting_ensemble_estimator_binary_classification() -> None:
    X, y = make_classification(n_samples=150, n_features=6, random_state=42)

    rf = RandomForestClassifier(n_estimators=20, random_state=42)
    lr = LogisticRegression(random_state=42)

    ensemble = VotingEnsembleEstimator(
        estimators=[("rf", rf), ("lr", lr)],
        weights=[0.6, 0.4],
        task_type="binary_classification",
        voting="soft",
    )

    ensemble.fit(X, y)
    assert ensemble.is_fitted_ is True
    assert len(ensemble.classes_) == 2

    probs = ensemble.predict_proba(X)
    assert probs.shape == (150, 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    preds = ensemble.predict(X)
    assert len(preds) == 150
    assert np.mean(preds == y) > 0.8


def test_voting_ensemble_estimator_hard_voting() -> None:
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    ensemble = VotingEnsembleEstimator(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("lr", LogisticRegression(random_state=42)),
        ],
        task_type="binary_classification",
        voting="hard",
    )
    ensemble.fit(X, y)
    preds = ensemble.predict(X)
    assert len(preds) == 100


def test_voting_ensemble_estimator_regression() -> None:
    X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

    ensemble = VotingEnsembleEstimator(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=20, random_state=42)),
            ("ridge", Ridge()),
        ],
        weights=[0.7, 0.3],
        task_type="regression",
    )
    ensemble.fit(X, y)
    assert ensemble.is_fitted_ is True

    with pytest.raises(AttributeError, match="predict_proba is not available for regression"):
        ensemble.predict_proba(X)

    preds = ensemble.predict(X)
    assert len(preds) == 100
    r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
    assert r2 > 0.5


def test_voting_ensemble_estimator_pre_fitted_models() -> None:
    X, y = make_classification(n_samples=80, n_features=4, random_state=42)

    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    lr = LogisticRegression(random_state=42).fit(X, y)

    ensemble = VotingEnsembleEstimator(
        estimators=[rf, lr],
        task_type="binary_classification",
        refit=False,
    )
    ensemble.fit(X, y)
    assert ensemble.is_fitted_ is True
    # The models in fitted_estimators_ should be the exact pre-fitted ones
    assert ensemble.fitted_estimators_[0] is rf

    probs = ensemble.predict_proba(X)
    assert probs.shape == (80, 2)


def test_voting_ensemble_plugin_metadata_and_validation() -> None:
    plugin = VotingEnsemblePlugin()

    assert plugin.plugin_id == "voting_ensemble"
    assert plugin.plugin_type == PluginType.MODEL
    assert "binary_classification" in plugin.capabilities.supported_tasks
    assert "regression" in plugin.capabilities.supported_tasks
    assert "tabular" in plugin.capabilities.supported_modalities
    assert plugin.capabilities.supports_proba is True

    validator = CompatibilityValidator()
    assert validator.is_compatible(plugin.capabilities, task_type="binary_classification") is True
    assert validator.is_compatible(plugin.capabilities, task_type="regression") is True
    assert validator.is_compatible(plugin.capabilities, task_type="clustering") is False

    space = plugin.get_search_space("binary_classification")
    assert "voting" in space.parameters


def test_voting_ensemble_plugin_build_and_pipeline() -> None:
    plugin = VotingEnsemblePlugin()

    estimator = plugin.build_estimator(
        parameters={"voting": "soft", "weights": [0.5, 0.5]},
        task_type="binary_classification",
    )
    assert isinstance(estimator, VotingEnsembleEstimator)

    X, y = make_classification(n_samples=60, n_features=4, random_state=42)
    estimator.fit(X, y)
    probs = estimator.predict_proba(X)
    assert probs.shape == (60, 2)


def test_voting_ensemble_plugin_from_models_and_blend() -> None:
    X, y = make_classification(n_samples=50, n_features=4, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    lr = LogisticRegression(random_state=42).fit(X, y)

    plugin = VotingEnsemblePlugin()
    ensemble = plugin.from_models([rf, lr], weights=[0.8, 0.2], task_type="binary_classification")
    probs = ensemble.predict_proba(X)
    assert probs.shape == (50, 2)

    # Test plugin blend helper
    blended = plugin.blend([probs, probs], task_type="binary_classification")
    np.testing.assert_allclose(blended, probs)


def test_voting_ensemble_multiclass() -> None:
    X, y = make_classification(
        n_samples=90,
        n_features=6,
        n_informative=4,
        n_classes=3,
        random_state=42,
    )
    ensemble = VotingEnsembleEstimator(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
            ("lr", LogisticRegression(max_iter=500, random_state=42)),
        ],
        task_type="multiclass_classification",
        voting="soft",
    )
    ensemble.fit(X, y)
    assert len(ensemble.classes_) == 3
    probs = ensemble.predict_proba(X)
    assert probs.shape == (90, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    preds = ensemble.predict(X)
    assert len(preds) == 90
    assert set(np.unique(preds)).issubset(set(ensemble.classes_))


def test_voting_ensemble_workspace_and_trainer_integration(tmp_path) -> None:
    import pandas as pd
    from automl.application.bootstrap import build_application
    from automl.engine.training.sklearn_trainer import SklearnTrainer

    workspace, cmd_bus, q_bus = build_application(root_dir=str(tmp_path / "app"))
    assert workspace.plugin_registry.has("voting_ensemble") is True

    model_p = workspace.plugin_registry.get_model_plugin("voting_ensemble")
    assert model_p.plugin_id == "voting_ensemble"

    # Test with SklearnTrainer.fit_and_predict
    np.random.seed(42)
    X_train = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0] * 5, "f2": [0.5, 1.5, 2.5, 3.5] * 5})
    y_train = pd.Series([0, 1, 0, 1] * 5)
    X_test = pd.DataFrame({"f1": [2.0, 3.0], "f2": [1.0, 2.0]})

    trainer = SklearnTrainer(plugin_registry=workspace.plugin_registry)
    preds = trainer.fit_and_predict(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        model_id="voting_ensemble",
        task_type="binary_classification",
        predict_proba=True,
    )
    assert len(preds) == 2

