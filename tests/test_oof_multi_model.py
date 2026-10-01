"""Unit and integration tests for generalized N-model OOF blending and Level-2 stacking."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    GenerateOOFSubmissionCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import GetOOFResultQuery
from automl.engine.ensemble.blender import (
    blend_predictions,
    optimize_ensemble_weights,
    rank_average_predictions,
    stack_predictions,
)
from automl.engine.ensemble.oof import evaluate_oof
from automl.interfaces.cli.main import main


# ---------------------------------------------------------------------------
# Unit tests for stack_predictions in blender.py
# ---------------------------------------------------------------------------


def test_stack_predictions_validation_errors():
    """Verify input validation and error reporting in stack_predictions."""
    with pytest.raises(ValueError, match="Cannot stack empty predictions"):
        stack_predictions([], [0, 1])

    with pytest.raises(ValueError, match="does not match OOF rows"):
        stack_predictions([[0.1, 0.9]], [0, 1, 0])

    with pytest.raises(ValueError, match="test_predictions count"):
        stack_predictions([[0.1, 0.9], [0.2, 0.8]], [0, 1], test_predictions=[[0.3]])

    with pytest.raises(ValueError, match="unsupported ndim"):
        stack_predictions([np.ones((2, 2, 2))], [0, 1])

    with pytest.raises(ValueError, match="unsupported ndim"):
        stack_predictions([[0.1, 0.9]], [0, 1], test_predictions=[np.ones((2, 2, 2))])

    with pytest.raises(ValueError, match="Unsupported stacking meta_model"):
        stack_predictions([[0.1, 0.9]], [0, 1], meta_model="unknown_meta")


def test_stack_predictions_binary_classification_ridge_and_logistic():
    """Verify Level-2 stacking for binary classification with Ridge and Logistic Regression."""
    rng = np.random.default_rng(42)
    n = 100
    y = rng.integers(0, 2, size=n)

    # 3 model predictions with varying correlation to target
    p1 = np.clip(y * 0.7 + rng.normal(0, 0.2, size=n), 0.0, 1.0)
    p2 = np.clip(y * 0.5 + rng.normal(0, 0.3, size=n), 0.0, 1.0)
    p3 = np.clip(rng.uniform(0, 1, size=n), 0.0, 1.0)

    t1 = np.array([0.1, 0.9])
    t2 = np.array([0.2, 0.8])
    t3 = np.array([0.5, 0.5])

    # 1. Ridge meta-model
    stacked_oof, stacked_test, meta = stack_predictions(
        [p1, p2, p3], y, test_predictions=[t1, t2, t3], meta_model="ridge"
    )
    assert stacked_oof.shape == (n,)
    assert stacked_test.shape == (2,)
    assert np.all((stacked_oof >= 0.0) & (stacked_oof <= 1.0))
    assert np.all((stacked_test >= 0.0) & (stacked_test <= 1.0))
    assert hasattr(meta, "coef_")
    # p1 should have higher weight than random p3
    assert meta.coef_[0] > meta.coef_[2]

    # 2. Logistic regression meta-model
    stacked_oof_lr, stacked_test_lr, meta_lr = stack_predictions(
        [p1, p2, p3], y, test_predictions=[t1, t2, t3], meta_model="logistic_regression"
    )
    assert stacked_oof_lr.shape == (n,)
    assert stacked_test_lr.shape == (2,)
    assert np.all((stacked_oof_lr >= 0.0) & (stacked_oof_lr <= 1.0))
    assert np.all((stacked_test_lr >= 0.0) & (stacked_test_lr <= 1.0))

    # 3. Lasso meta-model
    stacked_oof_l, stacked_test_l, meta_l = stack_predictions(
        [p1, p2, p3], y, test_predictions=[t1, t2, t3], meta_model="lasso", alpha=0.01
    )
    assert stacked_oof_l.shape == (n,)
    assert stacked_test_l.shape == (2,)


def test_stack_predictions_regression_task():
    """Verify Level-2 stacking for continuous regression targets."""
    rng = np.random.default_rng(42)
    n = 80
    y = rng.normal(10.0, 2.0, size=n)

    p1 = y + rng.normal(0, 0.5, size=n)
    p2 = y + rng.normal(0, 1.0, size=n)
    t1 = np.array([9.5, 11.0])
    t2 = np.array([10.0, 10.5])

    stacked_oof, stacked_test, meta = stack_predictions(
        [p1, p2], y, test_predictions=[t1, t2], task_type="regression", meta_model="ridge"
    )
    assert stacked_oof.shape == (n,)
    assert stacked_test.shape == (2,)
    assert np.isclose(stacked_test.mean(), 10.0, atol=1.5)


def test_stack_predictions_with_2d_and_string_targets():
    """Verify stacking with 2D probability matrices and string target labels."""
    y = np.array(["negative", "positive"] * 25)
    p1_2d = np.column_stack([np.linspace(0.8, 0.2, 50), np.linspace(0.2, 0.8, 50)])
    p2_1d = np.linspace(0.3, 0.7, 50)

    stacked_oof, _, _ = stack_predictions(
        [p1_2d, p2_1d], y, task_type="binary_classification", meta_model="ridge"
    )
    assert stacked_oof.shape == (50,)
    assert np.all((stacked_oof >= 0.0) & (stacked_oof <= 1.0))


# ---------------------------------------------------------------------------
# Multi-Model evaluate_oof Tests (N >= 3)
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_model_data():
    rng = np.random.default_rng(123)
    n = 60
    y = pd.Series(["no", "yes"] * (n // 2))
    X = pd.DataFrame({
        "feat_a": rng.normal(size=n) + (y == "yes") * 1.5,
        "feat_b": rng.uniform(-1, 1, size=n) + (y == "yes") * 0.8,
        "feat_c": rng.standard_t(df=3, size=n),
    })
    X_test = pd.DataFrame({
        "feat_a": rng.normal(size=10),
        "feat_b": rng.uniform(-1, 1, size=10),
        "feat_c": rng.standard_t(df=3, size=10),
    })
    return X, y, X_test


def four_model_factories():
    return {
        "log_reg": lambda: LogisticRegression(max_iter=200),
        "rand_forest": lambda: RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=3, random_state=42),
        "naive_bayes": lambda: GaussianNB(),
    }


def test_oof_with_four_models_average(multi_model_data):
    """Test standard equal-weight averaging with 4 distinct models."""
    X, y, X_test = multi_model_data
    factories = four_model_factories()

    result = evaluate_oof(X, y, X_test, factories, folds=3, method="average")
    assert len(result.oof) == 4
    assert len(result.test) == 4
    assert set(result.oof.keys()) == set(factories.keys())
    assert result.method == "average"
    assert result.weights is not None
    assert len(result.weights) == 4
    assert pytest.approx(sum(result.weights.values())) == 1.0
    for w in result.weights.values():
        assert pytest.approx(w) == 0.25

    # Check that blend matches manual average of 4 models
    expected_oof = (
        result.oof["log_reg"]
        + result.oof["rand_forest"]
        + result.oof["decision_tree"]
        + result.oof["naive_bayes"]
    ) / 4.0
    np.testing.assert_allclose(result.blended_oof, expected_oof)
    assert result.scores["oof_blend"] > 0.5


def test_oof_with_four_models_rank_blending(multi_model_data):
    """Test rank-averaging blending with 4 distinct models."""
    X, y, X_test = multi_model_data
    factories = four_model_factories()

    result = evaluate_oof(X, y, X_test, factories, folds=3, method="rank")
    assert result.method == "rank"
    assert len(result.oof) == 4
    assert np.all((result.blended_oof >= 0.0) & (result.blended_oof <= 1.0))
    assert np.all((result.blended_test >= 0.0) & (result.blended_test <= 1.0))
    assert result.scores["oof_blend"] > 0.5
    assert len(result.fold_scores["oof_blend"]) == 3


def test_oof_with_four_models_simplex_optimization(multi_model_data):
    """Test Nelder-Mead simplex weight optimization with 4 distinct models."""
    X, y, X_test = multi_model_data
    factories = four_model_factories()

    result = evaluate_oof(X, y, X_test, factories, folds=3, method="simplex")
    assert result.method == "simplex"
    assert result.weights is not None
    assert len(result.weights) == 4
    assert pytest.approx(sum(result.weights.values()), rel=1e-3) == 1.0
    assert all(w >= 0.0 for w in result.weights.values())
    assert result.scores["oof_blend"] > 0.5


def test_oof_with_four_models_level2_stacking(multi_model_data):
    """Test Level-2 Stacking meta-estimator with 4 distinct models."""
    X, y, X_test = multi_model_data
    factories = four_model_factories()

    # Ridge stacking
    result_ridge = evaluate_oof(X, y, X_test, factories, folds=3, method="stacked", meta_model="ridge")
    assert result_ridge.method == "stacked"
    assert np.all((result_ridge.blended_oof >= 0.0) & (result_ridge.blended_oof <= 1.0))
    assert np.all((result_ridge.blended_test >= 0.0) & (result_ridge.blended_test <= 1.0))
    assert result_ridge.scores["oof_blend"] > 0.5
    assert result_ridge.weights is not None
    assert len(result_ridge.weights) == 4

    # Logistic Regression stacking
    result_lr = evaluate_oof(X, y, X_test, factories, folds=3, method="stacked", meta_model="logistic_regression")
    assert result_lr.method == "stacked"
    assert np.all((result_lr.blended_oof >= 0.0) & (result_lr.blended_oof <= 1.0))
    assert result_lr.scores["oof_blend"] > 0.5


def test_oof_single_model_support(multi_model_data):
    """Test that N=1 model works correctly without raising error."""
    X, y, X_test = multi_model_data
    factories = {"single_lr": lambda: LogisticRegression(max_iter=200)}

    result = evaluate_oof(X, y, X_test, factories, folds=3, method="average")
    assert len(result.oof) == 1
    assert result.scores["oof_blend"] == result.scores["single_lr"]
    np.testing.assert_allclose(result.blended_oof, result.oof["single_lr"])


def test_oof_invalid_method_rejected(multi_model_data):
    """Test that unsupported blending methods are rejected with clear message."""
    X, y, X_test = multi_model_data
    factories = four_model_factories()

    with pytest.raises(ValueError, match="Unknown OOF blending method"):
        evaluate_oof(X, y, X_test, factories, folds=3, method="quantum_blend")


# ---------------------------------------------------------------------------
# CQRS and Application Layer Integration Tests (3+ Models)
# ---------------------------------------------------------------------------


@pytest.fixture
def app_with_multi_models(tmp_path, multi_model_data):
    X, y, X_test = multi_model_data
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    template_path = tmp_path / "template.csv"

    pd.DataFrame({"id": range(len(X)), **X, "churn": y}).to_csv(train_path, index=False)
    pd.DataFrame({"id": range(100, 100 + len(X_test)), **X_test}).to_csv(test_path, index=False)
    pd.DataFrame({"id": range(100, 100 + len(X_test)), "churn": 0.5}).to_csv(template_path, index=False)

    ws, cmd, qry = build_application(root_dir=str(tmp_path / "workspace"))
    dataset = ws.register_dataset(name="churn_ds", path=train_path, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")

    # Source experiment with 3 models: logistic_regression, random_forest, decision_tree
    source = cmd.dispatch(CreateExperimentCommand(
        run_id=run.id,
        name="source_multi",
        feature_names=list(X.columns),
        model_ids=["logistic_regression", "random_forest", "extra_trees"],
    ))
    cmd.dispatch(RunExperimentCommand(run.id, source.id))

    return ws, cmd, qry, run, source, test_path, template_path


def test_cqrs_oof_three_models_rank_blending(app_with_multi_models, tmp_path):
    """Test full CQRS pipeline with 3 models and rank blending."""
    ws, cmd, qry, run, source, test_path, template_path = app_with_multi_models

    out_csv = tmp_path / "submission_rank.csv"
    exp_id = cmd.dispatch(GenerateOOFSubmissionCommand(
        run_id=run.id,
        experiment_id=source.id,
        test_dataset_path=str(test_path),
        output_path=str(out_csv),
        template_path=str(template_path),
        folds=3,
        model_ids=["logistic_regression", "random_forest", "extra_trees"],
        method="rank",
    ))

    report = qry.dispatch(GetOOFResultQuery(run_id=run.id, experiment_id=exp_id))
    assert report["models"] == ["logistic_regression", "random_forest", "extra_trees"]
    assert report["method"] == "rank"
    assert len(report["model_scores"]) == 4  # 3 models + oof_blend
    assert report["score"] > 0.5
    assert Path(report["output_path"]).exists()

    sub_df = pd.read_csv(out_csv)
    assert len(sub_df) == 10
    assert "churn" in sub_df.columns
    assert sub_df["churn"].between(0.0, 1.0).all()


def test_cqrs_oof_three_models_stacking(app_with_multi_models, tmp_path):
    """Test full CQRS pipeline with 3 models and Level-2 Stacking."""
    ws, cmd, qry, run, source, test_path, template_path = app_with_multi_models

    out_csv = tmp_path / "submission_stacked.csv"
    exp_id = cmd.dispatch(GenerateOOFSubmissionCommand(
        run_id=run.id,
        experiment_id=source.id,
        test_dataset_path=str(test_path),
        output_path=str(out_csv),
        folds=3,
        model_ids=["logistic_regression", "random_forest", "extra_trees"],
        method="stacked",
        meta_model="ridge",
    ))

    report = qry.dispatch(GetOOFResultQuery(run_id=run.id, experiment_id=exp_id))
    assert report["method"] == "stacked"
    assert report["models"] == ["logistic_regression", "random_forest", "extra_trees"]
    assert "weights_by_model" in report
    assert report["score"] > 0.5


# ---------------------------------------------------------------------------
# CLI Integration Tests
# ---------------------------------------------------------------------------


def test_cli_oof_multi_model_with_method_flag(app_with_multi_models, tmp_path, capsys):
    """Test CLI automl predict with 3 models and --method simplex."""
    ws, _, _, run, source, test_path, template_path = app_with_multi_models
    out_csv = tmp_path / "cli_oof.csv"

    exit_code = main([
        "predict",
        "--workspace", str(ws.root_dir),
        "--run-id", run.id,
        "--test-dataset", str(test_path),
        "--template", str(template_path),
        "--folds", "3",
        "--models", "logistic_regression,random_forest,extra_trees",
        "--method", "simplex",
        "--proba",
        "--output", str(out_csv),
        "--json",
    ])
    assert exit_code == 0
    stdout = capsys.readouterr().out
    result = json.loads(stdout)
    assert result["folds"] == 3
    assert result["models"] == ["logistic_regression", "random_forest", "extra_trees"]
    assert result["method"] == "simplex"
    assert Path(out_csv).exists()

