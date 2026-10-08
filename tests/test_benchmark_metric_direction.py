import pytest
from automl.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkScenario,
    _best_result,
    is_minimizing_metric,
)
from automl.domain.experiments.trial import TrialResult


def test_is_minimizing_metric_known_metrics():
    # Minimizing loss metrics
    for m in ["mae", "rmse", "mse", "loss", "log_loss", "MAE", "RMSE", "MSE", "Log_Loss"]:
        assert is_minimizing_metric(m) is True

    # Maximizing score metrics
    for m in ["roc_auc", "accuracy", "f1", "r2", "balanced_accuracy", "precision", "recall"]:
        assert is_minimizing_metric(m) is False


def test_best_result_maximizing_metric():
    results = [
        TrialResult(
            trial_id="t1",
            experiment_id="exp1",
            model_id="model_a",
            primary_metric="roc_auc",
            primary_score=0.82,
        ),
        TrialResult(
            trial_id="t2",
            experiment_id="exp1",
            model_id="model_b",
            primary_metric="roc_auc",
            primary_score=0.91,
        ),
        TrialResult(
            trial_id="t3",
            experiment_id="exp1",
            model_id="model_c",
            primary_metric="roc_auc",
            primary_score=0.75,
        ),
    ]
    best_score, best_model = _best_result(results, metric="roc_auc")
    assert best_score == 0.91
    assert best_model == "model_b"


def test_best_result_minimizing_metric():
    results = [
        TrialResult(
            trial_id="t1",
            experiment_id="exp1",
            model_id="model_a",
            primary_metric="rmse",
            primary_score=1.45,
        ),
        TrialResult(
            trial_id="t2",
            experiment_id="exp1",
            model_id="model_b",
            primary_metric="rmse",
            primary_score=0.38,
        ),
        TrialResult(
            trial_id="t3",
            experiment_id="exp1",
            model_id="model_c",
            primary_metric="rmse",
            primary_score=0.89,
        ),
    ]
    # For minimizing metrics like rmse, lowest score wins
    best_score, best_model = _best_result(results, metric="rmse")
    assert best_score == 0.38
    assert best_model == "model_b"


def test_best_result_ignores_failed_trials():
    results = [
        TrialResult(
            trial_id="t1",
            experiment_id="exp1",
            model_id="failed_model",
            primary_metric="roc_auc",
            primary_score=0.99,
            failure_reason="OOM Error",
        ),
        TrialResult(
            trial_id="t2",
            experiment_id="exp1",
            model_id="valid_model",
            primary_metric="roc_auc",
            primary_score=0.85,
        ),
    ]
    best_score, best_model = _best_result(results, metric="roc_auc")
    assert best_score == 0.85
    assert best_model == "valid_model"


def test_best_result_all_failed_returns_defaults():
    results = [
        TrialResult(
            trial_id="t1",
            experiment_id="exp1",
            model_id="failed_model",
            primary_metric="roc_auc",
            primary_score=0.0,
            failure_reason="Timeout",
        ),
    ]
    best_score, best_model = _best_result(results, metric="roc_auc")
    assert best_score == 0.0
    assert best_model is None
