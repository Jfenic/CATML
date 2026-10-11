"""Tests for PR 2 (Fase P0): Métricas, puntuaciones y comparabilidad en CATML Workbench.

Verifies:
1. Metric optimization direction (min vs max, RMSE/MAE vs ROC-AUC).
2. Selection of best trial in backend APIs (/api/experiments, /api/runs, /api/overview).
3. Incompatible comparison blocking and honest labeling in Workbench UI.
"""
from __future__ import annotations

import json
from pathlib import Path
from http import HTTPStatus
import pytest

from automl.domain.experiments.trial import TrialResult
from automl.interfaces.web.server import (
    _is_metric_minimizing,
    _get_best_trial_result,
)


def test_metric_minimizing_detection() -> None:
    """Verifies that error/loss metrics minimize and score metrics maximize."""
    # Minimizing metrics
    for m in ["rmse", "RMSE", "mse", "mae", "log_loss", "mean_squared_error", "mean_absolute_error", "neg_mean_squared_error"]:
        assert _is_metric_minimizing(m) is True, f"Expected {m} to be minimizing"

    # Maximizing metrics
    for m in ["roc_auc", "ROC-AUC", "accuracy", "f1", "r2", "balanced_accuracy", "precision", "recall"]:
        assert _is_metric_minimizing(m) is False, f"Expected {m} to be maximizing"

    # None and empty
    assert _is_metric_minimizing(None) is False
    assert _is_metric_minimizing("") is False


def test_get_best_trial_result_minimizing_vs_maximizing() -> None:
    """Verifies that _get_best_trial_result selects lowest for RMSE and highest for ROC-AUC."""
    trial_high = TrialResult(
        trial_id="t_high",
        experiment_id="exp_1",
        model_id="random_forest",
        primary_metric="rmse",
        primary_score=0.85,
        secondary_metrics={},
        training_time_seconds=1.2,
    )
    trial_low = TrialResult(
        trial_id="t_low",
        experiment_id="exp_1",
        model_id="lightgbm",
        primary_metric="rmse",
        primary_score=0.32,
        secondary_metrics={},
        training_time_seconds=0.8,
    )

    # For RMSE, lower is better: trial_low (0.32) must be selected!
    best_rmse = _get_best_trial_result([trial_high, trial_low], metric="rmse")
    assert best_rmse is not None
    assert best_rmse.trial_id == "t_low"
    assert best_rmse.primary_score == 0.32

    # For ROC-AUC, higher is better: trial_high (0.85) must be selected!
    trial_high.primary_metric = "roc_auc"
    trial_low.primary_metric = "roc_auc"
    best_auc = _get_best_trial_result([trial_high, trial_low], metric="roc_auc")
    assert best_auc is not None
    assert best_auc.trial_id == "t_high"
    assert best_auc.primary_score == 0.85


def test_get_best_trial_filters_failed_trials() -> None:
    """Verifies that successful trials take precedence over failed ones."""
    t_failed = TrialResult(
        trial_id="t_fail",
        experiment_id="exp_1",
        model_id="bad_model",
        primary_metric="roc_auc",
        primary_score=0.99,
        secondary_metrics={},
        training_time_seconds=0.1,
        failure_reason="ConvergenceError",
    )
    t_ok = TrialResult(
        trial_id="t_ok",
        experiment_id="exp_1",
        model_id="good_model",
        primary_metric="roc_auc",
        primary_score=0.82,
        secondary_metrics={},
        training_time_seconds=1.0,
    )

    best = _get_best_trial_result([t_failed, t_ok], metric="roc_auc")
    assert best is not None
    assert best.trial_id == "t_ok"
    assert best.primary_score == 0.82


def test_server_runs_and_experiments_metric_direction(tmp_path: Path) -> None:
    """Verifies that /api/runs and /api/experiments include metric direction metadata."""
    from http.server import ThreadingHTTPServer
    import threading
    import urllib.request
    from automl.application.bootstrap import build_application
    from automl.interfaces.web.server import AutoMLWebHandler

    ws_dir = tmp_path / "server_test"
    ws, cmd_bus, qry_bus = build_application(root_dir=str(ws_dir))

    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x,y,target\n1,2,10.5\n3,4,20.0\n5,6,15.2\n")

    ds = ws.register_dataset(name="data", path=str(csv_path), target="target", task_type="regression")
    run_rmse = ws.create_run(ds, metric="rmse")
    exp_rmse = ws.create_experiment(run_rmse, name="baseline_rmse", model_ids=["ridge"])

    # Add trial results with 2 scores: 1.50 and 0.45
    t1 = TrialResult(
        trial_id="t1",
        experiment_id=exp_rmse.id,
        model_id="ridge",
        primary_metric="rmse",
        primary_score=1.50,
        secondary_metrics={},
        training_time_seconds=0.5,
    )
    t2 = TrialResult(
        trial_id="t2",
        experiment_id=exp_rmse.id,
        model_id="ridge",
        primary_metric="rmse",
        primary_score=0.45,
        secondary_metrics={},
        training_time_seconds=0.6,
    )
    ws.repository.save_trial_result(t1)
    ws.repository.save_trial_result(t2)

    AutoMLWebHandler.workspace_dir = str(ws_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # Check /api/runs
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs") as resp:
            assert resp.status == HTTPStatus.OK
            runs_data = json.loads(resp.read().decode())
            assert len(runs_data) >= 1
            r_entry = next(r for r in runs_data if r["id"] == run_rmse.id)
            assert r_entry["metric"] == "rmse"
            assert r_entry["is_minimizing"] is True
            assert r_entry["metric_direction"] == "minimize"

        # Check /api/experiments
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/experiments?run_id={run_rmse.id}") as resp:
            assert resp.status == HTTPStatus.OK
            exps_data = json.loads(resp.read().decode())
            assert len(exps_data) == 1
            e_entry = exps_data[0]
            assert e_entry["metric"] == "rmse"
            assert e_entry["is_minimizing"] is True
            assert e_entry["metric_direction"] == "minimize"
            # Crucial: best_score must be 0.45 (minimum), NOT 1.50 (maximum)!
            assert e_entry["best_score"] == 0.45

        # Check /api/experiments/compare
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/experiments/compare?ids={exp_rmse.id}") as resp:
            assert resp.status == HTTPStatus.OK
            comp_data = json.loads(resp.read().decode())
            assert isinstance(comp_data, list)
            assert comp_data[0]["best_score"] == 0.45
            assert comp_data[0]["is_minimizing"] is True
            assert comp_data[0]["metric_direction"] == "minimize"

    finally:
        server.shutdown()
        server.server_close()


def test_frontend_utils_metric_helpers_exported() -> None:
    """Verifies that utils.js exports metric direction and delta comparison functions."""
    utils_path = Path("src/automl/interfaces/web/static/js/utils.js")
    content = utils_path.read_text(encoding="utf-8")

    assert "export function isMinimizingMetric" in content
    assert "export function getMetricDirectionInfo" in content
    assert "export function formatMetricComparison" in content


def test_compare_js_incompatibility_guard_and_dynamic_chart() -> None:
    """Verifies compare.js implements metric compatibility checks and honest labels."""
    compare_path = Path("src/automl/interfaces/web/static/js/views/compare.js")
    content = compare_path.read_text(encoding="utf-8")

    # Compatibility check
    assert "isComparable" in content
    assert "Comparación cuantitativa incompatible" in content
    assert "formatMetricComparison" in content
    assert "getMetricDirectionInfo" in content

    # No hardcoded 1.0 clipping on chart Y axis
    assert "max: Math.min(1.0, maxScore + pad)" not in content
    assert "maxScore + pad" in content


def test_studio_and_overview_honest_labeling() -> None:
    """Verifies that misleading labels 'Optimal' and unverified 'Verified' were replaced."""
    studio_path = Path("src/automl/interfaces/web/static/js/views/studio.js")
    studio_content = studio_path.read_text(encoding="utf-8")

    # In leaderboard row and trials:
    assert "Mejor de esta ejecución" in studio_content
    assert "Evaluado" in studio_content
    assert "Optimal Parameter Set" not in studio_content
    assert "Mejor conjunto de parámetros observado" in studio_content

    overview_path = Path("src/automl/interfaces/web/static/js/views/overview.js")
    overview_content = overview_path.read_text(encoding="utf-8")
    assert "Evaluado por Cross-Validation" in overview_content
    assert "Verified by CV" not in overview_content
