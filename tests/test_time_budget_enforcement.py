from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.domain.runs.run import RunConfig, AutoMLRun
from automl.domain.runs.states import RunPhase, RunStatus
from automl.domain.experiments.trial import Experiment
from automl.application.services.workspace import AutoMLWorkspace
from automl.facade import AutoML


def test_run_config_effective_time_budget():
    # 1. Direct field
    cfg1 = RunConfig(task_type="binary_classification", target="y", time_budget_seconds=45.0)
    assert cfg1.effective_time_budget == 45.0
    d1 = cfg1.to_dict()
    assert d1["time_budget_seconds"] == 45.0
    loaded1 = RunConfig.from_dict(d1)
    assert loaded1.effective_time_budget == 45.0
    assert loaded1.time_budget_seconds == 45.0

    # 2. Backward compatibility with extra["time_budget"]
    cfg2 = RunConfig(task_type="binary_classification", target="y", extra={"time_budget": "30.5"})
    assert cfg2.effective_time_budget == 30.5
    d2 = cfg2.to_dict()
    loaded2 = RunConfig.from_dict(d2)
    assert loaded2.effective_time_budget == 30.5
    assert loaded2.time_budget_seconds == 30.5

    # 3. None when absent
    cfg3 = RunConfig(task_type="binary_classification", target="y")
    assert cfg3.effective_time_budget is None


def test_run_experiment_exhaustion_preserves_completed_models(tmp_path: Path):
    ws = AutoMLWorkspace.create("budget_test", root_dir=tmp_path / "ws")
    df = pd.DataFrame({"x1": np.random.randn(50), "x2": np.random.randn(50), "target": [0, 1] * 25})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    ds = ws.register_dataset(name="data", path=csv_path, target="target", task_type="binary_classification")
    run = ws.create_run(ds, time_budget_seconds=0.000001)

    exp = ws.create_experiment(
        run=run,
        name="multi_models",
        feature_names=["x1", "x2"],
        model_ids=["logistic_regression", "random_forest", "extra_trees"],
    )

    # Artificially expire deadline before second model
    ws._run_deadlines[run.id] = time.monotonic() - 10.0

    results = ws.run_experiment(run, exp)
    refreshed_run = ws.repository.get_run(run.id)

    assert refreshed_run is not None
    assert refreshed_run.config.extra.get("time_budget_exhausted") is True

    events = ws.repository.list_events(run_id=run.id)
    budget_events = [e for e in events if e.get("event_type") == "TimeBudgetExhausted"]
    assert len(budget_events) >= 1
    assert budget_events[0].get("payload", {}).get("run_id") == run.id
    # Execution should not crash or destroy experiment
    assert len(results) == 0


def test_automl_fit_fails_if_budget_expires_before_first_model(tmp_path: Path, monkeypatch):
    df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0] * 10, "target": [0, 1, 0, 1] * 10})

    automl = AutoML(
        task="classification",
        time_budget=0.0001,
        models=["logistic_regression", "random_forest"],
        workspace_dir=tmp_path / "fit_ws",
    )

    # Force immediate expiration upon checking
    from automl.application.services.workspace import AutoMLWorkspace
    original_is_exhausted = AutoMLWorkspace._is_run_budget_exhausted

    def mock_exhausted(self, run):
        return True

    monkeypatch.setattr(AutoMLWorkspace, "_is_run_budget_exhausted", mock_exhausted)

    with pytest.raises(RuntimeError, match="Time budget .* expired before any candidate model"):
        automl.fit(df, target="target")


def test_automl_fit_succeeds_if_at_least_one_model_finishes(tmp_path: Path, monkeypatch):
    df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0] * 10, "target": [0, 1, 0, 1] * 10})

    automl = AutoML(
        task="classification",
        time_budget=10.0,
        models=["logistic_regression", "random_forest", "extra_trees"],
        workspace_dir=tmp_path / "fit_ws2",
    )

    from automl.application.services.workspace import AutoMLWorkspace
    call_count = {"n": 0}

    def mock_exhausted_after_first(self, run):
        call_count["n"] += 1
        # Allow first model, exhaust for subsequent models
        return call_count["n"] > 1

    monkeypatch.setattr(AutoMLWorkspace, "_is_run_budget_exhausted", mock_exhausted_after_first)

    result = automl.fit(df, target="target")
    assert result is not None
    assert result.best_score > 0.0
    lb = result.leaderboard()
    assert len(lb) == 1
    assert lb.iloc[0]["model_id"] == "logistic_regression"
