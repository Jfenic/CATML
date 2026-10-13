"""HPO fold telemetry: MedianPruner receives per-fold scores and can prune."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    OptimizeExperimentCommand,
)
from automl.engine.training.sklearn_trainer import _cross_val_scores_with_callback
from automl.facade import AutoML
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def test_cross_val_callback_can_prune_mid_folds():
    X, y = make_classification(n_samples=80, n_features=6, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y = pd.Series(y)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=200))])
    seen: list[int] = []

    def cb(step: int, score: float) -> bool:
        seen.append(step)
        return step >= 1  # prune after second fold

    scores, pruned_at = _cross_val_scores_with_callback(
        pipe,
        X,
        y,
        cv_splitter=KFold(n_splits=4, shuffle=True, random_state=0),
        groups=None,
        scoring="accuracy",
        metric_name="accuracy",
        plugin_registry=None,
        fold_callback=cb,
    )
    assert pruned_at == 1
    assert len(scores) == 2
    assert seen == [0, 1]


def test_optimize_experiment_wires_fold_callback(tmp_path: Path, sample_csv: str):
    """Smoke: Optuna HPO with CV completes and may prune without crashing."""
    ws, cmd, _ = build_application(root_dir=str(tmp_path / "ws"))
    ds = ws.register_dataset(name="hpo", path=sample_csv, target="churn")
    run = ws.create_run(ds, metric="roc_auc", validation_strategy="kfold", cv_folds=3)
    features = ws.get_feature_registry(ds.id).active_feature_names("churn")
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="hpo_fold",
            feature_names=features,
            model_ids=["logistic_regression"],
            validation_strategy="kfold",
        )
    )
    result = cmd.dispatch(
        OptimizeExperimentCommand(
            run_id=run.id,
            experiment_id=exp.id,
            model_id="logistic_regression",
            optimizer="optuna",
            n_trials=6,
            timeout_seconds=45,
            patience=10,
        )
    )
    assert result["trials_executed"] >= 1
    assert "best_score" in result


def test_fit_with_time_budget_attempts_hpo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When time_budget is set, AutoML.fit dispatches OptimizeExperimentCommand."""
    calls: list[object] = []
    import automl.facade as facade_mod

    real_build = facade_mod.build_application

    def wrapped_build(*args, **kwargs):
        ws, cmd, qry = real_build(*args, **kwargs)
        real_dispatch = cmd.dispatch

        def tracking_dispatch(command):
            calls.append(command)
            if isinstance(command, OptimizeExperimentCommand):
                # Avoid long HPO in unit test; assert wiring only.
                return {
                    "experiment_id": command.experiment_id,
                    "model_id": command.model_id,
                    "optimizer": command.optimizer,
                    "trials_executed": 0,
                    "successful_trials": 0,
                    "best_score": None,
                    "best_params": {},
                    "best_trial_id": "",
                    "param_importance": [],
                    "trials": [],
                }
            return real_dispatch(command)

        cmd.dispatch = tracking_dispatch  # type: ignore[method-assign]
        return ws, cmd, qry

    monkeypatch.setattr(facade_mod, "build_application", wrapped_build)

    df = pd.DataFrame(
        {
            "salary": [30000 + i * 500 for i in range(60)],
            "debt": [5000 + (i % 7) * 1000 for i in range(60)],
            "tenure": [i % 10 for i in range(60)],
            "churn": [1 if i % 4 == 0 else 0 for i in range(60)],
        }
    )
    automl = AutoML(
        models=["logistic_regression"],
        time_budget=120,
        workspace_dir=tmp_path / "fit_ws",
        cv_folds=3,
    )
    result = automl.fit(df, target="churn")
    assert result.best_model_id
    opt_cmds = [c for c in calls if isinstance(c, OptimizeExperimentCommand)]
    assert len(opt_cmds) == 1
    assert opt_cmds[0].optimizer == "optuna"
    assert opt_cmds[0].timeout_seconds is not None
    assert opt_cmds[0].timeout_seconds >= 20


@pytest.fixture
def sample_csv(tmp_path: Path) -> str:
    csv_path = tmp_path / "sample_churn.csv"
    df = pd.DataFrame(
        {
            "salary": [30000 + i * 500 for i in range(80)],
            "debt": [5000 + (i % 7) * 1000 for i in range(80)],
            "tenure": [i % 10 for i in range(80)],
            "churn": [1 if i % 4 == 0 else 0 for i in range(80)],
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)
