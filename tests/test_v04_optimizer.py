from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    OptimizeExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    GetBestTrialQuery,
    GetExperimentTrialsQuery,
)
from automl.domain.experiments.trial import Trial
from automl.domain.optimization.budget import OptimizationBudget
from automl.domain.optimization.search_space import (
    ParameterSpec,
    ParameterType,
    SearchSpace,
)
from automl.domain.ports import TrialExecution
from automl.engine.optimization.early_stopping import EarlyStoppingPolicy
from automl.engine.optimization.random_search import RandomSearchOptimizer
from automl.engine.optimization.search_space_builder import SearchSpaceBuilder
from automl.engine.optimization.trial_factory import TrialFactory
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.interfaces.cli.main import main
from automl.plugins.optimizers.optuna_optimizer import OptunaOptimizer


@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "sample_churn.csv"
    df = pd.DataFrame(
        {
            "salary": [30000 + i * 500 for i in range(100)],
            "debt": [5000 + (i % 7) * 1000 for i in range(100)],
            "tenure": [i % 10 for i in range(100)],
            "churn": [1 if i % 4 == 0 else 0 for i in range(100)],
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_parameter_spec_and_search_space():
    p_float = ParameterSpec.float("learning_rate", 0.001, 0.1, log=True, default=0.01)
    p_int = ParameterSpec.int("n_estimators", 10, 100, step=10, default=50)
    p_cat = ParameterSpec.categorical("penalty", ["l1", "l2"], default="l2")

    assert p_float.param_type == ParameterType.FLOAT
    assert p_int.param_type == ParameterType.INT
    assert p_cat.param_type == ParameterType.CATEGORICAL

    space = SearchSpace()
    space.add(p_float).add(p_int).add(p_cat)

    assert len(space) == 3
    assert "learning_rate" in space
    assert space["n_estimators"].default == 50

    sampled = space.sample(seed=42)
    assert 0.001 <= sampled["learning_rate"] <= 0.1
    assert 10 <= sampled["n_estimators"] <= 100
    assert sampled["penalty"] in ["l1", "l2"]

    d = space.to_dict()
    assert "learning_rate" in d
    assert d["learning_rate"]["type"] == "float"


def test_search_space_builder():
    lr_space = SearchSpaceBuilder.build("logistic_regression", "binary_classification")
    assert "C" in lr_space
    assert "solver" in lr_space

    rf_space = SearchSpaceBuilder.build("random_forest", "binary_classification")
    assert "n_estimators" in rf_space
    assert "max_depth" in rf_space

    ridge_space = SearchSpaceBuilder.build("ridge", "regression")
    assert "alpha" in ridge_space

    svc_space = SearchSpaceBuilder.build("svc", "binary_classification")
    assert "C" in svc_space
    assert "kernel" in svc_space

    kmeans_space = SearchSpaceBuilder.build("kmeans", "clustering")
    assert "n_clusters" in kmeans_space

    with pytest.raises(ValueError, match="No default search space"):
        SearchSpaceBuilder.build("unknown_model_type", "binary_classification")


def test_random_search_optimizer():
    space = SearchSpace()
    space.add(ParameterSpec.float("c", 0.1, 10.0))
    space.add(ParameterSpec.int("max_iter", 50, 200))

    opt = RandomSearchOptimizer(seed=42, patience=3, min_delta=0.01, mode="max")

    p0 = opt.suggest(0, space)
    assert 0.1 <= p0["c"] <= 10.0
    assert 50 <= p0["max_iter"] <= 200

    opt.observe(0, p0, score=0.75, succeeded=True)
    assert opt.best_score() == 0.75
    assert opt.best_parameters() == p0

    p1 = opt.suggest(1, space)
    opt.observe(1, p1, score=0.85, succeeded=True)
    assert opt.best_score() == 0.85

    # Early stopping test: 3 consecutive non-improving trials
    for trial_idx in range(2, 5):
        p = opt.suggest(trial_idx, space)
        opt.observe(trial_idx, p, score=0.80, succeeded=True)

    assert opt.should_stop() is True


def test_early_stopping_policy_minimize():
    policy = EarlyStoppingPolicy(patience=2, min_delta=0.05, mode="min")
    assert policy.step(1.0) is False
    assert policy.step(0.8) is False  # improved
    assert policy.step(0.79) is False  # improvement < min_delta
    assert policy.step(0.85) is True  # patience reached (2 bad steps)


def test_optuna_optimizer():
    space = SearchSpace()
    space.add(ParameterSpec.float("lr", 0.01, 1.0, log=True))
    space.add(ParameterSpec.int("depth", 2, 8))
    space.add(ParameterSpec.categorical("criterion", ["gini", "entropy"]))

    opt = OptunaOptimizer(seed=123, direction="maximize", patience=4)

    for i in range(5):
        params = opt.suggest(i, space)
        assert 0.01 <= params["lr"] <= 1.0
        assert 2 <= params["depth"] <= 8
        assert params["criterion"] in ["gini", "entropy"]

        score = 0.5 + 0.1 * i
        opt.observe(i, params, score, succeeded=True)

    assert opt.best_score() == pytest.approx(0.9)
    assert opt.best_parameters() is not None


def test_trial_factory_and_sqlite_persistence(tmp_path):
    ws, _, _ = build_application(root_dir=str(tmp_path / "ws_trials"))
    trial = TrialFactory.create(
        experiment_id="exp_123",
        model_id="logistic_regression",
        parameters={"C": 0.5, "solver": "lbfgs"},
        seed=42,
    )
    assert trial.parameters == {"C": 0.5, "solver": "lbfgs"}

    ws.repository.save_trial(trial)
    retrieved = ws.repository.get_trial(trial.id)
    assert retrieved is not None
    assert retrieved.parameters == {"C": 0.5, "solver": "lbfgs"}
    assert retrieved.model_id == "logistic_regression"


def test_cqrs_optimize_experiment_optuna(tmp_path, sample_csv):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws_opt"))
    dataset = ws.register_dataset("churn_opt", sample_csv, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="opt_lr_exp",
            feature_names=["salary", "debt", "tenure"],
            model_ids=["logistic_regression"],
        )
    )

    result = cmd.dispatch(
        OptimizeExperimentCommand(
            run_id=run.id,
            experiment_id=exp.id,
            model_id="logistic_regression",
            optimizer="optuna",
            n_trials=4,
            patience=3,
        )
    )

    assert result["experiment_id"] == exp.id
    assert result["model_id"] == "logistic_regression"
    assert result["optimizer"] == "optuna"
    assert result["trials_executed"] == 4
    assert result["best_score"] > 0.0
    assert "C" in result["best_params"]

    # Check queries
    best_trial = qry.dispatch(GetBestTrialQuery(exp.id))
    assert best_trial is not None
    assert best_trial["experiment_id"] == exp.id
    assert best_trial["best_score"] == result["best_score"]
    assert "C" in best_trial["parameters"]

    trials = qry.dispatch(GetExperimentTrialsQuery(exp.id))
    assert len(trials) == 4
    assert all(t["succeeded"] for t in trials)


def test_cqrs_optimize_experiment_random_search(tmp_path, sample_csv):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws_rs"))
    dataset = ws.register_dataset("churn_rs", sample_csv, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="opt_rf_exp",
            feature_names=["salary", "debt", "tenure"],
            model_ids=["random_forest"],
        )
    )

    result = cmd.dispatch(
        OptimizeExperimentCommand(
            run_id=run.id,
            experiment_id=exp.id,
            model_id="random_forest",
            optimizer="random_search",
            n_trials=3,
        )
    )

    assert result["optimizer"] == "random_search"
    assert result["trials_executed"] == 3
    assert result["best_score"] > 0.0
    assert "n_estimators" in result["best_params"]


def test_cli_optimize_command(tmp_path, sample_csv):
    ws_dir = str(tmp_path / "ws_cli")
    ret = main(
        [
            "optimize",
            "--dataset",
            sample_csv,
            "--workspace",
            ws_dir,
            "--model",
            "logistic_regression",
            "--optimizer",
            "optuna",
            "--trials",
            "2",
        ]
    )
    assert ret == 0


def test_parameter_spec_validation_edge_cases():
    p_int = ParameterSpec.int("n", low=5, high=15)
    assert p_int.validate(10) is True
    assert p_int.validate(4) is False
    assert p_int.validate(16) is False
    assert p_int.validate(True) is False
    assert p_int.validate("10") is False

    p_float = ParameterSpec.float("x", low=0.1, high=1.0)
    assert p_float.validate(0.5) is True
    assert p_float.validate(0.05) is False
    assert p_float.validate(1.5) is False
    assert p_float.validate(False) is False
    assert p_float.validate("0.5") is False

    p_cat = ParameterSpec.categorical("cat", choices=["a", "b"])
    assert p_cat.validate("a") is True
    assert p_cat.validate("c") is False

    space = SearchSpace()
    space.add(p_int).add(p_float).add(p_cat)
    res = space.validate_params({"n": 10, "x": 2.0, "cat": "b", "unknown": 123})
    assert res == {"n": True, "x": False, "cat": True}

    assert [p.name for p in space] == ["n", "x", "cat"]
    assert len(space.list()) == 3


def test_optuna_optimizer_minimize_and_failure():
    space = SearchSpace()
    space.add(ParameterSpec.float("alpha", 0.01, 10.0))

    opt = OptunaOptimizer(seed=42, direction="minimize", patience=2)
    p0 = opt.suggest(0, space)
    opt.observe(0, p0, score=0.5, succeeded=False)  # trial failed

    p1 = opt.suggest(1, space)
    opt.observe(1, p1, score=1.0, succeeded=True)
    assert opt.best_score() == 1.0

    p2 = opt.suggest(2, space)
    opt.observe(2, p2, score=0.8, succeeded=True)
    assert opt.best_score() == 0.8  # lower is better in minimize


def test_workspace_optimize_errors_and_edge_cases(tmp_path, sample_csv):
    ws, cmd, qry = build_application(root_dir=str(tmp_path / "ws_err"))
    dataset = ws.register_dataset("churn_err", sample_csv, target="churn")
    run = ws.create_run(dataset, metric="roc_auc")

    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="err_exp",
            feature_names=["salary", "debt"],
            model_ids=["logistic_regression"],
        )
    )

    # Unknown experiment
    with pytest.raises(KeyError, match="Experiment not found"):
        ws.optimize_experiment(run.id, "non_existent_exp")

    # Model not in experiment
    with pytest.raises(ValueError, match="not part of experiment"):
        ws.optimize_experiment(run.id, exp.id, model_id="random_forest")

    # Run is paused
    ws.pause_run(run.id)
    with pytest.raises(RuntimeError, match="PAUSED"):
        ws.optimize_experiment(run.id, exp.id)

    # Best trial when no trials exist
    assert ws.get_best_trial("empty_exp") is None


def test_optimization_budget():
    b = OptimizationBudget(max_trials=20, timeout_seconds=60.0, patience=3, min_delta=0.001)
    assert b.max_trials == 20
    assert b.timeout_seconds == 60.0
    d = b.to_dict()
    assert d["patience"] == 3

