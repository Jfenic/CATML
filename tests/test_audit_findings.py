import logging
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from automl import AutoML
from automl.application.bootstrap import build_application
from automl.domain.experiments.trial import Trial, TrialResult, TrialStatus
from automl.engine.training.sklearn_trainer import SklearnTrainer, _sklearn_scoring


def test_cv_mae_rmse_positive_sign_and_ascending_ranking(tmp_path):
    """
    Finding 1: MAE and RMSE in cross-validation must produce positive error values
    and be ranked in ascending order (lowest error = rank 1).
    """
    np.random.seed(42)
    n = 40
    df = pd.DataFrame({
        "x1": np.random.randn(n),
        "x2": np.random.randn(n) * 2,
        "y": 3.0 * np.random.randn(n) + 10.0,
    })

    # Test MAE
    automl = AutoML(
        task="regression",
        metric="mae",
        models=["ridge", "random_forest"],
        cv_folds=2,
        workspace_dir=tmp_path / "ws_mae",
    )
    result = automl.fit(df, target="y")
    assert result is not None
    lb = result.leaderboard()
    assert len(lb) > 0

    # Scores must be positive real errors (e.g. MAE > 0)
    for score in lb["score"]:
        assert score > 0.0, f"MAE score must be strictly positive, got {score}"

    # Verify ascending ranking: rank 1 has the lowest error
    scores = lb["score"].tolist()
    assert scores == sorted(scores), f"MAE leaderboard must be sorted ascendingly: {scores}"
    assert result.best_score == min(scores)

    # Test RMSE
    automl_rmse = AutoML(
        task="regression",
        metric="rmse",
        models=["ridge"],
        cv_folds=2,
        workspace_dir=tmp_path / "ws_rmse",
    )
    res_rmse = automl_rmse.fit(df, target="y")
    lb_rmse = res_rmse.leaderboard()
    assert lb_rmse.iloc[0]["score"] > 0.0
    assert lb_rmse.iloc[0]["metric"] == "rmse"


def test_failed_trials_excluded_from_leaderboard_and_winner(tmp_path):
    """
    Finding 2: Failed trials (with score 0.0) must not appear in the leaderboard
    by default and must never be chosen as the winning model in ascending ranking.
    """
    ws, cmd, qry = build_application(root_dir=str(tmp_path))
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    ds = ws.register_dataset(name="reg_data", path=df, target="y", task_type="regression")
    run = ws.create_run(dataset=ds, metric="mae")

    exp = ws.create_experiment(
        run=run,
        name="exp_mixed",
        feature_names=["x"],
        model_ids=["ridge", "random_forest"],
    )

    # Save Trial entities
    ws.repository.save_trial(
        Trial(id="trial_valid", experiment_id=exp.id, model_id="ridge", parameters={})
    )
    ws.repository.save_trial(
        Trial(id="trial_failed", experiment_id=exp.id, model_id="random_forest", parameters={})
    )

    # 1. Valid result with MAE = 5.2
    valid_res = TrialResult(
        trial_id="trial_valid",
        experiment_id=exp.id,
        model_id="ridge",
        primary_metric="mae",
        primary_score=5.2,
        training_time_seconds=1.0,
    )
    # 2. Failed result with score = 0.0 and failure_reason
    failed_res = TrialResult(
        trial_id="trial_failed",
        experiment_id=exp.id,
        model_id="random_forest",
        primary_metric="mae",
        primary_score=0.0,
        training_time_seconds=0.1,
        failure_reason="Training crashed on random_forest",
    )

    ws.repository.save_trial_result(valid_res)
    ws.repository.save_trial_result(failed_res)

    # By default, get_leaderboard must exclude failed trials
    clean_lb = ws.repository.get_leaderboard(run.id, include_failed=False)
    assert len(clean_lb) == 1
    assert clean_lb[0].trial_id == "trial_valid"
    assert clean_lb[0].primary_score == 5.2

    # Include_failed=True includes it
    all_lb = ws.repository.get_leaderboard(run.id, include_failed=True)
    assert len(all_lb) == 2

    # export_model_artifact must pick trial_valid, not trial_failed
    with patch.object(ws.inference_service, "predict", return_value=np.array([1.0])):
        winning_artifact = ws.export_model_artifact(run_id=run.id)
        assert winning_artifact.metadata["trial_id"] == "trial_valid"
        assert winning_artifact.model_id == "ridge"
        assert winning_artifact.score == 5.2


def test_unsupported_metric_raises_error_no_silent_substitution():
    """
    Finding 3: Unsupported metrics must raise ValueError instead of silently
    substituting with accuracy or r2. Newly added metrics (mse, log_loss) work.
    """
    # 1. Unsupported metric raises ValueError
    with pytest.raises(ValueError, match="Unsupported metric 'fake_metric'"):
        _sklearn_scoring("fake_metric", task_type="classification")

    with pytest.raises(ValueError, match="Unsupported metric 'invalid_reg_metric'"):
        _sklearn_scoring("invalid_reg_metric", task_type="regression")

    # 2. Supported standard metrics return valid scoring
    assert _sklearn_scoring("mse", task_type="regression") == "neg_mean_squared_error"
    assert _sklearn_scoring("log_loss", task_type="binary_classification") == "neg_log_loss"
    assert _sklearn_scoring("balanced_accuracy", task_type="binary_classification") == "balanced_accuracy"


def test_mcp_streamable_http_external_host_security(caplog):
    """
    Finding H3: MCP service must reject external binding without authentication
    (fail-closed) and log a warning only when insecure_no_auth=True.
    """
    from automl.interfaces.mcp.server import run_mcp_service

    with patch("automl.interfaces.mcp.server.create_mcp_server") as mock_server:
        mock_instance = MagicMock()
        mock_server.return_value = mock_instance

        # 1. External host without token and without insecure_no_auth must fail closed
        with pytest.raises(PermissionError, match="Refusing to bind MCP streamable-http server to external interface '0.0.0.0'"):
            run_mcp_service(transport="streamable-http", host="0.0.0.0", port=8000)

        # 2. External host with insecure_no_auth=True fails closed unless CATML_ALLOW_INSECURE=1
        with pytest.raises(PermissionError, match="CATML_ALLOW_INSECURE=1"):
            run_mcp_service(transport="streamable-http", host="0.0.0.0", port=8000, insecure_no_auth=True)

        # 3. External host with insecure_no_auth=True and CATML_ALLOW_INSECURE=1 logs critical warning and proceeds
        with patch.dict("os.environ", {"CATML_ALLOW_INSECURE": "1"}):
            with caplog.at_level(logging.WARNING):
                run_mcp_service(transport="streamable-http", host="0.0.0.0", port=8000, insecure_no_auth=True)

        assert any("CRITICAL SECURITY WARNING" in rec.message for rec in caplog.records)
        assert any("0.0.0.0" in rec.message for rec in caplog.records)


def test_custom_minimization_metric_cv_positive_sign(tmp_path):
    """
    Finding H2: Custom metric with greater_is_better=False in cross-validation
    must produce positive error/cost values (normalized sign).
    """
    from automl.application.plugins.registry import PluginRegistry
    from automl.engine.training.sklearn_trainer import SklearnTrainer
    from automl.domain.runs.run import RunConfig, AutoMLRun
    from automl.domain.experiments.trial import Trial, Experiment, ExperimentStatus
    from automl.domain.ports import TrialExecution

    from dataclasses import dataclass, field
    from automl.domain.ports import PluginType, PluginCapability

    @dataclass
    class CustomCostMetric:
        plugin_id: str = "custom_cost"
        name: str = "Custom Cost"
        version: str = "1.0.0"
        plugin_type: PluginType = PluginType.METRIC
        capabilities: PluginCapability = field(default_factory=PluginCapability)
        greater_is_better: bool = False
        requires_probabilities: bool = False

        def compute(self, y_true, y_pred, y_prob=None):
            return float(np.mean(np.abs(y_true - y_pred)))

    reg = PluginRegistry()
    reg.register(CustomCostMetric())

    np.random.seed(42)
    n = 30
    df = pd.DataFrame({
        "x": np.random.randn(n),
        "y": 2.0 * np.random.randn(n) + 5.0,
    })
    csv_file = tmp_path / "df_cost.csv"
    df.to_csv(csv_file, index=False)

    trainer = SklearnTrainer(plugin_registry=reg)
    exp = Experiment(
        id="e_cost",
        run_id="run_cost",
        name="exp_cost",
        hypothesis="test",
        feature_names=["x"],
        metric="custom_cost",
        model_ids=["ridge"],
        status=ExperimentStatus.RUNNING,
    )
    run_obj = AutoMLRun(
        id="run_cost",
        workspace_id="ws_cost",
        dataset_id="ds_cost",
        config=RunConfig(target="y", task_type="regression", metric="custom_cost"),
    )
    trial_obj = Trial(
        id="t_cost",
        experiment_id="e_cost",
        model_id="ridge",
        parameters={},
    )
    trial_exec = TrialExecution(
        trial=trial_obj,
        experiment=exp,
        run=run_obj,
        feature_names=["x"],
        dataset_path=str(csv_file),
        target_column="y",
        task_type="regression",
        metric="custom_cost",
        validation_strategy="kfold",
        test_size=0.2,
        cv_folds=3,
        random_seed=42,
    )

    result = trainer.run(trial_exec)
    assert result.succeeded
    # Score must be strictly positive (real positive cost, NOT inverted scikit-learn negative)
    assert result.primary_score > 0.0, f"Expected positive cost, got {result.primary_score}"

