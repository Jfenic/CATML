"""Unit and integration tests for Milestone H4 specialists, context builder, and fake LLM provider."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
import pytest

from automl.application.agents.contracts import ContextPayload
from automl.application.agents.ports import LLMProviderPort, SpecialistPort
from automl.application.agents.specialists.advisor import FeatureAdvisor
from automl.application.agents.specialists.context_builder import ContextBuilder
from automl.application.agents.specialists.critic import Critic
from automl.application.agents.specialists.planner import Planner
from automl.infrastructure.llm.fake_provider import FakeLLMProvider


@pytest.fixture
def mock_workspace():
    ws = MagicMock()
    # Mock Run
    run = MagicMock()
    run.id = "run-test-001"
    run.dataset_id = "ds-test-001"
    ws.get_run.return_value = run

    # Mock Dataset
    dataset = MagicMock()
    dataset.id = "ds-test-001"
    dataset.name = "churn_data"
    dataset.target_column = "churn"
    ws.get_dataset.return_value = dataset

    # Mock Profile
    profile = MagicMock()
    profile.n_rows = 500
    profile.n_columns = 6
    profile.numeric_columns = ["age", "balance", "credit_score"]
    profile.categorical_columns = ["country", "gender"]
    profile.task_type = "binary_classification"
    ws.get_profile.return_value = profile

    # Mock Leaderboard
    ws.get_leaderboard.return_value = [
        {"model_id": "logistic_regression", "metric": "roc_auc", "score": 0.81, "experiment_id": "exp-1"},
    ]

    # Mock Feature Sets
    fset = MagicMock()
    fset.name = "baseline_features"
    fset.features = ["age", "balance"]
    ws.list_feature_sets.return_value = [fset]

    return ws


# ==========================================
# 1. ContextBuilder Tests
# ==========================================

def test_context_builder_extraction(mock_workspace):
    builder = ContextBuilder(max_leaderboard_rows=5, max_feature_evidence=5)
    payload = builder.build(mock_workspace, "run-test-001")

    assert payload.run_id == "run-test-001"
    assert payload.dataset_id == "ds-test-001"
    assert payload.target_metric == "roc_auc"
    assert payload.metric_direction == "maximize"
    assert payload.dataset_profile["n_rows"] == 500
    assert payload.dataset_profile["numeric_columns"] == ["age", "balance", "credit_score"]
    assert len(payload.leaderboard) == 1
    assert payload.leaderboard[0]["model_id"] == "logistic_regression"
    assert "raw_dataframe_samples_excluded" in payload.omissions


def test_context_builder_truncation_omissions(mock_workspace):
    # Simulate a long leaderboard exceeding limit
    mock_workspace.get_leaderboard.return_value = [
        {"model_id": f"model_{i}", "metric": "roc_auc", "score": 0.80 + i * 0.01}
        for i in range(15)
    ]
    builder = ContextBuilder(max_leaderboard_rows=3)
    payload = builder.build(mock_workspace, "run-test-001")

    assert len(payload.leaderboard) == 3
    assert any("leaderboard_truncated_to_3" in o for o in payload.omissions)


def test_context_builder_missing_run_or_dataset(mock_workspace):
    mock_workspace.get_run.return_value = None
    builder = ContextBuilder()
    with pytest.raises(ValueError, match="Run 'unknown_run' not found"):
        builder.build(mock_workspace, "unknown_run")


# ==========================================
# 2. Planner Tests
# ==========================================

def test_planner_implements_protocol():
    planner = Planner()
    assert isinstance(planner, SpecialistPort)
    assert planner.name == "planner"


def test_planner_proposes_baseline_when_empty():
    planner = Planner()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        metric_direction="maximize",
        leaderboard=[],
    )
    proposal = planner.analyze(ctx)
    assert proposal.specialist_name == "planner"
    assert proposal.action_type == "create_experiment"
    assert proposal.action_payload["model_id"] == "logistic_regression"
    assert proposal.confidence >= 0.90


def test_planner_proposes_random_forest_after_baseline():
    planner = Planner()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        leaderboard=[{"model_id": "logistic_regression", "score": 0.78}],
    )
    proposal = planner.analyze(ctx)
    assert proposal.action_payload["model_id"] == "random_forest"


def test_planner_proposes_tuning_when_models_explored():
    planner = Planner()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        leaderboard=[
            {"model_id": "lightgbm", "score": 0.86},
            {"model_id": "random_forest", "score": 0.83},
            {"model_id": "logistic_regression", "score": 0.78},
        ],
    )
    proposal = planner.analyze(ctx)
    assert proposal.action_type == "tune_hyperparameters"
    assert proposal.action_payload["model_id"] == "lightgbm"
    assert proposal.action_payload["n_trials"] == 10


def test_planner_with_fake_llm_provider():
    provider = FakeLLMProvider()
    provider.register_response(
        r".*",
        {
            "hypothesis": "Test CatBoost on categorical features",
            "action_type": "create_experiment",
            "model_id": "catboost",
            "hyperparameters": {"iterations": 200},
            "estimated_trials": 1,
        },
    )
    planner = Planner(llm_provider=provider)
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
    )
    proposal = planner.analyze(ctx)
    assert proposal.action_payload["model_id"] == "catboost"
    assert "CatBoost" in proposal.hypothesis
    assert provider.total_tokens_consumed > 0


# ==========================================
# 3. FeatureAdvisor Tests
# ==========================================

def test_advisor_implements_protocol():
    advisor = FeatureAdvisor()
    assert isinstance(advisor, SpecialistPort)
    assert advisor.name == "feature_advisor"


def test_advisor_detects_target_leakage():
    advisor = FeatureAdvisor()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        dataset_profile={
            "target_column": "churn",
            "numeric_columns": ["age", "is_churned"],
            "categorical_columns": [],
        },
    )
    proposal = advisor.analyze(ctx)
    assert proposal.action_type == "exclude_features"
    assert "is_churned" in proposal.action_payload["excluded_features"]
    assert "leakage" in proposal.hypothesis.lower()


def test_advisor_proposes_pairwise_differences():
    advisor = FeatureAdvisor()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        dataset_profile={
            "target_column": "churn",
            "numeric_columns": ["income", "expenditure", "debt"],
            "categorical_columns": [],
        },
        feature_evidence=[],
    )
    proposal = advisor.analyze(ctx)
    assert proposal.action_type == "propose_feature_set"
    assert proposal.action_payload["feature_set_name"] == "interactions_differences"
    assert "inter_diff_income_minus_expenditure" in proposal.action_payload["candidate_features"]


def test_advisor_proposes_frequency_encoding_for_categoricals():
    advisor = FeatureAdvisor()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        dataset_profile={
            "target_column": "churn",
            "numeric_columns": [],
            "categorical_columns": ["zip_code", "city"],
        },
        feature_evidence=[],
    )
    proposal = advisor.analyze(ctx)
    assert proposal.action_type == "propose_feature_set"
    assert proposal.action_payload["feature_set_name"] == "categorical_frequency_encoding"


# ==========================================
# 4. Critic Tests
# ==========================================

def test_critic_implements_protocol():
    critic = Critic()
    assert isinstance(critic, SpecialistPort)
    assert critic.name == "critic"


def test_critic_initial_baseline():
    critic = Critic()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        metric_direction="maximize",
        leaderboard=[{"model_id": "logistic_regression", "score": 0.80, "experiment_id": "exp-1"}],
    )
    feedback = critic.analyze(ctx)
    assert feedback.is_improvement is True
    assert feedback.recommendation == "accept"
    assert feedback.baseline_score is None
    assert feedback.current_score == 0.80


def test_critic_detects_improvement_maximize():
    critic = Critic()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        metric_direction="maximize",
        leaderboard=[
            {"model_id": "random_forest", "score": 0.85, "experiment_id": "exp-2"},
            {"model_id": "logistic_regression", "score": 0.80, "experiment_id": "exp-1"},
        ],
    )
    feedback = critic.analyze(ctx)
    assert feedback.is_improvement is True
    assert feedback.recommendation == "accept"
    assert feedback.evidence_summary["delta"] == pytest.approx(0.05)


def test_critic_detects_degradation():
    critic = Critic()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        metric_direction="maximize",
        leaderboard=[{"model_id": "logistic_regression", "score": 0.80}],
    )
    # latest trial scored worse
    feedback = critic.analyze(ctx, latest_result={"score": 0.75, "experiment_id": "exp-bad"})
    assert feedback.is_improvement is False
    assert feedback.recommendation == "reject"
    assert feedback.evidence_summary["delta"] == pytest.approx(-0.05)


def test_critic_detects_high_variance_instability():
    critic = Critic()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        metric_direction="maximize",
        leaderboard=[{"model_id": "baseline", "score": 0.80}],
    )
    # Improved on average, but high CV fold spread (e.g. 0.95 vs 0.70)
    feedback = critic.analyze(
        ctx,
        latest_result={
            "score": 0.83,
            "experiment_id": "exp-volatile",
            "fold_scores": [0.95, 0.70, 0.84],
        },
    )
    assert feedback.is_improvement is True
    assert feedback.recommendation == "explore_alternative"
    assert "High cross-validation spread" in feedback.variance_observation


# ==========================================
# 5. FakeLLMProvider Tests
# ==========================================

def test_fake_llm_provider_protocol():
    provider = FakeLLMProvider()
    assert isinstance(provider, LLMProviderPort)


def test_fake_llm_provider_canned_regex_match():
    provider = FakeLLMProvider()
    provider.register_response(r"churn|customer", {"prediction": "high_risk", "score": 0.92})

    resp = provider.generate("Tell me about churn risk for customer 101")
    assert resp.parsed == {"prediction": "high_risk", "score": 0.92}
    assert resp.provider == "fake_provider"
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0


def test_fake_llm_provider_schema_fallback():
    provider = FakeLLMProvider()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "value": {"type": "integer"},
            "active": {"type": "boolean"},
        },
    }
    resp = provider.generate("Random unmapped query", response_schema=schema)
    assert resp.parsed is not None
    assert resp.parsed["name"] == "fake_name"
    assert resp.parsed["value"] == 1
    assert resp.parsed["active"] is True


def test_fake_llm_provider_error_simulation():
    provider = FakeLLMProvider()
    provider.set_simulate_error(RuntimeError("API quota exceeded simulation"))
    with pytest.raises(RuntimeError, match="API quota exceeded simulation"):
        provider.generate("Test query")


def test_advisor_with_llm_and_fallback():
    provider = FakeLLMProvider()
    provider.register_response(
        r".*",
        {
            "hypothesis": "LLM suggested target encoding",
            "action_type": "propose_feature_set",
            "feature_set_name": "target_encoded_features",
            "source_columns": ["country"],
        },
    )
    advisor = FeatureAdvisor(llm_provider=provider)
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
    )
    proposal = advisor.analyze(ctx)
    assert proposal.action_payload["feature_set_name"] == "target_encoded_features"

    # Fallback when LLM raises
    provider.set_simulate_error(RuntimeError("LLM failed"))
    fallback_prop = advisor.analyze(ctx)
    assert fallback_prop.specialist_name == "feature_advisor"


def test_advisor_robust_scaling_fallback():
    advisor = FeatureAdvisor()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        dataset_profile={
            "target_column": "target",
            "numeric_columns": ["feat_1"],
            "categorical_columns": [],
        },
        feature_evidence=[],
    )
    proposal = advisor.analyze(ctx)
    assert proposal.action_payload["feature_set_name"] == "robust_scaled_features"


def test_critic_with_llm_and_fallback():
    provider = FakeLLMProvider()
    provider.register_response(
        r".*",
        {
            "is_improvement": True,
            "recommendation": "accept",
            "variance_observation": "Model generalizes robustly across splits.",
        },
    )
    critic = Critic(llm_provider=provider)
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        metric_direction="maximize",
        leaderboard=[{"model_id": "lightgbm", "score": 0.88}],
    )
    feedback = critic.analyze(ctx)
    assert feedback.is_improvement is True
    assert feedback.recommendation == "accept"
    assert "generalizes robustly" in feedback.variance_observation

    # Fallback on LLM failure
    provider.set_simulate_error(RuntimeError("LLM timeout"))
    fb = critic.analyze(ctx)
    assert fb.specialist_name == "critic"


def test_critic_minimization_metric():
    critic = Critic()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="regression",
        target_metric="rmse",
        metric_direction="minimize",
        leaderboard=[
            {"model_id": "new_model", "score": 2.10, "experiment_id": "exp-2"},
            {"model_id": "baseline", "score": 2.50, "experiment_id": "exp-1"},
        ],
    )
    # 2.10 is lower than 2.50, which is an improvement for minimize
    fb = critic.analyze(ctx)
    assert fb.is_improvement is True
    assert fb.recommendation == "accept"

    # Degraded score: 2.80 vs baseline 2.50
    fb_deg = critic.analyze(ctx, latest_result={"score": 2.80, "experiment_id": "exp-3"})
    assert fb_deg.is_improvement is False
    assert fb_deg.recommendation == "reject"


def test_critic_empty_state():
    critic = Critic()
    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
        leaderboard=[],
    )
    fb = critic.analyze(ctx)
    assert fb.is_improvement is False
    assert fb.recommendation == "explore_alternative"


def test_context_builder_feature_and_history_truncation(mock_workspace):
    # Simulate multiple feature sets
    mock_workspace.list_feature_sets.return_value = [
        MagicMock(name=f"fset_{i}", features=[f"col_{i}"]) for i in range(10)
    ]
    builder = ContextBuilder(max_feature_evidence=2, max_history_entries=2)
    history = [{"step": i} for i in range(10)]
    payload = builder.build(mock_workspace, "run-test-001", history=history)

    assert len(payload.feature_evidence) == 2
    assert len(payload.history) == 2
    assert any("feature_evidence_truncated_to_2" in o for o in payload.omissions)
    assert any("history_truncated_to_2" in o for o in payload.omissions)


def test_context_builder_minimization_detection(mock_workspace):
    mock_workspace.get_leaderboard.return_value = [
        {"model_id": "ridge", "metric": "rmse", "score": 1.25, "experiment_id": "exp-1"}
    ]
    builder = ContextBuilder()
    payload = builder.build(mock_workspace, "run-test-001")
    assert payload.target_metric == "rmse"
    assert payload.metric_direction == "minimize"


def test_context_builder_profile_exception_graceful(mock_workspace):
    mock_workspace.get_profile.side_effect = RuntimeError("Database locked")
    builder = ContextBuilder()
    payload = builder.build(mock_workspace, "run-test-001")
    assert payload.dataset_profile["dataset_id"] == "ds-test-001"


def test_fake_llm_provider_string_canned_response():
    provider = FakeLLMProvider(default_responses={"hello": "plain text reply"})
    resp = provider.generate("hello world")
    assert resp.content == "plain text reply"
    assert resp.parsed is None


def test_fake_llm_provider_string_json_canned_response():
    provider = FakeLLMProvider()
    provider.register_response("test_json", '{"key": "value"}')
    resp = provider.generate("test_json query")
    assert resp.parsed == {"key": "value"}

