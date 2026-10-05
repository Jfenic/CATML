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



# A5: interchangeable real-provider adapters (no paid API requests).
"""Provider contracts verified without credentials or paid network requests."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from automl.application.agents.ports import LLMProviderPort
from automl.infrastructure.llm import (
    FakeLLMProvider, HTTPProvider, LLMProviderConfig, LLMProviderError, create_llm_provider,
)
from automl.infrastructure.llm.providers import _http_post

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


def envelope(provider, content='{"ok":true}'):
    if provider == "openai":
        return {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": content}]}], "usage": {"input_tokens": 10, "output_tokens": 2}}
    if provider == "anthropic":
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": content}], "usage": {"input_tokens": 10, "output_tokens": 2}}
    if provider == "ollama":
        return {"done": True, "message": {"content": content}, "prompt_eval_count": 10, "eval_count": 2}
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def config(provider="openai", **overrides):
    values = dict(provider=provider, model="explicit-test-model", api_key="secret-test-key")
    if provider == "openai-compatible":
        values["base_url"] = "http://localhost:9000/v1"
    return LLMProviderConfig(**{**values, **overrides})


@pytest.mark.parametrize("provider,suffix", [("openai", "/responses"), ("anthropic", "/messages"), ("ollama", "/api/chat"), ("openai-compatible", "/chat/completions")])
def test_interchangeable_contract_and_payload(provider, suffix):
    calls = []
    def transport(url, headers, payload, timeout, limit):
        calls.append((url, headers, payload, timeout, limit))
        return 200, envelope(provider)
    adapter = create_llm_provider(config(provider), transport=transport)
    assert isinstance(adapter, LLMProviderPort)
    result = adapter.generate("secret-test-key password=hidden", system_prompt="Analyze", response_schema=SCHEMA)
    assert result.parsed == {"ok": True}
    assert result.total_tokens == 12
    assert calls[0][0].endswith(suffix)
    wire = json.dumps(calls[0][2])
    assert "secret-test-key" not in wire and "hidden" not in wire
    assert "secret-test-key" not in repr(adapter.config)
    assert "REDACTED" in wire
    assert "hidden" not in repr(adapter.audit)
    assert adapter.audit[0]["total_tokens"] == 12
    if provider == "openai":
        assert calls[0][2]["store"] is False
    if provider == "anthropic":
        assert calls[0][1]["anthropic-version"] == "2023-06-01"
        assert calls[0][2]["messages"][0]["role"] == "user"
    if provider == "ollama":
        assert calls[0][2]["stream"] is False
        assert calls[0][2]["format"] == SCHEMA


def test_fake_default_has_no_network():
    assert isinstance(create_llm_provider(LLMProviderConfig.from_env({})), FakeLLMProvider)
    assert LLMProviderConfig.from_env({"CATML_LLM_PROVIDER": "anthropic", "CATML_LLM_MODEL": "test", "ANTHROPIC_API_KEY": "secret"}).api_key == "secret"


@pytest.mark.parametrize("changes", [{"provider": "unknown"}, {"model": ""}, {"api_key": ""}, {"timeout_seconds": float("nan")}, {"timeout_seconds": 301}, {"max_retries": 4}, {"max_output_tokens": True}, {"base_url": "http://example.com/v1"}, {"base_url": "https://secret@example.com/v1"}, {"base_url": "https://example.com/v1?key=secret"}])
def test_configuration_rejects_unsafe_or_invalid_values(changes):
    with pytest.raises(ValueError):
        config(**changes)


@pytest.mark.parametrize("content", ['{"ok":"yes"}', '{}', '{"ok":true,"extra":1}', '{"ok":true,"ok":false}', '{"ok":NaN}', '```json\n{"ok":true}\n```', '[]'])
def test_structured_output_fails_closed(content):
    adapter = create_llm_provider(config(), transport=lambda *args: (200, envelope("openai", content)))
    with pytest.raises(LLMProviderError):
        adapter.generate("test", response_schema=SCHEMA)
    assert adapter.audit[0]["status"] == "failed"
    assert adapter.audit[0]["total_tokens"] == 12
    assert adapter.audit[0]["attempts"] == 1


def test_preflight_validation_never_calls_transport():
    def fail(*args):
        pytest.fail("Unexpected outbound request")
    adapter = create_llm_provider(config(max_input_chars=20), transport=fail)
    for kwargs in ({"prompt": "x" * 21}, {"prompt": "x", "temperature": 1}, {"prompt": "x", "response_schema": {"type": "object", "oneOf": []}}, {"prompt": "x", "response_schema": {"type": "string"}}, {"prompt": 1}):
        with pytest.raises(ValueError):
            adapter.generate(**kwargs)
    assert not adapter.audit


@pytest.mark.parametrize("status,attempts", [(401, 1), (400, 1), (302, 1), (429, 2), (503, 2)])
def test_retries_bounded_and_only_transient_http(status, attempts, monkeypatch):
    monkeypatch.setattr("automl.infrastructure.llm.providers.time.sleep", lambda _: None)
    adapter = create_llm_provider(config(), transport=lambda *args: (status, {}))
    with pytest.raises(LLMProviderError, match=str(status)):
        adapter.generate("test")
    assert adapter.audit[0]["attempts"] == attempts
    assert adapter.audit[0]["total_tokens"] is None


def test_transient_recovery_and_unknown_usage(monkeypatch):
    monkeypatch.setattr("automl.infrastructure.llm.providers.time.sleep", lambda _: None)
    data = envelope("openai")
    del data["usage"]
    replies = iter([(429, {}), (200, data)])
    adapter = create_llm_provider(config(), transport=lambda *args: next(replies))
    assert adapter.generate("test").total_tokens == 0  # Existing DTO convention.
    assert adapter.audit[0]["total_tokens"] is None  # Never pretend usage is known.
    assert adapter.audit[0]["attempts"] == 2


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama", "openai-compatible"])
def test_incomplete_responses_rejected(provider):
    data = envelope(provider)
    if provider == "openai":
        data["status"] = "incomplete"
    elif provider == "anthropic":
        data["stop_reason"] = "max_tokens"
    elif provider == "ollama":
        data["done_reason"] = "length"
    else:
        data["choices"][0]["finish_reason"] = "length"
    adapter = create_llm_provider(config(provider), transport=lambda *args: (200, data))
    with pytest.raises(LLMProviderError, match="incomplete"):
        adapter.generate("test")


def test_refusal_and_invalid_envelopes():
    for data in ({}, envelope("openai", ""), {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}, {**envelope("openai"), "usage": {"input_tokens": -1}}):
        adapter = create_llm_provider(config(), transport=lambda *args: (200, data))
        with pytest.raises(LLMProviderError):
            adapter.generate("test")


def test_schema_arrays_numbers_and_enums():
    schema = {"type": "object", "properties": {"values": {"type": "array", "items": {"type": "integer", "enum": [1, 2]}}, "params": {"type": "object"}}, "required": ["values"]}
    for content, valid in [('{"values":[1,2],"params":{"depth":2}}', True), ('{"values":[true]}', False), ('{"values":[3]}', False)]:
        adapter = create_llm_provider(config(), transport=lambda *args: (200, envelope("openai", content)))
        if valid:
            assert adapter.generate("test", response_schema=schema).parsed["values"] == [1, 2]
        else:
            with pytest.raises(LLMProviderError):
                adapter.generate("test", response_schema=schema)


def test_real_http_transport_with_local_fixture():
    received = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/target")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(envelope("ollama")).encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        adapter = create_llm_provider(config("ollama", base_url=url))
        assert adapter.generate("Analyze", response_schema=SCHEMA).parsed == {"ok": True}
        assert received[0]["model"] == "explicit-test-model"
        assert _http_post(url + "/redirect", {}, {}, 1, 1000) == (302, {})
        assert len(received) == 2  # Never forwarded credentials to redirected URL.
        with pytest.raises(LLMProviderError, match="size"):
            _http_post(url, {}, {}, 1, 1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("provider", ["openai", "anthropic", "ollama", "openai-compatible"])
def test_real_provider_drives_existing_planner_without_contract_changes(provider):
    candidate = {"hypothesis": "Test nonlinear effects", "action_type": "create_experiment", "model_id": "random_forest", "hyperparameters": {"n_estimators": 20}}
    adapter = create_llm_provider(config(provider), transport=lambda *args: (200, envelope(provider, json.dumps(candidate))))
    planner = Planner(llm_provider=adapter)
    context = ContextPayload(run_id="run-test", dataset_id="dataset-test", task_type="binary_classification", target_metric="roc_auc")
    proposal = planner.analyze(context)
    assert proposal.hypothesis == candidate["hypothesis"]
    assert proposal.action_payload == {"model_id": "random_forest", "hyperparameters": {"n_estimators": 20}}
    assert adapter.audit[0]["status"] == "ok"


def test_invalid_provider_output_uses_existing_deterministic_fallback():
    adapter = create_llm_provider(config(), transport=lambda *args: (200, envelope("openai", '{"hypothesis":false}')))
    proposal = Planner(llm_provider=adapter).analyze(ContextPayload(run_id="r", dataset_id="d", task_type="binary_classification", target_metric="roc_auc"))
    assert proposal.action_payload["model_id"] == "logistic_regression"
    assert adapter.audit[0]["status"] == "failed"


def test_network_failure_is_not_retried_and_usage_stays_unknown():
    def transport(*args):
        raise LLMProviderError("LLM transport unavailable or timed out")
    adapter = create_llm_provider(config(max_retries=3), transport=transport)
    with pytest.raises(LLMProviderError, match="timed out"):
        adapter.generate("test")
    assert adapter.audit[0]["attempts"] == 1
    assert adapter.audit[0]["total_tokens"] is None


def test_anthropic_cached_tokens_counted():
    data = envelope("anthropic")
    data["usage"].update(cache_creation_input_tokens=5, cache_read_input_tokens=7)
    adapter = create_llm_provider(config("anthropic"), transport=lambda *args: (200, data))
    assert adapter.generate("test").prompt_tokens == 22


def test_hyperparameters_cannot_contain_overflowing_numbers():
    schema = {"type": "object", "properties": {"params": {"type": "object"}}}
    adapter = create_llm_provider(config(), transport=lambda *args: (200, envelope("openai", '{"params":{"rate":1e999}}')))
    with pytest.raises(LLMProviderError):
        adapter.generate("test", response_schema=schema)


@pytest.mark.parametrize("raw", [b"not-json", b"[]", b"\xff"])
def test_transport_rejects_malformed_envelopes(raw, monkeypatch):
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, limit):
            return raw
    opener = MagicMock()
    opener.open.return_value = Response()
    monkeypatch.setattr("automl.infrastructure.llm.providers.build_opener", lambda *args: opener)
    with pytest.raises(LLMProviderError):
        _http_post("http://127.0.0.1/api/chat", {}, {}, 1, 1000)
