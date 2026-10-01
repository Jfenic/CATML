"""Unit tests for Milestone H4 shared contracts, DTOs, and segregated ports."""
from __future__ import annotations

from typing import Any

from automl.application.agents.contracts import (
    CandidateProposal,
    ContextPayload,
    EvaluationFeedback,
    LLMResponse,
    SessionStepResult,
)
from automl.application.agents.ports import (
    AgentApprovalStorePort,
    AgentLedgerPort,
    AgentOperationStorePort,
    AgentSessionStorePort,
    LLMProviderPort,
    SpecialistPort,
)
from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger


def test_candidate_proposal_serialization():
    proposal = CandidateProposal(
        proposal_id="prop-001",
        specialist_name="planner",
        run_id="run-123",
        hypothesis="Testing LightGBM on high-variance numeric features",
        action_type="create_experiment",
        action_payload={"model_id": "lightgbm", "hyperparameters": {"n_estimators": 50}},
        verification_plan={"metric": "roc_auc", "direction": "maximize", "min_delta": 0.01},
        estimated_cost={"trials": 1, "duration_seconds": 30.0},
        confidence=0.85,
        created_at="2026-10-01T12:00:00Z",
    )
    d = proposal.to_dict()
    assert d["proposal_id"] == "prop-001"
    assert d["specialist_name"] == "planner"
    assert d["confidence"] == 0.85

    restored = CandidateProposal.from_dict(d)
    assert restored.proposal_id == proposal.proposal_id
    assert restored.action_payload == proposal.action_payload
    assert restored.verification_plan == proposal.verification_plan


def test_evaluation_feedback_serialization():
    feedback = EvaluationFeedback(
        feedback_id="feed-001",
        run_id="run-123",
        specialist_name="critic",
        is_improvement=True,
        metric_name="roc_auc",
        experiment_id="exp-456",
        baseline_score=0.812,
        current_score=0.845,
        variance_observation="Low cross-validation spread (+-0.005)",
        recommendation="accept",
        evidence_summary={"fold_scores": [0.84, 0.85, 0.845]},
        created_at="2026-10-01T12:05:00Z",
    )
    d = feedback.to_dict()
    assert d["is_improvement"] is True
    assert d["current_score"] == 0.845
    assert d["recommendation"] == "accept"

    restored = EvaluationFeedback.from_dict(d)
    assert restored.feedback_id == feedback.feedback_id
    assert restored.is_improvement is True
    assert restored.baseline_score == 0.812


def test_context_payload_serialization():
    payload = ContextPayload(
        run_id="run-123",
        dataset_id="ds-789",
        task_type="binary_classification",
        target_metric="roc_auc",
        metric_direction="maximize",
        dataset_profile={"n_rows": 1000, "n_features": 12},
        feature_evidence=[{"feature_set": "baseline", "score": 0.80}],
        leaderboard=[{"model_id": "logistic_regression", "score": 0.80}],
        budget_status={"remaining_experiments": 4, "remaining_trials": 15},
        history=[{"step": 1, "action": "baseline_evaluated"}],
        omissions=["raw_sample_data", "uncompressed_images"],
    )
    d = payload.to_dict()
    assert d["run_id"] == "run-123"
    assert d["target_metric"] == "roc_auc"
    assert "raw_sample_data" in d["omissions"]

    restored = ContextPayload.from_dict(d)
    assert restored.run_id == payload.run_id
    assert restored.dataset_profile == payload.dataset_profile
    assert restored.budget_status == payload.budget_status


def test_session_step_result_serialization():
    prop = CandidateProposal(
        proposal_id="prop-1",
        specialist_name="planner",
        run_id="run-1",
        hypothesis="Try RandomForest baseline",
        action_type="create_experiment",
        action_payload={"model_id": "random_forest"},
    )
    feed = EvaluationFeedback(
        feedback_id="feed-1",
        run_id="run-1",
        specialist_name="critic",
        is_improvement=False,
        metric_name="roc_auc",
        current_score=0.79,
    )
    step = SessionStepResult(
        session_id="sess-001",
        step_number=2,
        state="evaluate",
        action_taken="run_experiment",
        proposal=prop,
        feedback=feed,
        operation_id="op-999",
        checkpoint_id="chk-002",
        created_at="2026-10-01T12:10:00Z",
    )
    d = step.to_dict()
    assert d["session_id"] == "sess-001"
    assert d["state"] == "evaluate"
    assert d["proposal"]["proposal_id"] == "prop-1"
    assert d["feedback"]["feedback_id"] == "feed-1"

    restored = SessionStepResult.from_dict(d)
    assert restored.session_id == step.session_id
    assert restored.proposal is not None
    assert restored.proposal.proposal_id == "prop-1"
    assert restored.feedback is not None
    assert restored.feedback.feedback_id == "feed-1"


def test_llm_response_serialization():
    resp = LLMResponse(
        content='{"action": "create_experiment"}',
        provider="fake_provider",
        model="fake-model-v1",
        parsed={"action": "create_experiment"},
        prompt_tokens=150,
        completion_tokens=25,
        total_tokens=175,
        latency_seconds=0.12,
    )
    d = resp.to_dict()
    assert d["total_tokens"] == 175
    assert d["parsed"]["action"] == "create_experiment"

    restored = LLMResponse.from_dict(d)
    assert restored.provider == "fake_provider"
    assert restored.latency_seconds == 0.12


def test_sqlite_agent_ledger_implements_segregated_ports(tmp_path):
    """Verify SqliteAgentLedger satisfies the segregated store ports and the combined ledger port."""
    db_file = str(tmp_path / "test_ledger.db")
    ledger = SqliteAgentLedger(db_path=db_file)

    # Verify structural protocol compatibility (can be used wherever segregated port is typed)
    assert isinstance(ledger, AgentOperationStorePort)
    assert isinstance(ledger, AgentApprovalStorePort)
    assert isinstance(ledger, AgentSessionStorePort)
    assert isinstance(ledger, AgentLedgerPort)


class DummySpecialist:
    @property
    def name(self) -> str:
        return "dummy_planner"

    def analyze(self, context: ContextPayload, **kwargs: Any) -> CandidateProposal:
        return CandidateProposal(
            proposal_id="p-1",
            specialist_name=self.name,
            run_id=context.run_id,
            hypothesis="Baseline hypothesis",
            action_type="create_experiment",
            action_payload={"model_id": "dummy"},
        )


class DummyLLMProvider:
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        response_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(
            content="{}",
            provider="dummy",
            model="dummy-model",
            total_tokens=10,
        )


def test_specialist_and_llm_provider_ports():
    specialist = DummySpecialist()
    assert isinstance(specialist, SpecialistPort)

    ctx = ContextPayload(
        run_id="run-1",
        dataset_id="ds-1",
        task_type="binary_classification",
        target_metric="roc_auc",
    )
    proposal = specialist.analyze(ctx)
    assert proposal.proposal_id == "p-1"

    provider = DummyLLMProvider()
    assert isinstance(provider, LLMProviderPort)
    resp = provider.generate("hello")
    assert resp.provider == "dummy"
