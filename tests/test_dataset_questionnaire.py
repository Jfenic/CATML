from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from automl.application.bootstrap import build_application
from automl.domain.tasks.questionnaire import (
    DatasetQuestionnaire,
    ErrorCostPriority,
    ExplainabilityLevel,
    LatencyConstraint,
    TemporalStructure,
)
from automl.engine.planning.questionnaire_advisor import QuestionnaireAdvisor
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository
from automl.interfaces.web.server import AutoMLWebHandler


# ---------------------------------------------------------------------------
# 1. Pure Domain Entity Tests
# ---------------------------------------------------------------------------


def test_questionnaire_domain_defaults_and_roundtrip():
    q = DatasetQuestionnaire(
        dataset_id="ds_123",
        domain_hint="customer_churn",
        error_cost=ErrorCostPriority.AVOID_FALSE_NEGATIVES,
        temporal_structure=TemporalStructure.CROSS_SECTIONAL,
        latency_constraint=LatencyConstraint.STANDARD_INTERACTIVE,
        explainability=ExplainabilityLevel.MODERATE,
        primary_metric_override="pr_auc",
        split_strategy_recommendation="stratified_kfold",
        auto_generated=True,
        confidence=0.95,
        reasoning="Test reasoning explanation",
    )

    data = q.to_dict()
    assert data["dataset_id"] == "ds_123"
    assert data["domain_hint"] == "customer_churn"
    assert data["error_cost"] == "avoid_false_negatives"
    assert data["temporal_structure"] == "cross_sectional"
    assert data["latency_constraint"] == "standard_interactive"
    assert data["explainability"] == "moderate"
    assert data["primary_metric_override"] == "pr_auc"
    assert data["split_strategy_recommendation"] == "stratified_kfold"
    assert data["auto_generated"] is True
    assert data["confidence"] == 0.95

    restored = DatasetQuestionnaire.from_dict(data)
    assert restored.dataset_id == q.dataset_id
    assert restored.domain_hint == q.domain_hint
    assert restored.error_cost == ErrorCostPriority.AVOID_FALSE_NEGATIVES
    assert restored.temporal_structure == TemporalStructure.CROSS_SECTIONAL
    assert restored.latency_constraint == LatencyConstraint.STANDARD_INTERACTIVE
    assert restored.explainability == ExplainabilityLevel.MODERATE
    assert restored.primary_metric_override == "pr_auc"
    assert restored.reasoning == "Test reasoning explanation"


# ---------------------------------------------------------------------------
# 2. QuestionnaireAdvisor Heuristic & LLM Tests
# ---------------------------------------------------------------------------


def test_advisor_heuristics_churn():
    advisor = QuestionnaireAdvisor()
    q = advisor.infer(
        dataset_id="ds_churn",
        dataset_name="telecom_churn_data",
        column_names=["tenure", "monthly_charges", "total_charges", "churn"],
        target_column="churn",
    )
    assert q.domain_hint == "customer_churn"
    assert q.error_cost == ErrorCostPriority.AVOID_FALSE_NEGATIVES
    assert q.primary_metric_override == "pr_auc"
    assert "churn" in q.reasoning.lower()


def test_advisor_heuristics_fraud():
    advisor = QuestionnaireAdvisor()
    q = advisor.infer(
        dataset_id="ds_fraud",
        dataset_name="credit_card_fraud",
        column_names=["amount", "merchant", "card_type", "is_fraud"],
        target_column="is_fraud",
    )
    assert q.domain_hint == "fraud_detection"
    assert q.error_cost == ErrorCostPriority.AVOID_FALSE_NEGATIVES
    assert q.latency_constraint == LatencyConstraint.ULTRA_LOW_REALTIME
    assert q.primary_metric_override == "pr_auc"
    assert q.imbalance_strategy == "balanced_weights"


def test_advisor_heuristics_medical():
    advisor = QuestionnaireAdvisor()
    q = advisor.infer(
        dataset_id="ds_medical",
        dataset_name="patient_cancer_diagnosis",
        column_names=["age", "radius_mean", "texture_mean", "diagnosis"],
        target_column="diagnosis",
    )
    assert q.domain_hint == "healthcare_diagnosis"
    assert q.error_cost == ErrorCostPriority.AVOID_FALSE_NEGATIVES
    assert q.explainability == ExplainabilityLevel.HIGHLY_REGULATED
    assert q.primary_metric_override == "recall"


def test_advisor_heuristics_credit():
    advisor = QuestionnaireAdvisor()
    q = advisor.infer(
        dataset_id="ds_credit",
        dataset_name="retail_loan_default",
        column_names=["income", "loan_amount", "fico_score", "default"],
        target_column="default",
    )
    assert q.domain_hint == "credit_risk"
    assert q.explainability == ExplainabilityLevel.HIGHLY_REGULATED
    assert q.primary_metric_override == "roc_auc"


def test_advisor_heuristics_temporal_time_series():
    advisor = QuestionnaireAdvisor()
    q = advisor.infer(
        dataset_id="ds_weather",
        dataset_name="weather_sensor_stream",
        column_names=["timestamp", "temperature", "humidity", "rainfall"],
        target_column="rainfall",
    )
    assert q.temporal_structure == TemporalStructure.SEQUENTIAL_TIME_SERIES
    assert q.split_strategy_recommendation == "time_series_split"
    assert "sequential" in q.reasoning.lower() or "time" in q.reasoning.lower()


def test_advisor_heuristics_grouped_cohort():
    advisor = QuestionnaireAdvisor()
    q = advisor.infer(
        dataset_id="ds_users",
        dataset_name="app_session_engagement",
        column_names=["user_id", "session_duration", "clicks", "converted"],
        target_column="converted",
    )
    assert q.temporal_structure == TemporalStructure.GROUPED_COHORTS
    assert q.split_strategy_recommendation == "group_kfold"
    assert "groupkfold" in q.reasoning.lower() or "group" in q.reasoning.lower()


def test_advisor_heuristics_imbalanced_profile():
    advisor = QuestionnaireAdvisor()
    q = advisor.infer(
        dataset_id="ds_imb",
        dataset_name="sample_data",
        column_names=["feat1", "feat2", "target"],
        target_column="target",
        profile_summary={"target_rate": 0.03},
    )
    assert q.imbalance_strategy == "balanced_weights"
    assert q.primary_metric_override == "pr_auc"
    assert "imbalanced" in q.reasoning.lower()


def test_advisor_fallback_on_unrecognized():
    advisor = QuestionnaireAdvisor()
    q = advisor.infer(
        dataset_id="ds_plain",
        dataset_name="generic_benchmark",
        column_names=["x1", "x2", "y"],
        target_column="y",
    )
    assert q.domain_hint == "general_tabular"
    assert q.error_cost == ErrorCostPriority.BALANCED
    assert q.temporal_structure == TemporalStructure.CROSS_SECTIONAL
    assert q.split_strategy_recommendation == "stratified_kfold"


def test_advisor_llm_inference_and_fallback():
    class DummyLLMResponse:
        def __init__(self, data: dict):
            self.parsed = data
            self.content = json.dumps(data)

    class DummyLLMProvider:
        def __init__(self, succeed: bool = True):
            self.succeed = succeed

        def generate(self, prompt, system_prompt, response_schema):
            if not self.succeed:
                raise RuntimeError("LLM service unavailable")
            return DummyLLMResponse({
                "domain_hint": "cybersecurity_intrusion",
                "error_cost": "avoid_false_negatives",
                "temporal_structure": "sequential_time_series",
                "latency_constraint": "ultra_low_realtime",
                "explainability": "moderate",
                "imbalance_strategy": "focal_loss",
                "primary_metric_override": "f1_score",
                "split_strategy_recommendation": "time_series_split",
                "confidence": 0.98,
                "reasoning": "Network intrusion detected via sequential packet flow.",
            })

    # Successful LLM provider
    adv_llm = QuestionnaireAdvisor(llm_provider=DummyLLMProvider(succeed=True))
    q_llm = adv_llm.infer(
        dataset_id="ds_sec",
        dataset_name="firewall_logs",
        column_names=["bytes_sent", "duration", "attack"],
        target_column="attack",
    )
    assert q_llm.domain_hint == "cybersecurity_intrusion"
    assert q_llm.confidence == 0.98
    assert q_llm.error_cost == ErrorCostPriority.AVOID_FALSE_NEGATIVES

    # Failing LLM provider falls back to heuristics
    adv_fail = QuestionnaireAdvisor(llm_provider=DummyLLMProvider(succeed=False))
    q_fallback = adv_fail.infer(
        dataset_id="ds_fail",
        dataset_name="customer_churn",
        column_names=["tenure", "churn"],
        target_column="churn",
    )
    assert q_fallback.domain_hint == "customer_churn"
    assert q_fallback.error_cost == ErrorCostPriority.AVOID_FALSE_NEGATIVES


# ---------------------------------------------------------------------------
# 3. Repository Persistence Tests
# ---------------------------------------------------------------------------


def test_sqlite_repository_questionnaire_crud(tmp_path: Path):
    db_path = str(tmp_path / "automl.db")
    repo = SQLiteExperimentRepository(db_path=db_path)

    assert repo.get_dataset_questionnaire("missing_ds") is None

    q = DatasetQuestionnaire(
        dataset_id="ds_persist",
        domain_hint="customer_churn",
        error_cost=ErrorCostPriority.AVOID_FALSE_NEGATIVES,
        temporal_structure=TemporalStructure.CROSS_SECTIONAL,
        latency_constraint=LatencyConstraint.STANDARD_INTERACTIVE,
        explainability=ExplainabilityLevel.MODERATE,
        primary_metric_override="pr_auc",
    )
    repo.save_dataset_questionnaire(q)

    loaded = repo.get_dataset_questionnaire("ds_persist")
    assert loaded is not None
    assert loaded.dataset_id == "ds_persist"
    assert loaded.domain_hint == "customer_churn"
    assert loaded.error_cost == ErrorCostPriority.AVOID_FALSE_NEGATIVES
    assert loaded.primary_metric_override == "pr_auc"

    # Update questionnaire
    q.domain_hint = "custom_enterprise_churn"
    q.confidence = 0.99
    repo.save_dataset_questionnaire(q)

    updated = repo.get_dataset_questionnaire("ds_persist")
    assert updated.domain_hint == "custom_enterprise_churn"
    assert updated.confidence == 0.99


# ---------------------------------------------------------------------------
# 4. Workspace Integration Tests
# ---------------------------------------------------------------------------


def test_workspace_questionnaire_lifecycle(tmp_path: Path):
    ws, _, _ = build_application(root_dir=str(tmp_path / "ws"))

    # Create dummy data
    csv_file = tmp_path / "churn_data.csv"
    pd.DataFrame({
        "tenure": [1, 12, 24, 36],
        "monthly_charges": [20.0, 50.0, 70.0, 90.0],
        "churn": [1, 0, 0, 0],
    }).to_csv(csv_file, index=False)

    ds = ws.register_dataset(
        name="telecom_churn",
        path=str(csv_file),
        target="churn",
        task_type="binary_classification",
    )

    # get_dataset_questionnaire with auto_generate=True generates and saves
    q = ws.get_dataset_questionnaire(ds.id)
    assert q is not None
    assert q.dataset_id == ds.id
    assert q.domain_hint == "customer_churn"
    assert q.error_cost == ErrorCostPriority.AVOID_FALSE_NEGATIVES

    # Modifying questionnaire and saving
    q.latency_constraint = LatencyConstraint.ULTRA_LOW_REALTIME
    q.auto_generated = False
    ws.save_dataset_questionnaire(q)

    reloaded = ws.get_dataset_questionnaire(ds.id, auto_generate=False)
    assert reloaded is not None
    assert reloaded.latency_constraint == LatencyConstraint.ULTRA_LOW_REALTIME
    assert reloaded.auto_generated is False


# ---------------------------------------------------------------------------
# 5. Web API HTTP Endpoints Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def running_test_server(tmp_path: Path):
    workspace_dir = str(tmp_path / "web_ws")
    ws, _, _ = build_application(root_dir=workspace_dir)

    csv_file = tmp_path / "customer_churn.csv"
    pd.DataFrame({
        "tenure": [1, 5, 20, 50],
        "churn": [1, 0, 1, 0],
    }).to_csv(csv_file, index=False)

    ds = ws.register_dataset(
        name="customer_churn",
        path=str(csv_file),
        target="churn",
        task_type="binary_classification",
    )

    AutoMLWebHandler.workspace_dir = workspace_dir
    server = ThreadingHTTPServer(("127.0.0.1", 0), AutoMLWebHandler)
    host, port = server.server_address

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://{host}:{port}"
    yield {"base_url": base_url, "dataset_id": ds.id}

    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


def test_web_api_questionnaire_get_and_post(running_test_server):
    base = running_test_server["base_url"]
    dataset_id = running_test_server["dataset_id"]

    # 1. GET /api/dataset/questionnaire (auto-generates)
    with urlopen(f"{base}/api/dataset/questionnaire?dataset_id={dataset_id}") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["dataset_id"] == dataset_id
        assert data["domain_hint"] == "customer_churn"
        assert data["error_cost"] == "avoid_false_negatives"

    # 2. POST /api/dataset/questionnaire (saves human modifications)
    post_payload = {
        "dataset_id": dataset_id,
        "questionnaire": {
            "dataset_id": dataset_id,
            "domain_hint": "telecom_b2b_churn",
            "error_cost": "custom_cost_matrix",
            "temporal_structure": "cross_sectional",
            "latency_constraint": "batch_offline",
            "explainability": "highly_regulated",
            "primary_metric_override": "pr_auc",
            "split_strategy_recommendation": "stratified_kfold",
            "confidence": 1.0,
            "reasoning": "Reviewed and tailored for enterprise tier contracts.",
        },
    }
    req = Request(
        f"{base}/api/dataset/questionnaire",
        data=json.dumps(post_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req) as resp:
        assert resp.status == 200
        result = json.loads(resp.read().decode("utf-8"))
        assert result["status"] == "success"
        assert result["questionnaire"]["domain_hint"] == "telecom_b2b_churn"
        assert result["questionnaire"]["error_cost"] == "custom_cost_matrix"
        assert result["questionnaire"]["latency_constraint"] == "batch_offline"

    # 3. GET again verifies updated persistence
    with urlopen(f"{base}/api/dataset/questionnaire?dataset_id={dataset_id}") as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data["domain_hint"] == "telecom_b2b_churn"
        assert data["error_cost"] == "custom_cost_matrix"

    # 4. POST /api/dataset/questionnaire/generate regenerates heuristic recommendations
    gen_req = Request(
        f"{base}/api/dataset/questionnaire/generate",
        data=json.dumps({"dataset_id": dataset_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(gen_req) as resp:
        assert resp.status == 200
        gen_res = json.loads(resp.read().decode("utf-8"))
        assert gen_res["status"] == "success"
        assert gen_res["questionnaire"]["domain_hint"] == "customer_churn"
