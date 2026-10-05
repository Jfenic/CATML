from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.artifacts.model_artifact import ModelArtifact
from automl.domain.modalities.modality import Modality
from automl.domain.problems import (
    BackendCapabilities,
    EvaluationResult,
    ExecutionPolicy,
    ProblemSpec,
    TabularSource,
    TargetSpec,
    TextSource,
    ValidationSpec,
)
from automl.domain.tasks.task_type import TaskType
from automl.engine.features.text import LightweightTextExtractor, is_text_column
from automl.engine.profiling.dataset_profiler import profile_dataset, Dataset
from automl.facade import AutoML


def test_problem_spec_and_capabilities_compatibility():
    prob = ProblemSpec(
        inputs=[
            TabularSource(columns=["age", "balance"]),
            TextSource(columns=["review"], max_features=40),
        ],
        target=TargetSpec(column="churn", task_type=TaskType.BINARY_CLASSIFICATION),
        validation=ValidationSpec(strategy="stratified_kfold", folds=5),
        policy=ExecutionPolicy(time_budget_seconds=60),
        name="churn_multimodal",
    )

    assert prob.name == "churn_multimodal"
    assert prob.modalities == {Modality.TABULAR, Modality.TEXT}
    assert prob.tabular_columns == ["age", "balance"]
    assert prob.text_columns == ["review"]
    assert prob.all_feature_columns == ["age", "balance", "review"]

    # Tabular + Text capable backend
    cap_multi = BackendCapabilities(
        backend_id="multimodal_blender",
        supported_modalities=frozenset({Modality.TABULAR, Modality.TEXT}),
        supported_tasks=frozenset({"binary_classification", "regression"}),
    )
    can_handle, reason = cap_multi.can_handle(prob)
    assert can_handle is True
    assert reason == "Compatible."

    # Pure tabular backend
    cap_tabular = BackendCapabilities(
        backend_id="lightgbm_native",
        supported_modalities=frozenset({Modality.TABULAR}),
        supported_tasks=frozenset({"binary_classification"}),
    )
    can_handle_tab, reason_tab = cap_tabular.can_handle(prob)
    assert can_handle_tab is False
    assert "modality 'text'" in reason_tab.lower()


def test_evaluation_result_heterogeneous_metrics():
    res = EvaluationResult(
        primary_metric="roc_auc",
        primary_score=0.9125,
        secondary_metrics={"f1": 0.88, "accuracy": 0.89},
        resource_metrics={"duration_s": 2.45, "vram_mb": 0.0, "latency_ms": 12.3},
    )

    d = res.to_dict()
    assert d["primary_metric"] == "roc_auc"
    assert d["primary_score"] == 0.9125
    assert d["secondary_metrics"]["f1"] == 0.88
    assert d["resource_metrics"]["latency_ms"] == 12.3


def test_is_text_column_heuristic():
    # Freeform natural language text
    text_s = pd.Series([
        "Customer complained about intermittent wifi signal and billing error in March.",
        "Everything was completely fine and the technician arrived on time.",
        "Very frustrated with long wait times on the phone support queue.",
        "High quality internet connection with exceptional bandwidth.",
    ] * 20)
    assert is_text_column(text_s) is True

    # Categorical string codes
    cat_s = pd.Series(["US", "FR", "ES", "DE"] * 25)
    assert is_text_column(cat_s) is False

    # Numeric series
    num_s = pd.Series([10.5, 20.3, 15.2, 8.4] * 25)
    assert is_text_column(num_s) is False


def test_lightweight_text_extractor_transform():
    extractor = LightweightTextExtractor(max_features=10, ngram_range=(1, 2), column_prefix="feedback")
    df_text = pd.DataFrame({
        "feedback": [
            "Great customer service and fast delivery",
            "Terrible slow delivery and bad customer support",
            "Fast service and amazing quality",
            np.nan,
        ]
    })

    extractor.fit(df_text)
    out = extractor.transform(df_text)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 4
    assert out.shape[1] > 0

    feature_names = extractor.get_feature_names_out()
    assert len(feature_names) == out.shape[1]
    assert all(fn.startswith("feedback_tfidf_") for fn in feature_names)


def test_profiler_detects_text_feature():
    df = pd.DataFrame({
        "customer_id": [f"ID_{i:04d}" for i in range(60)],
        "age": np.random.randint(18, 70, size=60),
        "notes": [
            f"Note {i}: Customer reported issues with service {i % 5} and requested ticket callback"
            for i in range(60)
        ],
        "churn": np.random.randint(0, 2, size=60),
    })

    dataset = Dataset(
        id="ds_text",
        workspace_id="ws_1",
        name="Text Test",
        path="",
        target_column="churn",
        task_type="binary_classification",
    )

    profile = profile_dataset(dataset, df)
    notes_col = next(c for c in profile.columns if c.name == "notes")
    assert notes_col.is_text is True
    assert notes_col.is_identifier is False
    assert "notes" in profile.text_column_names

    # Ensure recommendations include NLP text badge
    text_recs = [r for r in profile.recommendations if r.get("badge") == "Text Feature"]
    assert len(text_recs) == 1
    assert text_recs[0]["column"] == "notes"


def test_automl_fit_with_tabular_and_text_columns(tmp_path: Path):
    df_train = pd.DataFrame({
        "age": [25, 45, 35, 52, 28, 62, 38, 41, 29, 55],
        "balance": [1200.0, 4500.0, 2300.0, 8900.0, 950.0, 7200.0, 3100.0, 5400.0, 1100.0, 6800.0],
        "feedback": [
            "Super satisfied with the quick answers and excellent support",
            "Terrible customer service with long unresolved downtime",
            "Very good experience overall and clean billing",
            "Horrible experience representative was unhelpful and rude",
            "Great platform very easy to use and intuitive",
            "Cancelled subscription due to unexpected fee increases",
            "Fast technical resolution highly recommend this service",
            "Frustrated with constant connection dropouts every night",
            "Helpful customer support resolved my issue in minutes",
            "Slow connection and bad customer service will switch",
        ],
        "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    automl = AutoML(
        task="classification",
        cv_folds=2,
        models=["logistic_regression"],
        random_state=42,
    )

    result = automl.fit(df_train, target="churn", text_columns=["feedback"])

    assert result.best_model is not None
    assert isinstance(result.best_model, ModelArtifact)
    assert result.best_score > 0.5

    # Test inference on new data with text column
    df_test = pd.DataFrame({
        "age": [30, 58],
        "balance": [2000.0, 7500.0],
        "feedback": [
            "Super satisfied with quick support",
            "Terrible customer service and bad experience",
        ],
    })

    preds = result.predict(df_test)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})

    probs = result.predict_proba(df_test)
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # Save and reload artifact
    save_path = tmp_path / "text_tabular_model.pkl"
    result.save_model(save_path)
    assert save_path.exists()

    loaded_artifact = ModelArtifact.load(save_path)
    loaded_preds = loaded_artifact.predict(df_test)
    assert np.array_equal(preds, loaded_preds)

    # Verify provenance and description
    desc = loaded_artifact.describe()
    assert "provenance" in desc
    assert desc["provenance"]["catml_version"] == "0.7.0"
    assert "dependencies" in desc["provenance"]
