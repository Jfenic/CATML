"""
Tests for Fase 3 (v0.9.x): Anti-Leakage Guardian Avanzado (Group & Entity Leakage).
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl.domain.datasets.profile import Dataset, DatasetProfile
from automl.domain.experiments.trial import Experiment, Trial
from automl.domain.ports import TrialExecution
from automl.domain.runs.run import AutoMLRun, RunConfig
from automl.engine.profiling.dataset_profiler import (
    detect_group_leakage,
    detect_is_group_candidate,
    profile_dataset,
)
from automl.engine.training.sklearn_trainer import SklearnTrainer
from automl.facade import AutoML


def test_detect_is_group_candidate():
    n = 100
    # Patient ID with 20 unique patients (5 rows per patient)
    patient_series = pd.Series([f"PAT_{i % 20:03d}" for i in range(n)])
    assert detect_is_group_candidate("patient_id", patient_series, n, target_column="target") is True
    assert detect_is_group_candidate("subject", patient_series, n, target_column="target") is True
    assert detect_is_group_candidate("hospital_center", patient_series, n, target_column="target") is True

    # User ID with 25 unique users
    user_series = pd.Series([f"USR_{i % 25:03d}" for i in range(n)])
    assert detect_is_group_candidate("user_id", user_series, n, target_column="target") is True

    # 1-to-1 unique row identifier (no repeating groups)
    unique_ids = pd.Series([f"ROW_{i:04d}" for i in range(n)])
    assert detect_is_group_candidate("row_id", unique_ids, n, target_column="target") is False
    assert detect_is_group_candidate("patient_id", unique_ids, n, target_column="target") is False

    # Target column should never be group candidate
    assert detect_is_group_candidate("target", patient_series, n, target_column="target") is False

    # Short dataset (< 10 rows)
    assert detect_is_group_candidate("patient_id", patient_series[:5], 5, target_column="target") is False


def test_detect_group_leakage_simulation():
    n = 100
    df = pd.DataFrame({
        "patient_id": [f"PAT_{i % 15:03d}" for i in range(n)],
        "measurement": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n),
    })

    report = detect_group_leakage(df, "patient_id", test_size=0.2, random_seed=42)
    assert report is not None
    assert report["column"] == "patient_id"
    assert report["leakage_detected"] is True
    assert report["overlapping_groups_count"] > 0
    assert report["affected_rows_count"] > 0
    assert report["strategy_recommendation"] == "GroupKFold(patient_id)"
    assert "Group entity leakage detected" in report["description"]


def test_profile_dataset_detects_group_leakage(tmp_path: Path):
    n = 100
    df = pd.DataFrame({
        "patient_id": [f"PAT_{i % 20:03d}" for i in range(n)],
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
        "target": np.random.choice([0, 1], size=n),
    })
    csv_file = tmp_path / "clinical_trial.csv"
    df.to_csv(csv_file, index=False)

    dataset = Dataset(
        id="ds_clinical",
        workspace_id="ws_clinical",
        name="Clinical Trial",
        path=str(csv_file),
        target_column="target",
        task_type="binary_classification",
    )

    profile = profile_dataset(dataset, df=df)

    # Group candidate and leakage properties
    assert profile.has_group_leakage is True
    assert profile.has_leakage is True
    assert "patient_id" in profile.group_candidates
    assert "patient_id" in profile.leakage_column_names

    # Recommended features must exclude patient_id
    assert "patient_id" not in profile.recommended_feature_names
    assert "feature_a" in profile.recommended_feature_names
    assert "feature_b" in profile.recommended_feature_names

    # Group leakage recommendation details
    group_recs = [r for r in profile.recommendations if r.get("badge") == "Group Leakage"]
    assert len(group_recs) == 1
    rec = group_recs[0]
    assert rec["column"] == "patient_id"
    assert rec["severity"] == "danger"
    assert rec["action"] == "enforce_group_split"
    assert rec["strategy"] == "GroupKFold(patient_id)"

    # Serialization
    d = profile.to_dict()
    assert d["has_group_leakage"] is True
    assert d["group_candidates"] == ["patient_id"]
    assert len(d["group_leakage_reports"]) == 1
    assert d["group_leakage_reports"][0]["column"] == "patient_id"


def test_sklearn_trainer_group_kfold_execution(tmp_path: Path):
    n = 120
    csv_file = tmp_path / "device_telemetry.csv"
    df = pd.DataFrame({
        "device_id": [f"DEV_{i % 12:02d}" for i in range(n)],
        "temp": np.random.randn(n),
        "vibration": np.random.randn(n),
        "failure": np.random.choice([0, 1], size=n),
    })
    df.to_csv(csv_file, index=False)

    trial = Trial(id="trial_grp_1", experiment_id="exp_grp_1", model_id="logistic_regression")
    exp = Experiment(
        id="exp_grp_1",
        run_id="run_grp_1",
        name="Group CV Experiment",
        hypothesis="Test GroupKFold isolation",
        feature_names=["temp", "vibration"],
        model_ids=["logistic_regression"],
        metric="accuracy",
        validation_strategy="group_kfold",
        group_column="device_id",
    )
    run = AutoMLRun(
        id="run_grp_1",
        workspace_id="ws_grp",
        dataset_id="ds_grp",
        config=RunConfig(
            task_type="binary_classification",
            target="failure",
            metric="accuracy",
            validation_strategy="group_kfold",
            group_column="device_id",
        ),
    )

    execution = TrialExecution(
        trial=trial,
        experiment=exp,
        run=run,
        feature_names=["temp", "vibration"],
        dataset_path=str(csv_file),
        target_column="failure",
        task_type="binary_classification",
        metric="accuracy",
        validation_strategy="group_kfold",
        test_size=0.2,
        cv_folds=5,
        random_seed=42,
        group_column="device_id",
    )

    trainer = SklearnTrainer()
    result = trainer.run(execution)

    assert result.succeeded is True
    assert 0.0 <= result.primary_score <= 1.0
    assert result.artifacts["group_column"] == "device_id"
    assert result.secondary_metrics["n_groups"] == 12
    assert "cv_std" in result.secondary_metrics
    assert "cv_scores" in result.artifacts
    assert len(result.artifacts["cv_scores"]) == 5


def test_automl_fit_explicit_group_column(tmp_path: Path):
    n = 100
    df = pd.DataFrame({
        "patient_id": [f"P_{i % 10}" for i in range(n)],
        "age": np.random.uniform(20, 80, n),
        "blood_pressure": np.random.uniform(80, 160, n),
        "target": np.random.choice([0, 1], size=n),
    })

    automl = AutoML(
        task="binary_classification",
        models=["logistic_regression"],
        workspace_dir=tmp_path / "ws_fit_group",
    )
    result = automl.fit(df, target="target", group_column="patient_id")

    assert result.best_model is not None
    assert result.task_type == "binary_classification"
    leaderboard = result.leaderboard()
    assert len(leaderboard) >= 1

    # Predict with new observations
    X_new = pd.DataFrame({
        "age": [45.0, 60.0],
        "blood_pressure": [120.0, 140.0],
    })
    preds = result.predict(X_new)
    assert len(preds) == 2


def test_automl_fit_auto_enforces_detected_group_leakage(tmp_path: Path):
    n = 100
    # Dataset where user did NOT explicitly specify group_column, but patient_id is present
    df = pd.DataFrame({
        "patient_id": [f"PAT_{i % 15:03d}" for i in range(n)],
        "biomarker_a": np.random.randn(n),
        "biomarker_b": np.random.randn(n),
        "outcome": np.random.choice([0, 1], size=n),
    })

    automl = AutoML(
        models=["logistic_regression"],
        workspace_dir=tmp_path / "ws_auto_protect",
    )
    result = automl.fit(df, target="outcome")

    assert result.best_model is not None
    # Verify patient_id was not included in model training features
    assert "patient_id" not in result.best_model.feature_names
    assert "biomarker_a" in result.best_model.feature_names
    assert "biomarker_b" in result.best_model.feature_names
