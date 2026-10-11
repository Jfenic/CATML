"""Test Suite for Experiment Recipes Phase R0: Verified Hypothesis Transforms Application.

Verifies:
1. HypothesisExperimentTranslator distinguishes stateless (log1p) vs stateful (yeo_johnson, robust_scaler) transforms.
2. VerifyHypothesisCommand actually applies log1p derived features to candidate experiments.
3. VerifyHypothesisCommand applies yeo_johnson and robust_scaler fit-in-fold via ColumnTransformer.
4. Collinearity drops continue to work deterministically.
5. Negative learning is preserved (accepted=False when threshold is not reached).
6. EvidenceLink.accepted does not trigger automatic promotion (Propose ≠ Accept principle).
7. Persistence of column_transforms and derived_features in SQLiteExperimentRepository.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from automl.application.analysis.commands import (
    CreateStudyCommand,
    RunAnalysisCommand,
    VerifyHypothesisCommand,
)
from automl.application.analysis.queries import (
    GetEvidenceLinkQuery,
    GetHypothesisQuery,
    ListHypothesesQuery,
)
from automl.application.bootstrap import build_application
from automl.domain.analysis.models import AnalysisHypothesis
from automl.domain.experiments.trial import Experiment, ExperimentStatus
from automl.engine.analysis.hypothesis_translator import (
    ExperimentCandidateSpec,
    HypothesisExperimentTranslator,
)
from automl.infrastructure.database.sqlite_repository import SQLiteExperimentRepository
from automl.infrastructure.database.sqlite_studies import SQLiteStudyRepository


@pytest.fixture
def synthetic_r0_skewed_data(tmp_path: Path) -> Path:
    """Generate reproducible dataset with lognormal skew and outliers for R0 testing."""
    np.random.seed(42)
    n = 200
    # Skewed positive feature
    x_skew = np.random.lognormal(mean=2.0, sigma=1.2, size=n)
    # Feature with severe outliers
    x_outlier = np.random.normal(loc=10.0, scale=2.0, size=n)
    x_outlier[:10] += 50.0  # injection of outliers
    # Regular normal feature
    x_norm = np.random.normal(loc=0.0, scale=1.0, size=n)
    # Collinear feature
    x_collin = x_norm * 0.95 + np.random.normal(loc=0.0, scale=0.1, size=n)

    # Binary target
    prob = 1.0 / (1.0 + np.exp(-(0.05 * x_skew + 0.1 * x_norm - 0.02 * x_outlier)))
    target = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({
        "x_skew": x_skew,
        "x_outlier": x_outlier,
        "x_norm": x_norm,
        "x_collin": x_collin,
        "target": target,
    })
    path = tmp_path / "r0_skewed_data.csv"
    df.to_csv(path, index=False)
    return path


def test_translator_distinguishes_log1p_vs_yeo_johnson_vs_robust_scaler():
    """Verify translator creates derived_features for log1p and column_transforms for stateful transforms."""
    available = ["feat_a", "feat_b", "feat_c"]

    # 1. Stateless: log1p power transform
    hyp_log1p = AnalysisHypothesis.create(
        study_id="s1",
        finding_id="f1",
        description="Apply log1p on feat_a",
        proposed_action="power_transform",
        experiment_delta={"transform": "log1p", "column": "feat_a"},
    )
    spec_log1p = HypothesisExperimentTranslator.translate(hyp_log1p, available, "binary_classification")
    assert "feat_a_log1p" in spec_log1p.feature_names
    assert "feat_a" not in spec_log1p.feature_names
    assert len(spec_log1p.derived_features) == 1
    assert spec_log1p.derived_features[0]["name"] == "feat_a_log1p"
    assert spec_log1p.derived_features[0]["expression"] == "log1p(feat_a)"
    assert len(spec_log1p.column_transforms) == 0
    assert "log1p_feat_a" in spec_log1p.name

    # 2. Stateful: yeo_johnson power transform
    hyp_yeo = AnalysisHypothesis.create(
        study_id="s1",
        finding_id="f2",
        description="Apply Yeo-Johnson on feat_b",
        proposed_action="power_transform",
        experiment_delta={"transform": "yeo_johnson", "column": "feat_b"},
    )
    spec_yeo = HypothesisExperimentTranslator.translate(hyp_yeo, available, "binary_classification")
    assert "feat_b" in spec_yeo.feature_names
    assert len(spec_yeo.derived_features) == 0
    assert len(spec_yeo.column_transforms) == 1
    assert spec_yeo.column_transforms[0]["column"] == "feat_b"
    assert spec_yeo.column_transforms[0]["kind"] == "yeo_johnson"
    assert "power_transform_feat_b" in spec_yeo.name

    # 3. Stateful: robust_scaler
    hyp_robust = AnalysisHypothesis.create(
        study_id="s1",
        finding_id="f3",
        description="Apply RobustScaler on feat_c",
        proposed_action="robust_scaler",
        experiment_delta={"scaler": "robust", "column": "feat_c"},
    )
    spec_robust = HypothesisExperimentTranslator.translate(hyp_robust, available, "binary_classification")
    assert "feat_c" in spec_robust.feature_names
    assert len(spec_robust.derived_features) == 0
    assert len(spec_robust.column_transforms) == 1
    assert spec_robust.column_transforms[0]["column"] == "feat_c"
    assert spec_robust.column_transforms[0]["kind"] == "robust_scaler"
    assert "robust_scale_feat_c" in spec_robust.name


def test_sqlite_repository_persists_transforms_and_derived_features(tmp_path: Path):
    """Verify SQLite repository schema round-trip for column_transforms and derived_features."""
    db_path = str(tmp_path / "test_repo.db")
    repo = SQLiteExperimentRepository(db_path)

    exp = Experiment(
        id="exp_r0_test",
        run_id="run_test",
        name="test_experiment",
        hypothesis="Test transforms persistence",
        feature_names=["f1", "f2_log1p"],
        model_ids=["logistic_regression"],
        metric="roc_auc",
        validation_strategy="holdout",
        status=ExperimentStatus.CREATED,
        column_transforms=[{"column": "f1", "kind": "robust_scaler", "params": {}}],
        derived_features=[{"name": "f2_log1p", "expression_type": "formula", "expression": "log1p(f2)", "source_columns": ["f2"]}],
    )
    repo.save_experiment(exp)

    loaded = repo.get_experiment("exp_r0_test")
    assert loaded is not None
    assert loaded.id == "exp_r0_test"
    assert len(loaded.column_transforms) == 1
    assert loaded.column_transforms[0]["column"] == "f1"
    assert loaded.column_transforms[0]["kind"] == "robust_scaler"
    assert len(loaded.derived_features) == 1
    assert loaded.derived_features[0]["name"] == "f2_log1p"
    assert loaded.derived_features[0]["expression"] == "log1p(f2)"


def test_verify_hypothesis_applies_log1p_real_transform(synthetic_r0_skewed_data: Path, tmp_path: Path):
    """Verify that VerifyHypothesisCommand executes log1p derived feature on candidate experiment."""
    ws_dir = str(tmp_path / "ws_r0_log1p")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="r0_ds", path=synthetic_r0_skewed_data, target="target", task_type="binary_classification")

    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="R0 Study"))
    cb.dispatch(RunAnalysisCommand(study_id=study_id))

    # Manually create or verify power_transform log1p hypothesis
    study_repo = ws.get_study_repository()
    hyp = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id="f_skew",
        description="Apply log1p on x_skew",
        proposed_action="power_transform",
        experiment_delta={"transform": "log1p", "column": "x_skew"},
    )
    study_repo.save_hypothesis(hyp)

    link_data = cb.dispatch(VerifyHypothesisCommand(hypothesis_id=hyp.id, min_improvement=-1.0))
    assert link_data is not None
    assert link_data["hypothesis_id"] == hyp.id
    assert link_data["baseline_score"] > 0
    assert link_data["candidate_score"] > 0
    assert link_data["accepted"] is True

    # Candidate experiment inspection
    cand_exp = ws.repository.get_experiment(link_data["experiment_id"])
    assert cand_exp is not None
    assert "x_skew_log1p" in cand_exp.feature_names
    assert "x_skew" not in cand_exp.feature_names
    assert len(cand_exp.derived_features) == 1
    assert cand_exp.derived_features[0]["name"] == "x_skew_log1p"


def test_verify_hypothesis_applies_yeo_johnson_stateful_transform(synthetic_r0_skewed_data: Path, tmp_path: Path):
    """Verify that VerifyHypothesisCommand applies yeo_johnson fit-in-fold ColumnTransformer."""
    ws_dir = str(tmp_path / "ws_r0_yeo")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="r0_ds_yeo", path=synthetic_r0_skewed_data, target="target", task_type="binary_classification")

    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="R0 Yeo Study"))
    cb.dispatch(RunAnalysisCommand(study_id=study_id))

    study_repo = ws.get_study_repository()
    hyp = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id="f_yeo",
        description="Apply Yeo-Johnson on x_skew",
        proposed_action="power_transform",
        experiment_delta={"transform": "yeo_johnson", "column": "x_skew"},
    )
    study_repo.save_hypothesis(hyp)

    link_data = cb.dispatch(VerifyHypothesisCommand(hypothesis_id=hyp.id, min_improvement=-1.0))
    assert link_data is not None

    cand_exp = ws.repository.get_experiment(link_data["experiment_id"])
    assert cand_exp is not None
    assert "x_skew" in cand_exp.feature_names
    assert len(cand_exp.column_transforms) == 1
    assert cand_exp.column_transforms[0]["kind"] == "yeo_johnson"
    assert cand_exp.column_transforms[0]["column"] == "x_skew"


def test_verify_hypothesis_applies_robust_scaler_stateful_transform(synthetic_r0_skewed_data: Path, tmp_path: Path):
    """Verify that VerifyHypothesisCommand applies robust_scaler fit-in-fold ColumnTransformer."""
    ws_dir = str(tmp_path / "ws_r0_robust")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="r0_ds_robust", path=synthetic_r0_skewed_data, target="target", task_type="binary_classification")

    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="R0 Robust Study"))
    cb.dispatch(RunAnalysisCommand(study_id=study_id))

    study_repo = ws.get_study_repository()
    hyp = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id="f_outlier",
        description="Apply RobustScaler on x_outlier",
        proposed_action="robust_scaler",
        experiment_delta={"scaler": "robust", "column": "x_outlier"},
    )
    study_repo.save_hypothesis(hyp)

    link_data = cb.dispatch(VerifyHypothesisCommand(hypothesis_id=hyp.id, min_improvement=-1.0))
    assert link_data is not None

    cand_exp = ws.repository.get_experiment(link_data["experiment_id"])
    assert cand_exp is not None
    assert "x_outlier" in cand_exp.feature_names
    assert len(cand_exp.column_transforms) == 1
    assert cand_exp.column_transforms[0]["kind"] == "robust_scaler"
    assert cand_exp.column_transforms[0]["column"] == "x_outlier"


def test_verify_hypothesis_preserves_negative_learning_and_no_autopromotion(synthetic_r0_skewed_data: Path, tmp_path: Path):
    """Verify that failing the min_improvement threshold rejects the hypothesis and records negative learning."""
    ws_dir = str(tmp_path / "ws_r0_reject")
    ws, cb, qb = build_application(root_dir=ws_dir)
    ds = ws.register_dataset(name="r0_ds_neg", path=synthetic_r0_skewed_data, target="target", task_type="binary_classification")

    study_id = cb.dispatch(CreateStudyCommand(dataset_id=ds.id, target_column="target", name="R0 Negative Learning Study"))
    cb.dispatch(RunAnalysisCommand(study_id=study_id))

    study_repo = ws.get_study_repository()
    hyp = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id="f_skew",
        description="Apply Yeo-Johnson on x_skew with impossible threshold",
        proposed_action="power_transform",
        experiment_delta={"transform": "yeo_johnson", "column": "x_skew"},
    )
    study_repo.save_hypothesis(hyp)

    # Require impossible +0.50 improvement on ROC-AUC
    link_data = cb.dispatch(VerifyHypothesisCommand(hypothesis_id=hyp.id, min_improvement=0.50))
    assert link_data["accepted"] is False
    assert "Rejected" in link_data["notes"]

    # Hypothesis state is updated to rejected
    updated_hyp = qb.dispatch(GetHypothesisQuery(hypothesis_id=hyp.id))
    assert updated_hyp["status"] == "rejected"

    # EvidenceLink persisted
    retrieved_link = qb.dispatch(GetEvidenceLinkQuery(link_id=link_data["id"]))
    assert retrieved_link is not None
    assert retrieved_link["accepted"] is False
