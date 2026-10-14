"""Audit: export membership, derived predict, stateful apply block, trial-gated infer, nested FS."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl.application.services.workspace import AutoMLWorkspace
from automl.artifacts.model_artifact import ModelArtifact
from automl.domain.features.selection_strategy import FeatureSelectionStrategy
from automl.engine.training.sklearn_trainer import _build_pipeline
from sklearn.linear_model import LogisticRegression


def _binary_ws(tmp_path: Path, n: int = 60, n_noise: int = 0):
    ws = AutoMLWorkspace.create("rel", root_dir=tmp_path / "ws")
    rng = np.random.default_rng(0)
    data = {
        "num1": rng.normal(size=n),
        "num2": np.abs(rng.normal(size=n)) + 0.1,
        "target": [0, 1] * (n // 2),
    }
    for i in range(n_noise):
        data[f"noise_{i}"] = rng.normal(size=n)
    df = pd.DataFrame(data)
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    ds = ws.register_dataset(
        name="d", path=path, target="target", task_type="binary_classification"
    )
    run = ws.create_run(ds, metric="roc_auc")
    return ws, run, ds, path, df


def test_export_rejects_trial_from_other_experiment(tmp_path: Path):
    ws, run, _, _, _ = _binary_ws(tmp_path)
    exp_a = ws.create_experiment(
        run=run,
        name="a",
        feature_names=["num1", "num2"],
        model_ids=["logistic_regression"],
    )
    exp_b = ws.create_experiment(
        run=run,
        name="b",
        feature_names=["num1", "num2"],
        model_ids=["logistic_regression"],
    )
    res_a = ws.run_experiment(run, exp_a)
    res_b = ws.run_experiment(run, exp_b)
    assert any(r.succeeded for r in res_a)
    assert any(r.succeeded for r in res_b)
    trial_b = next(r.trial_id for r in res_b if r.succeeded)

    with pytest.raises(ValueError, match="belongs to experiment"):
        ws.export_model_artifact(run_id=run.id, experiment_id=exp_a.id, trial_id=trial_b)


def test_export_rejects_unevaluated_experiment(tmp_path: Path):
    ws, run, _, _, _ = _binary_ws(tmp_path)
    exp = ws.create_experiment(
        run=run,
        name="uneval",
        feature_names=["num1", "num2"],
        model_ids=["logistic_regression"],
    )
    with pytest.raises(ValueError, match="no successful trials"):
        ws.export_model_artifact(run_id=run.id, experiment_id=exp.id)


def test_predict_rejects_experiment_without_successful_trial(tmp_path: Path):
    ws, run, _, path, _ = _binary_ws(tmp_path)
    exp = ws.create_experiment(
        run=run,
        name="uneval",
        feature_names=["num1", "num2"],
        model_ids=["logistic_regression"],
    )
    with pytest.raises(ValueError, match="no successful trials|unevaluated"):
        ws.predict(run_id=run.id, test_dataset_path=path, experiment_id=exp.id)


def test_artifact_predict_applies_derived_from_original_columns(tmp_path: Path):
    ws, run, _, _, df = _binary_ws(tmp_path)
    derived = [
        {"name": "num2_log1p", "expression_type": "formula", "expression": "log1p(num2)"}
    ]
    exp = ws.create_experiment(
        run=run,
        name="derived",
        feature_names=["num1", "num2", "num2_log1p"],
        model_ids=["logistic_regression"],
        derived_features=derived,
    )
    results = ws.run_experiment(run, exp)
    assert any(r.succeeded for r in results)

    artifact = ws.export_model_artifact(run_id=run.id, experiment_id=exp.id)
    out = tmp_path / "model.pkl"
    artifact.save(out)
    loaded = ModelArtifact.load(out)

    original_only = df[["num1", "num2"]].head(5).copy()
    preds = loaded.predict(original_only)
    assert len(preds) == 5

    # Parity with materializing derived columns manually then predicting.
    with_derived = original_only.copy()
    with_derived["num2_log1p"] = np.log1p(with_derived["num2"])
    preds_manual = loaded.predict(with_derived)
    np.testing.assert_allclose(preds, preds_manual, rtol=1e-6, atol=1e-6)


def test_apply_derived_blocks_stateful_zscore_persist(tmp_path: Path):
    ws, _, ds, _, _ = _binary_ws(tmp_path)
    with pytest.raises(ValueError, match="stateful aggregate"):
        ws.apply_derived_feature(
            dataset_id=ds.id,
            name="z_num1",
            expression="zscore(num1)",
            expression_type="formula",
        )


def test_apply_then_cv_rowwise_formula_ok(tmp_path: Path):
    ws, run, ds, _, _ = _binary_ws(tmp_path)
    result, _ = ws.apply_derived_feature(
        dataset_id=ds.id,
        name="num2_log1p",
        expression="log1p(num2)",
        expression_type="formula",
    )
    assert result.is_valid
    exp = ws.create_experiment(
        run=run,
        name="after_apply",
        feature_names=["num1", "num2", "num2_log1p"],
        model_ids=["logistic_regression"],
        validation_strategy="kfold",
    )
    results = ws.run_experiment(run, exp)
    assert any(r.succeeded for r in results)


def test_supervised_select_candidates_are_nested_in_fold(tmp_path: Path):
    ws, run, _, _, _ = _binary_ws(tmp_path, n_noise=8)
    cands = ws.select_features(
        run.id,
        strategy=FeatureSelectionStrategy(
            methods=["mutual_information"],
            top_k=[3],
            include_reduction=False,
        ),
    )
    assert cands
    c = cands[0]
    assert c.metadata.get("evaluation_mode") == "nested_in_fold"
    assert c.metadata.get("supervised_selection_leakage_risk") is False
    assert len(c.feature_names) > 3  # full active set retained
    transforms = c.metadata.get("column_transforms") or []
    assert transforms and transforms[0]["kind"] == "supervised_select"

    exp = ws.create_experiment(
        run=run,
        name="nested_fs",
        feature_names=c.feature_names,
        model_ids=["logistic_regression"],
        column_transforms=transforms,
        validation_strategy="kfold",
    )
    results = ws.run_experiment(run, exp)
    assert any(r.succeeded for r in results)


def test_promote_blocks_leakage_risk_without_opt_in(tmp_path: Path):
    ws, run, _, _, _ = _binary_ws(tmp_path)
    from automl.domain.features.selection_strategy import FeatureSetCandidate

    leaky = FeatureSetCandidate.create(
        name="leaky_top",
        feature_names=["num1"],
        method="mutual_information",
        k=1,
        metadata={
            "evaluation_mode": "exploratory_full_dataset",
            "supervised_selection_leakage_risk": True,
        },
    )
    ws._candidate_feature_sets[run.id] = [leaky]
    with pytest.raises(ValueError, match="optimistic bias|Refuse promotion"):
        ws.promote_candidate_feature_set(run.id, leaky.id)


def test_build_pipeline_includes_supervised_select():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    pipe = _build_pipeline(
        X,
        LogisticRegression(max_iter=200),
        column_transforms=[
            {"column": "*", "kind": "supervised_select", "params": {"method": "mutual_information", "k": 1}}
        ],
        task_type="binary_classification",
    )
    step_names = [name for name, _ in pipe.steps]
    assert step_names == ["preprocessor", "supervised_select", "model"]
