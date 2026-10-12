"""Audit-3 contracts: recipe fidelity, CV aliases, metric direction, HPO seed, FDR, hash."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from automl.application.recipes.commands import (
    CreateExperimentRecipeFromEvidenceCommand,
    SubmitExperimentFromRecipeCommand,
)
from automl.application.recipes.recipe_service import RecipeService
from automl.application.services.workspace import AutoMLWorkspace
from automl.domain.analysis.models import AnalysisHypothesis, EvidenceLink
from automl.domain.datasets.content_hash import compute_file_content_hash
from automl.domain.experiments.trial import TrialResult
from automl.domain.experiments.validation import (
    normalize_validation_strategy,
    require_known_validation_strategy,
)
from automl.domain.metrics.direction import is_minimizing_metric_name
from automl.domain.ports import TrialExecution
from automl.domain.recipes.recipe import ExperimentRecipe, RecipeStatus
from automl.engine.analysis.hypothesis_testing import apply_benjamini_hochberg_correction
from automl.engine.analysis.hypothesis_translator import HypothesisExperimentTranslator
from automl.engine.training.sklearn_trainer import SklearnTrainer


def _binary_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "target": [0, 1] * (n // 2),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_normalize_k_fold_alias_and_reject_unknown():
    assert normalize_validation_strategy("k_fold") == "kfold"
    assert normalize_validation_strategy("K-Fold") == "kfold"
    assert require_known_validation_strategy("k_fold") == "kfold"
    with pytest.raises(ValueError, match="Unknown validation_strategy"):
        require_known_validation_strategy("not_a_real_strategy")


def test_recipe_create_normalizes_k_fold_default():
    recipe = ExperimentRecipe.create(
        name="r",
        dataset_id="ds",
        feature_names=["a"],
        validation_strategy="k_fold",
    )
    assert recipe.validation_strategy == "kfold"


def test_translator_accepts_default_models_and_exposes_drop_features():
    hyp = AnalysisHypothesis(
        id="hyp_x",
        study_id="st",
        finding_id="f",
        proposed_action="resolve_collinearity",
        description="drop collinear",
        experiment_delta={"collinear_pair": ["a", "b"]},
    )
    spec = HypothesisExperimentTranslator.translate(
        hyp,
        ["a", "b", "c"],
        default_models=["logistic_regression"],
    )
    assert "b" not in spec.feature_names
    assert "b" in spec.drop_features
    assert spec.model_id == "logistic_regression"
    assert spec.model_ids == ["logistic_regression"]


def test_recipe_fallback_translator_no_typeerror(tmp_path: Path):
    """Regression: create_recipe_from_evidence must not pass base_model= to translate()."""
    ws = AutoMLWorkspace.create("audit3_recipe", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )

    evidence = MagicMock(spec=EvidenceLink)
    evidence.id = "ev1"
    evidence.hypothesis_id = "hyp1"
    evidence.experiment_id = None
    evidence.metric = "roc_auc"
    evidence.baseline_score = 0.5
    evidence.candidate_score = 0.6
    evidence.accepted = True
    evidence.notes = ""

    hyp = AnalysisHypothesis(
        id="hyp1",
        study_id="st1",
        finding_id="f1",
        proposed_action="prioritize_feature",
        description="prioritize a",
        experiment_delta={"feature": "a"},
    )
    study = MagicMock()
    study.id = "st1"
    study.data_source.dataset_id = ds.id
    study.target_column = "target"

    study_repo = MagicMock()
    study_repo.get_evidence_link.return_value = evidence
    study_repo.get_hypothesis.return_value = hyp
    study_repo.get_study.return_value = study

    recipe_repo = MagicMock()
    saved = []
    recipe_repo.save.side_effect = lambda r: saved.append(r)

    svc = RecipeService(recipe_repo, study_repo, ws)
    rid = svc.create_recipe_from_evidence(
        CreateExperimentRecipeFromEvidenceCommand(evidence_link_id="ev1")
    )
    assert rid
    assert saved
    assert saved[0].dataset_content_hash == compute_file_content_hash(csv_path)
    assert saved[0].validation_strategy == "kfold"
    assert "a" in saved[0].feature_names


def test_submit_recipe_restores_metric_cv_seed(tmp_path: Path):
    ws = AutoMLWorkspace.create("audit3_submit", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="accuracy", validation_strategy="holdout", cv_folds=3, random_seed=1)

    recipe = ExperimentRecipe.create(
        name="replay",
        dataset_id=ds.id,
        feature_names=["a", "b"],
        model_ids=["logistic_regression"],
        validation_strategy="k_fold",
        cv_folds=5,
        metric="roc_auc",
        random_seed=99,
        dataset_content_hash=compute_file_content_hash(csv_path),
    )
    recipe_repo = MagicMock()
    recipe_repo.get.return_value = recipe
    recipe_repo.save.side_effect = lambda r: None
    study_repo = MagicMock()
    svc = RecipeService(recipe_repo, study_repo, ws)

    captured: list[TrialExecution] = []
    real_run = SklearnTrainer.run

    def spy(self, execution: TrialExecution) -> TrialResult:
        captured.append(execution)
        return real_run(self, execution)

    with patch.object(SklearnTrainer, "run", spy):
        ids = svc.submit_experiment_from_recipe(
            SubmitExperimentFromRecipeCommand(recipe_id=recipe.id, run_id=run.id)
        )

    assert ids
    refreshed = ws.get_run(run.id)
    assert refreshed.config.metric == "roc_auc"
    assert refreshed.config.cv_folds == 5
    assert refreshed.config.random_seed == 99
    assert refreshed.config.validation_strategy == "kfold"
    assert captured
    assert captured[0].metric == "roc_auc"
    assert captured[0].cv_folds == 5
    assert captured[0].random_seed == 99
    assert captured[0].validation_strategy == "kfold"


def test_archived_recipe_cannot_submit(tmp_path: Path):
    ws = AutoMLWorkspace.create("audit3_arch", root_dir=tmp_path / "ws")
    recipe = ExperimentRecipe.create(name="x", dataset_id="ds", feature_names=["a"])
    recipe.archive()
    recipe_repo = MagicMock()
    recipe_repo.get.return_value = recipe
    svc = RecipeService(recipe_repo, MagicMock(), ws)
    with pytest.raises(ValueError, match="archived"):
        svc.submit_experiment_from_recipe(SubmitExperimentFromRecipeCommand(recipe_id=recipe.id))


def test_run_scheduled_experiments_respects_minimizing_metric(tmp_path: Path):
    ws = AutoMLWorkspace.create("audit3_sched", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    # Use regression-like MAE via synthetic scores on a classification dataset run.
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="mae")

    fake_results = [
        TrialResult(
            trial_id="t1",
            experiment_id="e",
            model_id="m",
            primary_metric="mae",
            primary_score=9.0,
        ),
        TrialResult(
            trial_id="t2",
            experiment_id="e",
            model_id="m",
            primary_metric="mae",
            primary_score=4.0,
        ),
    ]
    exp = MagicMock()
    exp.id = "exp_mae"
    exp.name = "mae_exp"
    exp.priority = "normal"
    exp.metric = "mae"

    with patch.object(ws, "get_experiment_queue", side_effect=[["c1"], []]):
        with patch.object(ws, "execute_next_experiment", return_value=(exp, fake_results)):
            with patch.object(ws, "_is_run_budget_exhausted", return_value=False):
                out = ws.run_scheduled_experiments(run.id)

    assert out[0]["best_score"] == 4.0


def test_hpo_uses_fixed_cv_seed(tmp_path: Path):
    ws = AutoMLWorkspace.create("audit3_hpo", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path, n=60)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds, metric="roc_auc", random_seed=42, cv_folds=3)
    exp = ws.create_experiment(
        run=run,
        name="hpo",
        feature_names=["a", "b"],
        model_ids=["logistic_regression"],
        validation_strategy="kfold",
    )
    seeds: list[int] = []
    real_run = SklearnTrainer.run

    def spy(self, execution: TrialExecution) -> TrialResult:
        seeds.append(execution.random_seed)
        return real_run(self, execution)

    with patch.object(SklearnTrainer, "run", spy):
        ws.optimize_experiment(run_id=run.id, experiment_id=exp.id, n_trials=3, optimizer="random")

    assert len(seeds) >= 2
    assert len(set(seeds)) == 1
    assert seeds[0] == 42


def test_fdr_preserves_zero_p_value():
    from automl.domain.analysis.models import StatisticalFinding

    findings = [
        StatisticalFinding(
            id="f0",
            study_id="s",
            analysis_run_id="ar",
            finding_type="association",
            summary="zero p",
            column_name="a",
            p_value=0.0,
            metrics={},
            limitations=[],
        ),
        StatisticalFinding(
            id="f1",
            study_id="s",
            analysis_run_id="ar",
            finding_type="association",
            summary="mid p",
            column_name="b",
            p_value=0.04,
            metrics={},
            limitations=[],
        ),
    ]
    apply_benjamini_hochberg_correction(findings)
    assert findings[0].metrics["p_value_fdr"] == 0.0
    assert findings[0].metrics["fdr_significant"] is True


def test_is_minimizing_metric_name_builtins():
    assert is_minimizing_metric_name("mae")
    assert is_minimizing_metric_name("RMSE")
    assert not is_minimizing_metric_name("roc_auc")


def test_trainer_rejects_unknown_validation_strategy(tmp_path: Path):
    ws = AutoMLWorkspace.create("audit3_val", root_dir=tmp_path / "ws")
    csv_path = _binary_csv(tmp_path)
    ds = ws.register_dataset(
        name="data",
        path=csv_path,
        target="target",
        task_type="binary_classification",
    )
    run = ws.create_run(ds)
    with pytest.raises(ValueError, match="Unknown validation_strategy"):
        ws.create_experiment(
            run=run,
            name="bad",
            feature_names=["a", "b"],
            model_ids=["logistic_regression"],
            validation_strategy="mystery_cv",
        )


def test_escape_html_utils_present_in_views():
    """Static guard: audited XSS hotspots import/use escapeHtml."""
    root = Path(__file__).resolve().parents[1]
    new_exp = (root / "src/automl/interfaces/web/static/js/views/new_experiment.js").read_text()
    from workbench_static_sources import datasets_view_sources

    datasets = datasets_view_sources()
    compare = (root / "src/automl/interfaces/web/static/js/views/compare.js").read_text()
    assert "escapeHtml(f)" in new_exp
    assert "escapeHtml(activeDs" in new_exp
    assert "escapeHtml(r.name || r.id)" in new_exp
    assert "escapeHtml(r.column)" in datasets
    assert "escapeHtml(c)" in datasets
    assert "escapeHtml((selectedExps[0].model_ids" in compare
