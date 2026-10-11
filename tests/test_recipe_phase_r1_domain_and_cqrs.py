"""Tests for Phase R1: ExperimentRecipe domain entity, SQLite persistence, and CQRS handlers."""
from __future__ import annotations

from pathlib import Path
import pytest
import pandas as pd

from automl.application.analysis.commands import CreateStudyCommand
from automl.application.bootstrap import build_application
from automl.application.recipes.commands import (
    ArchiveExperimentRecipeCommand,
    CreateExperimentRecipeFromEvidenceCommand,
    PromoteExperimentRecipeCommand,
    SelectExperimentRecipeCommand,
    SubmitExperimentFromRecipeCommand,
)
from automl.application.recipes.queries import (
    GetExperimentRecipeQuery,
    ListExperimentRecipesQuery,
    ListVerifiedImprovementsQuery,
)
from automl.domain.recipes.recipe import ExperimentRecipe, RecipeStatus
from automl.infrastructure.database.sqlite_recipes import SQLiteRecipeRepository
from automl.domain.analysis.models import (
    AnalysisHypothesis,
    AnalysisRun,
    EvidenceLink,
    StatisticalFinding,
)


def test_recipe_domain_entity_lifecycle_and_serialization():
    recipe = ExperimentRecipe.create(
        name="Power Transform Fare Recipe",
        dataset_id="ds_123",
        feature_names=["age", "fare_log1p"],
        derived_features=[{"name": "fare_log1p", "expression": "log1p(fare)", "source_columns": ["fare"]}],
        column_transforms=[{"column": "age", "kind": "robust_scaler", "params": {}}],
        drop_features=["ticket"],
        model_ids=["lightgbm", "random_forest"],
        model_params={"max_depth": 5},
        dataset_content_hash="hash_abc123",
        validation_strategy="k_fold",
        cv_folds=5,
        metric="roc_auc",
        random_seed=42,
        evidence_link_id="ev_001",
        hypothesis_id="hyp_001",
        study_id="study_001",
        lineage={"creator": "agent", "source": "explore"},
    )

    assert recipe.status == RecipeStatus.DRAFT
    assert recipe.feature_names == ["age", "fare_log1p"]
    assert recipe.derived_features[0]["expression"] == "log1p(fare)"
    assert recipe.column_transforms[0]["kind"] == "robust_scaler"
    assert recipe.drop_features == ["ticket"]

    # 1. Select
    recipe.select(notes="Approved for trial")
    assert recipe.status == RecipeStatus.SELECTED
    assert any(h["action"] == "selected" for h in recipe.lineage.get("history", []))

    # 2. Submit
    recipe.submit(experiment_id="exp_run_1")
    assert recipe.status == RecipeStatus.SUBMITTED
    assert "exp_run_1" in recipe.experiment_ids

    # 3. Promote
    recipe.promote(reason="Outperformed baseline by +0.02 ROC-AUC", notes="Production candidate")
    assert recipe.status == RecipeStatus.PROMOTED
    assert recipe.lineage["promoted_reason"] == "Outperformed baseline by +0.02 ROC-AUC"

    # 4. Round-trip dictionary serialization
    data = recipe.to_dict()
    assert data["status"] == "promoted"
    assert data["name"] == "Power Transform Fare Recipe"

    restored = ExperimentRecipe.from_dict(data)
    assert restored.id == recipe.id
    assert restored.status == RecipeStatus.PROMOTED
    assert restored.lineage["promoted_reason"] == "Outperformed baseline by +0.02 ROC-AUC"


def test_sqlite_recipe_repository_crud(tmp_path: Path):
    db_path = tmp_path / "recipes.db"
    repo = SQLiteRecipeRepository(db_path)

    recipe1 = ExperimentRecipe.create(
        name="Recipe 1",
        dataset_id="ds_a",
        feature_names=["f1", "f2"],
        status=RecipeStatus.DRAFT,
    )
    recipe2 = ExperimentRecipe.create(
        name="Recipe 2",
        dataset_id="ds_b",
        feature_names=["f3"],
        status=RecipeStatus.PROMOTED,
    )

    repo.save(recipe1)
    repo.save(recipe2)

    # Get
    fetched1 = repo.get(recipe1.id)
    assert fetched1 is not None
    assert fetched1.name == "Recipe 1"
    assert fetched1.status == RecipeStatus.DRAFT

    # Update on conflict
    recipe1.select(notes="Selected for evaluation")
    repo.save(recipe1)
    updated1 = repo.get(recipe1.id)
    assert updated1 is not None
    assert updated1.status == RecipeStatus.SELECTED

    # List all
    all_recipes = repo.list()
    assert len(all_recipes) == 2

    # List filtered by dataset
    ds_a_recipes = repo.list(dataset_id="ds_a")
    assert len(ds_a_recipes) == 1
    assert ds_a_recipes[0].id == recipe1.id

    # List filtered by status
    promoted_recipes = repo.list(status=RecipeStatus.PROMOTED)
    assert len(promoted_recipes) == 1
    assert promoted_recipes[0].id == recipe2.id

    # Delete
    repo.delete(recipe1.id)
    assert repo.get(recipe1.id) is None
    assert len(repo.list()) == 1


def test_recipe_cqrs_create_from_evidence_and_no_autopromotion(tmp_path: Path):
    ws, cmd_bus, qry_bus = build_application(root_dir=str(tmp_path))

    # 1. Setup Dataset and Explore Study
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0],
        "num2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 0, 0, 1, 1, 1],
    })
    ds = ws.register_dataset("rec_ds", df, target="target")
    run = ws.create_run(ds)

    study_id = cmd_bus.dispatch(CreateStudyCommand(dataset_id=ds.id, name="Study 1", target_column="target"))
    study_repo = ws.get_study_repository()

    analysis_run = AnalysisRun.create(study_id=study_id)
    study_repo.save_analysis_run(analysis_run)

    finding = StatisticalFinding.create(
        study_id=study_id,
        analysis_run_id=analysis_run.id,
        finding_type="skewed_distribution",
        summary="num1 is highly skewed",
        column_name="num1",
    )
    study_repo.save_finding(finding)

    hypothesis = AnalysisHypothesis.create(
        study_id=study_id,
        finding_id=finding.id,
        description="Apply power transform to num1",
        proposed_action="power_transform",
        experiment_delta={
            "column": "num1",
            "method": "yeo_johnson",
            "transform_kind": "yeo_johnson",
            "suggested_action": "power_transform",
        },
    )
    study_repo.save_hypothesis(hypothesis)

    # 2. Simulate baseline and candidate experiments
    base_exp = ws.create_experiment(
        run=run,
        name="Baseline",
        model_ids=["random_forest"],
        feature_names=["num1", "num2"],
        validation_strategy="holdout",
    )
    ws.run_experiment(run, base_exp)

    cand_exp = ws.create_experiment(
        run=run,
        name="Candidate",
        model_ids=["random_forest"],
        feature_names=["num1", "num2"],
        validation_strategy="holdout",
        column_transforms=[{"column": "num1", "kind": "yeo_johnson", "params": {}}],
    )
    ws.run_experiment(run, cand_exp)

    # 3. Create EvidenceLink marked as accepted
    evidence = EvidenceLink.create(
        hypothesis_id=hypothesis.id,
        experiment_id=cand_exp.id,
        baseline_score=0.80,
        candidate_score=0.90,
        metric="accuracy",
        accepted=True,  # Empirical improvement verified
        notes="Verified +0.10 improvement",
    )
    study_repo.save_evidence_link(evidence)

    # 4. Create Recipe from EvidenceLink via CommandBus
    cmd = CreateExperimentRecipeFromEvidenceCommand(
        evidence_link_id=evidence.id,
        recipe_name="Yeo-Johnson num1 Recipe",
        notes="Created from verified study evidence",
    )
    recipe_id = cmd_bus.dispatch(cmd)
    assert recipe_id is not None
    assert recipe_id.startswith("rcp_")

    # 5. Query the recipe: rule "EvidenceLink.accepted != Promoted"
    recipe_dto = qry_bus.dispatch(GetExperimentRecipeQuery(recipe_id))
    assert recipe_dto is not None
    assert recipe_dto.name == "Yeo-Johnson num1 Recipe"
    assert recipe_dto.status == "draft"  # STRICT: remains DRAFT, NOT PROMOTED
    assert recipe_dto.column_transforms == [{"column": "num1", "kind": "yeo_johnson", "params": {}}]
    assert recipe_dto.lineage["accepted"] is True
    assert recipe_dto.lineage["delta_primary_metric"] == pytest.approx(0.10)


def test_recipe_cqrs_submit_and_promote_lifecycle(tmp_path: Path):
    ws, cmd_bus, qry_bus = build_application(root_dir=str(tmp_path))

    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        "target": [0, 0, 1, 1, 0, 1],
    })
    ds = ws.register_dataset("lifecycle_ds", df, target="target")

    # Manually create a draft recipe
    recipe_repo = ws.get_recipe_repository()
    recipe = ExperimentRecipe.create(
        name="Lifecycle Recipe",
        dataset_id=ds.id,
        feature_names=["x1", "x2"],
        column_transforms=[{"column": "x1", "kind": "robust_scaler", "params": {}}],
        model_ids=["random_forest"],
        metric="accuracy",
        validation_strategy="holdout",
    )
    recipe_repo.save(recipe)

    # 1. Select Recipe via CommandBus
    cmd_bus.dispatch(SelectExperimentRecipeCommand(recipe_id=recipe.id, notes="Selected by user"))
    dto_sel = qry_bus.dispatch(GetExperimentRecipeQuery(recipe.id))
    assert dto_sel.status == "selected"

    # 2. Submit Recipe via CommandBus (runs training)
    submitted_ids = cmd_bus.dispatch(SubmitExperimentFromRecipeCommand(recipe_id=recipe.id))
    assert len(submitted_ids) == 1
    dto_sub = qry_bus.dispatch(GetExperimentRecipeQuery(recipe.id))
    assert dto_sub.status == "submitted"
    assert submitted_ids[0] in dto_sub.experiment_ids

    # Verify executed experiment exists in workspace
    exp = ws.get_experiment(submitted_ids[0])
    assert exp is not None
    assert exp.column_transforms == [{"column": "x1", "kind": "robust_scaler", "params": {}}]

    # 3. Promote Recipe via CommandBus
    cmd_bus.dispatch(PromoteExperimentRecipeCommand(
        recipe_id=recipe.id,
        reason="Consistent holdout accuracy above 0.85",
        notes="Promoted to curated recipe catalog",
    ))
    dto_prom = qry_bus.dispatch(GetExperimentRecipeQuery(recipe.id))
    assert dto_prom.status == "promoted"
    assert dto_prom.lineage["promoted_reason"] == "Consistent holdout accuracy above 0.85"

    # 4. List recipes query
    all_recipes = qry_bus.dispatch(ListExperimentRecipesQuery(dataset_id=ds.id))
    assert len(all_recipes) == 1
    assert all_recipes[0].id == recipe.id


def test_list_verified_improvements_query(tmp_path: Path):
    ws, cmd_bus, qry_bus = build_application(root_dir=str(tmp_path))

    df = pd.DataFrame({"col": [1, 2, 3], "target": [0, 1, 0]})
    ds = ws.register_dataset("imp_ds", df, target="target")

    study_id = cmd_bus.dispatch(CreateStudyCommand(dataset_id=ds.id, name="Study Improvements", target_column="target"))
    study_repo = ws.get_study_repository()

    h1 = AnalysisHypothesis.create(study_id=study_id, finding_id="f1", description="H1 accepted", proposed_action="robust_scaler")
    h2 = AnalysisHypothesis.create(study_id=study_id, finding_id="f2", description="H2 rejected", proposed_action="drop_collinear")
    study_repo.save_hypothesis(h1)
    study_repo.save_hypothesis(h2)

    ev1 = EvidenceLink.create(hypothesis_id=h1.id, experiment_id="exp_1", baseline_score=0.80, candidate_score=0.85, metric="ROC-AUC", accepted=True)
    ev2 = EvidenceLink.create(hypothesis_id=h2.id, experiment_id="exp_2", baseline_score=0.80, candidate_score=0.78, metric="ROC-AUC", accepted=False)
    study_repo.save_evidence_link(ev1)
    study_repo.save_evidence_link(ev2)

    # Query accepted only
    accepted_imps = qry_bus.dispatch(ListVerifiedImprovementsQuery(study_id=study_id, accepted_only=True))
    assert len(accepted_imps) == 1
    assert accepted_imps[0].hypothesis_id == h1.id
    assert accepted_imps[0].accepted is True
    assert accepted_imps[0].delta_primary_metric == pytest.approx(0.05)

    # Query all (including rejected)
    all_imps = qry_bus.dispatch(ListVerifiedImprovementsQuery(study_id=study_id, accepted_only=False))
    assert len(all_imps) == 2
    assert any(imp.hypothesis_id == h2.id and not imp.accepted for imp in all_imps)
