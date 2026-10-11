"""CLI commands for CATML Experiment Recipes."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys

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


def run_recipe_cli(args: argparse.Namespace) -> int:
    """Dispatches `automl recipe` subcommands via CommandBus and QueryBus."""
    ws_dir = getattr(args, "workspace", None)
    ws, cmd, qry = build_application(root_dir=ws_dir)

    action = getattr(args, "recipe_action", None)

    if action == "list-verified":
        study_id = getattr(args, "study", None) or getattr(args, "study_id", None)
        dataset_id = getattr(args, "dataset", None) or getattr(args, "dataset_id", None)
        accepted_only = not getattr(args, "all", False)

        try:
            improvements = qry.dispatch(
                ListVerifiedImprovementsQuery(
                    study_id=study_id,
                    dataset_id=dataset_id,
                    accepted_only=accepted_only,
                )
            )
            if getattr(args, "json", False):
                print(json.dumps([asdict(i) for i in improvements], indent=2, default=str))
            else:
                if not improvements:
                    print("No verified improvements found.")
                    return 0
                print(f"Found {len(improvements)} verified improvement(s):")
                print("-" * 88)
                header = f"{'Evidence ID':<14} {'Dataset':<12} {'Suggested Action':<20} {'Baseline':<10} {'Candidate':<10} {'Delta':<8} {'Status'}"
                print(header)
                print("-" * 88)
                for imp in improvements:
                    status_str = "ACCEPTED" if imp.accepted else "REJECTED"
                    sign = "+" if imp.delta_primary_metric > 0 else ""
                    delta_str = f"{sign}{imp.delta_primary_metric:.4f}"
                    print(
                        f"{imp.evidence_link_id:<14} {imp.dataset_id:<12} {imp.suggested_action[:20]:<20} "
                        f"{imp.baseline_score:<10.4f} {imp.candidate_score:<10.4f} {delta_str:<8} [{status_str}]"
                    )
            return 0
        except Exception as exc:
            print(f"Error listing verified improvements: {exc}", file=sys.stderr)
            return 1

    elif action == "create-from-evidence":
        evidence_id = args.evidence_id
        name = getattr(args, "name", None)
        notes = getattr(args, "notes", None)
        models = getattr(args, "model", None)
        model_ids = [models] if isinstance(models, str) else models

        try:
            recipe_id = cmd.dispatch(
                CreateExperimentRecipeFromEvidenceCommand(
                    evidence_link_id=evidence_id,
                    recipe_name=name,
                    notes=notes,
                    model_ids=model_ids,
                )
            )
            recipe_dto = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
            if getattr(args, "json", False):
                print(json.dumps(asdict(recipe_dto) if recipe_dto else {"recipe_id": recipe_id}, indent=2, default=str))
            else:
                print(f"Successfully created recipe: {recipe_id}")
                if recipe_dto:
                    print(f"  Name:    {recipe_dto.name}")
                    print(f"  Dataset: {recipe_dto.dataset_id}")
                    print(f"  Status:  {recipe_dto.status.upper()}")
                    if recipe_dto.column_transforms:
                        print(f"  Transforms: {recipe_dto.column_transforms}")
                    if recipe_dto.derived_features:
                        print(f"  Derived:    {recipe_dto.derived_features}")
            return 0
        except Exception as exc:
            print(f"Error creating recipe from evidence: {exc}", file=sys.stderr)
            return 1

    elif action == "list":
        dataset_id = getattr(args, "dataset", None) or getattr(args, "dataset_id", None)
        status = getattr(args, "status", None)

        try:
            recipes = qry.dispatch(
                ListExperimentRecipesQuery(
                    dataset_id=dataset_id,
                    status=status,
                )
            )
            if getattr(args, "json", False):
                print(json.dumps([asdict(r) for r in recipes], indent=2, default=str))
            else:
                if not recipes:
                    print("No experiment recipes found.")
                    return 0
                print(f"Experiment Recipes ({len(recipes)}):")
                print("-" * 80)
                print(f"{'Recipe ID':<16} {'Dataset':<14} {'Status':<12} {'Models':<18} {'Name'}")
                print("-" * 80)
                for r in recipes:
                    models_str = ",".join(r.model_ids) if r.model_ids else "default"
                    print(f"{r.id:<16} {r.dataset_id:<14} {r.status.upper():<12} {models_str:<18} {r.name}")
            return 0
        except Exception as exc:
            print(f"Error listing experiment recipes: {exc}", file=sys.stderr)
            return 1

    elif action == "show":
        recipe_id = args.recipe_id
        try:
            recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
            if not recipe:
                print(f"Recipe not found: {recipe_id}", file=sys.stderr)
                return 1

            if getattr(args, "json", False):
                print(json.dumps(asdict(recipe), indent=2, default=str))
            else:
                print(f"Experiment Recipe: {recipe.id}")
                print("=" * 60)
                print(f"  Name:                 {recipe.name}")
                print(f"  Dataset ID:           {recipe.dataset_id}")
                print(f"  Status:               {recipe.status.upper()}")
                print(f"  Content Hash:         {recipe.dataset_content_hash or 'N/A'}")
                print(f"  Validation Strategy:  {recipe.validation_strategy} (cv={recipe.cv_folds})")
                print(f"  Metric:               {recipe.metric}")
                print(f"  Random Seed:          {recipe.random_seed}")
                print(f"  Models:               {recipe.model_ids}")
                print(f"  Base Features:        {len(recipe.feature_names)} features")
                print(f"  Derived Features:     {recipe.derived_features or 'None'}")
                print(f"  Column Transforms:    {recipe.column_transforms or 'None'}")
                print(f"  Dropped Features:     {recipe.drop_features or 'None'}")
                print(f"  Evidence Link ID:     {recipe.evidence_link_id or 'None'}")
                print(f"  Executed Experiments: {recipe.experiment_ids or 'None'}")
                print(f"  Created At:           {recipe.created_at}")
                print(f"  Updated At:           {recipe.updated_at}")
                if recipe.lineage:
                    print(f"  Lineage Metadata:     {json.dumps(recipe.lineage, indent=4, default=str)}")
            return 0
        except Exception as exc:
            print(f"Error showing recipe: {exc}", file=sys.stderr)
            return 1

    elif action == "select":
        recipe_id = args.recipe_id
        notes = getattr(args, "notes", None)
        try:
            cmd.dispatch(SelectExperimentRecipeCommand(recipe_id=recipe_id, notes=notes))
            recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
            if getattr(args, "json", False):
                print(json.dumps(asdict(recipe) if recipe else {"status": "success"}, indent=2, default=str))
            else:
                print(f"Recipe {recipe_id} transitioned to [SELECTED].")
            return 0
        except Exception as exc:
            print(f"Error selecting recipe: {exc}", file=sys.stderr)
            return 1

    elif action == "submit":
        recipe_id = args.recipe_id
        run_id = getattr(args, "run", None) or getattr(args, "run_id", None)
        model = getattr(args, "model", None)
        model_ids = [model] if isinstance(model, str) else model
        notes = getattr(args, "notes", None)

        try:
            exp_ids = cmd.dispatch(
                SubmitExperimentFromRecipeCommand(
                    recipe_id=recipe_id,
                    run_id=run_id,
                    model_ids=model_ids,
                    notes=notes,
                )
            )
            recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
            if getattr(args, "json", False):
                print(json.dumps({"recipe_id": recipe_id, "experiment_ids": exp_ids}, indent=2, default=str))
            else:
                print(f"Successfully submitted recipe: {recipe_id} -> [SUBMITTED]")
                print(f"  Executed Experiments: {', '.join(exp_ids)}")
            return 0
        except Exception as exc:
            print(f"Error submitting recipe: {exc}", file=sys.stderr)
            return 1

    elif action == "promote":
        recipe_id = args.recipe_id
        reason = getattr(args, "reason", None)
        notes = getattr(args, "notes", None)

        try:
            cmd.dispatch(
                PromoteExperimentRecipeCommand(
                    recipe_id=recipe_id,
                    reason=reason,
                    notes=notes,
                )
            )
            recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
            if getattr(args, "json", False):
                print(json.dumps(asdict(recipe) if recipe else {"status": "promoted"}, indent=2, default=str))
            else:
                print(f"Successfully promoted recipe: {recipe_id} -> [PROMOTED]")
                if reason:
                    print(f"  Reason: {reason}")
            return 0
        except Exception as exc:
            print(f"Error promoting recipe: {exc}", file=sys.stderr)
            return 1

    elif action == "archive":
        recipe_id = args.recipe_id
        try:
            cmd.dispatch(ArchiveExperimentRecipeCommand(recipe_id=recipe_id))
            if getattr(args, "json", False):
                print(json.dumps({"status": "archived", "recipe_id": recipe_id}))
            else:
                print(f"Recipe {recipe_id} archived.")
            return 0
        except Exception as exc:
            print(f"Error archiving recipe: {exc}", file=sys.stderr)
            return 1

    else:
        print("Error: Unknown recipe action. Use --help for usage.", file=sys.stderr)
        return 1
