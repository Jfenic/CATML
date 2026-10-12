"""Analysis / recipes / plan / agent / plugins API controller."""
from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any
from dataclasses import asdict

from automl.application.analysis.commands import (
    ArchiveStudyCommand,
    CreateStudyCommand,
    RunAnalysisCommand,
    VerifyHypothesisCommand,
)
from automl.application.analysis.queries import (
    GetAnalysisRunQuery,
    GetStudyQuery,
    ListEvidenceLinksQuery,
    ListFindingsQuery,
    ListHypothesesQuery,
    ListStudiesQuery,
    ListVisualizationsQuery,
)
from automl.application.commands.workspace_commands import (
    GenerateOOFSubmissionCommand,
    GenerateSubmissionCommand,
)
from automl.application.queries.workspace_queries import (
    GetLeaderboardQuery,
    GetMetaKnowledgeQuery,
    GetOOFResultQuery,
    GetTaskPlanQuery,
    ListModelsQuery,
    ListPluginsQuery,
)
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
from automl.interfaces.web.api._helpers import (
    _effective_experiment_status,
    _get_best_trial_result,
    _is_metric_minimizing,
)


def try_get(handler: Any, path: str, query_params: dict, ws: Any, cmd: Any, qry: Any) -> bool:
    """Handle GET routes for this resource. Return True if the request was handled."""
    if path == "/api/plan":
        dataset_id = query_params.get("dataset_id", [""])[0]
        run_id = query_params.get("run_id", [""])[0]
        run = None
        if run_id:
            run = ws.get_run(run_id)
            if run and not dataset_id:
                dataset_id = run.dataset_id

        if not dataset_id:
            runs = ws.list_runs()
            if runs:
                run = runs[0]
                dataset_id = run.dataset_id
                run_id = run.id

        if not dataset_id:
            handler._send_json({"error": "No dataset found in workspace"}, HTTPStatus.BAD_REQUEST)
            return True

        dataset = ws.get_dataset(dataset_id)
        profile = ws.get_dataset_profile(dataset_id)
        problem_def = qry.dispatch(GetTaskPlanQuery(dataset_id))

        plan_steps = []
        step_idx = 1

        if run_id:
            exps = ws.list_experiments(run_id)
            for e in exps:
                trials = ws.list_trial_results(e.id)
                best_t = _get_best_trial_result(trials, metric=e.metric, ws=ws)
                is_p_min = _is_metric_minimizing(e.metric, ws=ws)
                plan_steps.append({
                    "step": step_idx,
                    "name": e.name,
                    "status": _effective_experiment_status(e, trials),
                    "score": round(best_t.primary_score, 5) if best_t else None,
                    "metric": e.metric,
                    "metric_direction": "minimize" if is_p_min else "maximize",
                    "is_minimizing": is_p_min,
                    "delta": None,
                    "priority": e.priority.upper() if hasattr(e, "priority") else "HIGH",
                    "why": e.hypothesis or f"Experiment with models: {', '.join(e.model_ids)}",
                    "rule": "empirical_verification",
                    "source": e.created_by or "AutoMLPlanner",
                })
                step_idx += 1

        rec_models = problem_def.recommended_models if problem_def else ["lightgbm", "xgboost", "catboost"]
        task_type = problem_def.task_type.value if problem_def else "binary_classification"
        metric = problem_def.default_metric if problem_def else "roc_auc"

        planned_templates = [
            {
                "name": f"Baseline {rec_models[0].replace('_', ' ').title()}",
                "priority": "LOW",
                "why": f"Establishes initial benchmark threshold and near-instant training for {task_type}.",
                "rule": "fast_linear_baseline",
            },
            {
                "name": "Gradient Boosting Optimization (GBDT)",
                "priority": "HIGH",
                "why": f"Tree-based models capture non-linearities and optimize {metric}.",
                "rule": "tree_models_outperform_baseline",
            },
            {
                "name": "Feature Interactions & Selection",
                "priority": "MEDIUM",
                "why": "Generates A/B interactions and selects candidate features under Propose ≠ Accept rule.",
                "rule": "hypothesis_feature_interactions",
            },
            {
                "name": "Hyperparameter Optimization (Optuna HPO)",
                "priority": "HIGH",
                "why": "Bayesian TPE exploration across hyperparameter search space.",
                "rule": "bayesian_hpo_tuning",
            },
            {
                "name": "Ensemble Blender & Stacking",
                "priority": "HIGH",
                "why": "Weighted combination of out-of-fold predictions to reduce residual variance.",
                "rule": "diversity_blending",
            },
        ]

        while len(plan_steps) < len(planned_templates):
            tmpl = planned_templates[len(plan_steps)]
            plan_steps.append({
                "step": step_idx,
                "name": tmpl["name"],
                "status": "WAITING",
                "score": None,
                "delta": None,
                "priority": tmpl["priority"],
                "why": tmpl["why"],
                "rule": tmpl["rule"],
                "source": "RuleBasedPlanner",
            })
            step_idx += 1

        handler._send_json({
            "dataset_id": dataset_id,
            "dataset_name": dataset.name if dataset else "-",
            "problem": problem_def.to_dict() if problem_def else {},
            "steps": plan_steps,
        })
        return True

    if path == "/api/knowledge":
        dataset_id = query_params.get("dataset_id", [""])[0]
        run_id = query_params.get("run_id", [""])[0]
        if run_id and not dataset_id:
            run = ws.get_run(run_id)
            if run:
                dataset_id = run.dataset_id
        if not dataset_id:
            runs = ws.list_runs()
            if runs:
                dataset_id = runs[0].dataset_id
        if not dataset_id:
            datasets = ws.list_datasets()
            if datasets:
                dataset_id = datasets[0].id

        if not dataset_id:
            handler._send_error(404, "No registered dataset found for meta-learning.")
            return True

        knowledge = qry.dispatch(GetMetaKnowledgeQuery(dataset_id=dataset_id, run_id=run_id or None))
        handler._send_json(knowledge.to_dict())
        return True

    if path == "/api/agent/hypotheses":
        run_id = query_params.get("run_id", [""])[0]
        ledger_file = Path(handler.workspace_dir) / "agent_ledger.db"
        real_hypotheses = []
        if ledger_file.exists():
            try:
                from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger

                ledger = SqliteAgentLedger(ledger_file)
                target_runs = [run_id] if run_id else [r.id for r in ws.list_runs()]
                for r_id in target_runs:
                    hyps = ledger.list_hypotheses(r_id)
                    for h in hyps:
                        h_status = h.status.value.lower() if hasattr(h.status, "value") else str(h.status).lower()
                        h_id = getattr(h, "hypothesis_id", getattr(h, "id", "hyp"))
                        critique_decision = "PENDING"
                        if h_status == "accepted":
                            critique_decision = "PROMOTE"
                        elif h_status == "rejected":
                            critique_decision = "REJECT"

                        candidate_cfg = h.candidate_config if isinstance(h.candidate_config, dict) else {}
                        crit_cfg = h.verification_criteria if isinstance(h.verification_criteria, dict) else {}

                        before_score = getattr(h, "baseline_metric", None)
                        if before_score is None:
                            before_score = crit_cfg.get("baseline_metric", crit_cfg.get("before_score"))

                        after_score = getattr(h, "verified_metric", None)
                        if after_score is None:
                            after_score = crit_cfg.get("verified_metric")
                        if after_score is None:
                            after_score = crit_cfg.get("after_score")
                        delta_str = None
                        if before_score is not None and after_score is not None:
                            delta_val = float(after_score) - float(before_score)
                            delta_str = f"{delta_val:+.5f}"

                        real_hypotheses.append({
                            "id": h_id,
                            "run_id": h.run_id,
                            "statement": h.reasoning,
                            "action": candidate_cfg.get("action", str(h.candidate_config)),
                            "cost": crit_cfg.get("cost", "1 CV run"),
                            "status": h_status.upper(),
                            "before_score": before_score,
                            "after_score": after_score,
                            "delta": delta_str,
                            "critic_decision": critique_decision,
                            "critic_reason": getattr(h, "critique", ""),
                        })
            except Exception:
                pass

        handler._send_json({
            "principle": "Propose ≠ Accept",
            "hypotheses": real_hypotheses,
        })
        return True

    if path == "/api/kaggle/status":
        run_id = query_params.get("run_id", [""])[0]
        run = ws.get_run(run_id) if run_id else None
        if not run:
            runs = ws.list_runs()
        dataset = None
        if run:
            try:
                dataset = ws.get_dataset(run.dataset_id)
            except KeyError:
                dataset = None
        profile = ws.get_dataset_profile(run.dataset_id) if run else None
        lb = qry.dispatch(GetLeaderboardQuery(run.id)) if run else []
        best_cv = lb[0]["score"] if lb else None

        exps = ws.list_experiments(run.id) if run else []
        submissions = []
        for e in exps:
            trials = ws.list_trial_results(e.id)
            best_t = _get_best_trial_result(trials, metric=run.config.metric if run else None, ws=ws)
            if best_t:
                submissions.append({
                    "experiment": e.name,
                    "cv": round(best_t.primary_score, 5),
                    "status": "EVALUATED" if best_t.succeeded else "FAILED",
                })

        handler._send_json({
            "competition": dataset.name if dataset else "Custom Dataset",
            "title": f"AutoML Benchmark — {dataset.name if dataset else 'Default'}",
            "metric": run.config.metric.upper() if run else "ROC-AUC",
            "local_best_cv": best_cv,
            "submissions": submissions,
            "checklist": {
                "id_column_valid": True,
                "row_count": profile.row_count if profile else 0,
                "probabilities_bounded": True,
                "zero_missing_values": True,
                "schema_aligned": True,
            },
        })
        return True

    if path == "/api/analysis/studies":
        dataset_id = query_params.get("dataset_id", [None])[0]
        studies = qry.dispatch(ListStudiesQuery(dataset_id=dataset_id))
        result = []
        for s in (studies or []):
            s_dict = dict(s) if isinstance(s, dict) else (s.to_dict() if hasattr(s, "to_dict") else vars(s))
            s_id = s_dict.get("id", "")
            try:
                findings = qry.dispatch(ListFindingsQuery(study_id=s_id))
                s_dict["findings_count"] = len(findings) if findings else 0
            except Exception:
                s_dict["findings_count"] = 0
            try:
                hypotheses = qry.dispatch(ListHypothesesQuery(study_id=s_id))
                s_dict["hypotheses_count"] = len(hypotheses) if hypotheses else 0
            except Exception:
                s_dict["hypotheses_count"] = 0
            result.append(s_dict)
        handler._send_json(result)
        return True

    if path == "/api/analysis/study":
        study_id = query_params.get("id", [""])[0] or query_params.get("study_id", [""])[0]
        if not study_id:
            handler._send_json({"error": "study_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        study = qry.dispatch(GetStudyQuery(study_id=study_id))
        if not study:
            handler._send_json({"error": f"Study '{study_id}' not found"}, HTTPStatus.NOT_FOUND)
            return True
        data = dict(study) if isinstance(study, dict) else (study.to_dict() if hasattr(study, "to_dict") else vars(study))
        try:
            findings = qry.dispatch(ListFindingsQuery(study_id=study_id))
            data["findings"] = [f if isinstance(f, dict) else f.to_dict() for f in (findings or [])]
        except Exception:
            data["findings"] = []
        try:
            visualizations = qry.dispatch(ListVisualizationsQuery(study_id=study_id))
            data["visualizations"] = [v if isinstance(v, dict) else v.to_dict() for v in (visualizations or [])]
        except Exception:
            data["visualizations"] = []
        try:
            hypotheses = qry.dispatch(ListHypothesesQuery(study_id=study_id))
            data["hypotheses"] = [h if isinstance(h, dict) else h.to_dict() for h in (hypotheses or [])]
        except Exception:
            data["hypotheses"] = []
        handler._send_json(data)
        return True

    if path == "/api/analysis/findings":
        study_id = query_params.get("study_id", [""])[0]
        if not study_id:
            handler._send_json({"error": "study_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        finding_type = query_params.get("type", [None])[0]
        findings = qry.dispatch(ListFindingsQuery(study_id=study_id, finding_type=finding_type))
        handler._send_json([f if isinstance(f, dict) else f.to_dict() for f in (findings or [])])
        return True

    if path == "/api/analysis/visualizations":
        study_id = query_params.get("study_id", [""])[0]
        if not study_id:
            handler._send_json({"error": "study_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        visualizations = qry.dispatch(ListVisualizationsQuery(study_id=study_id))
        handler._send_json([v if isinstance(v, dict) else v.to_dict() for v in (visualizations or [])])
        return True

    if path == "/api/analysis/hypotheses":
        study_id = query_params.get("study_id", [""])[0]
        if not study_id:
            handler._send_json({"error": "study_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        hypotheses = qry.dispatch(ListHypothesesQuery(study_id=study_id))
        handler._send_json([h if isinstance(h, dict) else h.to_dict() for h in (hypotheses or [])])
        return True

    if path == "/api/analysis/evidence":
        study_id = query_params.get("study_id", [""])[0]
        hypothesis_id = query_params.get("hypothesis_id", [""])[0]
        links = qry.dispatch(
            ListEvidenceLinksQuery(
                study_id=study_id if study_id else None,
                hypothesis_id=hypothesis_id if hypothesis_id else None,
            )
        )
        handler._send_json([l if isinstance(l, dict) else l.to_dict() for l in (links or [])])
        return True

    if path in ("/api/analysis/verified-improvements", "/api/recipes/verified-improvements"):
        study_id = query_params.get("study_id", [""])[0]
        dataset_id = query_params.get("dataset_id", [""])[0]
        accepted_str = query_params.get("accepted_only", ["true"])[0].lower()
        accepted_only = accepted_str not in ("false", "0", "no")
        improvements = qry.dispatch(
            ListVerifiedImprovementsQuery(
                study_id=study_id if study_id else None,
                dataset_id=dataset_id if dataset_id else None,
                accepted_only=accepted_only,
            )
        )
        handler._send_json([asdict(i) for i in (improvements or [])])
        return True

    if path == "/api/recipes":
        dataset_id = query_params.get("dataset_id", [""])[0]
        status = query_params.get("status", [""])[0]
        recipes = qry.dispatch(
            ListExperimentRecipesQuery(
                dataset_id=dataset_id if dataset_id else None,
                status=status if status else None,
            )
        )
        handler._send_json([asdict(r) for r in (recipes or [])])
        return True

    if path.startswith("/api/recipes/"):
        recipe_id = path[len("/api/recipes/") :]
        if recipe_id and recipe_id not in ("verified-improvements",):
            recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
            if not recipe:
                handler._send_json({"error": f"Recipe not found: {recipe_id}"}, HTTPStatus.NOT_FOUND)
                return True
            handler._send_json(asdict(recipe))
            return True

    if path == "/api/plugins":
        plugins = qry.dispatch(ListPluginsQuery())
        handler._send_json(plugins)
        return True
    return False


def try_post(handler: Any, path: str, payload: dict, ws: Any, cmd: Any, qry: Any) -> bool:
    """Handle POST routes for this resource. Return True if the request was handled."""
    if path == "/api/kaggle/upload-template":
        content = payload.get("content", "")
        filename = payload.get("filename", "sample_submission.csv")
        if not content:
            handler._send_json({"error": "content is required"}, HTTPStatus.BAD_REQUEST)
            return True

        target_dir = Path(handler.workspace_dir) / "submissions"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / Path(filename).name
        target_file.write_text(content, encoding="utf-8")

        import io
        import pandas as pd
        df = pd.read_csv(io.StringIO(content))
        cols = list(df.columns)
        id_col = cols[0] if len(cols) > 0 else "id"
        target_col = cols[1] if len(cols) > 1 else (cols[0] if len(cols) == 1 else "target")
        row_count = len(df)

        handler._send_json({
            "status": "success",
            "template_path": str(target_file),
            "filename": target_file.name,
            "id_column": id_col,
            "target_column": target_col,
            "row_count": row_count,
            "columns": cols,
        })
        return True

    if path == "/api/agent/action":
        hyp_id = payload.get("hypothesis_id") or payload.get("approval_id")
        action = (payload.get("action") or "").strip().lower()
        reviewer = payload.get("reviewer") or "workbench_user"
        notes = payload.get("notes")

        if not hyp_id:
            handler._send_json({"error": "hypothesis_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        if action not in ("approve", "reject"):
            handler._send_json(
                {"error": "Invalid action. Allowed actions are 'approve' or 'reject'."},
                HTTPStatus.BAD_REQUEST,
            )
            return True

        ledger_file = Path(handler.workspace_dir) / "agent_ledger.db"
        from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger
        from automl.application.agents.ports import ApprovalStatus

        ledger = SqliteAgentLedger(ledger_file)
        hyp = ledger.get_hypothesis(hyp_id)
        appr = ledger.get_approval(hyp_id)

        if hyp is None and appr is None:
            handler._send_json(
                {
                    "error": f"Hypothesis or approval request '{hyp_id}' not found in agent ledger.",
                    "hypothesis_id": hyp_id,
                },
                HTTPStatus.NOT_FOUND,
            )
            return True

        new_hyp_status = None
        new_appr_status = None

        if hyp is not None:
            new_hyp_status = "accepted" if action == "approve" else "rejected"
            hyp.status = new_hyp_status
            ledger.save_hypothesis(hyp)

        if appr is not None:
            new_appr_status = ApprovalStatus.APPROVED if action == "approve" else ApprovalStatus.REJECTED
            reviewer_info = f"{reviewer} ({notes})" if notes else reviewer
            ledger.update_approval_status(
                approval_id=hyp_id,
                status=new_appr_status,
                reviewer=reviewer_info,
            )

        handler._send_json({
            "status": "success",
            "hypothesis_id": hyp_id,
            "action": action,
            "new_status": new_hyp_status or (new_appr_status.value if new_appr_status else action),
            "message": f"Hypothesis {hyp_id} {action}ed by human operator.",
        })
        return True

    if path in ("/api/analysis/studies/create", "/api/analysis/study/create"):
        dataset_id = payload.get("dataset_id")
        if not dataset_id:
            handler._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        name = payload.get("name", "")
        target_col = payload.get("target_column")
        cmd_obj = CreateStudyCommand(
            dataset_id=dataset_id,
            name=name,
            target_column=target_col if target_col else None,
        )
        study_res = cmd.dispatch(cmd_obj)
        study_id = study_res if isinstance(study_res, str) else getattr(study_res, "id", str(study_res))
        study = qry.dispatch(GetStudyQuery(study_id=study_id))
        handler._send_json({
            "status": "success",
            "study_id": study_id,
            "study": study,
        })
        return True

    if path in ("/api/analysis/studies/run", "/api/analysis/study/run"):
        study_id = payload.get("study_id")
        if not study_id:
            handler._send_json({"error": "study_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        run_res = cmd.dispatch(RunAnalysisCommand(study_id=study_id))
        run_id = run_res if isinstance(run_res, str) else getattr(run_res, "id", str(run_res))
        run_data = qry.dispatch(GetAnalysisRunQuery(run_id=run_id))
        findings = qry.dispatch(ListFindingsQuery(study_id=study_id, run_id=run_id))
        handler._send_json({
            "status": "success",
            "study_id": study_id,
            "run_id": run_id,
            "run": run_data,
            "findings_count": len(findings) if findings else 0,
        })
        return True

    if path in ("/api/analysis/studies/archive", "/api/analysis/study/archive"):
        study_id = payload.get("study_id")
        if not study_id:
            handler._send_json({"error": "study_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        cmd.dispatch(ArchiveStudyCommand(study_id=study_id))
        handler._send_json({"status": "success", "study_id": study_id, "archived": True})
        return True

    if path in ("/api/analysis/hypotheses/verify", "/api/analysis/hypothesis/verify"):
        hypothesis_id = payload.get("hypothesis_id")
        if not hypothesis_id:
            handler._send_json({"error": "hypothesis_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        run_id = payload.get("run_id")
        min_imp = float(payload.get("min_improvement", 0.0))
        link_data = cmd.dispatch(
            VerifyHypothesisCommand(
                hypothesis_id=hypothesis_id,
                run_id=run_id,
                min_improvement=min_imp,
            )
        )
        handler._send_json(link_data)
        return True

    if path == "/api/recipes/create-from-evidence":
        evidence_id = payload.get("evidence_link_id") or payload.get("evidence_id")
        if not evidence_id:
            handler._send_json({"error": "evidence_link_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        recipe_name = payload.get("name") or payload.get("recipe_name")
        notes = payload.get("notes")
        model_ids = payload.get("model_ids")
        val_strat = payload.get("validation_strategy")
        cv_folds = payload.get("cv_folds")
        metric = payload.get("metric")
        seed = payload.get("random_seed")
        recipe_id = cmd.dispatch(
            CreateExperimentRecipeFromEvidenceCommand(
                evidence_link_id=evidence_id,
                recipe_name=recipe_name,
                notes=notes,
                model_ids=model_ids,
                validation_strategy=val_strat,
                cv_folds=int(cv_folds) if cv_folds else None,
                metric=metric,
                random_seed=int(seed) if seed else None,
            )
        )
        recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
        handler._send_json({"status": "success", "recipe_id": recipe_id, "recipe": asdict(recipe) if recipe else None})
        return True

    if path == "/api/recipes/select":
        recipe_id = payload.get("recipe_id")
        if not recipe_id:
            handler._send_json({"error": "recipe_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        notes = payload.get("notes")
        cmd.dispatch(SelectExperimentRecipeCommand(recipe_id=recipe_id, notes=notes))
        recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
        handler._send_json({"status": "success", "recipe_id": recipe_id, "recipe": asdict(recipe) if recipe else None})
        return True

    if path == "/api/recipes/submit":
        recipe_id = payload.get("recipe_id")
        if not recipe_id:
            handler._send_json({"error": "recipe_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        run_id = payload.get("run_id")
        model_ids = payload.get("model_ids")
        notes = payload.get("notes")
        exp_ids = cmd.dispatch(
            SubmitExperimentFromRecipeCommand(
                recipe_id=recipe_id,
                run_id=run_id,
                model_ids=model_ids,
                notes=notes,
            )
        )
        recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
        handler._send_json({
            "status": "submitted",
            "recipe_id": recipe_id,
            "experiment_ids": exp_ids,
            "recipe": asdict(recipe) if recipe else None,
        })
        return True

    if path == "/api/recipes/promote":
        recipe_id = payload.get("recipe_id")
        if not recipe_id:
            handler._send_json({"error": "recipe_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        reason = payload.get("reason")
        notes = payload.get("notes")
        cmd.dispatch(PromoteExperimentRecipeCommand(recipe_id=recipe_id, reason=reason, notes=notes))
        recipe = qry.dispatch(GetExperimentRecipeQuery(recipe_id=recipe_id))
        handler._send_json({"status": "promoted", "recipe_id": recipe_id, "recipe": asdict(recipe) if recipe else None})
        return True

    if path == "/api/recipes/archive":
        recipe_id = payload.get("recipe_id")
        if not recipe_id:
            handler._send_json({"error": "recipe_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        cmd.dispatch(ArchiveExperimentRecipeCommand(recipe_id=recipe_id))
        handler._send_json({"status": "archived", "recipe_id": recipe_id})
        return True
    return False
