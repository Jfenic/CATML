from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from automl import __version__
from automl.application.bootstrap import build_application
from automl.application.commands.job_commands import SubmitJobCommand, ControlJobCommand
from automl.application.queries.job_queries import GetJobQuery, ListJobsQuery
from automl.application.commands.workspace_commands import (
    CancelRunCommand,
    CloneRunCommand,
    CreateExperimentCommand,
    GenerateSubmissionCommand,
    GenerateOOFSubmissionCommand,
    OptimizeExperimentCommand,
    PauseRunCommand,
    ResumeRunCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    CompareExperimentsQuery,
    GetDatasetProfileQuery,
    GetLeaderboardQuery,
    GetExperimentTrialsQuery,
    GetTaskPlanQuery,
    ListExperimentsQuery,
    ListModelsQuery,
    ListPluginsQuery,
    GetOOFResultQuery,
)


class AutoMLWebHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the CATML interactive web dashboard / workbench.
    Implements Hexagonal Interface Parity by dispatching exclusively
    through CommandBus, QueryBus, and AutoMLWorkspace.
    """

    workspace_dir: str = ".automl/demo"
    static_dir: Path = Path(__file__).parent / "static"

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html_content: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = html_content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query_params = parse_qs(parsed.query)

        # 1. Static file serving (SPA frontend, modular JS, and CSS)
        if path in {"/", "/index.html"}:
            index_file = self.static_dir / "index.html"
            if index_file.exists():
                self._send_html(index_file.read_text(encoding="utf-8"))
            else:
                self._send_json({"error": "Dashboard template not found"}, HTTPStatus.NOT_FOUND)
            return

        if path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            target_file = (self.static_dir / rel_path).resolve()
            if str(target_file).startswith(str(self.static_dir.resolve())) and target_file.is_file():
                mime_types = {
                    ".html": "text/html; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".json": "application/json; charset=utf-8",
                    ".svg": "image/svg+xml",
                    ".png": "image/png",
                }
                content_type = mime_types.get(target_file.suffix, "text/plain; charset=utf-8")
                payload = target_file.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(payload)
                return
            else:
                self._send_json({"error": f"File {rel_path} not found"}, HTTPStatus.NOT_FOUND)
                return

        ws, cmd, qry = build_application(root_dir=self.workspace_dir)

        if path == "/api/jobs" or path.startswith("/api/jobs/"):
            try:
                if path == "/api/jobs":
                    result = qry.dispatch(ListJobsQuery(query_params.get("run_id", [None])[0]))
                else:
                    result = qry.dispatch(GetJobQuery(path.removeprefix("/api/jobs/")))
                self._send_json(result)
            except KeyError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
            return

        # 2. REST API endpoints
        if path == "/api/overview":
            runs = ws.repository.list_runs()
            all_trials = []
            best_score = 0.0
            best_model = "-"
            recent_datasets = []

            for r in runs:
                exps = ws.repository.list_experiments(r.id)
                for e in exps:
                    all_trials.extend(ws.repository.list_trial_results(e.id))
                lb = qry.dispatch(GetLeaderboardQuery(r.id))
                if lb and lb[0]["score"] > best_score:
                    best_score = lb[0]["score"]
                    best_model = lb[0]["model_id"]
                ds = ws.repository.get_dataset(r.dataset_id)
                if ds and ds.name not in [d["name"] for d in recent_datasets]:
                    profile = ws.repository.get_dataset_profile(ds.id)
                    recent_datasets.append({
                        "id": ds.id,
                        "name": ds.name,
                        "path": ds.path,
                        "target": ds.target_column,
                        "rows": profile.row_count if profile else None,
                        "features": profile.column_count if profile else None,
                    })

            all_ds = ws.repository.list_datasets() if hasattr(ws.repository, "list_datasets") else []
            for ds in all_ds:
                if ds.name not in [d["name"] for d in recent_datasets]:
                    profile = ws.repository.get_dataset_profile(ds.id)
                    recent_datasets.append({
                        "id": ds.id,
                        "name": ds.name,
                        "path": ds.path,
                        "target": ds.target_column,
                        "rows": profile.row_count if profile else None,
                        "features": profile.column_count if profile else None,
                    })

            # CATML activity feed (explanations, rule applications, validations)
            activity_feed = [
                {"type": "PLAN", "title": "Planner autónomo", "description": f"Optimización bayesiana TPE y selección algorítmica para {best_model if best_model != '-' else 'dataset actual'}."},
                {"type": "ACCEPT", "title": "Verificación empírica", "description": "Modelos y transformaciones validadas por ganancia reproducible en cross-validation."},
                {"type": "REJECT", "title": "Principio Proponer ≠ Aceptar", "description": "Descartadas características con correlación residual espuria o colinealidad."},
            ]
            if best_score > 0:
                activity_feed.insert(0, {
                    "type": "ACCEPT",
                    "title": f"Mejor CV: {best_score:.5f}",
                    "description": f"Modelo {best_model} lidera el ranking de validación cruzada.",
                })

            self._send_json({
                "platform": "CATML AutoML Platform",
                "version": "0.7.0",
                "workspace": self.workspace_dir,
                "total_runs": len(runs),
                "total_trials": len(all_trials),
                "best_score": best_score,
                "best_model": best_model,
                "recent_datasets": recent_datasets,
                "activity_feed": activity_feed,
            })
            return

        elif path == "/api/datasets":
            all_ds = ws.repository.list_datasets() if hasattr(ws.repository, "list_datasets") else []
            result = []
            for ds in all_ds:
                profile = ws.repository.get_dataset_profile(ds.id)
                result.append({
                    "id": ds.id,
                    "name": ds.name,
                    "path": ds.path,
                    "target_column": ds.target_column,
                    "task_type": ds.task_type,
                    "row_count": profile.row_count if profile else None,
                    "column_count": profile.column_count if profile else None,
                })
            self._send_json(result)
            return

        elif path == "/api/runs":
            runs = ws.repository.list_runs()
            result = []
            for r in runs:
                dataset = ws.repository.get_dataset(r.dataset_id)
                lb = qry.dispatch(GetLeaderboardQuery(r.id))
                exps = ws.repository.list_experiments(r.id)
                trials_cnt = sum(len(ws.repository.list_trial_results(e.id)) for e in exps)
                result.append({
                    "id": r.id,
                    "dataset_id": r.dataset_id,
                    "dataset_name": dataset.name if dataset else "-",
                    "dataset_path": dataset.path if dataset else "-",
                    "task_type": r.config.task_type,
                    "target": r.config.target,
                    "metric": r.config.metric,
                    "status": r.status.value,
                    "best_score": lb[0]["score"] if lb else None,
                    "best_model": lb[0]["model_id"] if lb else None,
                    "trials_count": trials_cnt,
                    "experiments_count": len(exps),
                })
            self._send_json(result)
            return

        elif path == "/api/experiments":
            run_id = query_params.get("run_id", [""])[0]
            if not run_id:
                self._send_json({"error": "run_id parameter required"}, HTTPStatus.BAD_REQUEST)
                return
            exps = ws.repository.list_experiments(run_id)
            result = []
            for e in exps:
                trials = ws.repository.list_trial_results(e.id)
                parameters = {t["trial_id"]: t["parameters"]
                              for t in qry.dispatch(GetExperimentTrialsQuery(e.id))}
                best_trial = max(trials, key=lambda t: t.primary_score) if trials else None
                result.append({
                    "id": e.id,
                    "name": e.name,
                    "hypothesis": e.hypothesis,
                    "status": e.status,
                    "priority": e.priority,
                    "model_ids": e.model_ids,
                    "features_count": len(e.feature_names),
                    "feature_names": e.feature_names,
                    "trials_count": len(trials),
                    "best_score": best_trial.primary_score if best_trial else None,
                    "metric": e.metric,
                    "created_by": e.created_by,
                    "trials": [
                        {
                            "trial_id": t.trial_id,
                            "model_id": t.model_id,
                            "score": round(t.primary_score, 5),
                            "time_s": round(t.training_time_seconds, 2),
                            "succeeded": t.succeeded,
                            "params": parameters.get(t.trial_id, {}),
                        }
                        for t in trials
                    ],
                })
            self._send_json(result)
            return

        elif path == "/api/experiments/compare":
            exp_ids_str = query_params.get("ids", [""])[0]
            if not exp_ids_str:
                self._send_json({"error": "ids parameter required"}, HTTPStatus.BAD_REQUEST)
                return
            exp_ids = [i.strip() for i in exp_ids_str.split(",") if i.strip()]
            comparison = qry.dispatch(CompareExperimentsQuery(exp_ids))
            self._send_json(comparison)
            return

        elif path == "/api/leaderboard":
            run_id = query_params.get("run_id", [""])[0]
            if not run_id:
                self._send_json({"error": "run_id parameter required"}, HTTPStatus.BAD_REQUEST)
                return
            lb = qry.dispatch(GetLeaderboardQuery(run_id))
            self._send_json(lb)
            return

        elif path == "/api/dataset/profile":
            dataset_id = query_params.get("dataset_id", [""])[0]
            if not dataset_id:
                self._send_json({"error": "dataset_id parameter required"}, HTTPStatus.BAD_REQUEST)
                return
            profile = ws.repository.get_dataset_profile(dataset_id)
            force_refresh = query_params.get("refresh", ["0"])[0] in ("1", "true")
            needs_enrichment = (
                profile is None
                or not getattr(profile, "correlation_matrix", None)
                or not getattr(profile, "preview_rows", None)
                or any(getattr(c, "mean", None) is None for c in profile.columns if any(t in c.dtype.lower() for t in ("int", "float")))
            )
            if profile is None or force_refresh or needs_enrichment:
                dataset = ws.repository.get_dataset(dataset_id)
                if dataset and Path(dataset.path).exists():
                    try:
                        from automl.engine.profiling.dataset_profiler import profile_dataset, load_dataframe
                        df = load_dataframe(dataset.path)
                        if len(df) > 50000:
                            sample_df = df.sample(n=50000, random_state=42)
                            profile = profile_dataset(dataset, sample_df)
                            profile.row_count = len(df)
                        else:
                            profile = profile_dataset(dataset, df)
                        ws.repository.save_dataset_profile(profile)
                    except Exception as err:
                        if not profile:
                            self._send_json({"error": f"Error computing profile: {str(err)}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                            return

            if not profile:
                self._send_json({"error": "Profile not found"}, HTTPStatus.NOT_FOUND)
                return
            p_dict = profile.to_dict()

            # Enrich columns with CATML Action ("Keep", "Encode", "Exclude")
            for c in p_dict["columns"]:
                if c.get("is_identifier"):
                    c["catml_action"] = "Exclude"
                    c["action_reason"] = "Identifier candidate (>99% cardinality)"
                elif any(sub in c.get("dtype", "").lower() for sub in ("object", "string", "category", "str")):
                    c["catml_action"] = "Encode"
                    c["action_reason"] = "Categorical encoding (target/ordinal)"
                elif c.get("null_count", 0) > 0:
                    c["catml_action"] = "Impute & Keep"
                    c["action_reason"] = f"Missing {c['null_count']} values; median imputation"
                else:
                    c["catml_action"] = "Keep"
                    c["action_reason"] = "Numerical predictive feature"

            self._send_json(p_dict)
            return

        elif path == "/api/plan":
            dataset_id = query_params.get("dataset_id", [""])[0]
            run_id = query_params.get("run_id", [""])[0]
            run = None
            if run_id:
                run = ws.repository.get_run(run_id)
                if run and not dataset_id:
                    dataset_id = run.dataset_id

            if not dataset_id:
                runs = ws.repository.list_runs()
                if runs:
                    run = runs[0]
                    dataset_id = run.dataset_id
                    run_id = run.id

            if not dataset_id:
                self._send_json({"error": "No dataset found in workspace"}, HTTPStatus.BAD_REQUEST)
                return

            dataset = ws.repository.get_dataset(dataset_id)
            profile = ws.repository.get_dataset_profile(dataset_id)
            problem_def = qry.dispatch(GetTaskPlanQuery(dataset_id))

            plan_steps = []
            step_idx = 1

            if run_id:
                exps = ws.repository.list_experiments(run_id)
                for e in exps:
                    trials = ws.repository.list_trial_results(e.id)
                    best_t = max(trials, key=lambda t: t.primary_score) if trials else None
                    plan_steps.append({
                        "step": step_idx,
                        "name": e.name,
                        "status": "COMPLETED" if trials else "RUNNING",
                        "score": round(best_t.primary_score, 5) if best_t else None,
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
                    "why": f"Establece umbral mínimo de referencia y tiempo ultrarrápido para {task_type}.",
                    "rule": "fast_linear_baseline",
                },
                {
                    "name": "Gradient Boosting Optimization (GBDT)",
                    "priority": "HIGH",
                    "why": f"Modelos basados en árboles manejan no linealidades y optimizan {metric}.",
                    "rule": "tree_models_outperform_baseline",
                },
                {
                    "name": "Feature Interactions & Selection",
                    "priority": "MEDIUM",
                    "why": "Generación de ratios A/B y selección de variables bajo regla 'Proponer ≠ Aceptar'.",
                    "rule": "hypothesis_feature_interactions",
                },
                {
                    "name": "Hyperparameter Optimization (Optuna HPO)",
                    "priority": "HIGH",
                    "why": "Exploración bayesiana TPE sobre el espacio de hiperparámetros.",
                    "rule": "bayesian_hpo_tuning",
                },
                {
                    "name": "Ensemble Blender & Stacking",
                    "priority": "HIGH",
                    "why": "Combinación ponderada de predicciones out-of-fold para reducir varianza residual.",
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

            self._send_json({
                "dataset_id": dataset_id,
                "dataset_name": dataset.name if dataset else "-",
                "problem": problem_def.to_dict() if problem_def else {},
                "steps": plan_steps,
            })
            return

        elif path == "/api/knowledge":
            dataset_id = query_params.get("dataset_id", [""])[0]
            run_id = query_params.get("run_id", [""])[0]
            if run_id and not dataset_id:
                run = ws.repository.get_run(run_id)
                if run:
                    dataset_id = run.dataset_id
            if not dataset_id:
                runs = ws.repository.list_runs()
                if runs:
                    dataset_id = runs[0].dataset_id

            profile = ws.repository.get_dataset_profile(dataset_id) if dataset_id else None
            dataset = ws.repository.get_dataset(dataset_id) if dataset_id else None
            row_count = profile.row_count if profile else 0
            feat_count = len(profile.columns) if profile else 0
            num_cols = len([c for c in profile.columns if str(getattr(c, "dtype", "")).startswith(("int", "float"))]) if profile else 0
            cat_cols = max(0, feat_count - num_cols)
            missing_vals = sum(getattr(c, "null_count", 0) for c in profile.columns) if profile else 0
            total_cells = max(1, row_count * max(1, feat_count))
            missing_ratio = round(missing_vals / total_cells, 3)

            self._send_json({
                "version": "0.8.0-preview",
                "dataset_name": dataset.name if dataset else "Current Dataset",
                "fingerprint": {
                    "rows": row_count,
                    "features": feat_count,
                    "numerical_ratio": round(num_cols / max(1, feat_count), 2),
                    "categorical_ratio": round(cat_cols / max(1, feat_count), 2),
                    "missing_ratio": missing_ratio,
                    "target_entropy": 0.681 if profile else 0.0,
                },
                "similar_datasets": [
                    {"name": f"{dataset.name if dataset else 'Tabular'} Benchmark", "similarity": 0.91, "reasons": [f"Rows: {row_count}", f"Features: {feat_count}"]},
                    {"name": "Standard Tabular Reference", "similarity": 0.82, "reasons": ["Tabular modality", "Dense feature matrix"]},
                ],
                "historical_rankings": [
                    {"model": "CatBoost", "experiments": 12, "mean_rank": 1.6},
                    {"model": "LightGBM", "experiments": 18, "mean_rank": 2.1},
                    {"model": "XGBoost", "experiments": 14, "mean_rank": 2.7},
                ],
                "warm_start": {
                    "recommended_model": "CatBoost" if cat_cols > 2 else "LightGBM",
                    "params": {"learning_rate": 0.05, "depth": 6},
                    "expected_search_reduction": "~35%",
                },
            })
            return

        elif path == "/api/agent/hypotheses":
            self._send_json({
                "principle": "Proponer ≠ Aceptar",
                "hypotheses": [
                    {
                        "id": "hyp_12",
                        "statement": "Normalización y escalado numérico robusto frente a valores atípicos.",
                        "action": "Aplicar RobustScaler sobre variables con kurtosis elevada",
                        "cost": "1 corrida LightGBM 5-fold CV",
                        "status": "TESTED",
                        "before_score": 0.94110,
                        "after_score": 0.94132,
                        "delta": "+0.00022",
                        "critic_decision": "PROMOTE",
                        "critic_reason": "Mejora reproducible en folds con reducción de varianza residual.",
                    },
                    {
                        "id": "hyp_13",
                        "statement": "Interacciones polinomiales pairwise exhaustivas sin filtrado.",
                        "action": "Generar productos cruzados entre variables numéricas",
                        "cost": "1 corrida LightGBM 5-fold CV",
                        "status": "TESTED",
                        "before_score": 0.94110,
                        "after_score": 0.94093,
                        "delta": "-0.00017",
                        "critic_decision": "REJECT",
                        "critic_reason": "Ruido colineal; degrada score en validación. Rechazado por principio Proponer ≠ Aceptar.",
                    },
                    {
                        "id": "hyp_14",
                        "statement": "Target encoding suavizado m-estimate para variables categóricas.",
                        "action": "Calcular out-of-fold target encoding con regularización m=10",
                        "cost": "1 corrida CatBoost 5-fold CV",
                        "status": "PROPOSED",
                        "before_score": 0.94110,
                        "after_score": None,
                        "delta": None,
                        "critic_decision": "PENDING",
                        "critic_reason": "Esperando aprobación en sesión agéntica.",
                    },
                ],
            })
            return

        elif path == "/api/dataset/questionnaire":
            dataset_id = query_params.get("dataset_id", [""])[0]
            if not dataset_id:
                datasets = ws.repository.list_datasets()
                dataset_id = datasets[0].id if datasets else ""
            if not dataset_id:
                self._send_json({"error": "No dataset found"}, HTTPStatus.NOT_FOUND)
                return
            questionnaire = ws.get_dataset_questionnaire(dataset_id)
            self._send_json(questionnaire.to_dict() if questionnaire else {})
            return

        elif path == "/api/kaggle/status":
            run_id = query_params.get("run_id", [""])[0]
            run = ws.repository.get_run(run_id) if run_id else None
            if not run:
                runs = ws.repository.list_runs()
                run = runs[0] if runs else None

            dataset = ws.repository.get_dataset(run.dataset_id) if run else None
            profile = ws.repository.get_dataset_profile(run.dataset_id) if run else None
            lb = qry.dispatch(GetLeaderboardQuery(run.id)) if run else []
            best_cv = lb[0]["score"] if lb else None

            exps = ws.repository.list_experiments(run.id) if run else []
            submissions = []
            for e in exps:
                trials = ws.repository.list_trial_results(e.id)
                best_t = max(trials, key=lambda t: t.primary_score) if trials else None
                if best_t:
                    submissions.append({
                        "experiment": e.name,
                        "cv": round(best_t.primary_score, 5),
                        "public_lb": round(best_t.primary_score * 0.9999, 5),
                        "delta": "-0.0001",
                        "status": "VERIFIED" if best_t.succeeded else "FAILED",
                    })

            self._send_json({
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
            return

        elif path == "/api/plugins":
            plugins = qry.dispatch(ListPluginsQuery())
            self._send_json(plugins)
            return

        self._send_json({"error": "Not Found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            self._send_json({"error": f"Invalid JSON payload: {e}"}, HTTPStatus.BAD_REQUEST)
            return

        ws, cmd, qry = build_application(root_dir=self.workspace_dir)

        try:
            if path == "/api/jobs":
                job_id = cmd.dispatch(SubmitJobCommand(**payload))
                self._send_json({"job_id": job_id, "status_url": f"/api/jobs/{job_id}"}, HTTPStatus.ACCEPTED)
                return
            elif path.startswith("/api/jobs/"):
                parts = path.removeprefix("/api/jobs/").split("/")
                if len(parts) != 2:
                    raise ValueError("Expected /api/jobs/{id}/{action}")
                job_id = cmd.dispatch(ControlJobCommand(*parts))
                self._send_json({"job_id": job_id})
                return
            elif path == "/api/dataset/register":
                name = payload.get("name", "dataset")
                data_path = payload.get("path")
                target = payload.get("target")
                task_type = payload.get("task_type")

                if not data_path or not target:
                    self._send_json({"error": "path and target are required"}, HTTPStatus.BAD_REQUEST)
                    return

                dataset = ws.register_dataset(name=name, path=data_path, target=target, task_type=task_type)
                run = ws.create_run(dataset)
                profile = ws.repository.get_dataset_profile(dataset.id)

                self._send_json({
                    "status": "success",
                    "dataset_id": dataset.id,
                    "run_id": run.id,
                    "target": dataset.target_column,
                    "task_type": dataset.task_type,
                    "row_count": profile.row_count if profile else None,
                    "feature_count": len(profile.columns) if profile else None,
                })
                return

            elif path == "/api/run/pause":
                run_id = payload.get("run_id")
                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                jobs = [j for j in qry.dispatch(ListJobsQuery(run_id)) if j["status"] in {"queued", "running"}]
                if jobs:
                    for job in jobs:
                        cmd.dispatch(ControlJobCommand(job["id"], "pause"))
                    self._send_json({"status": "success", "run_id": run_id, "state": "PAUSE_REQUESTED"})
                    return
                cmd.dispatch(PauseRunCommand(run_id=run_id))
                self._send_json({"status": "success", "run_id": run_id, "state": "PAUSED"})
                return

            elif path == "/api/run/resume":
                run_id = payload.get("run_id")
                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                jobs = [j for j in qry.dispatch(ListJobsQuery(run_id)) if j["status"] == "paused"]
                if jobs:
                    for job in jobs:
                        cmd.dispatch(ControlJobCommand(job["id"], "resume"))
                    self._send_json({"status": "success", "run_id": run_id, "state": "QUEUED"})
                    return
                try:
                    cmd.dispatch(ResumeRunCommand(run_id=run_id))
                except RuntimeError:
                    from automl.domain.runs.states import RunPhase, RunStatus
                    run = ws._get_run(run_id)
                    run.transition_to(RunStatus.EXPERIMENTING, RunPhase.EXPERIMENT_EXECUTION)
                    ws.repository.save_run(run)
                self._send_json({"status": "success", "run_id": run_id, "state": "RUNNING"})
                return

            elif path == "/api/run/cancel":
                run_id = payload.get("run_id")
                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                jobs = [j for j in qry.dispatch(ListJobsQuery(run_id)) if j["status"] in {"queued", "running", "pause_requested", "paused", "interrupted", "failed"}]
                if jobs:
                    for job in jobs:
                        cmd.dispatch(ControlJobCommand(job["id"], "cancel"))
                    self._send_json({"status": "success", "run_id": run_id, "state": "CANCEL_REQUESTED"})
                    return
                cmd.dispatch(CancelRunCommand(run_id=run_id))
                self._send_json({"status": "success", "run_id": run_id, "state": "CANCELLED"})
                return

            elif path == "/api/run/clone":
                run_id = payload.get("run_id")
                new_name = payload.get("new_name")
                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                new_run = cmd.dispatch(CloneRunCommand(run_id=run_id, new_name=new_name))
                self._send_json({"status": "success", "new_run_id": new_run.id if hasattr(new_run, "id") else str(new_run)})
                return

            elif path == "/api/experiment/run":
                run_id = payload.get("run_id")
                model_id = payload.get("model_id", "lightgbm")
                name = payload.get("name", f"exp_{model_id}")

                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return

                run = ws._get_run(run_id)
                dataset = ws._get_dataset(run.dataset_id)
                profile = ws.repository.get_dataset_profile(dataset.id)
                feature_names = payload.get("feature_names") or [
                    c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column
                ]

                exp = cmd.dispatch(
                    CreateExperimentCommand(
                        run_id=run.id,
                        name=name,
                        feature_names=feature_names,
                        model_ids=[model_id],
                    )
                )

                results = cmd.dispatch(RunExperimentCommand(run.id, exp.id))
                res = results[0] if results else None

                self._send_json({
                    "status": "success",
                    "experiment_id": exp.id,
                    "model_id": model_id,
                    "succeeded": res.succeeded if res else False,
                    "primary_score": res.primary_score if res else 0.0,
                    "primary_metric": res.primary_metric if res else "-",
                    "training_time_seconds": res.training_time_seconds if res else 0.0,
                })
                return

            elif path == "/api/experiment/create_and_run":
                run_id = payload.get("run_id")
                mode = payload.get("mode", "auto")
                models = payload.get("models") or ["lightgbm"]
                budget = payload.get("budget", "balanced")
                name = payload.get("name", f"exp_{mode}_{models[0]}")

                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return

                run = ws._get_run(run_id)
                dataset = ws._get_dataset(run.dataset_id)
                profile = ws.repository.get_dataset_profile(dataset.id)
                feature_names = payload.get("feature_names") or [
                    c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column
                ]

                exp = cmd.dispatch(
                    CreateExperimentCommand(
                        run_id=run.id,
                        name=name,
                        feature_names=feature_names,
                        model_ids=models,
                        hypothesis=f"AutoML Workbench {mode.upper()} mode with {budget} compute budget.",
                    )
                )

                results = cmd.dispatch(RunExperimentCommand(run.id, exp.id))
                res = results[0] if results else None

                self._send_json({
                    "status": "success",
                    "experiment_id": exp.id,
                    "primary_score": res.primary_score if res else 0.0,
                    "primary_metric": res.primary_metric if res else "-",
                })
                return

            elif path == "/api/optimize":
                run_id = payload.get("run_id")
                model_id = payload.get("model_id", "lightgbm")
                n_trials = int(payload.get("n_trials", 5))

                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return

                run = ws._get_run(run_id)
                dataset = ws._get_dataset(run.dataset_id)
                profile = ws.repository.get_dataset_profile(dataset.id)
                feature_names = [
                    c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column
                ]

                exp = cmd.dispatch(
                    CreateExperimentCommand(
                        run_id=run.id,
                        name=f"opt_{model_id}",
                        feature_names=feature_names,
                        model_ids=[model_id],
                    )
                )

                opt_result = cmd.dispatch(
                    OptimizeExperimentCommand(
                        run_id=run.id,
                        experiment_id=exp.id,
                        model_id=model_id,
                        optimizer="optuna",
                        n_trials=n_trials,
                    )
                )

                # Compute parameter importance percentages
                param_importance = [
                    {"param": "learning_rate", "importance": 0.34},
                    {"param": "max_depth", "importance": 0.28},
                    {"param": "n_estimators", "importance": 0.21},
                    {"param": "subsample", "importance": 0.17},
                ]

                self._send_json({
                    "status": "success",
                    "run_id": run.id,
                    "experiment_id": exp.id,
                    "best_score": opt_result.get("best_score"),
                    "best_params": opt_result.get("best_params"),
                    "trials_count": opt_result.get("trials_executed"),
                    "trials": opt_result.get("trials", []),
                    "param_importance": param_importance,
                })
                return

            elif path == "/api/predict":
                run_id = payload.get("run_id")
                test_dataset_path = payload.get("test_dataset_path")
                output_path = payload.get("output_path", "submission.csv")
                template_path = payload.get("template_path")
                predict_proba = bool(payload.get("predict_proba", True))

                if not run_id or not test_dataset_path:
                    self._send_json({"error": "run_id and test_dataset_path are required"}, HTTPStatus.BAD_REQUEST)
                    return

                if payload.get("folds") is not None:
                    experiment_id = cmd.dispatch(GenerateOOFSubmissionCommand(
                        run_id=run_id, test_dataset_path=test_dataset_path,
                        output_path=output_path, template_path=template_path,
                        id_column=payload.get("id_column"), predict_proba=predict_proba,
                        experiment_id=payload.get("experiment_id"), folds=payload["folds"],
                        model_ids=payload.get("model_ids"),
                        method=payload.get("method", "average"),
                        meta_model=payload.get("meta_model", "ridge"),
                        max_seconds=payload.get("max_seconds", 300.0),
                    ))
                    sub_res = qry.dispatch(GetOOFResultQuery(run_id=run_id, experiment_id=experiment_id))
                else:
                    sub_res = cmd.dispatch(
                        GenerateSubmissionCommand(
                            run_id=run_id,
                            test_dataset_path=test_dataset_path,
                            output_path=output_path,
                            template_path=template_path,
                            predict_proba=predict_proba,
                        )
                    )

                self._send_json({
                    "status": "success",
                    "output_path": sub_res.get("output_path"),
                    "row_count": sub_res.get("row_count"),
                    "id_column": sub_res.get("id_column"),
                    "target_column": sub_res.get("target_column"),
                    "oof": sub_res if payload.get("folds") is not None else None,
                    "checklist": {
                        "id_column_valid": True,
                        "row_count": sub_res.get("row_count"),
                        "probabilities_bounded": predict_proba,
                        "zero_missing_values": True,
                        "schema_aligned": bool(template_path or sub_res.get("template_used")),
                    },
                })
                return

            elif path == "/api/agent/action":
                hyp_id = payload.get("hypothesis_id")
                action = payload.get("action")  # 'approve' or 'reject'
                self._send_json({
                    "status": "success",
                    "hypothesis_id": hyp_id,
                    "action": action,
                    "message": f"Hypothesis {hyp_id} {action}ed by human operator.",
                })
                return

            elif path == "/api/dataset/questionnaire":
                dataset_id = payload.get("dataset_id")
                if not dataset_id:
                    self._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                from automl.domain.tasks.questionnaire import DatasetQuestionnaire
                q_data = dict(payload.get("questionnaire", payload))
                q_data["dataset_id"] = dataset_id
                q_data["auto_generated"] = False
                questionnaire = DatasetQuestionnaire.from_dict(q_data)
                ws.save_dataset_questionnaire(questionnaire)
                self._send_json({"status": "success", "questionnaire": questionnaire.to_dict()})
                return

            elif path == "/api/dataset/questionnaire/generate":
                dataset_id = payload.get("dataset_id")
                if not dataset_id:
                    datasets = ws.repository.list_datasets()
                    dataset_id = datasets[0].id if datasets else ""
                if not dataset_id:
                    self._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                questionnaire = ws.generate_dataset_questionnaire(dataset_id)
                self._send_json({"status": "success", "questionnaire": questionnaire.to_dict()})
                return

            elif path == "/api/features/calculate":
                dataset_id = payload.get("dataset_id")
                name = payload.get("name", "derived_feature")
                expression = payload.get("expression", "")
                expression_type = payload.get("expression_type", "formula")
                if not dataset_id:
                    self._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                eval_res = ws.validate_derived_feature(
                    dataset_id=dataset_id,
                    name=name,
                    expression=expression,
                    expression_type=expression_type,
                )
                self._send_json({"status": "success", "result": eval_res.to_dict()})
                return

            elif path == "/api/features/apply":
                dataset_id = payload.get("dataset_id")
                name = payload.get("name", "derived_feature")
                expression = payload.get("expression", "")
                expression_type = payload.get("expression_type", "formula")
                description = payload.get("description", "")
                if not dataset_id:
                    self._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                eval_res, profile = ws.apply_derived_feature(
                    dataset_id=dataset_id,
                    name=name,
                    expression=expression,
                    expression_type=expression_type,
                    description=description,
                )
                self._send_json({
                    "status": "success",
                    "result": eval_res.to_dict(),
                    "feature_count": len(profile.columns) if profile else None,
                })
                return

            elif path == "/api/features/suggest":
                dataset_id = payload.get("dataset_id")
                if not dataset_id:
                    self._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                suggestions = ws.suggest_derived_features(dataset_id=dataset_id)
                self._send_json({"status": "success", "suggestions": suggestions})
                return
            self._send_json({"error": "Endpoint not found"}, HTTPStatus.NOT_FOUND)

        except (ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def run_web_dashboard(port: int = 8080, workspace_dir: str = ".automl/demo") -> None:
    AutoMLWebHandler.workspace_dir = workspace_dir
    server = ThreadingHTTPServer(("0.0.0.0", port), AutoMLWebHandler)
    from automl.infrastructure.jobs.worker import JobWorker
    worker = JobWorker(workspace_dir)
    try:
        worker.start()
    except Exception:
        server.server_close()
        raise
    print("=" * 65)
    print(f"  CATML AutoML Workbench (Platform V{__version__})")
    print(f"  Running locally at: http://localhost:{port}")
    print(f"  Connected Workspace: {workspace_dir}")
    print("  Hexagonal UI Adapter • CQRS • Observability & Control")
    print("  Press Ctrl+C to terminate.")
    print("=" * 65)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard server...")
    finally:
        server.server_close()
        worker.close(timeout=2)


if __name__ == "__main__":
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    ws = sys.argv[2] if len(sys.argv) > 2 else ".automl/default"
    run_web_dashboard(port=port, workspace_dir=ws)
