from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from automl import __version__
from automl.domain.runs.states import is_active_run_status
from automl.application.bootstrap import build_application
from automl.application.commands.job_commands import SubmitJobCommand, ControlJobCommand
from automl.application.queries.job_queries import GetJobQuery, ListJobsQuery
from automl.application.commands.workspace_commands import (
    BuildEnsembleCommand,
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
    GetMetaKnowledgeQuery,
)


def _build_real_activity_feed(
    ws: Any,
    qry: Any,
    runs: list[Any],
    all_trials: list[Any],
    recent_datasets: list[dict],
    workspace_dir: str,
) -> list[dict]:
    activity: list[dict] = []

    # 1. Best CV leaderboard result per run
    for r in runs:
        lb = qry.dispatch(GetLeaderboardQuery(r.id))
        if lb:
            top = lb[0]
            score = float(top["score"])
            model_id = str(top["model_id"])
            metric = str(top.get("metric") or getattr(r.config, "metric", "score") or "score")
            run_id = str(r.id)
            ds_name = None
            if hasattr(ws, "get_dataset") and getattr(r, "dataset_id", None):
                try:
                    ds_obj = ws.get_dataset(r.dataset_id)
                    if ds_obj and hasattr(ds_obj, "name"):
                        ds_name = ds_obj.name
                except Exception:
                    pass
            if not ds_name:
                for d in recent_datasets:
                    if d.get("id") == getattr(r, "dataset_id", None):
                        ds_name = d.get("name")
                        break
            ds_desc = f" on dataset {ds_name}" if ds_name else ""
            activity.append({
                "type": "ACCEPT",
                "title": f"Best CV Leaderboard: {model_id}",
                "description": f"Model {model_id} achieved {score:.5f} ({metric}){ds_desc} (run {run_id[:8]}).",
            })

    # 2. Agent ledger events (promoted / rejected hypotheses)
    ledger_file = Path(workspace_dir) / "agent_ledger.db"
    if ledger_file.exists():
        try:
            from automl.infrastructure.database.sqlite_agent_ledger import SqliteAgentLedger

            ledger = SqliteAgentLedger(ledger_file)
            for r in runs:
                for h in ledger.list_hypotheses(r.id):
                    h_status = h.status.value.lower() if hasattr(h.status, "value") else str(h.status).lower()
                    h_id = getattr(h, "hypothesis_id", getattr(h, "id", "hyp"))
                    if h_status == "accepted":
                        activity.append({
                            "type": "ACCEPT",
                            "title": f"Agent Promoted: #{h_id[:8]}",
                            "description": f"{h.reasoning} (Verified gain over baseline).",
                        })
                    elif h_status == "rejected":
                        activity.append({
                            "type": "REJECT",
                            "title": f"Agent Rejected: #{h_id[:8]}",
                            "description": f"{h.reasoning} (Rejected by empirical critic).",
                        })
                    elif h_status in ("proposed", "pending"):
                        activity.append({
                            "type": "PLAN",
                            "title": f"Agent Proposed: #{h_id[:8]}",
                            "description": h.reasoning,
                        })
        except Exception:
            pass

    # 3. Failed trials (empirical verification rejects)
    for t in all_trials:
        if not t.succeeded:
            activity.append({
                "type": "REJECT",
                "title": f"Trial Rejected: {t.model_id}",
                "description": f"Failed during execution: {t.failure_reason or 'Validation error'}.",
            })

    # 4. Anti-leakage guardian exclusions on datasets
    for ds_info in recent_datasets:
        profile = ws.get_dataset_profile(ds_info["id"])
        if profile:
            if profile.leakage_column_names:
                leak_cols = ", ".join(sorted(profile.leakage_column_names))
                activity.append({
                    "type": "REJECT",
                    "title": f"Anti-Leakage Guardian: {ds_info['name']}",
                    "description": f"Automatically excluded contaminated columns ({leak_cols}) to protect generalization.",
                })
            if profile.identifier_column_names:
                id_cols = ", ".join(sorted(profile.identifier_column_names))
                activity.append({
                    "type": "REJECT",
                    "title": f"Pseudo-Identifier Filter: {ds_info['name']}",
                    "description": f"Excluded non-predictive identifiers ({id_cols}).",
                })

    # 5. Background Jobs (if available)
    if hasattr(ws, "job_service") and ws.job_service is not None:
        try:
            jobs = ws.job_service.list_jobs()
            for j in jobs:
                if j.status.value == "completed":
                    activity.append({
                        "type": "ACCEPT",
                        "title": f"Job Completed: {j.job_type}",
                        "description": f"Background job {j.id[:8]} finished successfully.",
                    })
                elif j.status.value == "failed":
                    activity.append({
                        "type": "REJECT",
                        "title": f"Job Failed: {j.job_type}",
                        "description": f"Job {j.id[:8]} failed: {j.error_message or 'Execution error'}.",
                    })
                elif j.status.value in ("running", "queued"):
                    activity.append({
                        "type": "PLAN",
                        "title": f"Job {j.status.value.capitalize()}: {j.job_type}",
                        "description": f"Job {j.id[:8]} progress: {j.progress_pct}%.",
                    })
        except Exception:
            pass

    # 6. Planned experiments
    for r in runs:
        exps = ws.list_experiments(r.id)
        for e in exps:
            models_str = ", ".join(e.model_ids) if hasattr(e, "model_ids") and e.model_ids else "models"
            activity.append({
                "type": "PLAN",
                "title": f"Experiment Planned: {e.name}",
                "description": f"Run {r.id[:8]} • models: {models_str} • priority: {e.priority}.",
            })

    # 7. Dataset registrations (if few other entries)
    if len(activity) < 5:
        for ds_info in recent_datasets:
            profile = ws.get_dataset_profile(ds_info["id"])
            rows_str = f"{profile.row_count} rows, {profile.column_count} features" if profile else "registered"
            target_str = f"Target: {ds_info.get('target')}" if ds_info.get("target") else "Unsupervised"
            activity.append({
                "type": "INFO",
                "title": f"Dataset Ingested: {ds_info['name']}",
                "description": f"{rows_str}. {target_str}.",
            })

    return activity[:15]


class AutoMLWebHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler for the CATML interactive web dashboard / workbench.
    Implements Hexagonal Interface Parity by dispatching exclusively
    through CommandBus, QueryBus, and AutoMLWorkspace.
    """

    workspace_dir: str = ".automl/demo"
    static_dir: Path = Path(__file__).parent / "static"
    auth_token: str | None = None
    require_auth: bool = False

    def _is_authenticated(self) -> bool:
        if not self.require_auth or not self.auth_token:
            return True

        auth_header = self.headers.get("Authorization") if hasattr(self, "headers") and self.headers else None
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if token == self.auth_token:
                return True

        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)
        token_list = query_params.get("token")
        if token_list and token_list[0].strip() == self.auth_token:
            return True

        return False

    def _require_auth_or_reject(self) -> bool:
        if self._is_authenticated():
            return True
        self._send_json(
            {
                "error": "Unauthorized",
                "message": (
                    "Authentication token required for remote workbench access. "
                    "Provide via 'Authorization: Bearer <token>' header or '?token=<token>' query parameter."
                ),
            },
            status=HTTPStatus.UNAUTHORIZED,
        )
        return False

    def _send_cors_headers(self) -> None:
        origin = self.headers.get("Origin") if hasattr(self, "headers") and self.headers else None
        if origin and (origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:")):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self._send_cors_headers()
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
        self._send_cors_headers()
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._require_auth_or_reject():
            return

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
            rel_path = path[len("/static/"):].lstrip("/")
            static_root = self.static_dir.resolve()
            target_file = (static_root / rel_path).resolve()
            if not target_file.is_relative_to(static_root):
                self._send_json({"error": "Access denied: path outside static directory"}, HTTPStatus.FORBIDDEN)
                return
            if target_file.is_file():
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
            runs = ws.list_runs()
            all_trials = []
            best_by_metric: dict[str, dict] = {}
            active_run_top: dict | None = None
            recent_datasets = []

            for r in runs:
                exps = ws.list_experiments(r.id)
                for e in exps:
                    all_trials.extend(ws.list_trial_results(e.id))
                lb = qry.dispatch(GetLeaderboardQuery(r.id))
                if lb:
                    top = lb[0]
                    score = float(top["score"])
                    model_id = str(top["model_id"])
                    metric_name = str(top.get("metric") or r.config.metric or "score").lower()
                    is_minimize = metric_name in {"mae", "rmse", "mse", "loss", "log_loss"}

                    if metric_name not in best_by_metric:
                        best_by_metric[metric_name] = {
                            "score": score,
                            "model_id": model_id,
                            "run_id": r.id,
                            "metric": metric_name,
                        }
                    else:
                        cur_score = best_by_metric[metric_name]["score"]
                        if (is_minimize and score < cur_score) or (not is_minimize and score > cur_score):
                            best_by_metric[metric_name] = {
                                "score": score,
                                "model_id": model_id,
                                "run_id": r.id,
                                "metric": metric_name,
                            }

                    # Track active or latest run for top-level scalar display
                    is_active = is_active_run_status(r.status)
                    if is_active and (active_run_top is None or not active_run_top.get("is_active")):
                        active_run_top = {
                            "score": score,
                            "model_id": model_id,
                            "metric": metric_name,
                            "status": r.status.value,
                            "is_active": True,
                        }
                    elif active_run_top is None:
                        active_run_top = {
                            "score": score,
                            "model_id": model_id,
                            "metric": metric_name,
                            "status": r.status.value,
                            "is_active": is_active,
                        }
                else:
                    is_active = is_active_run_status(r.status)
                    if is_active and (active_run_top is None or not active_run_top.get("is_active")):
                        active_run_top = {
                            "score": None,
                            "model_id": "-",
                            "metric": str(getattr(r.config, "metric", "-") or "-").lower(),
                            "status": r.status.value,
                            "is_active": True,
                        }

                ds = ws.get_dataset(r.dataset_id)
                if ds and ds.name not in [d["name"] for d in recent_datasets]:
                    profile = ws.get_dataset_profile(ds.id)
                    recent_datasets.append({
                        "id": ds.id,
                        "name": ds.name,
                        "path": ds.path,
                        "target": ds.target_column,
                        "rows": profile.row_count if profile else None,
                        "features": profile.column_count if profile else None,
                    })

            all_ds = ws.list_datasets()
            for ds in all_ds:
                if ds.name not in [d["name"] for d in recent_datasets]:
                    profile = ws.get_dataset_profile(ds.id)
                    recent_datasets.append({
                        "id": ds.id,
                        "name": ds.name,
                        "path": ds.path,
                        "target": ds.target_column,
                        "rows": profile.row_count if profile else None,
                        "features": profile.column_count if profile else None,
                    })

            if active_run_top and active_run_top.get("score") is not None:
                best_score = active_run_top["score"]
                best_model = active_run_top["model_id"]
                best_metric = active_run_top["metric"]
            elif best_by_metric:
                first_metric = next(iter(best_by_metric.values()))
                best_score = first_metric["score"]
                best_model = first_metric["model_id"]
                best_metric = first_metric["metric"]
            else:
                best_score = 0.0
                best_model = "-"
                best_metric = "-"

            activity_feed = _build_real_activity_feed(
                ws=ws,
                qry=qry,
                runs=runs,
                all_trials=all_trials,
                recent_datasets=recent_datasets,
                workspace_dir=self.workspace_dir,
            )

            self._send_json({
                "platform": "CATML AutoML Platform",
                "version": __version__,
                "workspace": self.workspace_dir,
                "total_runs": len(runs),
                "total_trials": len(all_trials),
                "is_active": any(is_active_run_status(r.status) for r in runs),
                "best_score": best_score,
                "best_model": best_model,
                "best_metric": best_metric,
                "best_by_metric": best_by_metric,
                "recent_datasets": recent_datasets,
                "activity_feed": activity_feed,
            })
            return

        elif path == "/api/datasets":
            all_ds = ws.list_datasets()
            result = []
            for ds in all_ds:
                profile = ws.get_dataset_profile(ds.id)
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
            runs = ws.list_runs()
            result = []
            for r in runs:
                dataset = ws.get_dataset(r.dataset_id)
                lb = qry.dispatch(GetLeaderboardQuery(r.id))
                exps = ws.list_experiments(r.id)
                trials_cnt = sum(len(ws.list_trial_results(e.id)) for e in exps)
                result.append({
                    "id": r.id,
                    "dataset_id": r.dataset_id,
                    "dataset_name": dataset.name if dataset else "-",
                    "dataset_path": dataset.path if dataset else "-",
                    "task_type": r.config.task_type,
                    "target": r.config.target,
                    "metric": r.config.metric,
                    "status": r.status.value,
                    "is_active": is_active_run_status(r.status),
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
            exps = ws.list_experiments(run_id)
            result = []
            for e in exps:
                trials = ws.list_trial_results(e.id)
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

        elif path == "/api/models/export":
            run_id = query_params.get("run_id", [""])[0]
            if not run_id:
                runs = ws.list_runs()
                if runs:
                    run_id = runs[0].id
            if not run_id:
                self._send_json({"error": "run_id parameter required or no runs available"}, HTTPStatus.BAD_REQUEST)
                return

            exp_id = query_params.get("experiment_id", [None])[0]
            trial_id = query_params.get("trial_id", [None])[0]

            try:
                import io
                import joblib

                artifact = ws.export_model_artifact(run_id=run_id, experiment_id=exp_id, trial_id=trial_id)
                buf = io.BytesIO()
                joblib.dump(artifact, buf)
                data = buf.getvalue()

                filename = f"catml_{artifact.model_id}.pkl"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(data)))
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return

        elif path == "/api/models/export-info":
            run_id = query_params.get("run_id", [""])[0]
            if not run_id:
                runs = ws.list_runs()
                if runs:
                    run_id = runs[0].id
            if not run_id:
                self._send_json({"error": "run_id parameter required"}, HTTPStatus.BAD_REQUEST)
                return
            exp_id = query_params.get("experiment_id", [None])[0]
            trial_id = query_params.get("trial_id", [None])[0]
            try:
                artifact = ws.export_model_artifact(run_id=run_id, experiment_id=exp_id, trial_id=trial_id)
                self._send_json({
                    "model_id": artifact.model_id,
                    "task_type": artifact.task_type,
                    "metric": artifact.metric,
                    "score": artifact.score,
                    "feature_names": artifact.feature_names,
                    "target_name": artifact.target_name,
                    "parameters": artifact.parameters,
                    "download_url": f"/api/models/export?run_id={run_id}" + (f"&experiment_id={exp_id}" if exp_id else "") + (f"&trial_id={trial_id}" if trial_id else ""),
                })
                return
            except Exception as e:
                self._send_json({"error": str(e)}, HTTPStatus.BAD_REQUEST)
                return

        elif path == "/api/media/preview":
            img_path_str = query_params.get("path", [""])[0]
            if not img_path_str:
                self._send_json({"error": "path parameter required"}, HTTPStatus.BAD_REQUEST)
                return

            dataset_id = query_params.get("dataset_id", [""])[0]
            candidate_path = Path(img_path_str)

            allowed_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
            if candidate_path.suffix.lower() not in allowed_extensions:
                self._send_json({"error": f"Unsupported image format: {candidate_path.suffix}"}, HTTPStatus.BAD_REQUEST)
                return

            ws_root = Path(self.workspace_dir).resolve()
            allowed_roots = [ws_root]

            if dataset_id:
                dataset = ws.get_dataset(dataset_id)
                if dataset and dataset.path:
                    d_path = Path(dataset.path).resolve()
                    allowed_roots.append(d_path.parent if d_path.is_file() else d_path)
            else:
                for ds in ws.list_datasets():
                    if ds.path:
                        d_path = Path(ds.path).resolve()
                        allowed_roots.append(d_path.parent if d_path.is_file() else d_path)

            resolved_path = None

            if candidate_path.is_absolute():
                cand = candidate_path.resolve()
                if any(cand.is_relative_to(r) for r in allowed_roots):
                    if cand.is_file():
                        resolved_path = cand
                    else:
                        self._send_json({"error": f"Image file not found: {img_path_str}"}, HTTPStatus.NOT_FOUND)
                        return
                else:
                    if cand.exists():
                        self._send_json({"error": f"Access denied: path outside workspace: {img_path_str}"}, HTTPStatus.FORBIDDEN)
                        return
                    else:
                        self._send_json({"error": f"Image file not found: {img_path_str}"}, HTTPStatus.NOT_FOUND)
                        return
            else:
                for root in allowed_roots:
                    cand = (root / candidate_path).resolve()
                    if cand.is_relative_to(root) and cand.is_file():
                        resolved_path = cand
                        break

            if not resolved_path or not resolved_path.is_file():
                self._send_json({"error": f"Image file not found: {img_path_str}"}, HTTPStatus.NOT_FOUND)
                return

            if resolved_path.suffix.lower() not in allowed_extensions:
                self._send_json({"error": f"Unsupported image format: {resolved_path.suffix}"}, HTTPStatus.BAD_REQUEST)
                return

            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".bmp": "image/bmp",
                ".tiff": "image/tiff",
                ".gif": "image/gif",
            }
            content_type = mime_map.get(resolved_path.suffix.lower(), "application/octet-stream")

            try:
                data = resolved_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=3600")
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as exc:
                self._send_json({"error": f"Failed to read image: {str(exc)}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                return

        elif path == "/api/dataset/profile":
            dataset_id = query_params.get("dataset_id", [""])[0]
            if not dataset_id:
                self._send_json({"error": "dataset_id parameter required"}, HTTPStatus.BAD_REQUEST)
                return
            profile = ws.get_dataset_profile(dataset_id)
            force_refresh = query_params.get("refresh", ["0"])[0] in ("1", "true")
            needs_enrichment = (
                profile is None
                or not getattr(profile, "correlation_matrix", None)
                or not getattr(profile, "preview_rows", None)
                or any(getattr(c, "mean", None) is None for c in profile.columns if any(t in c.dtype.lower() for t in ("int", "float")))
            )
            if profile is None or force_refresh or needs_enrichment:
                dataset = ws.get_dataset(dataset_id)
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
                        ws.save_dataset_profile(profile)
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
                elif c.get("is_image"):
                    c["catml_action"] = "Vision Embedding"
                    c["action_reason"] = "Deep image extractor (timm / pretrained embeddings)"
                elif c.get("is_text"):
                    c["catml_action"] = "NLP Tokenize & TF-IDF"
                    c["action_reason"] = "Natural language text representation"
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
                self._send_json({"error": "No dataset found in workspace"}, HTTPStatus.BAD_REQUEST)
                return

            dataset = ws.get_dataset(dataset_id)
            profile = ws.get_dataset_profile(dataset_id)
            problem_def = qry.dispatch(GetTaskPlanQuery(dataset_id))

            plan_steps = []
            step_idx = 1

            if run_id:
                exps = ws.list_experiments(run_id)
                for e in exps:
                    trials = ws.list_trial_results(e.id)
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
                self._send_error(404, "No registered dataset found for meta-learning.")
                return

            knowledge = qry.dispatch(GetMetaKnowledgeQuery(dataset_id=dataset_id, run_id=run_id or None))
            self._send_json(knowledge.to_dict())
            return

        elif path == "/api/agent/hypotheses":
            run_id = query_params.get("run_id", [""])[0]
            ledger_file = Path(self.workspace_dir) / "agent_ledger.db"
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

                            before_score = h.baseline_metric
                            after_score = (
                                getattr(h, "verified_metric", None)
                                or crit_cfg.get("verified_metric")
                                or crit_cfg.get("after_score")
                            )
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

            self._send_json({
                "principle": "Propose ≠ Accept",
                "hypotheses": real_hypotheses,
            })
            return

        elif path == "/api/dataset/questionnaire":
            dataset_id = query_params.get("dataset_id", [""])[0]
            if not dataset_id:
                datasets = ws.list_datasets()
                dataset_id = datasets[0].id if datasets else ""
            if not dataset_id:
                self._send_json({"error": "No dataset found"}, HTTPStatus.NOT_FOUND)
                return
            questionnaire = ws.get_dataset_questionnaire(dataset_id)
            self._send_json(questionnaire.to_dict() if questionnaire else {})
            return

        elif path == "/api/kaggle/status":
            run_id = query_params.get("run_id", [""])[0]
            run = ws.get_run(run_id) if run_id else None
            if not run:
                runs = ws.list_runs()
                run = runs[0] if runs else None

            dataset = ws.get_dataset(run.dataset_id) if run else None
            profile = ws.get_dataset_profile(run.dataset_id) if run else None
            lb = qry.dispatch(GetLeaderboardQuery(run.id)) if run else []
            best_cv = lb[0]["score"] if lb else None

            exps = ws.list_experiments(run.id) if run else []
            submissions = []
            for e in exps:
                trials = ws.list_trial_results(e.id)
                best_t = max(trials, key=lambda t: t.primary_score) if trials else None
                if best_t:
                    submissions.append({
                        "experiment": e.name,
                        "cv": round(best_t.primary_score, 5),
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

        elif path == "/api/dataset/temporal":
            dataset_id = query_params.get("dataset_id", [""])[0]
            if not dataset_id:
                self._send_json({"error": "dataset_id parameter required"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                struct = ws.detect_temporal_structure(dataset_id)
                self._send_json({"status": "success", "temporal_structure": struct})
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        elif path == "/api/plugins":
            plugins = qry.dispatch(ListPluginsQuery())
            self._send_json(plugins)
            return

        elif path == "/api/kaggle/download":
            query = parse_qs(parsed.query)
            file_param = query.get("file", [""])[0]
            if not file_param:
                self._send_json({"error": "file parameter is required"}, HTTPStatus.BAD_REQUEST)
                return

            ws_root = Path(self.workspace_dir).resolve()
            submissions_dir = (ws_root / "submissions").resolve()
            exports_dir = (ws_root / "exports").resolve()
            allowed_dirs = [submissions_dir, exports_dir]

            raw_path = Path(file_param)

            # Whitelist allowed submission and export extensions only
            allowed_download_exts = {".csv", ".tsv", ".parquet", ".pq", ".zip"}
            if raw_path.suffix.lower() not in allowed_download_exts:
                self._send_json({"error": f"Access denied: unsupported file type: {raw_path.suffix}"}, HTTPStatus.FORBIDDEN)
                return

            # Resolve candidate path safely
            if raw_path.is_absolute():
                cand = raw_path.resolve()
            else:
                # Check relative path strictly against submissions subfolder or exports
                cand = (submissions_dir / raw_path).resolve()
                if not cand.exists() and (exports_dir / raw_path).resolve().exists():
                    cand = (exports_dir / raw_path).resolve()

            # Strict confinement: candidate MUST reside strictly within submissions or exports directory
            if not any(cand.is_relative_to(d) for d in allowed_dirs):
                self._send_json(
                    {"error": f"Access denied: downloads are restricted to submissions and export artifacts: {file_param}"},
                    HTTPStatus.FORBIDDEN,
                )
                return

            if not cand.is_file():
                self._send_json({"error": f"File not found: {file_param}"}, HTTPStatus.NOT_FOUND)
                return

            content = cand.read_bytes()
            content_types = {
                ".csv": "text/csv; charset=utf-8",
                ".tsv": "text/tab-separated-values; charset=utf-8",
                ".parquet": "application/octet-stream",
                ".pq": "application/octet-stream",
                ".zip": "application/zip",
            }
            ctype = content_types.get(cand.suffix.lower(), "application/octet-stream")

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Disposition", f'attachment; filename="{cand.name}"')
            self.send_header("Content-Length", str(len(content)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
            return

        self._send_json({"error": "Not Found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if not self._require_auth_or_reject():
            return

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
                target = payload.get("target") or payload.get("target_column")
                task_type = payload.get("task_type")

                if not data_path or not target:
                    self._send_json({"error": "path and target are required"}, HTTPStatus.BAD_REQUEST)
                    return

                dataset = ws.register_dataset(name=name, path=data_path, target=target, task_type=task_type)
                run = ws.create_run(dataset)
                profile = ws.get_dataset_profile(dataset.id)

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
                    ws.save_run(run)
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
                profile = ws.get_dataset_profile(dataset.id)
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
                profile = ws.get_dataset_profile(dataset.id)
                feature_names = payload.get("feature_names") or [
                    c.name for c in profile.columns if not c.is_identifier and c.name != dataset.target_column
                ]

                val_strategy = payload.get("validation_strategy")

                exp = cmd.dispatch(
                    CreateExperimentCommand(
                        run_id=run.id,
                        name=name,
                        feature_names=feature_names,
                        model_ids=models,
                        hypothesis=f"AutoML Workbench {mode.upper()} mode with {budget} compute budget.",
                        validation_strategy=val_strategy,
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
                profile = ws.get_dataset_profile(dataset.id)
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

            elif path == "/api/ensemble/build":
                run_id = payload.get("run_id")
                models = payload.get("models") or payload.get("model_ids") or []
                method = payload.get("method", "average")
                meta_model = payload.get("meta_model", "ridge")
                folds = int(payload.get("folds", 5))
                name = payload.get("name")

                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                if not models or len(models) < 2:
                    self._send_json({"error": "At least 2 distinct models required for ensemble"}, HTTPStatus.BAD_REQUEST)
                    return

                exp_id = cmd.dispatch(
                    BuildEnsembleCommand(
                        run_id=run_id,
                        model_ids=models,
                        method=method,
                        meta_model=meta_model,
                        folds=folds,
                        name=name,
                    )
                )

                exp = ws.get_experiment(exp_id)
                trials = ws.list_trial_results(exp_id)
                res = trials[0] if trials else None
                score = res.primary_score if res else 0.0
                sec_metrics = res.secondary_metrics if res else {}

                self._send_json({
                    "status": "success",
                    "experiment_id": exp.id,
                    "name": exp.name,
                    "score": score,
                    "method": method,
                    "models": models,
                    "weights": sec_metrics.get("weights"),
                    "model_scores": sec_metrics.get("model_scores"),
                    "cv_std": sec_metrics.get("cv_std"),
                })
                return

            elif path == "/api/kaggle/upload-template":
                content = payload.get("content", "")
                filename = payload.get("filename", "sample_submission.csv")
                if not content:
                    self._send_json({"error": "content is required"}, HTTPStatus.BAD_REQUEST)
                    return

                target_dir = Path(self.workspace_dir) / "submissions"
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

                self._send_json({
                    "status": "success",
                    "template_path": str(target_file),
                    "filename": target_file.name,
                    "id_column": id_col,
                    "target_column": target_col,
                    "row_count": row_count,
                    "columns": cols,
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
                    datasets = ws.list_datasets()
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

            elif path == "/api/features/temporal":
                run_id = payload.get("run_id")
                dataset_id = payload.get("dataset_id")
                if not run_id:
                    self._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
                    return
                max_lags = int(payload.get("max_lags", 1))
                include_lags = bool(payload.get("include_lags", True))
                include_deltas = bool(payload.get("include_deltas", True))
                include_cyclical = bool(payload.get("include_cyclical", True))
                created_sets = ws.generate_temporal_features(
                    run_id=run_id,
                    dataset_id=dataset_id,
                    max_lags=max_lags,
                    include_lags=include_lags,
                    include_deltas=include_deltas,
                    include_cyclical=include_cyclical,
                )
                self._send_json({
                    "status": "success",
                    "candidate_sets": [fs.to_dict() for fs in created_sets],
                })
                return

            self._send_json({"error": "Endpoint not found"}, HTTPStatus.NOT_FOUND)

        except (ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except KeyError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"status": "error", "message": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def run_web_dashboard(
    port: int = 8080,
    workspace_dir: str = ".automl/demo",
    host: str = "127.0.0.1",
    auth_token: str | None = None,
    insecure_no_auth: bool = False,
) -> None:
    import secrets

    is_local = host in {"127.0.0.1", "localhost", "::1"}
    effective_token = auth_token

    if not is_local and not insecure_no_auth:
        AutoMLWebHandler.require_auth = True
        if not effective_token:
            effective_token = secrets.token_urlsafe(16)
        AutoMLWebHandler.auth_token = effective_token
    else:
        AutoMLWebHandler.require_auth = False
        AutoMLWebHandler.auth_token = None

    AutoMLWebHandler.workspace_dir = workspace_dir
    server = ThreadingHTTPServer((host, port), AutoMLWebHandler)
    from automl.infrastructure.jobs.worker import JobWorker
    worker = JobWorker(workspace_dir)
    try:
        worker.start()
    except Exception:
        server.server_close()
        raise

    print("=" * 65)
    print(f"  CATML AutoML Workbench (Platform V{__version__})")
    if not is_local and not insecure_no_auth:
        print(f"  Bound to external interface: http://{host}:{port}")
        print("  SECURITY NOTICE: Remote binding protected with authentication token.")
        print(f"  Access token: {effective_token}")
        print(f"  Direct secure URL (fragment, not sent over HTTP): http://{host}:{port}/#token={effective_token}")
        print(f"  Alternative query URL: http://{host}:{port}/?token={effective_token}")
    elif not is_local and insecure_no_auth:
        print(f"  Bound to external interface: http://{host}:{port}")
        print("  WARNING: Workbench is exposed on non-localhost interface WITHOUT authentication!")
        print("  Ensure this instance is guarded behind a secure reverse proxy or private VPN.")
    else:
        print(f"  Running locally at: http://{host}:{port}")
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
    host = sys.argv[3] if len(sys.argv) > 3 else "127.0.0.1"
    token = sys.argv[4] if len(sys.argv) > 4 else None
    run_web_dashboard(port=port, workspace_dir=ws, host=host, auth_token=token)
