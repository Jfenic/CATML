"""Datasets / files / features API controller."""
from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from typing import Any
import uuid


def try_get(handler: Any, path: str, query_params: dict, ws: Any, cmd: Any, qry: Any) -> bool:
    """Handle GET routes for this resource. Return True if the request was handled."""
    if path == "/api/datasets":
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
        handler._send_json(result)
        return True

    if path == "/api/files/browse":
        target_rel = query_params.get("dir", [""])[0].strip()
        search_query = query_params.get("search", [""])[0].strip().lower()

        base_dir = Path.cwd().resolve()
        if target_rel == ".":
            target_path = base_dir
        elif not target_rel:
            # If searching without specific dir, search base_dir
            target_path = base_dir if search_query else ((base_dir / "data").resolve() if (base_dir / "data").exists() else base_dir)
        elif Path(target_rel).is_absolute():
            target_path = Path(target_rel).resolve()
        else:
            target_path = (base_dir / target_rel).resolve()

        # Security boundary: must remain within base_dir or workspace_dir
        ws_path = Path(handler.workspace_dir).resolve()
        is_in_base = target_path == base_dir or target_path.is_relative_to(base_dir)
        is_in_ws = target_path == ws_path or target_path.is_relative_to(ws_path)

        if not (is_in_base or is_in_ws):
            target_path = (base_dir / "data").resolve() if (base_dir / "data").exists() else base_dir

        if not target_path.exists() or not target_path.is_dir():
            target_path = base_dir

        allowed_exts = {".csv", ".parquet", ".pq", ".tsv", ".json", ".jsonl"}
        folders = []
        files = []
        parent_dir = str(target_path.parent) if target_path != base_dir and (target_path.parent == base_dir or target_path.parent.is_relative_to(base_dir)) else None

        try:
            if search_query:
                # Recursive search from target_path, limiting to 60 matches
                for item in target_path.rglob("*"):
                    if item.name.startswith(".") or "venv" in str(item) or "__pycache__" in str(item) or "node_modules" in str(item):
                        continue
                    if search_query in item.name.lower():
                        if item.is_file() and item.suffix.lower() in allowed_exts:
                            try:
                                size_bytes = item.stat().st_size
                                size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1048576 else f"{size_bytes / 1048576:.1f} MB"
                                files.append({
                                    "name": item.name,
                                    "path": str(item),
                                    "rel_path": str(item.relative_to(base_dir)) if item.is_relative_to(base_dir) else str(item),
                                    "size_bytes": size_bytes,
                                    "size_formatted": size_str,
                                    "extension": item.suffix.lower(),
                                })
                            except Exception:
                                pass
                    if len(files) >= 60:
                        break
            else:
                for item in sorted(target_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                    if item.name.startswith(".") or item.name in {"venv", ".venv", "__pycache__", "node_modules", "dist", "build", ".git"}:
                        continue
                    if item.is_dir():
                        folders.append({
                            "name": item.name,
                            "path": str(item),
                            "rel_path": str(item.relative_to(base_dir)) if item.is_relative_to(base_dir) else str(item),
                        })
                    elif item.is_file() and item.suffix.lower() in allowed_exts:
                        try:
                            size_bytes = item.stat().st_size
                            size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1048576 else f"{size_bytes / 1048576:.1f} MB"
                            files.append({
                                "name": item.name,
                                "path": str(item),
                                "rel_path": str(item.relative_to(base_dir)) if item.is_relative_to(base_dir) else str(item),
                                "size_bytes": size_bytes,
                                "size_formatted": size_str,
                                "extension": item.suffix.lower(),
                            })
                        except Exception:
                            pass
        except Exception:
            pass

        handler._send_json({
            "current_dir": str(target_path),
            "rel_dir": str(target_path.relative_to(base_dir)) if target_path.is_relative_to(base_dir) else str(target_path),
            "parent_dir": parent_dir,
            "folders": folders,
            "files": files,
        })
        return True

    if path == "/api/media/preview":
        img_path_str = query_params.get("path", [""])[0]
        if not img_path_str:
            handler._send_json({"error": "path parameter required"}, HTTPStatus.BAD_REQUEST)
            return True

        dataset_id = query_params.get("dataset_id", [""])[0]
        candidate_path = Path(img_path_str)

        allowed_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}
        if candidate_path.suffix.lower() not in allowed_extensions:
            handler._send_json({"error": f"Unsupported image format: {candidate_path.suffix}"}, HTTPStatus.BAD_REQUEST)
            return True

        ws_root = Path(handler.workspace_dir).resolve()
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
                    handler._send_json({"error": f"Image file not found: {img_path_str}"}, HTTPStatus.NOT_FOUND)
                    return True
            else:
                if cand.exists():
                    handler._send_json({"error": f"Access denied: path outside workspace: {img_path_str}"}, HTTPStatus.FORBIDDEN)
                    return True
                else:
                    handler._send_json({"error": f"Image file not found: {img_path_str}"}, HTTPStatus.NOT_FOUND)
                    return True
        else:
            for root in allowed_roots:
                cand = (root / candidate_path).resolve()
                if cand.is_relative_to(root) and cand.is_file():
                    resolved_path = cand
                    break

        if not resolved_path or not resolved_path.is_file():
            handler._send_json({"error": f"Image file not found: {img_path_str}"}, HTTPStatus.NOT_FOUND)
            return True

        if resolved_path.suffix.lower() not in allowed_extensions:
            handler._send_json({"error": f"Unsupported image format: {resolved_path.suffix}"}, HTTPStatus.BAD_REQUEST)
            return True

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
            handler.send_response(HTTPStatus.OK)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            handler.send_header("Cache-Control", "public, max-age=3600")
            handler._send_cors_headers()
            handler.end_headers()
            handler.wfile.write(data)
            return True
        except Exception as exc:
            handler._send_json({"error": f"Failed to read image: {str(exc)}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return True

    if path == "/api/dataset/profile":
        dataset_id = query_params.get("dataset_id", [""])[0]
        if not dataset_id:
            handler._send_json({"error": "dataset_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
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
                        handler._send_json({"error": f"Error computing profile: {str(err)}"}, HTTPStatus.INTERNAL_SERVER_ERROR)
                        return True

        if not profile:
            handler._send_json({"error": "Profile not found"}, HTTPStatus.NOT_FOUND)
            return True
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

        handler._send_json(p_dict)
        return True

    if path == "/api/dataset/questionnaire":
        dataset_id = query_params.get("dataset_id", [""])[0]
        if not dataset_id:
            datasets = ws.list_datasets()
            dataset_id = datasets[0].id if datasets else ""
        if not dataset_id:
            handler._send_json({"error": "No dataset found"}, HTTPStatus.NOT_FOUND)
            return True
        questionnaire = ws.get_dataset_questionnaire(dataset_id)
        handler._send_json(questionnaire.to_dict() if questionnaire else {})
        return True

    if path == "/api/dataset/temporal":
        dataset_id = query_params.get("dataset_id", [""])[0]
        if not dataset_id:
            handler._send_json({"error": "dataset_id parameter required"}, HTTPStatus.BAD_REQUEST)
            return True
        try:
            struct = ws.detect_temporal_structure(dataset_id)
            handler._send_json({"status": "success", "temporal_structure": struct})
        except Exception as exc:
            handler._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        return True

    if path == "/api/kaggle/download":
        file_param = query_params.get("file", [""])[0]
        if not file_param:
            handler._send_json({"error": "file parameter is required"}, HTTPStatus.BAD_REQUEST)
            return True

        ws_root = Path(handler.workspace_dir).resolve()
        submissions_dir = (ws_root / "submissions").resolve()
        exports_dir = (ws_root / "exports").resolve()
        allowed_dirs = [submissions_dir, exports_dir]

        raw_path = Path(file_param)

        # Whitelist allowed submission and export extensions only
        allowed_download_exts = {".csv", ".tsv", ".parquet", ".pq", ".zip"}
        if raw_path.suffix.lower() not in allowed_download_exts:
            handler._send_json({"error": f"Access denied: unsupported file type: {raw_path.suffix}"}, HTTPStatus.FORBIDDEN)
            return True

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
            handler._send_json(
                {"error": f"Access denied: downloads are restricted to submissions and export artifacts: {file_param}"},
                HTTPStatus.FORBIDDEN,
            )
            return True

        if not cand.is_file():
            handler._send_json({"error": f"File not found: {file_param}"}, HTTPStatus.NOT_FOUND)
            return True

        content = cand.read_bytes()
        content_types = {
            ".csv": "text/csv; charset=utf-8",
            ".tsv": "text/tab-separated-values; charset=utf-8",
            ".parquet": "application/octet-stream",
            ".pq": "application/octet-stream",
            ".zip": "application/zip",
        }
        ctype = content_types.get(cand.suffix.lower(), "application/octet-stream")

        handler.send_response(HTTPStatus.OK)
        handler.send_header("Content-Type", ctype)
        handler.send_header("Content-Disposition", f'attachment; filename="{cand.name}"')
        handler.send_header("Content-Length", str(len(content)))
        handler._send_cors_headers()
        handler.end_headers()
        handler.wfile.write(content)
        return True
    return False


def try_post(handler: Any, path: str, payload: dict, ws: Any, cmd: Any, qry: Any) -> bool:
    """Handle POST routes for this resource. Return True if the request was handled."""
    if path == "/api/dataset/register":
        name = payload.get("name", "dataset")
        data_path = payload.get("path")
        target = payload.get("target") or payload.get("target_column")
        task_type = payload.get("task_type")

        if not data_path:
            handler._send_json({"error": "path is required"}, HTTPStatus.BAD_REQUEST)
            return True

        if target and not str(target).strip():
            target = None

        try:
            from automl.infrastructure.security.workspace_paths import (
                PathOutsideWorkspaceError,
                resolve_allowed_path,
            )

            try:
                data_path = str(
                    resolve_allowed_path(data_path, workspace_dir=handler.workspace_dir)
                )
            except PathOutsideWorkspaceError as pe:
                handler._send_json({"error": str(pe)}, HTTPStatus.FORBIDDEN)
                return True

            dataset = ws.register_dataset(name=name, path=data_path, target=target, task_type=task_type)
            run = ws.create_run(dataset)
            profile = ws.get_dataset_profile(dataset.id)

            handler._send_json({
                "status": "success",
                "dataset_id": dataset.id,
                "run_id": run.id,
                "target": dataset.target_column,
                "task_type": dataset.task_type,
                "row_count": profile.row_count if profile else None,
                "feature_count": len(profile.columns) if profile else None,
            })
            return True
        except Exception as exc:
            import traceback
            traceback.print_exc()
            handler._send_json(
                {
                    "error": f"No se ha podido analizar o registrar el dataset: {exc}",
                    "message": str(exc),
                },
                HTTPStatus.BAD_REQUEST,
            )
            return True

    if path == "/api/dataset/upload":
        filename = payload.get("filename", "uploaded_dataset.csv")
        safe_filename = Path(filename).name
        if not safe_filename or safe_filename.startswith("."):
            safe_filename = f"dataset_{uuid.uuid4().hex[:6]}.csv"

        content_b64 = payload.get("content_base64")
        if not content_b64:
            handler._send_json({"error": "content_base64 is required"}, HTTPStatus.BAD_REQUEST)
            return True

        try:
            import base64
            data_bytes = base64.b64decode(content_b64)
        except Exception as b64_err:
            handler._send_json({"error": f"Invalid base64 payload: {b64_err}"}, HTTPStatus.BAD_REQUEST)
            return True

        upload_dir = Path(handler.workspace_dir) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Never overwrite a previous upload that may still be registered.
        stem = Path(safe_filename).stem or "dataset"
        suffix = Path(safe_filename).suffix or ".csv"
        dest_path = upload_dir / f"{stem}_{uuid.uuid4().hex[:10]}{suffix}"
        dest_path.write_bytes(data_bytes)

        size_bytes = len(data_bytes)
        size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1048576 else f"{size_bytes / 1048576:.1f} MB"

        detected_cols = []
        suggested_target = None
        try:
            from automl.engine.profiling.dataset_profiler import load_dataframe
            sample_df = load_dataframe(dest_path)
            detected_cols = list(sample_df.columns)
            candidates = ["target", "label", "class", "churn", "survived", "price", "is_fraud", "Will_Buy_EV"]
            for c in detected_cols:
                if c.lower() in [cand.lower() for cand in candidates]:
                    suggested_target = c
                    break
            if not suggested_target and len(detected_cols) > 1:
                suggested_target = detected_cols[-1]
        except Exception:
            pass

        handler._send_json({
            "status": "success",
            "path": str(dest_path.resolve()),
            "filename": safe_filename,
            "size_bytes": size_bytes,
            "size_formatted": size_str,
            "columns": detected_cols,
            "suggested_target": suggested_target,
            "suggested_name": Path(safe_filename).stem,
        })
        return True

    if path == "/api/dataset/inspect-file":
        file_path_str = payload.get("path")
        if not file_path_str:
            handler._send_json({"error": "path is required"}, HTTPStatus.BAD_REQUEST)
            return True

        from automl.infrastructure.security.workspace_paths import (
            PathOutsideWorkspaceError,
            resolve_allowed_path,
        )

        try:
            target_file = resolve_allowed_path(
                file_path_str, workspace_dir=handler.workspace_dir
            )
        except PathOutsideWorkspaceError as pe:
            handler._send_json({"error": str(pe)}, HTTPStatus.FORBIDDEN)
            return True
        if not target_file.exists() or not target_file.is_file():
            handler._send_json({"error": f"File not found: {file_path_str}"}, HTTPStatus.NOT_FOUND)
            return True

        try:
            from automl.engine.profiling.dataset_profiler import load_dataframe
            df = load_dataframe(target_file)
            cols = list(df.columns)
            suggested_target = None
            candidates = ["target", "label", "class", "churn", "survived", "price", "is_fraud", "Will_Buy_EV"]
            for c in cols:
                if c.lower() in [cand.lower() for cand in candidates]:
                    suggested_target = c
                    break
            if not suggested_target and len(cols) > 1:
                suggested_target = cols[-1]

            size_bytes = target_file.stat().st_size
            size_str = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1048576 else f"{size_bytes / 1048576:.1f} MB"

            handler._send_json({
                "status": "success",
                "path": str(target_file),
                "suggested_name": target_file.stem,
                "columns": cols,
                "suggested_target": suggested_target,
                "total_rows": len(df),
                "total_columns": len(cols),
                "size_formatted": size_str,
            })
            return True
        except Exception as load_err:
            handler._send_json({"error": f"Could not inspect file: {load_err}"}, HTTPStatus.BAD_REQUEST)
            return True

    if path == "/api/dataset/questionnaire":
        dataset_id = payload.get("dataset_id") or (
            payload.get("questionnaire", {}).get("dataset_id")
            if isinstance(payload.get("questionnaire"), dict)
            else None
        )
        if not dataset_id:
            handler._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        from automl.domain.tasks.questionnaire import DatasetQuestionnaire
        q_data = dict(payload.get("questionnaire", payload))
        q_data["dataset_id"] = dataset_id
        q_data["auto_generated"] = False
        q_data["user_confirmed"] = bool(payload.get("user_confirmed", q_data.get("user_confirmed", True)))
        questionnaire = DatasetQuestionnaire.from_dict(q_data)
        ws.save_dataset_questionnaire(questionnaire)
        handler._send_json({"status": "success", "questionnaire": questionnaire.to_dict()})
        return True

    if path == "/api/dataset/questionnaire/generate":
        dataset_id = payload.get("dataset_id")
        if not dataset_id:
            datasets = ws.list_datasets()
            dataset_id = datasets[0].id if datasets else ""
        if not dataset_id:
            handler._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        questionnaire = ws.generate_dataset_questionnaire(dataset_id)
        handler._send_json({"status": "success", "questionnaire": questionnaire.to_dict()})
        return True

    if path == "/api/features/calculate":
        dataset_id = payload.get("dataset_id")
        name = payload.get("name", "derived_feature")
        expression = payload.get("expression", "")
        expression_type = payload.get("expression_type", "formula")
        if not dataset_id:
            handler._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        eval_res = ws.validate_derived_feature(
            dataset_id=dataset_id,
            name=name,
            expression=expression,
            expression_type=expression_type,
        )
        handler._send_json({"status": "success", "result": eval_res.to_dict()})
        return True

    if path == "/api/features/apply":
        dataset_id = payload.get("dataset_id")
        name = payload.get("name", "derived_feature")
        expression = payload.get("expression", "")
        expression_type = payload.get("expression_type", "formula")
        description = payload.get("description", "")
        if not dataset_id:
            handler._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        eval_res, profile = ws.apply_derived_feature(
            dataset_id=dataset_id,
            name=name,
            expression=expression,
            expression_type=expression_type,
            description=description,
        )
        handler._send_json({
            "status": "success",
            "result": eval_res.to_dict(),
            "feature_count": len(profile.columns) if profile else None,
        })
        return True

    if path == "/api/features/suggest":
        dataset_id = payload.get("dataset_id")
        if not dataset_id:
            handler._send_json({"error": "dataset_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
        suggestions = ws.suggest_derived_features(dataset_id=dataset_id)
        handler._send_json({"status": "success", "suggestions": suggestions})
        return True

    if path == "/api/features/temporal":
        run_id = payload.get("run_id")
        dataset_id = payload.get("dataset_id")
        if not run_id:
            handler._send_json({"error": "run_id is required"}, HTTPStatus.BAD_REQUEST)
            return True
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
        handler._send_json({
            "status": "success",
            "candidate_sets": [fs.to_dict() for fs in created_sets],
        })
        return True

    if path == "/api/project/demo":
        try:
            ws_root = Path(handler.workspace_dir).resolve()
            demo_csv = ws_root / "demo_customers_churn.csv"

            if not demo_csv.exists():
                examples_csv = (
                    Path(__file__).resolve().parents[5]
                    / "examples"
                    / "data"
                    / "customers_churn.csv"
                )
                if examples_csv.exists():
                    import shutil
                    shutil.copyfile(examples_csv, demo_csv)
                else:
                    import numpy as np
                    import pandas as pd
                    rng = np.random.default_rng(42)
                    n = 60
                    df = pd.DataFrame({
                        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
                        "age": rng.integers(18, 70, size=n),
                        "tenure_months": rng.integers(1, 72, size=n),
                        "monthly_charges": np.round(rng.uniform(20.0, 120.0, size=n), 2),
                        "total_charges": np.round(rng.uniform(100.0, 5000.0, size=n), 2),
                        "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], size=n),
                        "churn": rng.integers(0, 2, size=n),
                    })
                    df.to_csv(demo_csv, index=False)

            existing_datasets = ws.list_datasets()
            demo_ds = next((d for d in existing_datasets if d.name == "Demo Churn de Clientes"), None)
            if not demo_ds:
                demo_ds = ws.register_dataset(
                    name="Demo Churn de Clientes",
                    path=str(demo_csv),
                    target="churn",
                    task_type="binary_classification",
                )

            runs = ws.list_runs(dataset_id=demo_ds.id)
            if runs:
                run = runs[0]
            else:
                run = ws.create_run(demo_ds, metric="roc_auc")

            quick_train = bool((payload or {}).get("quick_train", True))
            exp_id = None
            if quick_train:
                exps = ws.list_experiments(run.id)
                if not exps:
                    features = ws.get_feature_registry(demo_ds.id).active_feature_names(demo_ds.target_column)
                    features = [f for f in features if f != "customer_id"]
                    exp = ws.create_experiment(
                        run=run,
                        name="Baseline Comparativo",
                        hypothesis="Variables contractuales y financieras predicen la retención",
                        feature_names=features,
                        model_ids=["logistic_regression", "random_forest"],
                    )
                    ws.run_experiment(run, exp)
                    exp_id = exp.id
                else:
                    exp_id = exps[0].id

            profile = ws.get_dataset_profile(demo_ds.id)
            handler._send_json({
                "status": "success",
                "dataset_id": demo_ds.id,
                "dataset_name": demo_ds.name,
                "run_id": run.id,
                "experiment_id": exp_id,
                "row_count": profile.row_count if profile else None,
                "feature_count": len(profile.columns) if profile else None,
                "message": "Proyecto demo cargado con éxito",
            })
            return True
        except Exception as exc:
            import traceback
            traceback.print_exc()
            handler._send_json(
                {"error": f"Error cargando proyecto demo: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return True
    return False
