"""Context builder service for assembling bounded, structured evidence for specialists."""
from __future__ import annotations

from typing import Any

from automl.application.agents.contracts import ContextPayload


class ContextBuilder:
    """Extracts, bounds, and redacts workspace and run state into structured evidence payloads."""

    def __init__(
        self,
        max_leaderboard_rows: int = 10,
        max_feature_evidence: int = 10,
        max_history_entries: int = 10,
    ) -> None:
        self.max_leaderboard_rows = max_leaderboard_rows
        self.max_feature_evidence = max_feature_evidence
        self.max_history_entries = max_history_entries

    def build(
        self,
        workspace: Any,
        run_id: str,
        history: list[dict[str, Any]] | None = None,
        budget_status: dict[str, Any] | None = None,
    ) -> ContextPayload:
        """Build a bounded ContextPayload for a given run from the workspace."""
        run = workspace.get_run(run_id)
        if run is None:
            raise ValueError(f"Run '{run_id}' not found in workspace.")

        dataset = workspace.get_dataset(run.dataset_id)
        if dataset is None:
            raise ValueError(f"Dataset '{run.dataset_id}' for run '{run_id}' not found.")

        omissions: list[str] = [
            "raw_dataframe_samples_excluded",
            "uncompressed_feature_arrays_excluded",
        ]

        # 1. Profile extraction & bounding
        profile_summary: dict[str, Any] = {}
        task_type = "tabular_classification"
        target_metric = "roc_auc"
        metric_direction = "maximize"

        try:
            profile = workspace.get_profile(dataset.id)
            if profile:
                profile_summary = {
                    "dataset_id": dataset.id,
                    "dataset_name": dataset.name,
                    "target_column": getattr(dataset, "target_column", getattr(dataset, "target", "")),
                    "n_rows": getattr(profile, "n_rows", 0),
                    "n_columns": getattr(profile, "n_columns", 0),
                    "numeric_columns": getattr(profile, "numeric_columns", []),
                    "categorical_columns": getattr(profile, "categorical_columns", []),
                }
                if hasattr(profile, "task_type") and profile.task_type:
                    task_type = str(profile.task_type)
        except Exception:
            profile_summary = {
                "dataset_id": dataset.id,
                "dataset_name": dataset.name,
                "target_column": getattr(dataset, "target_column", getattr(dataset, "target", "")),
            }

        # 2. Leaderboard extraction & truncation
        leaderboard_summary: list[dict[str, Any]] = []
        try:
            raw_lb = workspace.get_leaderboard(run_id)
            if raw_lb:
                if len(raw_lb) > self.max_leaderboard_rows:
                    omissions.append(
                        f"leaderboard_truncated_to_{self.max_leaderboard_rows}_of_{len(raw_lb)}_entries"
                    )
                for entry in raw_lb[: self.max_leaderboard_rows]:
                    cleaned_entry = {
                        "model_id": entry.get("model_id"),
                        "metric": entry.get("metric", target_metric),
                        "score": entry.get("score"),
                        "experiment_id": entry.get("experiment_id"),
                    }
                    if "trial_count" in entry:
                        cleaned_entry["trial_count"] = entry["trial_count"]
                    leaderboard_summary.append(cleaned_entry)
        except Exception:
            leaderboard_summary = []

        # Determine metric & direction from top leaderboard entry or defaults
        if leaderboard_summary and leaderboard_summary[0].get("metric"):
            target_metric = leaderboard_summary[0]["metric"]
            if target_metric.lower() in ("rmse", "mae", "loss", "log_loss", "mse"):
                metric_direction = "minimize"

        # 3. Feature evidence extraction
        feature_evidence: list[dict[str, Any]] = []
        try:
            feature_sets = workspace.list_feature_sets(run_id)
            if feature_sets:
                if len(feature_sets) > self.max_feature_evidence:
                    omissions.append(
                        f"feature_evidence_truncated_to_{self.max_feature_evidence}_of_{len(feature_sets)}_sets"
                    )
                for fset in feature_sets[: self.max_feature_evidence]:
                    feature_evidence.append({
                        "name": getattr(fset, "name", str(fset)),
                        "feature_count": len(getattr(fset, "features", [])),
                        "features": [getattr(f, "name", str(f)) for f in getattr(fset, "features", [])[:10]],
                    })
        except Exception:
            feature_evidence = []

        # 4. History bounding
        bounded_history: list[dict[str, Any]] = []
        if history:
            if len(history) > self.max_history_entries:
                omissions.append(
                    f"history_truncated_to_{self.max_history_entries}_of_{len(history)}_entries"
                )
            bounded_history = history[-self.max_history_entries :]

        # 5. Budget status
        active_budget = budget_status or {
            "remaining_experiments": 5,
            "remaining_trials": 20,
            "remaining_duration_seconds": 600.0,
        }

        return ContextPayload(
            run_id=run_id,
            dataset_id=dataset.id,
            task_type=task_type,
            target_metric=target_metric,
            metric_direction=metric_direction,
            dataset_profile=profile_summary,
            feature_evidence=feature_evidence,
            leaderboard=leaderboard_summary,
            budget_status=active_budget,
            history=bounded_history,
            omissions=omissions,
        )
