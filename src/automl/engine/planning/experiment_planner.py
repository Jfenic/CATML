from __future__ import annotations

import uuid
from typing import Any

from automl.domain.datasets.profile import DatasetProfile
from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.experiments.priority import ExperimentPriority, Priority
from automl.domain.experiments.trial import TrialResult
from automl.domain.features.feature import FeatureStatus
from automl.domain.features.registry import FeatureRegistry
from automl.domain.models.registry import ModelRegistry
from automl.domain.ports import ExperimentPlannerPort
from automl.domain.runs.run import AutoMLRun
from automl.domain.tasks.task_type import TaskType


class RuleBasedExperimentPlanner(ExperimentPlannerPort):
    """
    Deterministic experiment planner that proposes candidates based on
    DatasetProfile, FeatureRegistry, ModelRegistry, and RunConfig.
    """

    def propose(
        self,
        run: AutoMLRun,
        profile: DatasetProfile,
        feature_registry: FeatureRegistry,
        model_registry: ModelRegistry,
        history: list[TrialResult] | None = None,
    ) -> list[ExperimentCandidate]:
        candidates: list[ExperimentCandidate] = []
        task_type = run.config.task_type

        # 1. Resolve active features and active models
        active_features = feature_registry.active_feature_names(run.config.target)
        if not active_features:
            return []

        # Exclude non-predictive identifier columns from automated candidate proposals
        if profile and hasattr(profile, "identifier_column_names"):
            identifiers = set(profile.identifier_column_names)
            non_identifiers = [f for f in active_features if f not in identifiers]
            if non_identifiers:
                active_features = non_identifiers

        active_models = model_registry.resolve_active(
            include=run.config.models_include,
            exclude=run.config.models_exclude,
            task_type=task_type,
        )
        if not active_models:
            return []

        fast_baseline_models = self._select_fast_baseline_models(active_models, task_type)
        complex_models = [m for m in active_models if m not in fast_baseline_models]

        # 2. Candidate 1: Fast baseline with all active features
        if fast_baseline_models:
            candidates.append(
                ExperimentCandidate.create(
                    run_id=run.id,
                    name="baseline_fast",
                    hypothesis=f"Fast baseline using {', '.join(fast_baseline_models)} across all active features.",
                    feature_names=active_features,
                    model_ids=fast_baseline_models,
                    metric=run.config.metric,
                    validation_strategy=run.config.validation_strategy,
                    tags=["baseline", "fast"],
                    created_by="rule_based_planner",
                )
            )

        # 3. Candidate 2: Tree-based / complex model exploration
        if complex_models:
            for model_id in complex_models:
                spec = model_registry.get(model_id)
                model_name = spec.name if spec else model_id
                candidates.append(
                    ExperimentCandidate.create(
                        run_id=run.id,
                        name=f"model_{model_id}",
                        hypothesis=f"Evaluate non-linear/ensemble capacity of {model_name}.",
                        feature_names=active_features,
                        model_ids=[model_id],
                        metric=run.config.metric,
                        validation_strategy=run.config.validation_strategy,
                        tags=["model_exploration", model_id],
                        created_by="rule_based_planner",
                    )
                )

        # 4. Candidate 3: User prioritized features subset (if any exist)
        prioritized_features = [
            f.name
            for f in feature_registry.list()
            if f.name in active_features and (f.status == FeatureStatus.PRIORITY or f.user_priority > 0)
        ]
        if prioritized_features and len(prioritized_features) < len(active_features):
            models_for_subset = fast_baseline_models or active_models[:1]
            candidates.append(
                ExperimentCandidate.create(
                    run_id=run.id,
                    name="prioritized_feature_subset",
                    hypothesis=(
                        f"Assess compact feature subset restricted to {len(prioritized_features)} "
                        f"user-prioritized feature(s): {', '.join(prioritized_features[:4])}."
                    ),
                    feature_names=prioritized_features,
                    model_ids=models_for_subset,
                    metric=run.config.metric,
                    validation_strategy=run.config.validation_strategy,
                    tags=["feature_subset", "prioritized"],
                    created_by="rule_based_planner",
                )
            )

        # 5. Candidate 4: Compact subset if feature space is large (> 6 features) and no prioritized subset was created
        if len(active_features) > 6 and not prioritized_features:
            top_compact = active_features[:5]
            models_for_subset = fast_baseline_models or active_models[:1]
            candidates.append(
                ExperimentCandidate.create(
                    run_id=run.id,
                    name="compact_feature_subset",
                    hypothesis=f"Evaluate reduced initial feature dimensionality ({len(top_compact)} features).",
                    feature_names=top_compact,
                    model_ids=models_for_subset,
                    metric=run.config.metric,
                    validation_strategy=run.config.validation_strategy,
                    tags=["feature_subset", "compact"],
                    created_by="rule_based_planner",
                )
            )

        return candidates

    def _select_fast_baseline_models(self, active_models: list[str], task_type: str) -> list[str]:
        preferred_baselines = {
            TaskType.BINARY_CLASSIFICATION.value: ["logistic_regression"],
            TaskType.MULTICLASS_CLASSIFICATION.value: ["logistic_regression"],
            TaskType.REGRESSION.value: ["ridge"],
            TaskType.CLUSTERING.value: ["kmeans"],
        }
        preferred = preferred_baselines.get(task_type, [])
        selected = [m for m in preferred if m in active_models]
        if not selected and active_models:
            selected = [active_models[0]]
        return selected
