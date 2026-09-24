from __future__ import annotations

from typing import Any

from automl.domain.experiments.candidate import ExperimentCandidate
from automl.domain.runs.run import AutoMLRun


class AblationPlanner:
    """Proposes leave-one-out feature ablation experiments to empirically validate

    the marginal contribution of individual features against a baseline.
    """

    def propose_ablation(
        self,
        run: AutoMLRun,
        base_feature_names: list[str],
        features_to_ablate: list[str] | None = None,
        model_id: str | None = None,
    ) -> list[ExperimentCandidate]:
        if len(base_feature_names) <= 1:
            return []

        target_features = (
            [f for f in features_to_ablate if f in base_feature_names]
            if features_to_ablate is not None
            else list(base_feature_names)
        )
        if not target_features:
            return []

        chosen_model = (
            model_id
            or (run.config.models_include[0] if run.config.models_include else None)
        )
        if not chosen_model:
            task_type = run.config.task_type.value if hasattr(run.config.task_type, "value") else str(run.config.task_type)
            chosen_model = "ridge" if task_type == "regression" else "logistic_regression"

        candidates: list[ExperimentCandidate] = []
        for feature in target_features:
            ablation_features = [f for f in base_feature_names if f != feature]
            candidates.append(
                ExperimentCandidate.create(
                    run_id=run.id,
                    name=f"ablation_without_{feature}",
                    hypothesis=(
                        f"Measure performance degradation when ablating feature '{feature}' "
                        f"to determine its marginal contribution."
                    ),
                    feature_names=ablation_features,
                    model_ids=[chosen_model],
                    metric=run.config.metric,
                    validation_strategy=run.config.validation_strategy,
                    tags=["ablation", f"without_{feature}"],
                    created_by="ablation_planner",
                )
            )

        return candidates
