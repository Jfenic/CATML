"""Deterministic mapper translating statistical hypotheses into AutoML experiment candidates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from automl.domain.analysis.models import AnalysisHypothesis


@dataclass(frozen=True)
class ExperimentCandidateSpec:
    """Concrete experiment specification translated from an AnalysisHypothesis."""
    name: str
    feature_names: list[str]
    model_ids: list[str]
    hypothesis_text: str
    action_type: str
    column_transforms: tuple[dict[str, Any], ...] = ()
    derived_features: tuple[dict[str, Any], ...] = ()
    drop_features: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_id(self) -> str:
        """Primary model id (first in model_ids) for single-model recipe fallbacks."""
        return self.model_ids[0] if self.model_ids else "random_forest"


class HypothesisExperimentTranslator:
    """Translates AnalysisHypothesis domain entities into executable AutoML experiment candidates."""

    @classmethod
    def translate(
        cls,
        hypothesis: AnalysisHypothesis,
        available_features: list[str],
        task_type: str | None = None,
        default_models: list[str] | None = None,
    ) -> ExperimentCandidateSpec:
        """Deterministically map hypothesis action and experiment delta to experiment features and models."""
        features = list(available_features)
        models = list(default_models or cls._default_models_for_task(task_type))
        action = hypothesis.proposed_action.lower()
        delta = dict(hypothesis.experiment_delta or {})

        candidate_name = f"hyp_{hypothesis.id[:8]}"
        metadata: dict[str, Any] = {"hypothesis_id": hypothesis.id, "action": action}
        column_transforms: list[dict[str, Any]] = []
        derived_features: list[dict[str, Any]] = []
        dropped: list[str] = []
        original_features = list(available_features)

        if "derived_features" in delta and isinstance(delta["derived_features"], list):
            derived_features.extend(delta["derived_features"])
        if "column_transforms" in delta and isinstance(delta["column_transforms"], list):
            column_transforms.extend(delta["column_transforms"])

        # 1. Custom / Explicit Delta Overrides
        if "features" in delta and isinstance(delta["features"], list) and delta["features"]:
            features = [f for f in delta["features"] if f in available_features or f in delta["features"]]
        if "drop_features" in delta and isinstance(delta["drop_features"], list):
            drop_set = set(delta["drop_features"])
            dropped.extend(f for f in delta["drop_features"] if f)
            features = [f for f in features if f not in drop_set]
        if "models" in delta and isinstance(delta["models"], list) and delta["models"]:
            models = list(delta["models"])

        # 2. Action-specific Translation Logic
        if action == "resolve_collinearity":
            collinear_pair = delta.get("collinear_pair")
            if isinstance(collinear_pair, (list, tuple)) and len(collinear_pair) >= 2:
                # Deterministically drop the second feature of the collinear pair to break multicollinearity
                drop_col = collinear_pair[1]
                if drop_col in features and len(features) > 1:
                    features = [f for f in features if f != drop_col]
                    dropped.append(drop_col)
                    candidate_name = f"drop_collinear_{drop_col}"
                    metadata["dropped_feature"] = drop_col
            elif "column" in delta and delta["column"] in features and len(features) > 1:
                drop_col = delta["column"]
                features = [f for f in features if f != drop_col]
                dropped.append(drop_col)
                candidate_name = f"drop_collinear_{drop_col}"
                metadata["dropped_feature"] = drop_col

        elif action == "prioritize_feature":
            feature = delta.get("feature") or delta.get("column")
            if feature and feature in features:
                # Reorder so prioritized feature leads the feature vector
                features = [feature] + [f for f in features if f != feature]
                candidate_name = f"prioritize_{feature}"
                metadata["prioritized_feature"] = feature

        elif action in ("nonlinear_transform_or_trees", "nonlinear_models"):
            suggested = delta.get("suggested_models")
            if isinstance(suggested, list) and suggested:
                models = list(suggested)
            else:
                models = ["random_forest", "extra_trees"]
            candidate_name = "nonlinear_tree_models"
            metadata["suggested_models"] = models

        elif action == "power_transform":
            col = delta.get("column")
            transform = str(delta.get("transform", "yeo_johnson")).lower()
            if col and col in features:
                if transform == "log1p":
                    derived_col = f"{col}_log1p"
                    features = [derived_col if f == col else f for f in features]
                    derived_spec = {
                        "name": derived_col,
                        "expression_type": "formula",
                        "expression": f"log1p({col})",
                        "source_columns": [col],
                    }
                    derived_features.append(derived_spec)
                    candidate_name = f"log1p_{col}"
                    metadata["transform"] = "log1p"
                    metadata["derived_features"] = [derived_spec]
                else:
                    ct_spec = {"column": col, "kind": "yeo_johnson", "params": {}}
                    column_transforms.append(ct_spec)
                    candidate_name = f"power_transform_{col}"
                    metadata["transform"] = "yeo_johnson"
                    metadata["column_transforms"] = [ct_spec]
            else:
                candidate_name = f"power_transform_{col or 'var'}"
                metadata["transform"] = transform

        elif action == "robust_scaler":
            col = delta.get("column")
            if col and col in features:
                ct_spec = {"column": col, "kind": "robust_scaler", "params": {}}
                column_transforms.append(ct_spec)
                candidate_name = f"robust_scale_{col}"
                metadata["scaler"] = "robust"
                metadata["column_transforms"] = [ct_spec]
            else:
                candidate_name = f"robust_scale_{col or 'var'}"
                metadata["scaler"] = "robust"

        elif action == "discretize_or_cluster":
            col = delta.get("column")
            if col and col in features:
                ct_spec = {"column": col, "kind": "quantile_discretizer", "params": {"n_bins": 5}}
                column_transforms.append(ct_spec)
                candidate_name = f"discretize_{col}"
                metadata["discretize"] = True
                metadata["column_transforms"] = [ct_spec]
            else:
                candidate_name = f"discretize_{col or 'var'}"
                metadata["discretize"] = True

        elif action == "categorical_interaction":
            pair = delta.get("interaction_pair")
            if isinstance(pair, (list, tuple)):
                candidate_name = f"interact_{'_'.join(pair)}"
                metadata["interaction_pair"] = pair

        # Ensure features is never empty
        if not features:
            features = list(available_features)

        # Ensure models is never empty
        if not models:
            models = cls._default_models_for_task(task_type)

        # Features removed relative to the original available set (stable, de-duplicated).
        if not dropped:
            feature_set = set(features)
            dropped = [f for f in original_features if f not in feature_set]
        seen_drop: set[str] = set()
        drop_unique: list[str] = []
        for name in dropped:
            if name and name not in seen_drop:
                seen_drop.add(name)
                drop_unique.append(name)

        return ExperimentCandidateSpec(
            name=candidate_name,
            feature_names=features,
            model_ids=models,
            hypothesis_text=hypothesis.description,
            action_type=action,
            column_transforms=tuple(column_transforms),
            derived_features=tuple(derived_features),
            drop_features=tuple(drop_unique),
            metadata=metadata,
        )

    @staticmethod
    def _default_models_for_task(task_type: str | None) -> list[str]:
        if task_type == "regression":
            return ["ridge", "random_forest"]
        return ["logistic_regression", "random_forest"]
