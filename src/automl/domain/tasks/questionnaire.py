"""Pure domain entities and enums for dataset framing and problem context questionnaires."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ErrorCostPriority(str, Enum):
    """Business cost priority for prediction errors."""
    BALANCED = "balanced"
    AVOID_FALSE_NEGATIVES = "avoid_false_negatives"  # E.g. Fraud, Churn, Medical: Missed positive is critical
    AVOID_FALSE_POSITIVES = "avoid_false_positives"  # E.g. Spam filtering, automated blocking
    CUSTOM_COST_MATRIX = "custom_cost_matrix"


class TemporalStructure(str, Enum):
    """Chronological and sequential ordering characteristics of the dataset."""
    CROSS_SECTIONAL = "cross_sectional"              # Independent rows (standard K-Fold)
    SEQUENTIAL_TIME_SERIES = "sequential_time_series"  # Autoregressive sequence (TimeSeriesSplit)
    GROUPED_COHORTS = "grouped_cohorts"              # Grouped observations (GroupKFold)
    UNKNOWN = "unknown"


class LatencyConstraint(str, Enum):
    """Operational latency budget for real-world serving."""
    ULTRA_LOW_REALTIME = "ultra_low_realtime"      # < 10ms (Linear models, shallow decision trees)
    STANDARD_INTERACTIVE = "standard_interactive"  # < 200ms (Gradient boosting trees: LightGBM, XGBoost)
    BATCH_OFFLINE = "batch_offline"                # No strict deadline (Super-Learner, stacked ensembles)


class ExplainabilityLevel(str, Enum):
    """Regulatory and interpretability constraints on model architectures."""
    HIGHLY_REGULATED = "highly_regulated"          # E.g. Banking credit scoring, strict auditable rules
    MODERATE = "moderate"                          # Tree SHAP, feature importances
    PERFORMANCE_FIRST = "performance_first"        # Kaggle, competitive, black-box allowed


_TIME_SERIES_SPLITS = frozenset({"time_series", "time_series_split"})
_GROUP_SPLITS = frozenset({"group_kfold", "group_cv", "grouped_kfold"})


def default_split_for_temporal(
    temporal: TemporalStructure,
    *,
    classification: bool = True,
) -> str:
    """Default validation strategy aligned with temporal framing."""
    if temporal == TemporalStructure.SEQUENTIAL_TIME_SERIES:
        return "time_series_split"
    if temporal == TemporalStructure.GROUPED_COHORTS:
        return "group_kfold"
    return "stratified_kfold" if classification else "kfold"


def coerce_split_strategy(
    temporal: TemporalStructure,
    split: str | None,
    *,
    classification: bool = True,
) -> str:
    """Keep split_strategy_recommendation coherent with temporal_structure.

    Cross-sectional datasets must not keep a leftover TimeSeriesSplit / GroupKFold
    recommendation (that combination made airline-style launches fail without time_column).
    """
    raw = (split or "").strip().lower()
    if not raw:
        return default_split_for_temporal(temporal, classification=classification)
    if temporal == TemporalStructure.CROSS_SECTIONAL:
        if raw in _TIME_SERIES_SPLITS or raw in _GROUP_SPLITS:
            return default_split_for_temporal(temporal, classification=classification)
        return raw
    if temporal == TemporalStructure.SEQUENTIAL_TIME_SERIES:
        return raw if raw in _TIME_SERIES_SPLITS else "time_series_split"
    if temporal == TemporalStructure.GROUPED_COHORTS:
        return raw if raw in _GROUP_SPLITS else "group_kfold"
    return raw


@dataclass
class DatasetQuestionnaire:
    """Problem context questionnaire guiding AutoML task planning and model selection."""
    dataset_id: str
    domain_hint: str
    error_cost: ErrorCostPriority
    temporal_structure: TemporalStructure
    latency_constraint: LatencyConstraint
    explainability: ExplainabilityLevel
    imbalance_strategy: str = "auto"
    primary_metric_override: str | None = None
    split_strategy_recommendation: str = "stratified_kfold"
    auto_generated: bool = True
    user_confirmed: bool = False
    engine_source: str = "heuristic"
    confidence: float = 1.0
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize questionnaire to dictionary."""
        d = asdict(self)
        d["error_cost"] = self.error_cost.value
        d["temporal_structure"] = self.temporal_structure.value
        d["latency_constraint"] = self.latency_constraint.value
        d["explainability"] = self.explainability.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetQuestionnaire:
        """Construct questionnaire from dictionary."""
        temporal = TemporalStructure(
            data.get("temporal_structure", TemporalStructure.CROSS_SECTIONAL.value)
        )
        split = coerce_split_strategy(
            temporal,
            data.get("split_strategy_recommendation", "stratified_kfold"),
        )
        return cls(
            dataset_id=data["dataset_id"],
            domain_hint=data.get("domain_hint", "general_tabular"),
            error_cost=ErrorCostPriority(data.get("error_cost", ErrorCostPriority.BALANCED.value)),
            temporal_structure=temporal,
            latency_constraint=LatencyConstraint(data.get("latency_constraint", LatencyConstraint.STANDARD_INTERACTIVE.value)),
            explainability=ExplainabilityLevel(data.get("explainability", ExplainabilityLevel.MODERATE.value)),
            imbalance_strategy=data.get("imbalance_strategy", "auto"),
            primary_metric_override=data.get("primary_metric_override"),
            split_strategy_recommendation=split,
            auto_generated=bool(data.get("auto_generated", True)),
            user_confirmed=bool(data.get("user_confirmed", False)),
            engine_source=str(data.get("engine_source", "heuristic")),
            confidence=float(data.get("confidence", 1.0)),
            reasoning=data.get("reasoning", ""),
        )
