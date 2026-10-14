"""MetricSpec: single source of truth for metric direction and best-score selection."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from automl.domain.metrics.exceptions import InvalidMetric


class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: MetricDirection

    @property
    def greater_is_better(self) -> bool:
        return self.direction is MetricDirection.MAXIMIZE

    @property
    def is_minimizing(self) -> bool:
        return self.direction is MetricDirection.MINIMIZE

    def is_better(self, candidate: float, baseline: float) -> bool:
        if self.direction is MetricDirection.MINIMIZE:
            return candidate < baseline
        return candidate > baseline

    def select_best_score(self, scores: Iterable[float]) -> float:
        values = list(scores)
        if not values:
            raise ValueError("Cannot select best score from an empty sequence")
        return min(values) if self.is_minimizing else max(values)

    def select_best_index(self, scores: Sequence[float]) -> int:
        if not scores:
            raise ValueError("Cannot select best index from an empty sequence")
        best = self.select_best_score(scores)
        for idx, score in enumerate(scores):
            if score == best:
                return idx
        raise RuntimeError("Best score not found in sequence")  # pragma: no cover

    # Spec alias used in ADR / docs
    pick_best = select_best_score


# Builtin registry keyed by lower-case canonical name.
_BUILTIN_SPECS: dict[str, MetricSpec] = {
    "accuracy": MetricSpec("accuracy", MetricDirection.MAXIMIZE),
    "balanced_accuracy": MetricSpec("balanced_accuracy", MetricDirection.MAXIMIZE),
    "f1": MetricSpec("f1", MetricDirection.MAXIMIZE),
    "precision": MetricSpec("precision", MetricDirection.MAXIMIZE),
    "recall": MetricSpec("recall", MetricDirection.MAXIMIZE),
    "roc_auc": MetricSpec("roc_auc", MetricDirection.MAXIMIZE),
    "r2": MetricSpec("r2", MetricDirection.MAXIMIZE),
    "mae": MetricSpec("mae", MetricDirection.MINIMIZE),
    "rmse": MetricSpec("rmse", MetricDirection.MINIMIZE),
    "mse": MetricSpec("mse", MetricDirection.MINIMIZE),
    "log_loss": MetricSpec("log_loss", MetricDirection.MINIMIZE),
    "loss": MetricSpec("loss", MetricDirection.MINIMIZE),
    # Common aliases / sklearn-style names (web + tooling)
    "logloss": MetricSpec("logloss", MetricDirection.MINIMIZE),
    "mean_squared_error": MetricSpec("mean_squared_error", MetricDirection.MINIMIZE),
    "mean_absolute_error": MetricSpec("mean_absolute_error", MetricDirection.MINIMIZE),
    "root_mean_squared_error": MetricSpec("root_mean_squared_error", MetricDirection.MINIMIZE),
    "neg_mean_absolute_error": MetricSpec("neg_mean_absolute_error", MetricDirection.MINIMIZE),
    "neg_mean_squared_error": MetricSpec("neg_mean_squared_error", MetricDirection.MINIMIZE),
    "neg_root_mean_squared_error": MetricSpec(
        "neg_root_mean_squared_error", MetricDirection.MINIMIZE
    ),
    "neg_log_loss": MetricSpec("neg_log_loss", MetricDirection.MINIMIZE),
    "silhouette": MetricSpec("silhouette", MetricDirection.MAXIMIZE),
    "calinski_harabasz": MetricSpec("calinski_harabasz", MetricDirection.MAXIMIZE),
}


def builtin_metric_specs() -> dict[str, MetricSpec]:
    """Return a copy of the builtin metric registry."""
    return dict(_BUILTIN_SPECS)


def resolve_metric_spec(
    name: str,
    *,
    greater_is_better: bool | None = None,
) -> MetricSpec:
    """Resolve a metric name to a MetricSpec.

    - If ``greater_is_better`` is provided (plugin / caller override), that direction wins.
    - Else look up the builtin registry.
    - Unknown builtin names fail fast with ``InvalidMetric``.
    """
    key = str(name).lower().strip()
    if not key:
        raise InvalidMetric("Metric name must be a non-empty string")

    if greater_is_better is not None:
        direction = (
            MetricDirection.MAXIMIZE if greater_is_better else MetricDirection.MINIMIZE
        )
        return MetricSpec(name=key, direction=direction)

    spec = _BUILTIN_SPECS.get(key)
    if spec is None:
        raise InvalidMetric(
            f"Unknown metric {name!r}. Register a plugin or pass greater_is_better. "
            f"Known builtins: {sorted(_BUILTIN_SPECS)}"
        )
    return spec


def is_minimizing_metric_name(metric: str) -> bool:
    """True when lower scores are better for a built-in metric name.

    Unknown names are treated as maximizing (legacy safe default for callers that
    only have a name and no plugin registry). Prefer ``resolve_metric_spec`` when
    fail-fast is desired.
    """
    key = str(metric).lower().strip()
    spec = _BUILTIN_SPECS.get(key)
    if spec is None:
        return False
    return spec.is_minimizing
