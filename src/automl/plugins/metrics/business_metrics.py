from __future__ import annotations

from typing import Any
import numpy as np

from automl.domain.plugins.plugin import PluginCapability, PluginType
from automl.domain.ports import MetricPluginPort


class CostSensitiveMetricPlugin(MetricPluginPort):
    """Calculates total business financial cost based on False Negatives and False Positives."""

    def __init__(
        self,
        cost_fn: float = 100.0,
        cost_fp: float = 20.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0,
        normalize: bool = False,
        version: str = "1.0.0",
    ) -> None:
        self.plugin_id = "churn_cost"
        self.name = "Cost-Sensitive Churn Loss"
        self.version = version
        self.plugin_type = PluginType.METRIC
        self.greater_is_better = False  # Minimize total financial loss
        self.capabilities = PluginCapability(
            supported_tasks=["binary_classification"],
            supported_modalities=["tabular"],
        )
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.cost_tp = cost_tp
        self.cost_tn = cost_tn
        self.normalize = normalize

    def compute(self, y_true: Any, y_pred: Any, y_prob: Any | None = None) -> float:
        y_t = np.asarray(y_true).astype(int)
        y_p = np.asarray(y_pred).astype(int)

        fn = np.sum((y_t == 1) & (y_p == 0))
        fp = np.sum((y_t == 0) & (y_p == 1))
        tp = np.sum((y_t == 1) & (y_p == 1))
        tn = np.sum((y_t == 0) & (y_p == 0))

        total_cost = (
            fn * self.cost_fn
            + fp * self.cost_fp
            + tp * self.cost_tp
            + tn * self.cost_tn
        )
        if self.normalize and len(y_t) > 0:
            return float(total_cost / len(y_t))
        return float(total_cost)


class WeightedF1MetricPlugin(MetricPluginPort):
    """Calculates custom F-Beta or weighted F1 score for imbalanced classification."""

    def __init__(self, beta: float = 2.0, version: str = "1.0.0") -> None:
        self.plugin_id = "f_beta"
        self.name = f"F-{beta} Score"
        self.version = version
        self.plugin_type = PluginType.METRIC
        self.greater_is_better = True  # Maximize
        self.capabilities = PluginCapability(
            supported_tasks=["binary_classification", "multiclass_classification"],
            supported_modalities=["tabular"],
        )
        self.beta = beta

    def compute(self, y_true: Any, y_pred: Any, y_prob: Any | None = None) -> float:
        from sklearn.metrics import fbeta_score

        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        return float(fbeta_score(y_t, y_p, beta=self.beta, average="weighted", zero_division=0))
