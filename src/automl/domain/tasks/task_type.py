from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"

    @classmethod
    def parse(cls, value: str) -> TaskType:
        normalized = value.strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        raise ValueError(
            f"Unknown task type: {value}. "
            f"Use: {', '.join(t.value for t in cls)}"
        )


TASK_CATALOG: dict[TaskType, dict] = {
    TaskType.BINARY_CLASSIFICATION: {
        "label": "Clasificación binaria",
        "description": "Predecir una de dos clases (p. ej. churn sí/no).",
        "default_metric": "roc_auc",
        "metrics": ["roc_auc", "accuracy", "f1"],
        "model_ids": ["logistic_regression", "random_forest", "svc"],
    },
    TaskType.MULTICLASS_CLASSIFICATION: {
        "label": "Clasificación multiclase",
        "description": "Predecir una entre tres o más clases discretas.",
        "default_metric": "accuracy",
        "metrics": ["accuracy", "f1", "roc_auc"],
        "model_ids": ["logistic_regression", "random_forest", "svc"],
    },
    TaskType.REGRESSION: {
        "label": "Regresión",
        "description": "Predecir un valor numérico continuo.",
        "default_metric": "r2",
        "metrics": ["r2", "mae", "rmse"],
        "model_ids": ["ridge", "random_forest", "svr"],
    },
    TaskType.CLUSTERING: {
        "label": "Agrupación (clustering)",
        "description": "Descubrir grupos en los datos sin variable objetivo supervisada.",
        "default_metric": "silhouette",
        "metrics": ["silhouette", "calinski_harabasz"],
        "model_ids": ["kmeans", "agglomerative", "dbscan"],
    },
}


def default_metric_for(task_type: TaskType) -> str:
    return TASK_CATALOG[task_type]["default_metric"]


def models_for_task(task_type: TaskType) -> list[str]:
    return list(TASK_CATALOG[task_type]["model_ids"])


def metrics_for_task(task_type: TaskType) -> list[str]:
    return list(TASK_CATALOG[task_type]["metrics"])
