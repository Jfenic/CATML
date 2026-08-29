from __future__ import annotations

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR


def build_sklearn_model(model_id: str, task_type: str, parameters: dict | None = None):
    params = parameters or {}
    is_regression = task_type == "regression"
    is_clustering = task_type == "clustering"

    if is_clustering:
        clustering = {
            "kmeans": KMeans(
                n_clusters=int(params.get("n_clusters", 3)),
                random_state=int(params.get("random_state", 42)),
                n_init=int(params.get("n_init", 10)),
            ),
            "agglomerative": AgglomerativeClustering(
                n_clusters=int(params.get("n_clusters", 3)),
            ),
            "dbscan": DBSCAN(
                eps=float(params.get("eps", 0.5)),
                min_samples=int(params.get("min_samples", 5)),
            ),
        }
        if model_id not in clustering:
            raise ValueError(f"Unknown clustering model: {model_id}")
        return clustering[model_id]

    supervised = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": (
            RandomForestRegressor(n_estimators=100, random_state=42)
            if is_regression
            else RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        "svc": SVR() if is_regression else SVC(probability=True, random_state=42),
        "ridge": Ridge(alpha=1.0),
        "svr": SVR(),
    }

    if model_id not in supervised:
        raise ValueError(f"Unknown model plugin: {model_id}")
    return supervised[model_id]


def default_model_specs():
    from automl.domain.models.registry import ModelSpec
    from automl.domain.tasks.task_type import TaskType, models_for_task

    specs: list[ModelSpec] = []
    seen: dict[str, set[str]] = {}

    for task in TaskType:
        for model_id in models_for_task(task):
            seen.setdefault(model_id, set()).add(task.value)

    labels = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "svc": "Support Vector Classifier",
        "ridge": "Ridge Regression",
        "svr": "Support Vector Regressor",
        "kmeans": "K-Means",
        "agglomerative": "Agglomerative Clustering",
        "dbscan": "DBSCAN",
    }

    for model_id, task_types in sorted(seen.items()):
        specs.append(
            ModelSpec(
                id=model_id,
                name=labels.get(model_id, model_id),
                task_types=sorted(task_types),
                description=f"Compatible with: {', '.join(sorted(task_types))}",
            )
        )
    return specs
