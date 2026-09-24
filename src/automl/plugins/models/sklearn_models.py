from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR


def build_sklearn_model(model_id: str, task_type: str, parameters: dict | None = None):
    params = parameters or {}
    is_regression = task_type == "regression"
    is_clustering = task_type == "clustering"

    if is_clustering:
        if model_id == "kmeans":
            return KMeans(
                n_clusters=int(params.get("n_clusters", 3)),
                random_state=int(params.get("random_state", 42)),
                n_init=int(params.get("n_init", 10)),
            )
        elif model_id == "agglomerative":
            return AgglomerativeClustering(
                n_clusters=int(params.get("n_clusters", 3)),
            )
        elif model_id == "dbscan":
            return DBSCAN(
                eps=float(params.get("eps", 0.5)),
                min_samples=int(params.get("min_samples", 5)),
            )
        else:
            raise ValueError(f"Unknown clustering model: {model_id}")

    if model_id == "logistic_regression":
        lr_params = {"max_iter": 1000, "random_state": 42}
        lr_params.update(params)
        return LogisticRegression(**lr_params)
    elif model_id == "random_forest":
        rf_params = {"n_estimators": 100, "random_state": 42}
        rf_params.update(params)
        return (
            RandomForestRegressor(**rf_params)
            if is_regression
            else RandomForestClassifier(**rf_params)
        )
    elif model_id in {"svc", "svr"}:
        if is_regression or model_id == "svr":
            svr_params = {}
            svr_params.update(params)
            return SVR(**svr_params)
        else:
            svc_params = {"random_state": 42}
            svc_params.update(params)
            return CalibratedClassifierCV(SVC(**svc_params), method="sigmoid", cv=3)
    elif model_id == "ridge":
        ridge_params = {"alpha": 1.0}
        ridge_params.update(params)
        return Ridge(**ridge_params)
    else:
        raise ValueError(f"Unknown supervised model: {model_id}")


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
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
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
