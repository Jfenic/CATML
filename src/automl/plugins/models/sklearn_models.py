from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR


def build_sklearn_model(model_id: str, task_type: str):
    is_regression = task_type == "regression"

    models = {
        "logistic_regression": (
            Ridge(alpha=1.0) if is_regression else LogisticRegression(max_iter=1000, random_state=42)
        ),
        "random_forest": (
            RandomForestRegressor(n_estimators=100, random_state=42)
            if is_regression
            else RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        "svc": SVR() if is_regression else SVC(probability=True, random_state=42),
    }

    if model_id not in models:
        raise ValueError(f"Unknown model plugin: {model_id}")
    return models[model_id]


def default_model_specs():
    from automl.domain.models.registry import ModelSpec

    return [
        ModelSpec(
            id="logistic_regression",
            name="Logistic Regression / Ridge",
            task_types=["binary_classification", "multiclass_classification", "regression"],
        ),
        ModelSpec(
            id="random_forest",
            name="Random Forest",
            task_types=["binary_classification", "multiclass_classification", "regression"],
        ),
        ModelSpec(
            id="svc",
            name="Support Vector Machine",
            task_types=["binary_classification", "multiclass_classification", "regression"],
        ),
    ]
