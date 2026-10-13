"""Sklearn baseline — fit on shared train, predict shared test (external scored)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from benchmarks.frameworks.eval import ResourceTracker


@dataclass
class SklearnPilotResult:
    status: str
    y_pred: np.ndarray | None
    y_proba: np.ndarray | None
    wall_time_s: float
    peak_rss_mb: float | None = None
    cpu_user_s: float | None = None
    cpu_system_s: float | None = None
    failure_reason: str | None = None
    extra: dict[str, Any] | None = None


def _build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = list(X_train.select_dtypes(include="number").columns)
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def run_sklearn_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    *,
    task: str = "binary_classification",
) -> SklearnPilotResult:
    with ResourceTracker() as tracker:
        try:
            y_train = train[target]
            X_train = train.drop(columns=[target])
            X_test = test.drop(columns=[target])
            pre = _build_preprocessor(X_train)

            if task == "regression":
                pipe = Pipeline([("pre", pre), ("reg", Ridge())])
                pipe.fit(X_train, y_train)
                pred = np.asarray(pipe.predict(X_test))
                snap = tracker.snapshot()
                return SklearnPilotResult(
                    status="ok",
                    y_pred=pred,
                    y_proba=None,
                    wall_time_s=snap.wall_time_s,
                    peak_rss_mb=snap.peak_rss_mb,
                    cpu_user_s=snap.cpu_user_s,
                    cpu_system_s=snap.cpu_system_s,
                )

            le = LabelEncoder()
            y_enc = le.fit_transform(y_train.astype(str))
            pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
            pipe.fit(X_train, y_enc)
            pred_enc = pipe.predict(X_test)
            pred = le.inverse_transform(pred_enc)
            proba = pipe.predict_proba(X_test)
            snap = tracker.snapshot()
            return SklearnPilotResult(
                status="ok",
                y_pred=np.asarray(pred),
                y_proba=np.asarray(proba),
                wall_time_s=snap.wall_time_s,
                peak_rss_mb=snap.peak_rss_mb,
                cpu_user_s=snap.cpu_user_s,
                cpu_system_s=snap.cpu_system_s,
                extra={"classes": list(le.classes_)},
            )
        except Exception as exc:  # noqa: BLE001
            snap = tracker.snapshot()
            return SklearnPilotResult(
                status="failed",
                y_pred=None,
                y_proba=None,
                wall_time_s=snap.wall_time_s,
                peak_rss_mb=snap.peak_rss_mb,
                cpu_user_s=snap.cpu_user_s,
                cpu_system_s=snap.cpu_system_s,
                failure_reason=str(exc),
            )
