"""Simple sklearn baseline adapter for Benchmark v1 pilots."""
from __future__ import annotations

from dataclasses import dataclass
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class SklearnPilotResult:
    status: str
    primary_score: float | None
    wall_time_s: float
    failure_reason: str | None = None


def run_sklearn_baseline(df: pd.DataFrame, target: str, *, random_state: int = 42) -> SklearnPilotResult:
    started = time.perf_counter()
    try:
        y = df[target]
        X = df.drop(columns=[target])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y if y.nunique() > 1 else None
        )
        num_cols = list(X_train.select_dtypes(include="number").columns)
        cat_cols = [c for c in X_train.columns if c not in num_cols]
        pre = ColumnTransformer(
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
        pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
        pipe.fit(X_train, y_train)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:, 1]
            score = float(roc_auc_score(y_test, proba))
        else:
            score = None
        return SklearnPilotResult(status="ok", primary_score=score, wall_time_s=time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001
        return SklearnPilotResult(
            status="failed",
            primary_score=None,
            wall_time_s=time.perf_counter() - started,
            failure_reason=str(exc),
        )
