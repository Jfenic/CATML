"""Shared holdout + external metrics + light resource tracking for Pilot 1A."""
from __future__ import annotations

from dataclasses import dataclass, field
import resource
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class HoldoutSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    target: str
    seed: int
    test_size: float = 0.2


@dataclass
class ExternalScores:
    primary: float | None
    secondary: dict[str, float] = field(default_factory=dict)


@dataclass
class ResourceSnapshot:
    wall_time_s: float
    peak_rss_mb: float | None
    cpu_user_s: float | None = None
    cpu_system_s: float | None = None


class ResourceTracker:
    """Wall clock + peak RSS (Linux ru_maxrss is KB) around a fit."""

    def __init__(self) -> None:
        self._t0 = 0.0
        self._cpu0: resource.struct_rusage | None = None

    def __enter__(self) -> ResourceTracker:
        self._t0 = time.perf_counter()
        self._cpu0 = resource.getrusage(resource.RUSAGE_SELF)
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def snapshot(self) -> ResourceSnapshot:
        wall = time.perf_counter() - self._t0
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # Linux: KB; macOS: bytes. Heuristic: values > 10M likely bytes.
        rss = float(usage.ru_maxrss)
        peak_mb = rss / (1024 * 1024) if rss > 10_000_000 else rss / 1024.0
        cpu_user = None
        cpu_sys = None
        if self._cpu0 is not None:
            cpu_user = float(usage.ru_utime - self._cpu0.ru_utime)
            cpu_sys = float(usage.ru_stime - self._cpu0.ru_stime)
        return ResourceSnapshot(
            wall_time_s=wall,
            peak_rss_mb=peak_mb,
            cpu_user_s=cpu_user,
            cpu_system_s=cpu_sys,
        )


def make_holdout(
    df: pd.DataFrame,
    target: str,
    *,
    task: str,
    seed: int,
    test_size: float = 0.2,
) -> HoldoutSplit:
    """Fix train/test indices once per (dataset, seed) before any framework runs."""
    y = df[target]
    stratify = None
    if task != "regression" and y.nunique() > 1:
        counts = y.value_counts()
        if int(counts.min()) >= 2:
            stratify = y
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify
    )
    return HoldoutSplit(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        target=target,
        seed=seed,
        test_size=test_size,
    )


def score_external(
    *,
    task: str,
    metric: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray | None,
    y_proba: np.ndarray | None = None,
) -> ExternalScores:
    """Score predictions with one evaluator shared across frameworks."""
    y_true_arr = np.asarray(y_true)
    secondary: dict[str, float] = {}

    if task == "regression" or metric in {"rmse", "mae", "r2", "mse"}:
        if y_pred is None:
            return ExternalScores(primary=None)
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred)))
        mae = float(mean_absolute_error(y_true_arr, y_pred))
        secondary["mae"] = mae
        secondary["rmse"] = rmse
        primary = rmse if metric != "mae" else mae
        return ExternalScores(primary=primary, secondary=secondary)

    # Classification — encode labels for stable multiclass metrics
    le = LabelEncoder()
    y_enc = le.fit_transform(pd.Series(y_true_arr).astype(str))
    pred_enc = None
    if y_pred is not None:
        pred_enc = le.transform(pd.Series(y_pred).astype(str))

    if pred_enc is not None:
        secondary["accuracy"] = float(accuracy_score(y_enc, pred_enc))
        secondary["f1_macro"] = float(f1_score(y_enc, pred_enc, average="macro", zero_division=0))

    if metric in {"f1_macro", "f1"}:
        primary = secondary.get("f1_macro")
        return ExternalScores(primary=primary, secondary=secondary)
    if metric == "accuracy":
        primary = secondary.get("accuracy")
        return ExternalScores(primary=primary, secondary=secondary)

    if metric == "log_loss" or task == "multiclass":
        if y_proba is None:
            # Fall back to f1 if no probabilities
            primary = secondary.get("f1_macro")
            return ExternalScores(primary=primary, secondary=secondary)
        ll = float(log_loss(y_enc, y_proba, labels=list(range(len(le.classes_)))))
        secondary["log_loss"] = ll
        if metric == "log_loss":
            return ExternalScores(primary=ll, secondary=secondary)
        # multiclass default primary still f1 if metric says so
        return ExternalScores(primary=secondary.get("f1_macro"), secondary=secondary)

    # binary roc_auc
    if y_proba is not None:
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            auc = float(roc_auc_score(y_enc, y_proba[:, 1]))
        elif y_proba.ndim == 2:
            auc = float(roc_auc_score(y_enc, y_proba, multi_class="ovr", average="macro"))
        else:
            auc = float(roc_auc_score(y_enc, y_proba))
        secondary["roc_auc"] = auc
        return ExternalScores(primary=auc, secondary=secondary)

    if pred_enc is not None:
        return ExternalScores(primary=secondary.get("accuracy"), secondary=secondary)
    return ExternalScores(primary=None, secondary=secondary)


def framework_version(name: str) -> str | None:
    try:
        if name == "sklearn_baseline":
            import sklearn

            return sklearn.__version__
        if name == "catml":
            import catml

            return getattr(catml, "__version__", None)
        if name == "flaml":
            import flaml

            return flaml.__version__
        if name == "autogluon":
            import autogluon.tabular as ag

            return getattr(ag, "__version__", None) or "installed"
    except ImportError:
        return None
    return None


def result_dict(
    *,
    dataset_id: str,
    task: str,
    metric: str,
    framework: str,
    seed: int,
    status: str,
    primary_score: float | None,
    wall_time_s: float,
    peak_rss_mb: float | None = None,
    cpu_user_s: float | None = None,
    cpu_system_s: float | None = None,
    secondary: dict[str, Any] | None = None,
    failure_reason: str | None = None,
    model_id: str | None = None,
    note: str | None = None,
    framework_version: str | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "task": task,
        "metric": metric,
        "framework": framework,
        "seed": seed,
        "status": status,
        "primary_score": primary_score,
        "wall_time_s": wall_time_s,
        "peak_rss_mb": peak_rss_mb,
        "cpu_user_s": cpu_user_s,
        "cpu_system_s": cpu_system_s,
        "framework_version": framework_version,
        "secondary": secondary or {},
    }
    if failure_reason is not None:
        out["failure_reason"] = failure_reason
    if model_id is not None:
        out["model_id"] = model_id
    if note is not None:
        out["note"] = note
    return out
