"""Shared helpers for CATML fire-test notebooks / scripts."""
from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"


@dataclass
class FireResult:
    test_id: str
    title: str
    status: str  # ok | failed
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    traceback: str | None = None


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_result(result: FireResult) -> Path:
    out = ensure_results_dir() / f"{result.test_id}.json"
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return out


def run_fire_test(test_id: str, title: str, fn: Callable[[], dict[str, Any]]) -> FireResult:
    try:
        metrics = fn() or {}
        result = FireResult(test_id=test_id, title=title, status="ok", metrics=metrics)
    except Exception as exc:  # noqa: BLE001 — fire tests must capture failures
        result = FireResult(
            test_id=test_id,
            title=title,
            status="failed",
            detail=str(exc),
            error_type=type(exc).__name__,
            traceback=traceback.format_exc(),
        )
    write_result(result)
    return result


def load_titanic_like() -> tuple[pd.DataFrame, str]:
    """Binary classification with mixed types (OpenML Titanic if reachable, else synthetic)."""
    try:
        from sklearn.datasets import fetch_openml

        bunch = fetch_openml(data_id=40945, as_frame=True, parser="auto")
        df = bunch.frame.copy()
        # Drop leakage / ids commonly excluded
        drop = [c for c in ("name", "ticket", "cabin", "boat", "body", "home.dest") if c in df.columns]
        df = df.drop(columns=drop, errors="ignore")
        target = "survived"
        if target not in df.columns and bunch.target is not None:
            df[target] = bunch.target
        for col in df.columns:
            if str(df[col].dtype) == "category":
                df[col] = df[col].astype(object)
        return df.dropna(subset=[target]).reset_index(drop=True), target
    except Exception:
        rng = np.random.default_rng(0)
        n = 400
        df = pd.DataFrame(
            {
                "pclass": rng.integers(1, 4, n),
                "sex": rng.choice(["male", "female"], n),
                "age": rng.normal(30, 12, n).clip(1, 80),
                "sibsp": rng.integers(0, 4, n),
                "fare": rng.lognormal(3, 0.8, n),
                "embarked": rng.choice(["S", "C", "Q"], n),
            }
        )
        logits = (
            -0.8 * (df["sex"] == "male").astype(float)
            + 0.4 * (4 - df["pclass"])
            + 0.01 * df["fare"]
            - 0.02 * df["age"]
        )
        df["survived"] = (logits + rng.normal(0, 0.5, n) > 0).astype(int)
        return df, "survived"


def load_house_prices_like() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_california_housing

    df = fetch_california_housing(as_frame=True).frame
    return df, "MedHouseVal"


def load_credit_risk() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=31, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    target = "class"
    if target not in df.columns and bunch.target is not None:
        df[target] = bunch.target
    for col in df.columns:
        if str(df[col].dtype) == "category":
            df[col] = df[col].astype(object)
    return df, target


def load_multiclass() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_wine

    return load_wine(as_frame=True).frame, "target"


def load_high_cardinality() -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(7)
    n = 800
    df = pd.DataFrame(
        {
            "user_id": [f"u{i}" for i in range(n)],  # near-unique id
            "city": rng.choice([f"city_{i}" for i in range(40)], n),
            "amount": rng.lognormal(3, 1, n),
            "tenure": rng.integers(0, 60, n),
        }
    )
    df["churn"] = (
        (df["amount"] < 10).astype(int) | (df["tenure"] < 3).astype(int)
    ).astype(int)
    return df, "churn"


def load_ordered_split() -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(11)
    n = 600
    t = np.arange(n)
    df = pd.DataFrame(
        {
            "time_idx": t,
            "x1": rng.normal(size=n) + 0.01 * t,
            "x2": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], n),
        }
    )
    df["label"] = ((df["x1"] + 0.002 * df["time_idx"] + rng.normal(0, 0.3, n)) > 0).astype(int)
    return df, "label"
