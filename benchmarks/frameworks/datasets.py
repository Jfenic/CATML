"""Load active Study A / pilot datasets from the Benchmark v1 manifest."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

MANIFEST = Path(__file__).resolve().parents[1] / "datasets-manifest.csv"


@dataclass(frozen=True)
class PilotDataset:
    dataset_id: str
    task: str
    metric: str
    target: str
    frame: pd.DataFrame


def _maybe_subsample(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=0).reset_index(drop=True)


def _parse_max_rows(raw: str | None) -> int | None:
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def _load_sklearn_breast_cancer() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    return load_breast_cancer(as_frame=True).frame


def _load_sklearn_wine() -> pd.DataFrame:
    from sklearn.datasets import load_wine

    return load_wine(as_frame=True).frame


def _load_sklearn_california_housing() -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    return fetch_california_housing(as_frame=True).frame


def _load_sklearn_diabetes() -> pd.DataFrame:
    from sklearn.datasets import load_diabetes

    return load_diabetes(as_frame=True).frame


def _load_openml(data_id: int, target: str) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    frame = bunch.frame.copy()
    if target not in frame.columns and bunch.target is not None:
        # Prefer requested name; else use series name or 'target'
        series = bunch.target
        col = target if target else (getattr(series, "name", None) or "target")
        frame[col] = series.values
        if target != col and target not in frame.columns:
            frame[target] = frame[col]
    # Case-insensitive target resolve
    if target not in frame.columns:
        lower = {c.lower(): c for c in frame.columns}
        if target.lower() in lower:
            frame[target] = frame[lower[target.lower()]]
    for col in frame.columns:
        if str(frame[col].dtype) == "category":
            frame[col] = frame[col].astype(object)
    return frame


def _resolve_frame(dataset_id: str, source: str, target: str) -> pd.DataFrame:
    if dataset_id == "sklearn_breast_cancer":
        return _load_sklearn_breast_cancer()
    if dataset_id == "sklearn_wine":
        return _load_sklearn_wine()
    if dataset_id == "sklearn_california_housing":
        return _load_sklearn_california_housing()
    if dataset_id == "sklearn_diabetes":
        return _load_sklearn_diabetes()
    if source.startswith("openml:"):
        data_id = int(source.split(":", 1)[1])
        return _load_openml(data_id, target)
    raise KeyError(f"Unknown dataset source for {dataset_id}: {source}")


def load_pilot_datasets(
    *,
    only_active: bool = True,
    study: str | None = None,
) -> list[PilotDataset]:
    """Load tabular datasets for pilot or Study A.

    ``study='A'`` / ``study='pilot'`` both use ``status=active`` rows excluding Study C fixtures.
    """
    rows: list[PilotDataset] = []
    with MANIFEST.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if only_active and row.get("status") != "active":
                continue
            if row.get("task") == "*" or row.get("source") == "fixture":
                continue
            dataset_id = row["dataset_id"]
            # Pilot mode: original trio only
            if study == "pilot" and dataset_id not in {
                "sklearn_breast_cancer",
                "sklearn_wine",
                "sklearn_california_housing",
            }:
                continue
            task = row["task"]
            metric = row["metric"]
            target = row["target"]
            source = row["source"]
            max_rows = _parse_max_rows(row.get("max_rows"))
            frame = _resolve_frame(dataset_id, source, target)
            frame = _maybe_subsample(frame, max_rows)
            if target not in frame.columns:
                # last resort: single leftover target-like column
                raise KeyError(
                    f"{dataset_id}: target column {target!r} missing; cols={list(frame.columns)}"
                )
            rows.append(
                PilotDataset(
                    dataset_id=dataset_id,
                    task=task,
                    metric=metric,
                    target=target,
                    frame=frame,
                )
            )
    return rows


def load_study_a_datasets(*, only_active: bool = True) -> list[PilotDataset]:
    return load_pilot_datasets(only_active=only_active, study="A")
