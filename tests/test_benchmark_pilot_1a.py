"""Pilot 1A / Study A harness: shared holdout, freeze, dry-run, optional-deps skip."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_protocol_is_frozen():
    text = (ROOT / "benchmarks" / "protocol-v1.yaml").read_text(encoding="utf-8")
    assert "status: frozen" in text
    assert "shared_external_holdout" in text
    assert "f1_macro" in text
    assert "skipped_does_not_count_as_comparison: true" in text


def test_load_pilot_trio():
    from benchmarks.frameworks.datasets import load_pilot_datasets

    datasets = load_pilot_datasets(study="pilot")
    ids = {d.dataset_id for d in datasets}
    assert ids == {
        "sklearn_breast_cancer",
        "sklearn_wine",
        "sklearn_california_housing",
    }


def test_load_study_a_has_binary_multi_regression():
    from benchmarks.frameworks.datasets import load_study_a_datasets

    datasets = load_study_a_datasets()
    assert len(datasets) >= 9
    tasks = {d.task for d in datasets}
    assert tasks == {"binary_classification", "multiclass", "regression"}


def test_shared_holdout_identical_indices():
    from sklearn.datasets import load_breast_cancer

    from benchmarks.frameworks.eval import make_holdout

    df = load_breast_cancer(as_frame=True).frame
    a = make_holdout(df, "target", task="binary_classification", seed=42)
    b = make_holdout(df, "target", task="binary_classification", seed=42)
    assert list(a.train.index) == list(b.train.index)
    assert list(a.test.index) == list(b.test.index)
    assert len(a.train) + len(a.test) == len(df)


def test_external_score_rmse():
    from benchmarks.frameworks.eval import score_external

    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 4.0])
    scores = score_external(task="regression", metric="rmse", y_true=y, y_pred=pred)
    assert scores.primary is not None
    assert abs(scores.primary - float(np.sqrt(((pred - y) ** 2).mean()))) < 1e-9
    assert "mae" in scores.secondary


def test_study_a_dry_run_writes_artifacts(tmp_path: Path):
    from benchmarks.run import _campaign

    out = tmp_path / "v1"
    payload = _campaign(
        study="A", dry_run=True, out_dir=out, time_budget_s=60.0, seeds=[42]
    )
    assert payload["study"] == "A"
    assert payload["protocol_status"] == "frozen"
    assert (out / "study-a-raw.json").is_file()
    assert (out / "environment.json").is_file()
    assert (out / "checksums.sha256").is_file()
    raw = json.loads((out / "study-a-raw.json").read_text(encoding="utf-8"))
    assert raw["rows"]
    frameworks = {r["framework"] for r in raw["rows"]}
    assert {"sklearn_baseline", "catml", "flaml", "autogluon"} <= frameworks


def test_study_b_scaffold(tmp_path: Path):
    from benchmarks.run import _study_b_scaffold

    payload = _study_b_scaffold(dry_run=True, out_dir=tmp_path)
    assert payload["study"] == "B"
    assert "catml_agent" in payload["configurations"]
    assert (tmp_path / "study-b-scaffold.json").is_file()


def test_flaml_adapter_skips_without_install(monkeypatch: pytest.MonkeyPatch):
    import builtins

    from sklearn.datasets import load_breast_cancer

    from benchmarks.frameworks.eval import make_holdout

    real_import = builtins.__import__

    def _blocker(name, *args, **kwargs):
        if name == "flaml" or name.startswith("flaml."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocker)
    from benchmarks.frameworks import flaml_adapter

    df = load_breast_cancer(as_frame=True).frame
    holdout = make_holdout(df, "target", task="binary_classification", seed=0)
    result = flaml_adapter.run_flaml(
        holdout.train, holdout.test, "target", time_budget=5.0
    )
    assert result.status == "skipped"
    assert result.y_pred is None
    assert "flaml_not_installed" in (result.failure_reason or "")


def test_sklearn_and_catml_share_external_metric(tmp_path: Path):
    from sklearn.datasets import load_breast_cancer

    from benchmarks.frameworks.catml_adapter import run_catml_fit
    from benchmarks.frameworks.eval import make_holdout, score_external
    from benchmarks.frameworks.sklearn_adapter import run_sklearn_baseline

    df = load_breast_cancer(as_frame=True).frame
    holdout = make_holdout(df, "target", task="binary_classification", seed=42)
    sk = run_sklearn_baseline(holdout.train, holdout.test, "target", task="binary_classification")
    cat = run_catml_fit(
        holdout.train,
        holdout.test,
        "target",
        metric="roc_auc",
        workspace_dir=tmp_path / "ws",
        time_budget=30.0,
        random_state=42,
    )
    assert sk.status == "ok" and cat.status == "ok"
    sk_score = score_external(
        task="binary_classification",
        metric="roc_auc",
        y_true=holdout.test["target"],
        y_pred=sk.y_pred,
        y_proba=sk.y_proba,
    )
    cat_score = score_external(
        task="binary_classification",
        metric="roc_auc",
        y_true=holdout.test["target"],
        y_pred=cat.y_pred,
        y_proba=cat.y_proba,
    )
    assert sk_score.primary is not None and cat_score.primary is not None
    assert 0.9 <= sk_score.primary <= 1.0
    assert 0.9 <= cat_score.primary <= 1.0
