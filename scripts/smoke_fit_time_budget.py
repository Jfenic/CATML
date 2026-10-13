#!/usr/bin/env python3
"""Quick smoke: AutoML.fit without vs with time_budget (post-#116 wiring)."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from automl.facade import AutoML


def _run(budget: int | None, ws: Path) -> tuple[float, float, str]:
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    t0 = time.perf_counter()
    am = AutoML(
        models=["logistic_regression", "random_forest"],
        metric="roc_auc",
        time_budget=budget,
        workspace_dir=ws,
        cv_folds=3,
        random_state=42,
    )
    res = am.fit(df, target="target")
    wall = time.perf_counter() - t0
    return wall, float(res.best_score), str(res.best_model_id)


def main() -> None:
    root = Path("/tmp/catml_hpo_smoke")
    w0, s0, m0 = _run(None, root / "nobudget")
    print(f"NO_BUDGET wall={w0:.1f}s score={s0:.4f} model={m0}", flush=True)
    w1, s1, m1 = _run(60, root / "budget60")
    print(f"BUDGET_60 wall={w1:.1f}s score={s1:.4f} model={m1}", flush=True)
    print(f"DELTA_WALL={w1 - w0:.1f}s", flush=True)
    print("SMOKE_DONE", flush=True)


if __name__ == "__main__":
    main()
