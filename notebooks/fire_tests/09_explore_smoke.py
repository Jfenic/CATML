"""09 — Explore analysis study smoke (CQRS AnalysisStudyService)."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import run_fire_test  # noqa: E402


def main() -> None:
    def body():
        import numpy as np
        import pandas as pd

        from automl.application.analysis.commands import CreateStudyCommand, RunAnalysisCommand
        from automl.application.bootstrap import build_application

        rng = np.random.default_rng(42)
        n = 120
        price = np.linspace(10.0, 500.0, n)
        views = price * 2.5 + rng.normal(0, 5.0, n)
        df = pd.DataFrame(
            {
                "item_id": [f"id_{i:04d}" for i in range(n)],
                "price_usd": price,
                "monthly_views": views,
                "is_bestseller": (views > np.median(views)).astype(int),
            }
        )
        ws_dir = Path(__file__).resolve().parent / "results" / "ws_09"
        ws_dir.mkdir(parents=True, exist_ok=True)
        csv_path = ws_dir / "catalog.csv"
        df.to_csv(csv_path, index=False)

        ws, command_bus, _query_bus = build_application(root_dir=str(ws_dir))
        ds = ws.register_dataset(
            name="catalog",
            path=str(csv_path),
            target="is_bestseller",
            task_type="binary_classification",
        )
        study_id = command_bus.dispatch(
            CreateStudyCommand(
                dataset_id=ds.id,
                name="fire_explore_09",
                target_column="is_bestseller",
                analysis_types=("descriptive", "association"),
            )
        )
        run_id = command_bus.dispatch(RunAnalysisCommand(study_id=study_id))
        assert study_id
        assert run_id
        return {"study_id": study_id, "run_id": run_id, "dataset_id": ds.id}

    r = run_fire_test("09_explore_smoke", "Explore study smoke", body)
    print(r.status, r.detail or r.metrics)


if __name__ == "__main__":
    main()
