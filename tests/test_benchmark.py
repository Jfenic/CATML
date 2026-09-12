from pathlib import Path

import pytest

from automl.benchmarks.runner import BenchmarkRunner


@pytest.fixture
def sample_dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "data" / "customers_churn.csv"


def test_benchmark_scenarios_improve_or_match_baseline(tmp_path: Path, sample_dataset_path: Path) -> None:
    runner = BenchmarkRunner(
        dataset_path=str(sample_dataset_path),
        workspace_root=str(tmp_path / "bench"),
    )
    results = runner.run_all()
    assert len(results) == len(runner.scenarios())
    assert len(results) >= 6

    by_id = {r["scenario_id"]: r for r in results}
    baseline = by_id["baseline_all_features"]["best_score"]
    full = by_id["full_v02_pipeline"]["best_score"]
    auto_v03 = by_id["automated_planning_v03"]["best_score"]

    assert baseline > 0.5
    assert full >= baseline - 0.05
    assert auto_v03 > 0.5
    assert by_id["full_v02_pipeline"]["delta_vs_baseline"] is not None
    assert by_id["automated_planning_v03"]["delta_vs_baseline"] is not None



def test_benchmark_single_scenario(tmp_path: Path, sample_dataset_path: Path) -> None:
    runner = BenchmarkRunner(
        dataset_path=str(sample_dataset_path),
        workspace_root=str(tmp_path / "bench_one"),
    )
    results = runner.run_all(scenario_filter="exclude_leaky_id")
    assert len(results) == 1
    assert results[0]["scenario_id"] == "exclude_leaky_id"
