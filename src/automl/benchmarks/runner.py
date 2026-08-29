from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    CreateFeatureSetCommand,
    ExcludeFeatureCommand,
    ExcludeModelCommand,
    PrioritizeFeatureCommand,
    RunExperimentCommand,
)
from automl.application.services.workspace import PLATFORM_VERSION, AutoMLWorkspace
from automl.domain.experiments.trial import TrialResult


@dataclass
class BenchmarkScenario:
    id: str
    version: str
    description: str
    setup: Callable[[AutoMLWorkspace, object, object], tuple[str, str]]
    metric: str = "roc_auc"


def _best_result(results: list[TrialResult]) -> tuple[float, str | None]:
    ok = [r for r in results if r.succeeded]
    if not ok:
        return 0.0, None
    best = max(ok, key=lambda r: r.primary_score)
    return best.primary_score, best.model_id


@dataclass
class BenchmarkRunner:
    dataset_path: str
    target: str = "churn"
    workspace_root: str | None = None

    def scenarios(self) -> list[BenchmarkScenario]:
        return [
            BenchmarkScenario(
                id="baseline_all_features",
                version="0.1",
                description="All features, all models — V0.1 baseline",
                setup=self._setup_baseline,
            ),
            BenchmarkScenario(
                id="exclude_leaky_id",
                version="0.2",
                description="Exclude customer_id (non-predictive identifier)",
                setup=self._setup_exclude_id,
            ),
            BenchmarkScenario(
                id="financial_feature_set",
                version="0.2",
                description="Curated FeatureSet: salary, debt, account_balance, tenure, age",
                setup=self._setup_financial_feature_set,
            ),
            BenchmarkScenario(
                id="curated_models",
                version="0.2",
                description="Exclude slow SVC; keep logistic_regression + random_forest",
                setup=self._setup_curated_models,
            ),
            BenchmarkScenario(
                id="full_v02_pipeline",
                version="0.2",
                description="Exclude ID + financial FeatureSet + curated models + priorities",
                setup=self._setup_full_pipeline,
            ),
        ]

    def _register_base(self, ws: AutoMLWorkspace):
        dataset = ws.register_dataset(
            name="customers_churn",
            path=self.dataset_path,
            target=self.target,
            task_type="binary_classification",
        )
        run = ws.create_run(dataset, metric="roc_auc")
        return dataset, run

    def _setup_baseline(self, ws, _cmd, _qry):
        dataset, run = self._register_base(ws)
        exp = ws.create_experiment(
            run,
            name="baseline_all",
            model_ids=["logistic_regression", "random_forest", "svc"],
        )
        return run.id, exp.id

    def _setup_exclude_id(self, ws, cmd, _qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        exp = ws.create_experiment(
            run,
            name="exclude_id",
            model_ids=["logistic_regression", "random_forest", "svc"],
        )
        return run.id, exp.id

    def _setup_financial_feature_set(self, ws, cmd, _qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        fset = cmd.dispatch(
            CreateFeatureSetCommand(
                dataset_id=dataset.id,
                name="financial_core",
                feature_names=["salary", "debt", "account_balance", "tenure", "age"],
                lineage="benchmark_v02",
            )
        )
        exp = ws.create_experiment(
            run,
            name="financial_set",
            feature_set_id=fset.id,
            model_ids=["logistic_regression", "random_forest"],
        )
        return run.id, exp.id

    def _setup_curated_models(self, ws, cmd, _qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        cmd.dispatch(ExcludeModelCommand(run.id, "svc"))
        exp = ws.create_experiment(
            run,
            name="curated_models",
            model_ids=["logistic_regression", "random_forest"],
        )
        return run.id, exp.id

    def _setup_full_pipeline(self, ws, cmd, _qry):
        dataset, run = self._register_base(ws)
        cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
        cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "salary", score=1.0, run_id=run.id))
        cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "debt", score=0.9, run_id=run.id))
        cmd.dispatch(ExcludeModelCommand(run.id, "svc"))
        fset = cmd.dispatch(
            CreateFeatureSetCommand(
                dataset_id=dataset.id,
                name="financial_plus_country",
                feature_names=["salary", "debt", "account_balance", "tenure", "age", "country"],
                lineage="benchmark_v02_full",
            )
        )
        exp = ws.create_experiment(
            run,
            name="full_v02",
            feature_set_id=fset.id,
            model_ids=["logistic_regression", "random_forest"],
            priority="high",
        )
        return run.id, exp.id

    def run_all(self, scenario_filter: str | None = None) -> list[dict]:
        import time

        results: list[dict] = []

        for scenario in self.scenarios():
            if scenario_filter and scenario.id != scenario_filter:
                continue

            scenario_root = None
            if self.workspace_root:
                scenario_root = str(Path(self.workspace_root) / scenario.id)
            ws, cmd, qry = build_application(root_dir=scenario_root)
            started = time.perf_counter()
            run_id, experiment_id = scenario.setup(ws, cmd, qry)
            trial_results = cmd.dispatch(RunExperimentCommand(run_id, experiment_id))
            elapsed = time.perf_counter() - started
            best_score, best_model = _best_result(trial_results)

            details = {
                "description": scenario.description,
                "scenario_version": scenario.version,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "trials": [
                    {
                        "model_id": r.model_id,
                        "score": round(r.primary_score, 4),
                        "failed": not r.succeeded,
                    }
                    for r in trial_results
                ],
            }
            ws.repository.save_benchmark_result(
                platform_version=PLATFORM_VERSION,
                scenario_id=scenario.id,
                metric=scenario.metric,
                best_score=best_score,
                best_model=best_model,
                total_time_s=elapsed,
                details=details,
            )
            results.append(
                {
                    "scenario_id": scenario.id,
                    "version": scenario.version,
                    "description": scenario.description,
                    "metric": scenario.metric,
                    "best_score": round(best_score, 4),
                    "best_model": best_model,
                    "total_time_s": round(elapsed, 3),
                    "delta_vs_baseline": None,
                }
            )

        baseline = next((r for r in results if r["scenario_id"] == "baseline_all_features"), None)
        if baseline and baseline["best_score"]:
            for row in results:
                row["delta_vs_baseline"] = round(row["best_score"] - baseline["best_score"], 4)

        return results

    def compare_history(self) -> list[dict]:
        root = self.workspace_root or str(Path.cwd() / ".automl" / "benchmark")
        rows: list[dict] = []
        base = Path(root)
        if not base.exists():
            return rows
        for scenario_dir in sorted(base.iterdir()):
            if not scenario_dir.is_dir():
                continue
            db = scenario_dir / "automl.db"
            if not db.exists():
                continue
            ws, _, _ = build_application(root_dir=str(scenario_dir))
            rows.extend(ws.repository.list_benchmark_results(platform_version=PLATFORM_VERSION))
        rows.sort(key=lambda r: r["timestamp"], reverse=True)
        return rows
