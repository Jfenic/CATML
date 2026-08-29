from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    ExcludeFeatureCommand,
    ExcludeModelCommand,
    PrioritizeFeatureCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    GetDatasetProfileQuery,
    GetLeaderboardQuery,
    GetTaskPlanQuery,
    ListTaskTypesQuery,
)
from automl.application.services.workspace import PLATFORM_VERSION
from automl.benchmarks.runner import BenchmarkRunner


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_dataset() -> Path:
    return _project_root() / "examples" / "data" / "customers_churn.csv"


def _print_table(rows: list[dict], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows:
        delta = row.get("delta_vs_baseline")
        delta_str = f"  Δ={delta:+.4f}" if delta is not None else ""
        print(
            f"  {row['scenario_id']:28s}  {row['metric']}={row['best_score']:.4f}  "
            f"model={row.get('best_model') or '-':22s}  "
            f"{row['total_time_s']:.2f}s{delta_str}"
        )


def run_demo(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    workspace_dir = Path(args.workspace) if args.workspace else _project_root() / ".automl" / "demo"

    ws, cmd, qry = build_application(root_dir=str(workspace_dir))
    dataset = ws.register_dataset(
        name="customers_churn",
        path=dataset_path,
        target="churn",
        task_type="binary_classification",
    )
    profile = qry.dispatch(GetDatasetProfileQuery(dataset.id))
    print(f"Workspace: {workspace_dir}")
    print(f"Dataset:   {dataset_path}")
    print(f"Profile:   {profile.row_count} rows, {profile.column_count} columns")

    run = ws.create_run(dataset, metric="roc_auc")
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))
    cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "salary", run_id=run.id))
    cmd.dispatch(PrioritizeFeatureCommand(dataset.id, "debt", run_id=run.id))
    cmd.dispatch(ExcludeModelCommand(run.id, "svc"))

    features = ws.get_feature_registry(dataset.id).active_feature_names(dataset.target_column)
    experiment = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="financial_baseline",
            feature_names=features,
            model_ids=["logistic_regression", "random_forest"],
            hypothesis="Salary and debt features predict churn",
            priority="high",
        )
    )

    print(f"Run:         {run.id}")
    print(f"Experiment:  {experiment.id}")
    print("\nRunning trials...\n")
    results = cmd.dispatch(RunExperimentCommand(run.id, experiment.id))
    for result in results:
        status = "OK" if result.succeeded else f"FAIL ({result.failure_reason})"
        print(
            f"  {result.model_id:22s} {result.primary_metric}={result.primary_score:.4f}  "
            f"[{status}]  {result.training_time_seconds:.2f}s"
        )

    leaderboard = qry.dispatch(GetLeaderboardQuery(run.id))
    print("\nLeaderboard:")
    for row in leaderboard:
        print(f"  {row['model_id']:22s} {row['metric']}={row['score']:.4f}")

    if args.json:
        print(json.dumps(leaderboard, indent=2))
    return 0


def run_benchmark(args: argparse.Namespace) -> int:
    dataset = Path(args.dataset) if args.dataset else _default_dataset()
    ws_dir = str(Path(args.workspace)) if args.workspace else str(_project_root() / ".automl" / "benchmark")

    runner = BenchmarkRunner(dataset_path=str(dataset), workspace_root=ws_dir)
    results = runner.run_all(scenario_filter=args.scenario)
    _print_table(results, f"CATML Benchmark (platform {PLATFORM_VERSION})")

    if args.json:
        print(json.dumps(results, indent=2))
    return 0


def show_benchmark_history(args: argparse.Namespace) -> int:
    ws_dir = str(Path(args.workspace)) if args.workspace else str(_project_root() / ".automl" / "benchmark")
    runner = BenchmarkRunner(dataset_path=str(_default_dataset()), workspace_root=ws_dir)
    rows = runner.compare_history()
    if not rows:
        print("No benchmark history yet. Run: automl benchmark run")
        return 0
    print(f"\nBenchmark history (platform {PLATFORM_VERSION})")
    for row in rows[: args.limit]:
        print(
            f"  {row['timestamp'][:19]}  {row['scenario_id']:28s}  "
            f"{row['metric']}={row['best_score']:.4f}  model={row.get('best_model') or '-'}"
        )
    return 0


def list_tasks(args: argparse.Namespace) -> int:
    _, _, qry = build_application(root_dir=args.workspace)
    tasks = qry.dispatch(ListTaskTypesQuery())
    print("\nTask types and compatible models\n")
    for task in tasks:
        print(f"  {task['task_type']}")
        print(f"    {task['label']} — {task['description']}")
        print(f"    metrics: {', '.join(task['metrics'])}  (default: {task['default_metric']})")
        print(f"    models:  {', '.join(task['models'])}\n")
    return 0


def plan_task(args: argparse.Namespace) -> int:
    ws_dir = str(Path(args.workspace)) if args.workspace else str(_project_root() / ".automl" / "demo")
    ws, _, qry = build_application(root_dir=ws_dir)

    if args.dataset_id:
        plan = qry.dispatch(GetTaskPlanQuery(args.dataset_id))
    else:
        dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
        dataset = ws.register_dataset(
            name="plan_preview",
            path=dataset_path,
            target=args.target,
            task_type=args.task_type,
        )
        plan = qry.dispatch(GetTaskPlanQuery(dataset.id))

    print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="automl", description=f"AutoML Platform CLI (V{PLATFORM_VERSION})")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("run-demo", help="Run demo experiment via CommandBus")
    demo.add_argument("--dataset")
    demo.add_argument("--workspace")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=run_demo)

    bench = sub.add_parser("benchmark", help="Benchmark harness")
    bench_sub = bench.add_subparsers(dest="bench_cmd", required=True)

    bench_run = bench_sub.add_parser("run", help="Run all benchmark scenarios")
    bench_run.add_argument("--dataset")
    bench_run.add_argument("--workspace")
    bench_run.add_argument("--scenario", help="Run a single scenario id")
    bench_run.add_argument("--json", action="store_true")
    bench_run.set_defaults(func=run_benchmark)

    bench_hist = bench_sub.add_parser("history", help="Show stored benchmark results")
    bench_hist.add_argument("--workspace")
    bench_hist.add_argument("--limit", type=int, default=20)
    bench_hist.set_defaults(func=show_benchmark_history)

    task = sub.add_parser("task", help="Task types and problem planning")
    task_sub = task.add_subparsers(dest="task_cmd", required=True)

    task_list = task_sub.add_parser("list", help="List task types and model catalogs")
    task_list.add_argument("--workspace")
    task_list.set_defaults(func=list_tasks)

    task_plan = task_sub.add_parser("plan", help="Plan task from dataset (infer classification/regression/clustering)")
    task_plan.add_argument("--dataset")
    task_plan.add_argument("--dataset-id", help="Use an already registered dataset")
    task_plan.add_argument("--target", default="churn")
    task_plan.add_argument("--task-type", help="Force: binary_classification, multiclass_classification, regression, clustering")
    task_plan.add_argument("--workspace")
    task_plan.set_defaults(func=plan_task)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
