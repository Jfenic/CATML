from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from automl.application.services.workspace import AutoMLWorkspace


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_dataset() -> Path:
    return _project_root() / "examples" / "data" / "customers_churn.csv"


def run_demo(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    workspace_dir = Path(args.workspace) if args.workspace else _project_root() / ".automl" / "demo"

    print(f"Workspace: {workspace_dir}")
    print(f"Dataset:   {dataset_path}")

    workspace = AutoMLWorkspace.create("demo", root_dir=workspace_dir)
    dataset = workspace.register_dataset(
        name="customers_churn",
        path=dataset_path,
        target="churn",
        task_type="binary_classification",
    )
    profile = workspace.repository.get_dataset_profile(dataset.id)
    print(f"Profile:   {profile.row_count} rows, {profile.column_count} columns")

    run = workspace.create_run(dataset, metric="roc_auc")
    workspace.exclude_feature(dataset.id, "customer_id")
    workspace.prioritize_feature(dataset.id, "salary")
    workspace.prioritize_feature(dataset.id, "debt")
    workspace.exclude_model(run, "svc")

    registry = workspace.get_feature_registry(dataset.id)
    features = registry.active_feature_names(dataset.target_column)

    experiment = workspace.create_experiment(
        run,
        name="financial_baseline",
        feature_names=features,
        model_ids=["logistic_regression", "random_forest"],
        hypothesis="Salary and debt features predict churn",
        priority="high",
    )

    print(f"Run:         {run.id}")
    print(f"Experiment:  {experiment.id}")
    print(f"Models:      {', '.join(experiment.model_ids)}")
    print(f"Features:    {', '.join(experiment.feature_names)}")
    print("\nRunning trials...\n")

    results = workspace.run_experiment(run, experiment)
    for result in results:
        status = "OK" if result.succeeded else f"FAIL ({result.failure_reason})"
        print(
            f"  {result.model_id:22s} {result.primary_metric}={result.primary_score:.4f}  "
            f"[{status}]  {result.training_time_seconds:.2f}s"
        )

    print("\nLeaderboard:")
    for row in workspace.leaderboard(run):
        print(
            f"  {row['model_id']:22s} {row['metric']}={row['score']:.4f}  "
            f"(trial {row['trial_id']})"
        )

    if args.json:
        print("\nJSON output:")
        print(json.dumps(workspace.leaderboard(run), indent=2))

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="automl", description="AutoML Platform CLI (V0.1)")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("run-demo", help="Run a demo experiment on the sample dataset")
    demo.add_argument("--dataset", help="Path to CSV dataset")
    demo.add_argument("--workspace", help="Workspace directory (SQLite + artifacts)")
    demo.add_argument("--json", action="store_true", help="Print leaderboard as JSON")
    demo.set_defaults(func=run_demo)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
