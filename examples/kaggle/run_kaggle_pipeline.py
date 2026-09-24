"""End-to-End Kaggle AutoML Pipeline using CATML.

Demonstrates:
1. Dataset registration and automatic problem profiling
2. Advanced feature selection (Mutual Information, Tree Importance)
3. Experiment planning and priority scheduling
4. Hyperparameter optimization with Optuna (LightGBM/XGBoost)
5. Generation of submission.csv ready for Kaggle evaluation
"""
from pathlib import Path
import pandas as pd

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    GenerateSubmissionCommand,
    OptimizeExperimentCommand,
    PlanExperimentsCommand,
    RunScheduledExperimentsCommand,
    SelectFeaturesCommand,
)
from automl.application.queries.workspace_queries import (
    GetLeaderboardQuery,
    GetTaskPlanQuery,
)


def run_kaggle_pipeline() -> Path:
    root = Path(__file__).resolve().parents[2]
    train_csv = root / "examples" / "data" / "kaggle_churn" / "train.csv"
    test_csv = root / "examples" / "data" / "kaggle_churn" / "test.csv"
    submission_csv = root / "examples" / "data" / "kaggle_churn" / "submission.csv"
    ws_dir = root / ".automl" / "kaggle_demo_ws"

    print("================================================================")
    print("  CATML — Kaggle End-to-End Automated Pipeline")
    print("================================================================")

    # 1. Initialize CQRS Application
    ws, cmd, qry = build_application(root_dir=str(ws_dir))

    # 2. Register Dataset
    print(f"\n[1/5] Registering Kaggle training dataset: {train_csv.name}")
    dataset = ws.register_dataset(
        name="kaggle_churn",
        path=train_csv,
        target="churn",
    )
    plan = qry.dispatch(GetTaskPlanQuery(dataset.id))
    print(f"      Inferred Task: {plan.task_type.value} (default metric: {plan.default_metric})")
    print(f"      Recommended:   {len(plan.recommended_models)} models recommended ({', '.join(plan.recommended_models[:3])})")

    # 3. Create Run & Run Feature Selection
    print("\n[2/5] Running Multi-Method Feature Discovery...")
    run = ws.create_run(dataset, metric="roc_auc")
    feature_candidates = cmd.dispatch(SelectFeaturesCommand(run_id=run.id))
    for fc in feature_candidates[:3]:
        print(f"      Candidate: {fc.name:22s} ({fc.k} vars): {', '.join(fc.feature_names)}")

    # 4. Plan & Execute Scheduled Experiments
    print("\n[3/5] Planning & Prioritizing Experiments...")
    candidates = cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
    for c in candidates[:3]:
        prio_score = c.priority.effective_score if c.priority else 0.0
        print(f"      {c.name:28s} models={','.join(c.model_ids):20s} prio={prio_score:.2f}")

    print("\n      Running Scheduled Experiments...")
    cmd.dispatch(RunScheduledExperimentsCommand(run_id=run.id, max_experiments=3))
    lb = qry.dispatch(GetLeaderboardQuery(run.id))
    print("\n      Leaderboard after initial exploration:")
    print("      --------------------------------------------------")
    for r in lb:
        print(f"      {r['model_id']:18s} {r['metric']}={r['score']:.4f} time={r['training_time_s']:.2f}s")

    # 5. Hyperparameter Optimization with Optuna on Best Model
    best_entry = lb[0] if lb else {"model_id": "logistic_regression", "experiment_id": "baseline_fast"}
    best_model_id = best_entry["model_id"]
    best_exp_id = best_entry["experiment_id"]
    print(f"\n[4/5] Hyperparameter Optimization with Optuna on '{best_model_id}' (10 trials)...")
    opt_result = cmd.dispatch(
        OptimizeExperimentCommand(
            run_id=run.id,
            experiment_id=best_exp_id,
            model_id=best_model_id,
            optimizer="optuna",
            n_trials=10,
            patience=5,
        )
    )
    print(f"      Optuna Best Score: {opt_result['best_score']:.4f}")
    print(f"      Best Hyperparameters: {opt_result['best_params']}")

    # 6. Generate Kaggle Submission CSV
    print(f"\n[5/5] Generating Kaggle Submission on test set: {test_csv.name}")
    sub_result = cmd.dispatch(
        GenerateSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_csv),
            output_path=str(submission_csv),
            id_column="customer_id",
            predict_proba=True,  # Output probabilities for ROC-AUC
        )
    )

    print("\n================================================================")
    print("  Submission Generated Successfully!")
    print("================================================================")
    print(f"  File:       {sub_result['output_path']}")
    print(f"  Rows:       {sub_result['row_count']}")
    print(f"  ID Column:  {sub_result['id_column']}")
    print(f"  Target:     {sub_result['target_column']}")

    sub_df = pd.read_csv(submission_csv)
    print("\n  Preview of submission.csv (first 5 rows):")
    print("  -----------------------------------------")
    for _, row in sub_df.head(5).iterrows():
        print(f"    {row['customer_id']}  ->  {row['churn']:.5f}")

    # Evaluate against original ground truth if available (Simulating Kaggle Private Leaderboard)
    orig_csv = root / "examples" / "data" / "customers_churn.csv"
    if orig_csv.exists():
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        orig_df = pd.read_csv(orig_csv)
        y_true = orig_df.iloc[400:]["churn"].values
        y_probs = sub_df["churn"].values
        y_pred = (y_probs >= 0.5).astype(int)

        acc = accuracy_score(y_true, y_pred)
        roc = roc_auc_score(y_true, y_probs)
        f1 = f1_score(y_true, y_pred)

        print("\n================================================================")
        print("  Kaggle Private Leaderboard Simulation (Out-of-Sample Test)")
        print("================================================================")
        print(f"  Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"  Test ROC-AUC:  {roc:.4f}")
        print(f"  Test F1-Score: {f1:.4f}")
        print("================================================================\n")

    return submission_csv


if __name__ == "__main__":
    run_kaggle_pipeline()
