"""Kaggle Playground Series S4E1: Bank Churn Solver using CATML.

Runs:
1. Dataset registration and automatic problem profiling (165,034 rows)
2. Feature filtering (excludes identifier columns id, CustomerId, Surname)
3. Experiment planning and execution with Gradient Boosting models
4. Hyperparameter optimization using Optuna
5. Generation of submission_real.csv (exactly 110,023 rows)
6. Submission directly to Kaggle via CLI
"""
from pathlib import Path
import subprocess
import sys
import pandas as pd

from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    ExcludeFeatureCommand,
    GenerateSubmissionCommand,
    OptimizeExperimentCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    GetLeaderboardQuery,
    GetTaskPlanQuery,
)


def run_kaggle_solver(submit: bool = True) -> Path:
    root = Path(__file__).resolve().parents[1]
    comp_dir = root / "competitions" / "playground-series-s4e1"
    train_csv = comp_dir / "train.csv"
    test_csv = comp_dir / "test.csv"
    submission_csv = comp_dir / "submission_real.csv"
    ws_dir = root / ".automl" / "kaggle_s4e1_ws"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"Missing train.csv or test.csv in {comp_dir}")

    print("================================================================")
    print("  CATML — Kaggle Playground S4E1: Bank Churn Solver")
    print("================================================================")

    # 1. Initialize CATML Workspace
    print(f"\n[1/6] Inicializando espacio de trabajo AutoML en: {ws_dir.name}")
    ws, cmd, qry = build_application(root_dir=str(ws_dir))

    # 2. Register Dataset
    print(f"\n[2/6] Registrando dataset de entrenamiento (165,034 filas)...")
    dataset = ws.register_dataset(
        name="bank_churn_train",
        path=train_csv,
        target="Exited",
    )
    plan = qry.dispatch(GetTaskPlanQuery(dataset.id))
    print(f"      Tarea inferida:   {plan.task_type.value}")
    print(f"      Métrica recomendada: {plan.default_metric}")

    # 3. Create Run & Exclude Identifier / Text Columns
    print(f"\n[3/6] Configurando variables del modelo...")
    run = ws.create_run(dataset, metric="roc_auc")

    # Exclude ID and non-predictive identifiers to avoid data leakage & high cardinality overhead
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "id", run_id=run.id))
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "CustomerId", run_id=run.id))
    cmd.dispatch(ExcludeFeatureCommand(dataset.id, "Surname", run_id=run.id))

    feature_registry = ws.get_feature_registry(dataset.id)
    active_features = feature_registry.active_feature_names("Exited")
    print(f"      Variables activas ({len(active_features)}): {', '.join(active_features)}")

    # 4. Create & Run Baseline Experiments with Gradient Boosting
    print(f"\n[4/6] Evaluando modelos con Gradient Boosting (LightGBM y XGBoost)...")
    exp = cmd.dispatch(
        CreateExperimentCommand(
            run_id=run.id,
            name="gradient_boosting_baseline",
            feature_names=active_features,
            model_ids=["lightgbm", "xgboost", "logistic_regression"],
            hypothesis="Evaluate LightGBM and XGBoost on tabular bank features",
        )
    )
    cmd.dispatch(RunExperimentCommand(run_id=run.id, experiment_id=exp.id))

    lb = qry.dispatch(GetLeaderboardQuery(run.id))
    print("\n      Leaderboard inicial:")
    print("      --------------------------------------------------")
    for r in lb:
        print(f"      {r['model_id']:20s} {r['metric']}={r['score']:.4f}  (tiempo={r['training_time_s']:.2f}s)")

    # 5. Hyperparameter Optimization with Optuna on the Top Model
    best_entry = lb[0] if lb else {"model_id": "lightgbm", "experiment_id": exp.id}
    best_model_id = best_entry["model_id"]
    print(f"\n[5/6] Optimizando hiperparámetros con Optuna para '{best_model_id}' (10 trials)...")
    opt_result = cmd.dispatch(
        OptimizeExperimentCommand(
            run_id=run.id,
            experiment_id=exp.id,
            model_id=best_model_id,
            optimizer="optuna",
            n_trials=10,
            patience=5,
        )
    )
    print(f"      Optuna Best Score (ROC-AUC): {opt_result['best_score']:.4f}")
    print(f"      Mejores hiperparámetros:     {opt_result['best_params']}")

    # 6. Generate Official Kaggle Submission (110,023 rows)
    print(f"\n[6/6] Generando archivo de sumisión oficial: {submission_csv.name}")
    sub_result = cmd.dispatch(
        GenerateSubmissionCommand(
            run_id=run.id,
            test_dataset_path=str(test_csv),
            output_path=str(submission_csv),
            id_column="id",
            predict_proba=True,  # Mandatory for ROC-AUC
        )
    )

    print("\n================================================================")
    print("  Archivo de sumisión generado exitosamente!")
    print("================================================================")
    print(f"  Ruta:       {sub_result['output_path']}")
    print(f"  Filas:      {sub_result['row_count']} (esperadas: 110023)")
    print(f"  ID:         {sub_result['id_column']}")
    print(f"  Target:     {sub_result['target_column']}")

    sub_df = pd.read_csv(submission_csv)
    assert len(sub_df) == 110023, f"Error: esperadas 110023 filas, obtenidas {len(sub_df)}"
    print("\n  Primeras 5 filas del archivo:")
    for _, row in sub_df.head(5).iterrows():
        print(f"    id={int(row['id'])}  ->  Exited={row['Exited']:.5f}")

    if submit:
        print("\n================================================================")
        print("  Subiendo automáticamente a Kaggle vía CLI...")
        print("================================================================")
        submit_cmd = [
            ".venv/bin/kaggle",
            "competitions",
            "submit",
            "-c",
            "playground-series-s4e1",
            "-f",
            str(submission_csv),
            "-m",
            f"CATML v0.6 AutoML - {best_model_id} + Optuna (CV ROC-AUC: {opt_result['best_score']:.4f})",
        ]
        try:
            res = subprocess.run(submit_cmd, capture_output=True, text=True, check=True)
            print(res.stdout)
            print("[✓] ¡Sumisión enviada a Kaggle con éxito!")
            print("    Consulta tu puntuación en: https://www.kaggle.com/competitions/playground-series-s4e1/submissions")
        except subprocess.CalledProcessError as exc:
            print(f"[!] Error al subir: {exc.stderr}", file=sys.stderr)

    return submission_csv


if __name__ == "__main__":
    run_kaggle_solver()
