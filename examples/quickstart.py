"""
CATML 2-Minute Quickstart

Demonstrates automated machine learning with the ergonomic Python API:
1. Training multiple models and auto-selecting the best.
2. Viewing the experiment leaderboard.
3. Exporting a standalone, portable model artifact ('model.pkl').
4. Loading the model for standalone production inference anywhere.
"""

from pathlib import Path
import pandas as pd
from catml import AutoML, ModelArtifact


def main() -> None:
    print("=" * 65)
    print("  CATML Quickstart (Python API)")
    print("=" * 65)

    # 1. Load sample dataset
    data_path = Path(__file__).parent / "data" / "customers_churn.csv"
    if not data_path.exists():
        import numpy as np

        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "age": np.random.randint(18, 70, size=n),
            "balance": np.random.uniform(1000, 100000, size=n),
            "tenure_months": np.random.randint(1, 72, size=n),
            "churn": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        })
    else:
        df = pd.read_csv(data_path)

    print(f"Loaded dataset: {len(df)} rows, target='churn'")

    # 2. Fit AutoML models
    print("\nRunning AutoML...")
    automl = AutoML(task="classification")
    result = automl.fit(df, target="churn")

    # 3. Display Leaderboard
    print("\nModel Leaderboard:")
    print("-" * 55)
    lb = result.leaderboard()
    cols = [c for c in ["rank", "model_id", "metric", "score", "training_time_s"] if c in lb.columns]
    print(lb[cols].to_string(index=False))
    print("-" * 55)
    print(f"Winner: {result.best_model_id} ({result.metric}: {result.best_score:.4f})")

    # 4. Save Standalone Artifact
    model_path = Path("catml-runs") / "model.pkl"
    result.save_model(model_path)
    print(f"\nSaved standalone artifact to: {model_path.resolve()}")

    # 5. Load and Predict Anywhere (Zero workspace or SQLite dependency)
    print("\nLoading model artifact for independent production inference...")
    loaded_model = ModelArtifact.load(model_path)
    sample_data = df.drop(columns=["churn"]).head(3)
    predictions = loaded_model.predict(sample_data)
    probabilities = loaded_model.predict_proba(sample_data)

    print("\nSample Predictions:")
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        print(f"  Row {i}: Predicted={pred}, Probabilities={prob}")

    print("\nQuickstart finished successfully!")
    print("=" * 65)


if __name__ == "__main__":
    main()
