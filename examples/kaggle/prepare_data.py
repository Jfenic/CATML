"""Prepare train.csv and test.csv from customers_churn.csv for Kaggle demo."""
from pathlib import Path
import pandas as pd


def prepare_kaggle_data() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[2]
    source_csv = root / "examples" / "data" / "customers_churn.csv"
    dest_dir = root / "examples" / "data" / "kaggle_churn"
    dest_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_csv)

    # Split: first 400 rows as train, remaining 100 rows as test (without target)
    train_df = df.iloc[:400].copy()
    test_df = df.iloc[400:].copy()
    test_df = test_df.drop(columns=["churn"])

    train_path = dest_dir / "train.csv"
    test_path = dest_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Created {train_path} ({len(train_df)} rows, columns: {list(train_df.columns)})")
    print(f"Created {test_path} ({len(test_df)} rows, columns: {list(test_df.columns)})")
    return train_path, test_path


if __name__ == "__main__":
    prepare_kaggle_data()
