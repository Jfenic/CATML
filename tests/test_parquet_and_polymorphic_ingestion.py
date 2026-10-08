import io
from pathlib import Path
import unittest.mock as mock
import pytest
import pandas as pd
import numpy as np

from automl import AutoML
from automl.application.bootstrap import build_application
from automl.engine.profiling.dataset_profiler import load_dataframe


def test_load_dataframe_formats(tmp_path):
    df_sample = pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["x", "y", "z"]})

    # 1. CSV
    csv_file = tmp_path / "data.csv"
    df_sample.to_csv(csv_file, index=False)
    loaded_csv = load_dataframe(csv_file)
    pd.testing.assert_frame_equal(df_sample, loaded_csv)

    # 2. TSV
    tsv_file = tmp_path / "data.tsv"
    df_sample.to_csv(tsv_file, sep="\t", index=False)
    loaded_tsv = load_dataframe(tsv_file)
    pd.testing.assert_frame_equal(df_sample, loaded_tsv)

    # 3. Parquet
    parquet_file = tmp_path / "data.parquet"
    df_sample.to_parquet(parquet_file)
    loaded_pq = load_dataframe(parquet_file)
    pd.testing.assert_frame_equal(df_sample, loaded_pq)

    # 4. JSON
    json_file = tmp_path / "data.json"
    df_sample.to_json(json_file)
    loaded_json = load_dataframe(json_file)
    pd.testing.assert_frame_equal(df_sample, loaded_json)

    # 5. Non-existent file
    with pytest.raises(FileNotFoundError):
        load_dataframe(tmp_path / "non_existent.parquet")


def test_load_dataframe_parquet_missing_engine(tmp_path):
    parquet_file = tmp_path / "sample.parquet"
    parquet_file.write_bytes(b"PAR1dummycontent")

    with mock.patch("pandas.read_parquet", side_effect=ImportError("No module named 'pyarrow'")):
        with pytest.raises(ImportError, match="Reading parquet file 'sample.parquet' requires 'pyarrow'"):
            load_dataframe(parquet_file)


def test_workspace_register_dataset_polymorphism(tmp_path):
    ws, cmd, qry = build_application(root_dir=str(tmp_path))

    df = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0], "x2": [10, 20, 30, 40], "y": [0, 1, 0, 1]})

    # 1. Register DataFrame directly
    ds_df = ws.register_dataset(name="from_df", path=df, target="y")
    assert ds_df.id.startswith("ds_")
    assert Path(ds_df.path).exists()
    assert ws.get_dataset(ds_df.id) is not None

    # 2. Register from Parquet file
    pq_path = tmp_path / "features.parquet"
    df.to_parquet(pq_path)
    ds_pq = ws.register_dataset(name="from_pq", path=pq_path, target="y")
    assert ds_pq.id.startswith("ds_")
    assert ds_pq.path == str(pq_path.resolve())


def test_automl_fit_with_parquet_and_csv_paths(tmp_path):
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num1": np.random.randn(n),
        "num2": np.random.randn(n),
        "cat1": np.random.choice(["A", "B", "C"], size=n),
        "label": np.random.choice([0, 1], size=n),
    })

    pq_file = tmp_path / "train.parquet"
    df.to_parquet(pq_file)

    # Fit using file path directly
    automl = AutoML(time_budget=15, cv_folds=2, random_state=42)
    result = automl.fit(data=pq_file, target="label")

    assert result is not None
    lb = result.leaderboard()
    assert len(lb) > 0
    assert "rank" in lb.columns
    assert "score" in lb.columns

    # Verify predictions
    preds = automl.predict(df[["num1", "num2", "cat1"]])
    assert len(preds) == n
