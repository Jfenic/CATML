from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl import AutoML, AutoMLResult, ModelArtifact
import catml


def test_automl_fit_dataframe_classification(tmp_path: Path):
    # Create simple binary classification dataset
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "age": np.random.randint(18, 70, size=n),
        "income": np.random.uniform(20000, 100000, size=n),
        "credit_score": np.random.randint(300, 850, size=n),
        "churn": np.random.choice([0, 1], size=n, p=[0.6, 0.4]),
    })

    automl = AutoML(
        models=["logistic_regression", "random_forest"],
        workspace_dir=tmp_path / "ws_clf",
    )

    result = automl.fit(df, target="churn")

    assert isinstance(result, AutoMLResult)
    assert result.task_type == "binary_classification"
    assert result.best_model is not None
    assert result.best_model_id in {"logistic_regression", "random_forest"}
    assert isinstance(result.best_score, float)
    assert result.best_score > 0.0

    # Leaderboard checks
    lb = result.leaderboard()
    assert isinstance(lb, pd.DataFrame)
    assert len(lb) >= 2
    assert "model_id" in lb.columns
    assert "score" in lb.columns
    assert "rank" in lb.columns

    # Summary
    summary = result.summary()
    assert summary["best_model_id"] == result.best_model_id
    assert summary["total_models_evaluated"] >= 2

    # Inference via result
    test_df = df.drop(columns=["churn"]).head(5)
    preds = result.predict(test_df)
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})

    probs = result.predict_proba(test_df)
    assert probs.shape == (5, 2)

    # Inference via automl instance
    inst_preds = automl.predict(test_df)
    assert np.array_equal(preds, inst_preds)

    # Save and Load model
    saved_path = tmp_path / "model.pkl"
    result.save_model(saved_path)
    assert saved_path.exists()

    loaded = AutoML.load_model(saved_path)
    assert isinstance(loaded, ModelArtifact)
    loaded_preds = loaded.predict(test_df)
    assert np.array_equal(preds, loaded_preds)


def test_automl_fit_numpy_arrays(tmp_path: Path):
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    automl = AutoML(
        models=["logistic_regression"],
        workspace_dir=tmp_path / "ws_numpy",
    )
    result = automl.fit(X, y)

    assert result.best_model_id == "logistic_regression"
    preds = result.predict(X[:5])
    assert len(preds) == 5


def test_automl_fit_regression(tmp_path: Path):
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "price": np.random.uniform(100.0, 500.0, size=n),
    })

    automl = AutoML(
        models=["ridge", "random_forest"],
        workspace_dir=tmp_path / "ws_reg",
    )
    result = automl.fit(df, target="price")

    assert result.task_type == "regression"
    assert result.best_model_id in {"ridge", "random_forest"}
    preds = result.predict(df.drop(columns=["price"]).head(5))
    assert len(preds) == 5
    assert all(isinstance(p, (float, np.floating)) for p in preds)


def test_catml_alias_and_default_models(tmp_path: Path):
    # Verify catml alias imports and default model selection (without passing models)
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0] * 6,
        "x2": [10.0, 20.0, 30.0, 40.0, 50.0] * 6,
        "target": [0, 1, 0, 1, 0] * 6,
    })

    model = catml.AutoML(workspace_dir=tmp_path / "ws_catml")
    result = model.fit(df, target="target")

    assert isinstance(result, catml.AutoMLResult)
    assert len(result.leaderboard()) >= 1


def test_automl_errors_before_fit():
    automl = AutoML()
    with pytest.raises(RuntimeError, match="not fitted yet"):
        automl.predict(np.array([[1.0, 2.0]]))

    with pytest.raises(RuntimeError, match="not fitted yet"):
        automl.predict_proba(np.array([[1.0, 2.0]]))

    with pytest.raises(RuntimeError, match="not fitted yet"):
        automl.leaderboard()


def test_automl_invalid_input_errors():
    automl = AutoML()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    with pytest.raises(ValueError, match="not found in DataFrame"):
        automl.fit(df, target="non_existent")

    with pytest.raises(ValueError, match="Target must be specified"):
        automl.fit(df, target=None)

    with pytest.raises(TypeError, match="Unsupported input data type"):
        automl.fit("invalid_type", target="target")


def test_automl_cv_folds_stratified_kfold(tmp_path: Path):
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 5,
        "y": [0, 1, 0, 1, 0, 1] * 5,
    })
    automl = AutoML(
        cv_folds=3,
        models=["logistic_regression"],
        workspace_dir=tmp_path / "ws_cv3",
    )
    result = automl.fit(df, target="y")
    assert result.best_model is not None
    lb = result.leaderboard()
    assert len(lb) == 1
    assert lb.iloc[0]["cv_std"] is not None
