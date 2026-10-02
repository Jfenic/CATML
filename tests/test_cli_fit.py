from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

from automl.artifacts.model_artifact import ModelArtifact
from automl.interfaces.cli.main import main


def test_cli_fit_command_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    # Create test dataset
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] * 5,
        "feature_2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0] * 5,
        "target": [0, 1, 0, 1, 0, 1, 0, 1] * 5,
    })
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    out_model = tmp_path / "out_model.pkl"
    exit_code = main([
        "fit",
        str(csv_path),
        "--target",
        "target",
        "--models",
        "logistic_regression",
        "--output-model",
        str(out_model),
        "--workspace",
        str(tmp_path / "ws_cli"),
    ])

    assert exit_code == 0
    assert out_model.exists()

    # Verify loaded model
    artifact = ModelArtifact.load(out_model)
    assert artifact.model_id == "logistic_regression"

    captured = capsys.readouterr()
    assert "Leaderboard:" in captured.out
    assert "Best Model:" in captured.out


def test_cli_fit_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    df = pd.DataFrame({
        "x": [1, 2, 3, 4, 5, 6],
        "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })
    csv_path = tmp_path / "reg.csv"
    df.to_csv(csv_path, index=False)

    out_model = tmp_path / "reg_model.pkl"
    exit_code = main([
        "fit",
        str(csv_path),
        "--target",
        "y",
        "--models",
        "ridge",
        "--output-model",
        str(out_model),
        "--json",
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["task_type"] == "regression"
    assert parsed["best_model_id"] == "ridge"
    assert "leaderboard" in parsed


def test_cli_fit_error_missing_file(capsys: pytest.CaptureFixture[str]):
    exit_code = main(["fit", "non_existent.csv", "--target", "col"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_cli_fit_error_missing_target(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    exit_code = main(["fit", str(csv_path), "--target", "wrong_col"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "not found in dataset" in captured.err
