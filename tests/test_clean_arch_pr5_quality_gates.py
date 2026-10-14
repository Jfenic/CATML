"""Clean Architecture PR5: quality-gate contracts + export→predict E2E smoke."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_ruff_and_mypy_configured_in_pyproject():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.ruff]" in text
    assert "[tool.mypy]" in text
    assert "ruff>=" in text
    assert "mypy>=" in text


def test_eslint_and_prettier_tooling_present():
    assert (ROOT / "eslint.config.js").is_file()
    assert (ROOT / "package.json").is_file()
    assert (ROOT / ".prettierrc.json").is_file()
    pkg = (ROOT / "package.json").read_text(encoding="utf-8")
    assert "lint:js" in pkg
    assert "eslint" in pkg


def test_export_reload_predict_parity(tmp_path: Path):
    """Product gate: train → export → reload → predictions match in-memory."""
    from automl import AutoML
    from automl.artifacts.model_artifact import ModelArtifact

    rng = np.random.default_rng(7)
    n = 80
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.uniform(-1, 1, size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )
    X = df.drop(columns=["target"])

    automl = AutoML(
        task="classification",
        models=["logistic_regression", "random_forest"],
        cv_folds=2,
        time_budget=20,
    )
    result = automl.fit(df, target="target")
    assert result.best_model_id is not None

    live_preds = result.predict(X)
    art_path = tmp_path / "pr5_parity.catml"
    result.save_model(str(art_path))

    loaded = ModelArtifact.load(str(art_path), verify_checksum=True)
    reloaded_preds = loaded.predict(X)

    assert len(live_preds) == len(reloaded_preds) == n
    assert np.array_equal(live_preds, reloaded_preds)


def test_domain_quality_gate_modules_importable():
    """Smoke: domain modules touched by PR5 gates remain importable."""
    from automl.domain.features.registry import FeatureRegistry
    from automl.domain.models.registry import ModelRegistry, ModelSpec
    from automl.domain.optimization.search_space import ParameterSpec, SearchSpace

    reg = ModelRegistry()
    reg.register(ModelSpec(id="m1", name="M1", task_types=["binary_classification"]))
    assert len(reg.list()) == 1

    feats = FeatureRegistry()
    space = SearchSpace()
    space.add(ParameterSpec.int("n", 1, 3))
    assert len(space.list()) == 1
    assert feats.list() == []
