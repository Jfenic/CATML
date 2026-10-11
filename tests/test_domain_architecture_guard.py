"""Guard: domain/ must stay pure (no ML frameworks or persistence SDKs)."""
from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_MODULES = {
    "sklearn",
    "scikit-learn",
    "optuna",
    "sqlite3",
    "pandas",
    "numpy",
    "torch",
    "joblib",
    "scipy",
    "statsmodels",
    "lightgbm",
    "xgboost",
    "catboost",
}


def _top_level(module: str) -> str:
    return module.split(".", 1)[0]


def test_domain_has_no_forbidden_third_party_imports():
    domain_root = Path(__file__).resolve().parent.parent / "src" / "automl" / "domain"
    violations: list[str] = []

    for path in domain_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _top_level(alias.name) in FORBIDDEN_MODULES:
                        violations.append(f"{path.name}:{node.lineno} import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _top_level(node.module) in FORBIDDEN_MODULES:
                    violations.append(f"{path.name}:{node.lineno} from {node.module}")

    assert not violations, "Forbidden imports in domain/:\n" + "\n".join(violations)
