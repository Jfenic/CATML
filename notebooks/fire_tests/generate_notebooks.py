#!/usr/bin/env python3
"""Generate companion .ipynb notebooks that execute each fire-test script."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    ("01_titanic_like", "Titanic-like binary classification"),
    ("02_house_prices_like", "House prices-like regression"),
    ("03_credit_risk", "Credit risk (OpenML credit-g)"),
    ("04_multiclass", "Multiclass wine"),
    ("05_high_cardinality", "High-cardinality categoricals"),
    ("06_ordered_time_split", "Ordered / time split"),
    ("07_hpo_short", "Short HPO time_budget"),
    ("08_export_predict", "Export → predict"),
    ("09_explore_smoke", "Explore study smoke"),
    ("10_ensemble_compare", "Ensemble / multi-model compare"),
]


def notebook_for(stem: str, title: str) -> dict:
    md = f"# Fire test — {title}\n\nRuns `{stem}.py` and prints the JSON result under `results/`."
    code = f"""from pathlib import Path
import json
import runpy

runpy.run_path(str(Path("{stem}.py").resolve()), run_name="__main__")
result = json.loads(Path("results/{stem}.json").read_text(encoding="utf-8"))
print(result["status"])
result
"""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
    }


def main() -> None:
    for stem, title in SCRIPTS:
        path = HERE / f"{stem}.ipynb"
        path.write_text(json.dumps(notebook_for(stem, title), indent=1), encoding="utf-8")
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
