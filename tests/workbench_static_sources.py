"""Helpers to read Workbench static sources after modular splits."""
from __future__ import annotations

from pathlib import Path

VIEWS_DIR = Path("src/automl/interfaces/web/static/js/views")


def datasets_view_sources() -> str:
    """Concatenate DatasetsView entry + modules (Clean Architecture PR3)."""
    parts = [VIEWS_DIR / "datasets.js"]
    ds_dir = VIEWS_DIR / "datasets"
    if ds_dir.is_dir():
        parts.extend(sorted(ds_dir.glob("*.js")))
    return "\n".join(p.read_text(encoding="utf-8") for p in parts if p.exists())
