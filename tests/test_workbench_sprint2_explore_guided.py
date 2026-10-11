"""Sprint 2 — Explore guided/advanced UX and honest comparator deltas."""
from __future__ import annotations

from pathlib import Path


EXPLORE = Path("src/automl/interfaces/web/static/js/views/explore.js")
UTILS = Path("src/automl/interfaces/web/static/js/utils.js")
COMPARE = Path("src/automl/interfaces/web/static/js/views/compare.js")


def test_explore_has_guided_advanced_mode_and_top_findings() -> None:
    src = EXPLORE.read_text(encoding="utf-8")
    assert "showAdvancedExplore" in src
    assert "btnExploreModeGuided" in src
    assert "btnExploreModeAdvanced" in src
    assert "GUIDED_FINDINGS_LIMIT" in src
    assert "_visibleFindings" in src
    assert "Hallazgos prioritarios" in src
    assert "_renderGuidedAssociationsHtml" in src
    # Conclusions before dense viz in guided path
    assert src.index("Hallazgos prioritarios") < src.index("Asociaciones destacadas")


def test_explore_severity_not_blanket_critical_for_collinearity() -> None:
    src = EXPLORE.read_text(encoding="utf-8")
    assert 'type === "high_collinearity" && effect >= 0.90' in src
    assert 'if (type === "high_collinearity") return "WARNING"' in src
    assert "formatPValue" in src
    assert "toExponential" not in src


def test_utils_format_pvalue_and_wider_neutral_delta() -> None:
    src = UTILS.read_text(encoding="utf-8")
    assert "function formatPValue" in src
    assert "const eps = 0.001" in src
    assert "< 0.0001" in src


def test_compare_chart_avoids_tiny_delta_zoom() -> None:
    src = COMPARE.read_text(encoding="utf-8")
    assert "looksBounded01" in src
    assert "Math.max(range * 0.5, 0.05)" in src
    assert "inputMatrixColumnFilter" not in src  # belongs to explore
