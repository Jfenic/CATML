"""Workbench UX: guided correlation expand, questionnaire-on-load, temporal CV gating."""
from __future__ import annotations

from pathlib import Path

from workbench_static_sources import datasets_view_sources

EXPLORE = Path("src/automl/interfaces/web/static/js/views/explore.js")
NEW_EXP = Path("src/automl/interfaces/web/static/js/views/new_experiment.js")


def test_explore_guided_correlation_has_expand_for_matrices() -> None:
    src = EXPLORE.read_text(encoding="utf-8")
    assert "guidedCorrExpand" in src
    assert "Ampliar matrices de correlación" in src
    assert "Pearson · Spearman · Cramér's V" in src
    assert "_renderCorrelationMatrixHtml(corrViz)" in src
    assert "categorical_association_matrix" in src
    # Still keep guided top-pairs path
    assert "|r| ≥ 0.70" in src or "|r| &ge; 0.70" in src or "0.70" in src


def test_datasets_auto_opens_questionnaire_when_unconfirmed() -> None:
    src = datasets_view_sources()
    assert "generateDatasetQuestionnaire" in src
    assert 'this.activeMainTab = "cuestionario"' in src
    assert "_questionnaireGuidedSeen" in src
    assert "Tipo de datos (elige uno)" in src
    assert "Time-Series solo aparece si eliges serie temporal" in src


def test_new_experiment_hides_time_series_unless_framing_says_so() -> None:
    src = NEW_EXP.read_text(encoding="utf-8")
    assert "isTimeSeries = this.questionnaire?.temporal_structure === \"sequential_time_series\"" in src
    assert "isGrouped = this.questionnaire?.temporal_structure === \"grouped_cohorts\"" in src
    # Disabled greyed Time-Series card removed; only shown when isTimeSeries
    assert "Sin orden temporal" not in src
    assert "${isTimeSeries ? `" in src or "isTimeSeries ? `" in src
    assert "group_kfold" in src
    assert "01 Datos → Encuadre" in src
