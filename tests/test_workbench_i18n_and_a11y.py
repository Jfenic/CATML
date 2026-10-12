"""
Tests for Workbench Phase P2: Spanish Language Unification, WCAG AA Accessibility,
Contrast, and Responsive Layout.
"""
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "src" / "automl" / "interfaces" / "web" / "static"


def test_index_html_has_spanish_lang_and_a11y():
    index_path = STATIC_DIR / "index.html"
    assert index_path.exists(), "index.html does not exist"
    content = index_path.read_text(encoding="utf-8")

    # Document language
    assert '<html lang="es"' in content, "index.html must specify lang='es'"

    # Accessibility attributes
    assert 'aria-label="Navegación principal"' in content
    assert 'aria-label="Abrir panel del asistente copiloto"' in content
    assert 'aria-label="Sincronizar estado del laboratorio"' in content

    # Header branding and system status in Spanish
    assert "PROYECTO:" in content
    assert "Espacio de Trabajo Activo" in content
    assert "LISTO" in content
    assert "Motor Activo" in content
    assert "ESPACIO:" in content
    assert "Conectando..." in content
    assert "Sincronizar" in content
    assert "Agente" in content


def test_workbench_css_has_wcag_and_responsive_rules():
    css_path = STATIC_DIR / "css" / "workbench.css"
    assert css_path.exists(), "workbench.css does not exist"
    content = css_path.read_text(encoding="utf-8")

    # WCAG AA Focus visible outline
    assert ":focus-visible" in content
    assert "var(--catml-blue)" in content

    # Reduced motion
    assert "@media (prefers-reduced-motion: reduce)" in content

    # Responsive breakpoints
    assert "@media (max-width: 768px)" in content
    assert "@media (max-width: 1024px)" in content

    # Modal responsive containment
    assert ".modal-responsive" in content
    assert "max-height: calc(100vh - 2.5rem)" in content


def test_datasets_js_has_spanish_schema_and_calculator():
    from workbench_static_sources import datasets_view_sources

    datasets_path = STATIC_DIR / "js" / "views" / "datasets.js"
    assert datasets_path.exists()
    content = datasets_view_sources()

    # Schema table headers in Spanish
    assert "<th>Característica</th>" in content
    assert "<th>Tipo</th>" in content
    assert "<th>Nulos %</th>" in content
    assert "<th>Valores Únicos</th>" in content
    assert "<th>Correlación con Target (r)</th>" in content
    assert "<th>Acción del Motor</th>" in content
    assert "<th>Justificación</th>" in content
    assert '<th class="text-center">Visualización</th>' in content

    # Engine action badges in Spanish
    assert ">Excluir<" in content
    assert ">Codificar<" in content
    assert ">Conservar<" in content

    # Feature Calculator in Spanish
    assert "Nombre de Característica Derivada" in content
    assert "Fórmula Matemática" in content
    assert "Probar y Previsualizar" in content
    assert "Aplicar al Dataset y Guardar" in content
    assert "Características Disponibles" in content


def test_new_experiment_js_has_spanish_controls_and_modal_responsive():
    exp_path = STATIC_DIR / "js" / "views" / "new_experiment.js"
    assert exp_path.exists()
    content = exp_path.read_text(encoding="utf-8")

    # Modes in Spanish
    assert "Automático" in content
    assert "Guiado" in content
    assert "Manual" in content

    # Presets in Spanish
    assert "Rápido" in content
    assert "Equilibrado" in content
    assert "Exhaustivo" in content

    # Validation strategies in Spanish
    assert "K-Fold Estándar" in content
    assert "Stratified K-Fold" in content
    assert "Time-Series" in content
    assert "Holdout" in content

    # Preserved universal ML terms
    assert "LightGBM" in content
    assert "XGBoost" in content
    assert "CatBoost" in content
    assert "Random Forest" in content

    # Modal responsive class
    assert "modal-responsive" in content


def test_compare_js_has_spanish_ensemble_builder_and_modal_responsive():
    compare_path = STATIC_DIR / "js" / "views" / "compare.js"
    assert compare_path.exists()
    content = compare_path.read_text(encoding="utf-8")

    # Ensemble modal in Spanish
    assert "Constructor Visual de Ensambles" in content
    assert "Modelos Base a Incluir" in content
    assert "Estrategia de Ensamble" in content
    assert "Arquitectura del Meta-Aprendiz de Nivel 2" in content
    assert "Pliegues de Validación" in content

    # Modal responsive class
    assert "modal-responsive" in content


def test_kaggle_pipeline_knowledge_views_in_spanish():
    kaggle_content = (STATIC_DIR / "js" / "views" / "kaggle.js").read_text(encoding="utf-8")
    pipeline_content = (STATIC_DIR / "js" / "views" / "pipeline.js").read_text(encoding="utf-8")
    knowledge_content = (STATIC_DIR / "js" / "views" / "knowledge.js").read_text(encoding="utf-8")

    # Kaggle view
    assert "Benchmark Activo" in kaggle_content
    assert "Mejor CV Local" in kaggle_content
    assert "Validación Previa al Envío (Pre-Flight)" in kaggle_content
    assert "Generar y Validar submission.csv" in kaggle_content
    assert "Preparar Envío con Kaggle CLI" in kaggle_content
    assert "aria-label=" in kaggle_content

    # Pipeline view
    assert "Grafo Visual del Pipeline DAG" in pipeline_content
    assert "Inspector de Nodos" in pipeline_content
    assert "Contrato del Nodo" in pipeline_content
    assert "Pipeline Entrenado y Alineado" in pipeline_content
    assert "aria-label=" in pipeline_content

    # Knowledge view
    assert "Base de Conocimiento y Meta-Aprendizaje CATML" in knowledge_content
    assert "Huella Digital del Dataset Actual" in knowledge_content
    assert "Rendimiento Histórico de Algoritmos en Datasets Similares" in knowledge_content
    assert "aria-label=" in knowledge_content
