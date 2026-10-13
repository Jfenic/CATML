# ADR-012: Data Preparation subsystem (dataset vs training-pipeline scopes)

> **Estado:** Aceptada (dirección de producto; implementación post-`0.9.0rc1`)  
> **Fecha:** 2026-10-13  
> **Alcance:** Preparación de datos reproducible y anti-leakage  
> **Relacionado:** [ADR-001](001-hexagonal-architecture-and-cqrs.md), [ADR-002](002-hypothesis-driven-experiments.md), [ADR-009](009-experiment-recipes-and-interface-parity.md), [spec](../features/data-preparation/spec.md), [plan](../features/data-preparation/plan.md), Explore, Recipes, derived features.

---

## Contexto

CATML ya diagnostica calidad (Explore / profiler / anti-leakage) y preprocesa dentro de pipelines de entrenamiento. Falta un puente **accionable, auditable y con paridad** entre “hay un problema” y “aplicar una transformación reproducible” sin:

- mezclar limpieza de dataset con estadísticos aprendidos en todo el frame (leakage),
- sobrescribir CSV originales,
- duplicar lógica en Workbench vs CLI vs agentes,
- obligar un wizard largo antes de entrenar.

## Decisión

1. Introducir un **subsistema de aplicación** `PreparationService` (CQRS), no un conjunto de botones ad hoc en el frontend.  
2. Separar estrictamente:
   - `scope=dataset` → produce **nueva versión inmutable** del dataset;  
   - `scope=training_pipeline` → pasos que **aprenden parámetros** solo en folds de train.  
3. Contrato central: **`PreparationPlan`** (pasos tipados). Ciclo: analyze → suggest → preview → validate → apply (Propose ≠ Accept).  
4. **Paridad:** Python / CLI / MCP / Workbench consumen los mismos commands/queries.  
5. Agentes: proponer libremente; **apply** destructivo o semántico con aprobación; **sin** código Python arbitrario de limpieza.  
6. Implementación **después** de estabilizar la RC tabular; el flujo `cargar → entrenar` permanece válido con preprocess por defecto en el trainer.

## Alternativas rechazadas

| Alternativa | Motivo |
| --- | --- |
| Solo UI de limpieza en Workbench | Duplica lógica; agentes/CLI quedan fuera |
| Unificar toda limpieza en el trainer | No cubre tipado/dedupe/versiones; UX confusa |
| ETL generalista (grafo tipo KNIME) | Fuera de foco ML; coste enorme |
| Mutar el dataset in-place | Rompe auditabilidad y replay de recipes |

## Consecuencias

- **Más fácil:** trazabilidad datos→plan→experimento→predict; agentes seguros; menos leakage accidental.  
- **Más difícil:** versionado de datasets, validación de scopes, replay en export.  
- **Requerido:** tests de anti-leakage de scope; badge UX honesto (leakage ≠ “todo validado”).  
- **No requerido para Labs ADR-011:** el servicio es núcleo; la UI puede vivir primero en tabular.
