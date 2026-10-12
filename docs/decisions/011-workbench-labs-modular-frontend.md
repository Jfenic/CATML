# ADR-011: Workbench Labs — monolito frontend modular por modalidad

> **Estado:** Aceptada (dirección futura; ejecución diferida)  
> **Fecha:** 2026-10-12  
> **Alcance:** Workbench UI y descubrimiento de capacidades  
> **Relacionado con:** [ADR-001](001-hexagonal-architecture-and-cqrs.md), [ADR-008](008-catml-explore-modular-monolith.md), [ADR-010](010-progressive-clean-architecture-refactor.md), [plan Workbench Labs](../features/workbench-labs/plan.md), [MASTER_PLAN](../MASTER_PLAN.md).

---

## Contexto

El Workbench actual concentra navegación y vistas en un único árbol JS (`app.js` + `views/*`). Eso funciona para AutoML tabular, pero mezclar clustering, visión, NLP y forecasting en las mismas pantallas aumenta el coste de mantenimiento sin beneficio proporcional para el usuario tabular (el producto que se publica primero).

Al mismo tiempo, el motor ya es hexagonal/CQRS con `PluginCapability` (`supported_tasks`, `supported_modalities`, native/fallback). No hace falta un segundo backend por modalidad.

## Decisión

1. **Modular monolith frontend:** un solo Workbench empaquetado; varios **Labs** cargados bajo demanda.
2. **Un solo motor:** CommandBus / QueryBus / jobs / artefactos compartidos; labs solo construyen payloads válidos.
3. **Tabular first:** no se inicia la extracción de Labs hasta estabilizar el flujo tabular publicable (ver `TASKS.md`).
4. **Capabilities desde el dominio:** el frontend no duplica catálogos de modelos; consulta disponibilidad real (extensión de `/api/plugins` o futuro `/api/capabilities`).
5. **Sin React obligatorio** para esta modularización; ES modules + `mount`/`destroy` bastan en la primera oleada.

## Alternativas rechazadas

| Alternativa | Motivo de rechazo |
| --- | --- |
| Varios frontends (repos/apps) | Duplica store, jobs, empaquetado wheel y auth |
| Una sola UI mega-configurable | Conditional soup; peores UX por modalidad |
| Migrar a React ahora | Coste alto; no desbloquea el product gate tabular |

## Consecuencias

- Documentación y backlog reflejan Labs como **futuro**; el trabajo activo es tabular.
- Clean Architecture PR3 (`datasets.js` split) es el precedente de “vista = módulos”.
- Temporal validation (TimeSeriesSplit) sigue en Tabular; Temporal Lab = forecasting especializado.
- Plugins Python **no** inyectan JS arbitrario en el navegador en esta etapa.
