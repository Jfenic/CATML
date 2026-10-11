# ADR-010: Refactor progresivo hacia Clean Architecture modular

- **Status:** Accepted (dirección); implementación por fases
- **Date:** 2026-10-11
- **Deciders:** CATML maintainers (@Jfenic, @tvarmar)

## Context

CATML ya sigue hexagonal + CQRS (ADR-001) y monolito modular en Explore (ADR-008). Las auditorías y el crecimiento de `AutoMLWorkspace`, `server.py` y `datasets.js` muestran reglas de negocio duplicadas (métricas, CV, seeds) y god-objects, no falta de capas.

Una reescritura o un split a microservicios aumentaría coste sin beneficio para un AutoML local.

## Decision

1. **Refactor progresivo** en 5 PRs (contratos → fachada Workspace → API/UI → persistencia/estados → tooling), documentado en [`docs/features/clean-architecture-refactor/`](../features/clean-architecture-refactor/).
2. **Conservar** CQRS, plugins, SQLite local y paridad de interfaces.
3. **Priorizar contratos de dominio** (`MetricSpec`, validation strategy, `ExecutionSpec`) antes de mover archivos grandes.
4. **Fachada compatible:** extracciones de servicios dejan métodos thin en `AutoMLWorkspace` hasta que los callers migran.
5. **No** introducir microservicios, event bus distribuido ni framework web obligatorio en esta fase.

## Consequences

### Positive

- Elimina familias enteras de bugs de comparación / reproducibilidad.
- Permite trabajo en paralelo (Persona A contratos/servicios vs Persona B UI) con límites claros.
- Mantiene la inversión existente en motor y Workbench.

### Negative / risks

- Periodo con fachada + servicio duplicado temporalmente.
- Riesgo de mezclar refactor con features de producto si no se respeta el gate de merge (#99/#100).

### Neutral

- El árbol de directorios objetivo es guía; no se exige un move masivo único.

## Links

- Spec/plan: [`docs/features/clean-architecture-refactor/`](../features/clean-architecture-refactor/)
- ADR-001 hexagonal/CQRS, ADR-008 Explore modular monolith
- Audit-3 contracts: PR #100
