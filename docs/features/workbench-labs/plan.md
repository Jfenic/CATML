# Workbench Labs — Frontend modular por modalidad (plan futuro)

> **Estado:** Aceptado como dirección futura · **No implementar ahora**  
> **Prioridad inmediata:** terminar y estabilizar la experiencia **tabular** (ver [`TASKS.md`](../../../TASKS.md)).  
> **ADR:** [011 — Workbench Labs](../../decisions/011-workbench-labs-modular-frontend.md)  
> **Alineado con:** [ADR-001](../../decisions/001-hexagonal-architecture-and-cqrs.md), [ADR-008 Explore](../../decisions/008-catml-explore-modular-monolith.md), [MASTER_PLAN §7–14](../../MASTER_PLAN.md).

## Intención

Una plataforma CATML → varios **laboratorios** especializados → un único sistema de experimentos, trabajos, evidencias y artefactos.

- **No** varios proyectos frontend independientes.
- **No** una única pantalla mega-configurable para todas las modalidades.
- **Sí** App Shell compartido + módulos JS por lab (lazy load) + motor AutoML / CQRS único.

## Ahora vs después

| Ahora (tabular first) | Después (labs) |
| --- | --- |
| Cerrar PRs Clean Arch / UX (#109–#111) | Introducir `module-registry` + router sin cambiar UX |
| Completar flujo tabular CSV → train → compare → export | Extraer Tabular Lab desde vistas actuales |
| Audit-5 honesty (presupuesto, XSS, fallbacks, métricas) | `GET /api/capabilities` (proyección CQRS) |
| Demo confidence Sprint 3 + PyPI | Clustering Lab → Vision Lab → NLP Lab |

Temporal **CV** permanece en Tabular; un Temporal Lab de forecasting/backtesting es futuro opcional.

## Destino de carpetas (evolutivo)

```
src/automl/interfaces/web/static/js/
  core/          # router, module-registry, store, bus, context
  api/           # client + clientes por dominio (fachada compatible)
  shared/        # shell, jobs, dataset-selector, model-selector
  labs/
    tabular/     # primera extracción real
    clustering/  # primer lab nuevo (prueba de arquitectura)
    vision/      # tras contratos multimodales en motor
    nlp/
    explore/     # Explore como lab transversal o anclado a tabular
    temporal/    # forecasting (no solo TimeSeriesSplit tabular)
```

Cada extracción debe aportar responsabilidad clara + tests; no mover archivos por estética.

## Fases (tras tabular estable)

0. Consolidar merges actuales (#109, #110, #111) y Audit-5 en `main`.
1. Núcleo frontend: registry + router; rutas antiguas vía adaptadores.
2. Tabular Lab: Studio / New Experiment / leaderboard como módulos del lab.
3. Capabilities dinámicas desde dominio/plugins (native vs fallback, experimental).
4. Clustering Lab (sin target supervisado).
5. Vision Lab (un flujo train→export completo) antes de NLP.

## Criterios de aceptación (cuando se ejecute)

- Añadir un lab no obliga a editar labs existentes.
- Destruir un lab cancela timers/listeners/fetch obsoletos (`contextVersion`).
- Frontend no ofrece modelos incompatibles como entrenables.
- Python / CLI / REST / MCP siguen el mismo CQRS.
- Static JS nuevos siguen entrando en el wheel PyPI.
- Usuario tabular completa el flujo sin abrir Vision/NLP/Clustering.

## Fuera de alcance inmediato

- Migración a React/Vite solo para modularizar.
- Scaffold vacío de Vision/NLP.
- Stores por modalidad (contaminan el contexto global).
- JS arbitrario cargado desde plugins Python.
