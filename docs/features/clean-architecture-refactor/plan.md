# Plan: Refactor progresivo Clean Architecture

> **Spec:** [spec.md](spec.md) · **ADR:** [010](../../decisions/010-progressive-clean-architecture-refactor.md)  
> **Estado operativo:** [`TASKS.md`](../../../TASKS.md)  
> **Brief para agente implementador PR1:** [agent-brief-pr1.md](agent-brief-pr1.md)

---

## Principio

Refactor **progresivo**, no rewrite. Cada PR debe:

1. Mantener la API pública usable (fachada / CQRS).
2. Añadir o migrar tests de contrato.
3. Dejar el árbol más cerca de la estructura objetivo sin big-bang moves.

**Condición obligatoria:** mismos datasets + semillas + folds → resultados equivalentes, salvo corrección estadística documentada.

---

## Estructura objetivo (referencia)

```
src/automl/
├── domain/
│   ├── datasets/
│   ├── experiments/          # validation.py (ya existe post-audit-3)
│   ├── evaluation/           # metrics.py, policies.py (nuevo / consolidación)
│   ├── recipes/
│   └── runs/
├── application/
│   ├── commands/ / queries/ / registries/   # conservar CQRS
│   ├── services/
│   │   ├── workspace.py      # fachada delgada
│   │   ├── experiment_execution_service.py
│   │   ├── optimization_service.py
│   │   ├── dataset_service.py
│   │   └── … (feature / inference / pipeline ya existen)
│   └── ports/
├── engine/
├── infrastructure/
│   ├── database/             # repos por agregado (PR4)
│   ├── filesystem/
│   └── security/
├── interfaces/web/api/       # controllers (PR3)
└── plugins/
```

---

## Cadena de PRs

```mermaid
flowchart LR
  P0[Merge #99 #100 + rc1 gate] --> PR1[PR1 Contratos]
  PR1 --> PR2[PR2 Extraer ejecución/HPO de Workspace]
  PR2 --> PR3[PR3 API + datasets.js]
  PR3 --> PR4[PR4 Persistencia + estados + UoW]
  PR4 --> PR5[PR5 Calidad automática]
```

### Gate previo (humano / merge)

- Merge **#99** (Sprint 2 UX) y **#100** (audit-3 contracts) en `main`.
- No abrir PR2/PR3 sobre `main` viejo sin esos contratos.

---

## PR 1 — Contratos y reglas compartidas (bajo riesgo)

**Objetivo:** Una sola fuente de verdad para métricas, CV y especificación de ejecución.

| Trabajo | Detalle |
| --- | --- |
| `MetricSpec` + registro | Ampliar `domain/metrics/` (hoy solo `direction.py`); builtins + resolución plugin |
| CV | Usar `domain/experiments/validation.py` en todos los callers restantes |
| `ExecutionSpec` | Dataclass frozen; factory desde run+experiment+recipe; trainer consume |
| Errores de dominio | Fail fast tipado para strategy/metric/transition |
| Tests contrato | Mismo ganador MAE/ROC-AUC en schedule, `get_best_trial`, helper leaderboard |

**No hacer en PR1:** partir `server.py`, partir `datasets.js`, Unit of Work, mover SQLite a multi-repo.

**Dueño sugerido:** agente implementador (ver [agent-brief-pr1.md](agent-brief-pr1.md)).

**Criterio done:** tests nuevos verdes; consumidores críticos ya no usan `max()` ciego ni listas de métricas hardcodeadas duplicadas; cobertura global ≥ 85 %.

### PR1 acceptance checklist

- [x] `MetricSpec` + registro builtin + `resolve_metric_spec` / `InvalidMetric` (`domain/metrics/`)
- [x] `ExecutionSpec` frozen + `build_execution_spec` / `trial_execution_from_spec`
- [x] Consumidores cableados: Workspace (`run_experiment`, `optimize_experiment`, schedule, `get_best_trial`), benchmarks `is_minimizing_metric`, web `_is_metric_minimizing`, trainer helper
- [x] CV desconocido → `InvalidValidationStrategy` (fail closed)
- [x] HPO mantiene semilla de folds fija (`ExecutionSpec.random_seed` = run seed)
- [x] Tests: `tests/test_clean_arch_pr1_contracts.py` (+ suites audit-3 / métrica / HPO)
- [x] Cobertura global ≥ 85 % (suite completa)
- [x] **Fuera de alcance respetado:** no partir `server.py` ni `datasets.js`; no extraer servicios Workspace (PR2)
- [ ] Merge: preferir `#100` → `#101` (docs) → `#102` (PR1 código)

Implementación código: PR **#102** (`refactor/clean-arch-pr1-metric-execution-contracts`).

---

## PR 2 — Extraer ejecución y optimización de `AutoMLWorkspace`

**Objetivo:** Workspace como fachada.

1. Extraer `run_experiment` → `ExperimentExecutionService`.
2. Extraer `optimize_experiment` → `OptimizationService`.
3. Métodos antiguos en Workspace delegan 1:1 (sin cambiar CLI/MCP/Web).
4. Tests de regresión sobre CSV sintético binario + MAE regression smoke.

**No hacer:** reescribir planner/queue en el mismo PR salvo lo mínimo para compilar.

### PR2 acceptance checklist

- [x] `ExperimentExecutionService` con lógica de `run_experiment` (`application/services/experiment_execution_service.py`)
- [x] `OptimizationService` con lógica de `optimize_experiment` (`application/services/optimization_service.py`)
- [x] `AutoMLWorkspace` expone propiedades lazy + métodos thin (firmas públicas intactas)
- [x] Tests: `tests/test_clean_arch_pr2_execution_services.py` (binary smoke, MAE regression, HPO facade, fail-closed optimizer)
- [x] Imports HPO/trainer muertos eliminados de `workspace.py`
- [ ] Merge PR [#109](https://github.com/Jfenic/catml/pull/109) (cobertura ≥ 85 %, suite completa verde)

---

## PR 3 — Separar API y frontend

**Backend**

- `server.py` = HTTP glue (parse, status, JSON).
- Controllers: `api/datasets.py`, `experiments.py`, `analysis.py`, `jobs.py`.
- Ciclo de vida app ≠ ciclo de vida request (cuidado con caches mutables).

**Frontend**

```
views/datasets/
  datasets_view.js
  dataset_summary.js
  dataset_columns.js
  data_quality_panel.js
  feature_selector.js
  correlations_view.js
  questionnaire.js
  derived_features.js
```

- Preferir `textContent` / nodos DOM para datos de usuario; `escapeHtml` donde quede `innerHTML`.

### PR3 acceptance checklist

- [x] `server.py` thin glue + dispatch to `api/{jobs,datasets,experiments,analysis}.py`
- [x] Shared helpers in `api/_helpers.py` (re-exported from `server` for tests)
- [x] `views/datasets.js` re-exports `DatasetsView`; modules under `views/datasets/`
- [x] Tests: `tests/test_clean_arch_pr3_web_api_datasets_split.py` + updated static source helpers
- [ ] Merge PR [#110](https://github.com/Jfenic/catml/pull/110) (cobertura ≥ 85 %, suite completa verde)

---

## PR 4 — Persistencia y estados transaccionales

- Repositorios SQLite por responsabilidad; migraciones comunes.
- Máquina de estados explícita para `AutoMLRun` / experiment / job (cancelado ↛ completed por worker tardío).
- Unit of Work en operaciones críticas (crear experiment + trial + result).

### PR4 acceptance checklist

- [x] `domain/runs/transitions.py` + `domain/experiments/transitions.py` (CANCELLED sticky; retry from COMPLETED/FAILED allowed)
- [x] `AutoMLRun.transition_to` / `Experiment.transition_to` enforce maps; persist layer re-validates on save
- [x] `SqliteUnitOfWork` + `connection.py` + `mappers.py`; package `stores/` as aggregate boundary
- [x] `persist_trial_outcome(trial, result, experiment=?)` atomic; used by execution/HPO services
- [x] Tests: `tests/test_clean_arch_pr4_persistence_uow.py`
- [x] Brief: [agent-brief-pr4.md](agent-brief-pr4.md)
- [x] Merge PR (cobertura ≥ 85 %, suite completa verde) — `#129`

---

## PR 5 — Calidad automática

- [x] Ruff + mypy gradual en `domain/` y `application/` (`pyproject.toml`; CI `lint-python`).
- [x] ESLint/Prettier para `static/js` (`eslint.config.js`, `package.json`; CI `lint-js`).
- [x] Tests de contrato + E2E export→predict parity (`tests/test_clean_arch_pr5_quality_gates.py`).
- [x] Mantener gate pytest coverage ≥ 85 % en CI.
- [ ] Merge PR (suite verde + lint jobs).
- (Opcional) endurecer `pip-audit` cuando el equipo lo decida.

---

## Qué no cambiar

- CQRS + registries en `application/`.
- Sistema de plugins.
- SQLite como store local por defecto.
- Paridad Python / CLI / MCP / Web vía application layer.
- Brand system Tech Minimalista en UI.

---

## Tracking

Actualizar solo el track **Clean Architecture Refactor** en `TASKS.md` y `.agent/progress.md` al cerrar cada PR.
