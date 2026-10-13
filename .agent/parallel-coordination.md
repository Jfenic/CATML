# Parallel agents — coordinación (post–Study B smoke)

> Fuente de verdad para este split paralelo. Actualizar este archivo al reclamar, pausar o cerrar.
> Reglas globales: [`AGENTS.md`](../AGENTS.md) · blackboard: `gh issue list --label blackboard --state open`.
> Base de código: `origin/main` @ merge `#119` (`d7982bb`) + docs `#120` cuando entre.

**Estado del split:** 2026-10-14

| Agente | Rol | Estado | Rama |
| --- | --- | --- | --- |
| **A** | Study B lift | **PR** [#122](https://github.com/Jfenic/catml/pull/122) (smoke 24/0/0) | `feat/study-b-lift` |
| **B** | Demo export + SQLite #94 | **DONE** | `feat/demo-export-and-sqlite-94` |

---

## Zona prohibida (ninguno edita a la vez sin contract PR en `main`)

- `src/automl/domain/ports.py`
- `src/automl/application/bootstrap.py`
- `src/automl/application/agents/executor.py` (o path equivalente del executor monolítico)
- `src/automl/infrastructure/**/sqlite_agent_ledger.py`
- Merges: solo `git merge origin/main` (no rebase)

Hallazgos fuera de alcance → `gh issue create --label blackboard` (no arreglar en la otra rama).

---

## Agente A — Study B lift

| Campo | Valor |
| --- | --- |
| **Asignado** | Cursor agent (esta sesión) |
| **Estado** | **HACIENDO** |
| **Rama** | `feat/study-b-lift` |
| **Objetivo** | Que `catml_explore` aplique recipes/verificaciones de Explore al fit (lift real, Propose≠Accept respetado); cablear `catml_agent` (FakeLLM primero, Ollama opcional). Actualizar smoke/report. |

### Alcance de archivos (A)

**Principal (esperado):**

- `benchmarks/frameworks/study_b.py`
- `benchmarks/run.py` (solo si hace falta flags Study B)
- `tests/test_benchmark_study_b.py`
- `docs/experiments/study-b-report.md`
- `benchmarks/README.md` (resumen status lift, si aplica)

**Permitido si el lift lo exige (mínimo diff):**

- Cableado Explore → recipe → fit bajo `src/automl/application/` / explore / recipes (solo lo necesario para “verified recipe → fit”)
- Stub/adaptador `catml_agent` en `benchmarks/frameworks/` + tests asociados

**Fuera de alcance A:** Workbench UI, Demo Sprint 3, issue `#94`.

### Criterio de hecho

- [x] `catml_explore`: verify → recipe from evidence → fit/predict (fallback base si no hay evidence usable)
- [x] `catml_agent`: FakeLLM + session step (ya no `skipped`)
- [x] Tests Study B verdes (`tests/test_benchmark_study_b.py` 3 passed)
- [x] Smoke local breast/wine × 3 seeds + tabla en report (24 ok / 0 skipped / 0 failed)
- [x] PR hacia `main`: https://github.com/Jfenic/catml/pull/122

### Notas A

- Sync: actualizar solo la sección Agente A de este archivo + entrada corta en `.agent/progress.md`.
- Al pausar: dejar commit/PR link y checklist parcial aquí.

---

## Agente B — Demo export + blackboard #94

| Campo | Valor |
| --- | --- |
| **Asignado** | Agente B (Antigravity session) |
| **Estado** | **DONE** |
| **Rama** | `feat/demo-export-and-sqlite-94` |
| **Objetivo** | Completar las **dos** subtareas abajo sin solaparse con Agente A. |

### Subtarea B1 — Demo project + export ejecutivo (Sprint 3)

- Producto: proyecto demo + exportación de resultados ejecutivos (track Workbench UX Sprint 3 en `TASKS.md`).
- **Archivos esperados:** `src/automl/interfaces/web/**` (+ tests web relacionados, p.ej. `tests/test_workbench_*.py`).
- **No tocar:** `benchmarks/`, Study B, Explore fit path.

### Subtarea B2 — Blackboard [#94](https://github.com/Jfenic/catml/issues/94)

- Stress / lifetime de conexiones SQLite largas más allá del check unitario de FDs.
- **Archivos esperados:** `tests/` (nuevo o extendido) + cambio mínimo en `src/automl/infrastructure/` solo si el test lo exige.
- **No tocar:** Study B, Workbench salvo que #94 lo fuerce (no debería).

### Claim Agente B (rellenado)

- **Asignado:** Agente B (Antigravity)
- **Reclamado:** 2026-10-14T00:30:00Z
- **Rama:** `feat/demo-export-and-sqlite-94`
- **Archivos que tocaré (lista concreta):**
  - `src/automl/infrastructure/database/sqlite_recipes.py` (`__del__` y context manager)
  - `src/automl/infrastructure/database/sqlite_studies.py` (`__del__` y context manager)
  - `src/automl/infrastructure/database/sqlite_repository.py` (`close` hook)
  - `src/automl/application/services/workspace.py` (`dispose()` y caching de repos)
  - `src/automl/interfaces/web/api/datasets.py` (`POST /api/project/demo`)
  - `src/automl/interfaces/web/api/experiments.py` (`GET /api/reports/executive` en HTML, MD y JSON)
  - `src/automl/interfaces/web/static/js/api.js` (métodos `loadDemoProject` y `downloadExecutiveReport`)
  - `src/automl/interfaces/web/static/js/views/home.js` (botón cargar proyecto demo)
  - `src/automl/interfaces/web/static/js/views/register_dataset_modal.js` (botón cargar demo)
  - `src/automl/interfaces/web/static/js/views/studio.js` (enlaces informe ejecutivo)
  - `src/automl/interfaces/web/static/js/views/overview.js` (enlaces informe ejecutivo)
  - `tests/test_sqlite_connection_lifecycle.py` (soak test ResourceWarning y dispose)
  - `tests/test_workbench_demo_and_export.py` (tests exhaustivos B1)
  - `TASKS.md` (Track Workbench UX Sprint 3)
  - `.agent/progress.md` (Track Agente B)
- **Orden:** B2 primero (#94 soak + dispose lifecycle), luego B1 (demo project + export ejecutivo)
- **PR(s):** (completado y verificado en rama local)

### Criterio de hecho B

- [x] B1: demo + export usable o PR con aceptación del track Sprint 3
- [x] B2: #94 cerrado con comentario de commit/PR o justificación + test
- [x] Este markdown: claim relleno + estado `DONE`
- [x] Solo track B en `TASKS.md` / `.agent/progress.md` (no editar sección Agente A)

---

## Cómo arrancar Agente B (prompt listo)

> Lee `.agent/parallel-coordination.md`. Eres **Agente B**. Reclama el rol rellenando la plantilla Claim (asignado, rama, lista de archivos). No toques la zona prohibida ni `benchmarks/frameworks/study_b.py`. Haz B1 (demo/export web) y B2 (issue #94). Antes de codear: `gh issue list --label blackboard --state open`. Coordina solo vía este markdown, blackboard y PRs. Merge con `git merge origin/main` únicamente.

## Cómo sigue Agente A (esta sesión)

> Continúa en `feat/study-b-lift`. Actualiza la checklist de Agente A aquí al avanzar. No reclames B ni edites archivos de B1/B2.
