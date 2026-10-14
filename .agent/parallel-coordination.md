# Parallel agents — coordinación (post–Study B smoke)

> Fuente de verdad para este split paralelo. Actualizar este archivo al reclamar, pausar o cerrar.
> Reglas globales: [`AGENTS.md`](../AGENTS.md) · blackboard: `gh issue list --label blackboard --state open`.
> Base de código: `origin/main` @ merge `#119` (`d7982bb`) + docs `#120` cuando entre.

**Estado del split:** 2026-10-14

| Agente | Rol | Estado | Rama |
| --- | --- | --- | --- |
| **A** | Study B lift | **MERGED** [#122](https://github.com/Jfenic/catml/pull/122) (smoke 24/0/0) | `feat/study-b-lift` |
| **B** | Demo export + SQLite #94 | **MERGED** [#125](https://github.com/Jfenic/catml/pull/125), [#126](https://github.com/Jfenic/catml/pull/126) | `feat/demo-export-and-sqlite-94` |

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
| **Asignado** | Cursor agent |
| **Estado** | **DONE** (merged `#122`) |
| **Rama** | `feat/study-b-lift` |
| **Objetivo** | Que `catml_explore` aplique recipes/verificaciones de Explore al fit; cablear `catml_agent` (FakeLLM). |

### Criterio de hecho

- [x] `catml_explore`: verify → recipe from evidence → fit/predict
- [x] `catml_agent`: FakeLLM + session step
- [x] Tests Study B verdes
- [x] Smoke 24 ok / 0 skipped / 0 failed
- [x] PR `#122` merged

---

## Agente B — Demo export + blackboard #94

| Campo | Valor |
| --- | --- |
| **Asignado** | Agente B (Antigravity session) |
| **Estado** | **DONE** (PR `#125`) |
| **Rama** | `feat/demo-export-and-sqlite-94` |
| **Objetivo** | Completar B1 + B2 sin solaparse con Agente A. |

### Subtarea B1 — Demo project + export ejecutivo (Sprint 3)

- Producto: proyecto demo + exportación de resultados ejecutivos.
- **Archivos:** `src/automl/interfaces/web/**` (+ tests web).

### Subtarea B2 — Blackboard [#94](https://github.com/Jfenic/catml/issues/94)

- Stress / lifetime de conexiones SQLite largas.
- **Archivos:** `tests/` + infra SQLite mínima.

### Claim Agente B (rellenado)

- **Asignado:** Agente B (Antigravity)
- **Reclamado:** 2026-10-14T00:30:00Z
- **Rama:** `feat/demo-export-and-sqlite-94`
- **Archivos:**
  - `src/automl/infrastructure/database/sqlite_recipes.py`
  - `src/automl/infrastructure/database/sqlite_studies.py`
  - `src/automl/infrastructure/database/sqlite_repository.py`
  - `src/automl/application/services/workspace.py`
  - `src/automl/interfaces/web/api/datasets.py`
  - `src/automl/interfaces/web/api/experiments.py`
  - `src/automl/interfaces/web/static/js/api.js`
  - `src/automl/interfaces/web/static/js/views/home.js`
  - `src/automl/interfaces/web/static/js/views/register_dataset_modal.js`
  - `src/automl/interfaces/web/static/js/views/studio.js`
  - `src/automl/interfaces/web/static/js/views/overview.js`
  - `tests/test_sqlite_connection_lifecycle.py`
  - `tests/test_workbench_demo_and_export.py`
  - `TASKS.md` / `.agent/progress.md`
- **Orden:** B2 primero, luego B1
- **PR(s):** [#125](https://github.com/Jfenic/catml/pull/125), [#126](https://github.com/Jfenic/catml/pull/126)

### Criterio de hecho B

- [x] B1: demo + export usable
- [x] B2: #94 con soak test + dispose lifecycle
- [x] Claim relleno + estado `DONE`
- [x] Solo track B en docs de coordinación
