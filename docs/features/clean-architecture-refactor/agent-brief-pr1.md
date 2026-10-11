# Brief para agente implementador — PR1 Contratos

> **Rol:** implementar **solo PR1** del plan Clean Architecture.  
> **Documentación:** [spec.md](spec.md) · [plan.md](plan.md) · [ADR-010](../../decisions/010-progressive-clean-architecture-refactor.md)  
> **Agente documentador / orquestador:** no debe modificar código de producto; este brief es la fuente de verdad del alcance de código.

---

## Misión

Centralizar contratos de **métricas**, **validación CV** y **especificación de ejecución** para que todos los caminos (Workspace schedule, HPO, leaderboard/`get_best_trial`, recetas, trainer) usen la misma fuente de verdad.

## Base git (obligatorio)

1. Trabajar en rama nueva: `refactor/clean-arch-pr1-metric-execution-contracts`.
2. Partir de **`origin/fix/audit3-recipe-metric-cv-contracts`** (PR #100) si aún no está mergeado en `main`; si #100 ya está en `main`, partir de `origin/main`.
3. `git merge origin/main` si hace falta (nunca rebase).
4. No tocar la rama del orquestador ni mezclar Sprint 2 (#99) salvo conflictos mínimos de merge.

## Leer antes de codificar

- `AGENTS.md` (hexagonal, CQRS, cobertura ≥ 85 %, blackboard).
- `src/automl/domain/experiments/validation.py`
- `src/automl/domain/metrics/direction.py`
- `src/automl/domain/ports.py` (`TrialExecution`)
- `src/automl/application/services/workspace.py` (`run_scheduled_experiments`, `optimize_experiment`, `get_best_trial`, `run_experiment`)
- `src/automl/engine/training/sklearn_trainer.py`
- `src/automl/benchmarks/runner.py` (`is_minimizing_metric`)
- `src/automl/interfaces/web/server.py` (`_is_metric_minimizing`, `_get_best_trial_result`) — **solo adaptar a usar el registro**, no partir el server.
- Tests existentes: `tests/test_audit3_recipe_metric_cv_contracts.py`, `tests/test_workbench_metric_context.py`, `tests/test_hpo_recipe_fidelity.py`

## Entregables de código

### 1. Domain — MetricSpec

Crear (nombres orientativos; alinear con estilo del repo):

- `src/automl/domain/metrics/spec.py` (o consolidar en `evaluation/` **solo si** el move es trivial; preferir ampliar `domain/metrics/` para no big-bang).
- `MetricDirection`, `MetricSpec` frozen con `is_better` / `select_best_score` / `select_best_index`.
- Registro builtin: al menos `accuracy`, `f1`, `roc_auc`, `r2`, `mae`, `rmse`, `mse`, `log_loss`, `balanced_accuracy` (y los que ya use el trainer).
- Función `resolve_metric_spec(name: str, *, greater_is_better: bool | None = None) -> MetricSpec`  
  - Si viene de plugin: respetar `greater_is_better`.  
  - Si builtin desconocido: fail fast `InvalidMetric` (o ValueError de dominio documentado).

Deprecar uso disperso de sets `{"mae","rmse",...}` en application; dejar un único helper que delegue en `MetricSpec`.

### 2. Domain — ExecutionSpec

- `src/automl/domain/experiments/execution_spec.py` (frozen dataclass).
- Campos mínimos:  
  `dataset_id`, `dataset_content_hash` (opcional `None`), `target`, `feature_names`, `model_id`, `metric`, `validation_strategy`, `cv_folds`, `random_seed`, `test_size`, `group_column`, `column_transforms`, `derived_features`, `task_type`.
- Factory pura o en application: `build_execution_spec(run, experiment, dataset, *, model_id, params=...)` que:
  - normaliza CV con `require_known_validation_strategy`;
  - resuelve métrica con `resolve_metric_spec`;
  - no mutea el run (la mutación de recipe replay puede quedarse en RecipeService).
- Adaptar `SklearnTrainer.run` **o** el punto que construye `TrialExecution` para validar/rellenar desde `ExecutionSpec` sin cambiar el comportamiento estadístico bueno ya corregido en audit-3 (fold seed fija en HPO).

### 3. Cablear consumidores (mínimo viable)

Sustituir listas hardcodeadas / `max()` ciego donde aún queden:

- `AutoMLWorkspace.get_best_trial` / `_is_minimizing_metric` → `MetricSpec`
- `run_scheduled_experiments` → ya corregido en #100; unificar via `MetricSpec`
- `benchmarks/runner.is_minimizing_metric` → delegar
- Helper web `_is_metric_minimizing` → delegar al mismo contrato (import application/domain, sin duplicar sets)

### 4. Tests nuevos

Archivo sugerido: `tests/test_clean_arch_pr1_contracts.py`

- `MetricSpec` MAE elige 4 no 9; ROC-AUC elige el mayor.
- Plugin metric con `greater_is_better=False` respetado si hay fixture simple; si no hay plugin fácil, mock.
- `ExecutionSpec` rechaza `validation_strategy` desconocida.
- `build_execution_spec` + un `run_experiment` smoke: metric/cv/seed del spec llegan a `TrialExecution`.
- No borrar tests audit-3; deben seguir pasando.

### 5. Docs / TASKS (solo track de este refactor)

- Marcar PR1 items en `TASKS.md` como In progress → Done al cerrar.
- Añadir párrafo corto en `.agent/progress.md` bajo track Clean Architecture.

## Prohibido en este PR

- Partir `server.py` en controllers.
- Partir `datasets.js`.
- Extraer `ExperimentExecutionService` / `OptimizationService` (eso es **PR2**).
- Unit of Work / multi-repository SQLite.
- Cambiar stacking, shap naming, o hacer hard-fail de pip-audit.
- Reformat masivo / renombres cosméticos fuera del alcance.
- `--no-verify`, force push, amend de commits ajenos.

## Validación

```bash
.venv/bin/pytest tests/test_clean_arch_pr1_contracts.py tests/test_audit3_recipe_metric_cv_contracts.py tests/test_workbench_metric_context.py tests/test_hpo_recipe_fidelity.py -q
.venv/bin/pytest --cov=src/automl --cov-fail-under=85 -q
```

## Entrega

1. Commit(s) claros en la rama `refactor/clean-arch-pr1-metric-execution-contracts`.
2. Push + `gh pr create` hacia `main` con summary + test plan.
3. En el body del PR: enlace a este brief y a ADR-010.
4. Devolver al orquestador: URL del PR, lista de archivos tocados, resultado pytest.

## Comunicación

- Si encuentras un bug fuera de alcance → `gh issue create` con label `blackboard`, **no** lo arregles en este PR.
- Conflictos con #100/#99: merge, no rebase; reportar en el PR.
