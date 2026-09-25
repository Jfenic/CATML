# Guía para desarrolladores — CATML

Documento orientado a que otro programador pueda **continuar el proyecto de forma modular**, sin reescribir el núcleo.

**Versión de plataforma:** `0.6.0`  
**Última fase completada:** V0.6 (Plugin Architecture & Kaggle Inference)  
**Siguiente fase recomendada:** Backlog Tabular & V0.7 (Multimodal)  
**Guía de colaboración y ramas:** [`CONTRIBUTING.md`](CONTRIBUTING.md)

---

## 1. Punto de entrada rápido

```bash
pip install -e ".[dev]"
pytest                          # 64 tests — debe pasar todo con >= 85% coverage
automl task list                # catálogo tarea → modelos
automl plugin list              # plugins registrados (modelos, métricas)
automl run-demo --auto          # flujo automático con planner y scheduler
automl plan-experiments        # inspeccionar candidatos y explicabilidad
automl optimize --model logistic_regression --optimizer optuna --trials 10  # tuning bayesiano
automl benchmark run            # comparar escenarios de regresión
```


El wiring de la aplicación está en:

```text
src/automl/application/bootstrap.py   → build_application()
src/automl/application/services/workspace.py   → casos de uso
src/automl/interfaces/cli/main.py   → CLI
```

---

## 2. Arquitectura en capas (regla de dependencias)

```text
interfaces/  (CLI, futura API, futuros LLM tools)
      ↓
application/ (CommandBus, QueryBus, handlers, workspace)
      ↓
domain/      (Python puro — SIN sklearn, SIN SQLite)
      ↑
engine/      (profiling, planning, training)
plugins/     (modelos concretos)
infrastructure/ (SQLite, storage)
```

**Nunca importar sklearn o sqlite3 desde `domain/`.**

---

## 3. Módulos y responsabilidades

| Módulo | Ruta | Qué hace | Puedes extender… |
|--------|------|----------|------------------|
| **Dominio — Runs** | `domain/runs/` | AutoMLRun, RunConfig, estados | Nuevos estados, campos de config |
| **Dominio — Tasks** | `domain/tasks/` | TaskType, TASK_CATALOG, ProblemDefinition | Nuevas tareas, métricas, modelos en catálogo |
| **Dominio — Features** | `domain/features/` | Feature, FeatureSet, FeatureRegistry | Feature evidence (V0.5) |
| **Dominio — Experiments** | `domain/experiments/` | Experiment, Trial, TrialResult | Nuevos tipos de experimento |
| **Dominio — Models** | `domain/models/` | ModelSpec, ModelRegistry | Validación por tarea |
| **Application — CQRS** | `application/commands/`, `queries/`, `bus/` | Comandos y consultas | 1 comando = 1 handler en bootstrap |
| **Application — Workspace** | `application/services/workspace.py` | Orquestación | Nuevos casos de uso delegando aquí |
| **Engine — Planning** | `engine/planning/task_planner.py` | Inferir tarea desde dataset | RuleBasedExperimentPlanner (V0.3) |
| **Engine — Training** | `engine/training/sklearn_trainer.py` | Ejecutar Trial | Nuevas ramas por task_type |
| **Plugins — Models** | `plugins/models/sklearn_models.py` | build_sklearn_model, specs | LightGBM, XGBoost (V0.4) |
| **Infrastructure** | `infrastructure/database/` | SQLite, eventos, benchmark | Postgres adapter |
| **Benchmarks** | `benchmarks/runner.py` | Escenarios de regresión de calidad | Nuevos escenarios por versión |
| **CLI** | `interfaces/cli/` | Subcomandos | API REST reutilizando buses |

---

## 4. Cómo añadir algo sin romper la arquitectura

### A) Añadir un modelo nuevo

1. Añadir `model_id` en `TASK_CATALOG` (`domain/tasks/task_type.py`) para las tareas compatibles.
2. Implementar en `plugins/models/sklearn_models.py` → `build_sklearn_model()`.
3. El `ModelRegistry` se puebla solo vía `default_model_specs()`.
4. Test: experimento con el nuevo modelo + validación de incompatibilidad.

**No tocar** `domain/` salvo el catálogo de tareas.

### B) Añadir un Command (acción de usuario/agente)

1. Dataclass en `application/commands/workspace_commands.py`.
2. Handler en `application/bootstrap.py` → `register_handlers()`.
3. Método en `workspace.py` con la lógica.
4. (Opcional) Subcomando CLI.
5. Test en `tests/test_v02.py` o nuevo archivo.

### C) Añadir un Query (lectura)

1. Dataclass en `application/queries/workspace_queries.py`.
2. Handler en `bootstrap.py`.
3. Test.

### D) Añadir persistencia

1. Tabla/método en `infrastructure/database/sqlite_repository.py`.
2. Llamar desde `workspace.py`.
3. Migración en `_migrate()` si alteras tablas existentes.

### E) Añadir escenario de benchmark

1. Nuevo método `_setup_*` en `benchmarks/runner.py`.
2. Entrada en `scenarios()` con `id`, `version`, `description`.
3. Test en `tests/test_benchmark.py`.

---

## 5. Roadmap — qué hacer en cada fase

Referencia completa: `AutoML_Arquitectura_Tecnica.md` §8 y Anexo A.

### ✅ V0.1 — Hecho

- [x] Dominio: Run, Dataset, Feature, Experiment, Trial
- [x] Trainer sklearn + SQLite
- [x] Demo básico

### ✅ V0.2 — Hecho

- [x] CommandBus / QueryBus
- [x] FeatureSet, persistencia de features
- [x] Pause / Resume / Cancel / Clone
- [x] CompareExperiments, event log
- [x] Catálogo TaskType → modelos por tarea
- [x] ProblemDefinition + task planner
- [x] Benchmark harness (5 escenarios)
- [x] Validación de pertenencia Run → Experiment y Dataset → FeatureSet
- [x] CI en Python 3.10/3.12 con cobertura mínima del 85%

### ✅ V0.3 — Hecho

- [x] `ExperimentCandidate` (`domain/experiments/candidate.py`)
- [x] `Priority` y `PriorityScoreBreakdown` con scoring explicable (`domain/experiments/priority.py`)
- [x] `BudgetPolicy` (`domain/policies/budget.py`)
- [x] Ports: `ExperimentPlannerPort`, `PriorityScorerPort` (`domain/ports.py`)
- [x] `RuleBasedExperimentPlanner` (`engine/planning/experiment_planner.py`)
- [x] `RuleBasedPriorityScorer` (`engine/priority/scorer.py`)
- [x] `ExperimentQueue` y `Scheduler` con ordenamiento por `effective_score` y `pinned` (`engine/priority/scheduler.py`)
- [x] CQRS: `PlanExperimentsCommand`, `PrioritizeCandidateCommand`, `ExecuteNextExperimentCommand`, `RunScheduledExperimentsCommand`
- [x] Queries: `GetExperimentQueueQuery`, `ListCandidatesQuery`
- [x] CLI `plan-experiments` y soporte `--auto` en `run-demo`
- [x] Escenario de benchmark V0.3 (`automated_planning_v03`)
- [x] Test suite `tests/test_v03_planner.py` (31 tests totales, 89% coverage)

### ✅ V0.4 — Hecho (Optimización e Hiperparámetros con Optuna)

- [x] Dominio: `ParameterSpec`, `ParameterType`, `SearchSpace` declarativo (`domain/optimization/search_space.py`)
- [x] Dominio: `OptimizationBudget` (`domain/optimization/budget.py`)
- [x] Dominio: `OptimizerPort` protocol en `domain/ports.py` (`suggest/observe/should_stop/best_score/best_parameters`)
- [x] Engine: `EarlyStoppingPolicy` con soporte minimización/maximización (`engine/optimization/early_stopping.py`)
- [x] Engine: `RandomSearchOptimizer` con muestreo estocástico (`engine/optimization/random_search.py`)
- [x] Engine: `SearchSpaceBuilder` para modelos supervisados y no supervisados (`engine/optimization/search_space_builder.py`)
- [x] Engine: `TrialFactory` (`engine/optimization/trial_factory.py`)
- [x] Plugins: `OptunaOptimizer` adaptador ask-and-tell con TPESampler (`plugins/optimizers/optuna_optimizer.py`)
- [x] Plugins: Inyección de parámetros e instanciación condicional en `plugins/models/sklearn_models.py`
- [x] Application: `OptimizeExperimentCommand`, `GetBestTrialQuery`, `GetExperimentTrialsQuery`
- [x] Application: Orquestación en `workspace.optimize_experiment()`, `get_best_trial()`, `get_experiment_trials()`
- [x] CLI: Subcomando `automl optimize`
- [x] Benchmark: Escenario `optuna_optimization_v04` (+10.3% mejora de ROC AUC sobre baseline)
- [x] Test suite: `tests/test_v04_optimizer.py` (44 tests totales, 86% coverage)

### ✅ V0.5 — Hecho (Feature Discovery & Selection)

- [x] Ports: `FeatureSelectorPort`, `FeatureEvidenceRepositoryPort` en `src/automl/domain/ports.py`
- [x] Selectores estadísticos y ML (`MutualInfoSelector`, `TreeImportanceSelector`, `L1Selector`, `CorrelationSelector`, `VarianceSelector`, `EnsembleRankSelector`, `PCAReducer`) en `engine/features/`
- [x] `AblationPlanner` en `engine/planning/`
- [x] CQRS: `SelectFeaturesCommand`, `PlanAblationExperimentsCommand`, `PromoteCandidateFeatureSetCommand`
- [x] Persistencia de evidencia en SQLite y CLI `automl features select|ablation`
- [x] Test suite: `tests/test_v05_features.py` (52 tests, coverage >= 85%)

### ✅ V0.6 — Hecho (Plugin Architecture & Extensible Ecosystem)

- [x] Contratos: `PluginPort`, `ModelPluginPort`, `MetricPluginPort`, `PreprocessorPluginPort`
- [x] Application: `PluginRegistry` y `CompatibilityValidator` en engine
- [x] Adaptadores: `LightGBMPlugin` y `XGBoostPlugin` con fallback a scikit-learn
- [x] Métricas de negocio: `CostSensitiveMetricPlugin`, `WeightedF1MetricPlugin`
- [x] CLI `automl plugin list` y test suite `tests/test_v06_plugins.py`
- [x] Inferencia Kaggle: `GenerateSubmissionCommand`, `PredictDatasetQuery`, CLI `automl predict` (`tests/test_kaggle_prediction.py`)

### ⏳ Siguiente: Backlog Tabular & V0.7 (Multimodalidad)

Ver especificaciones de tareas para colaboradores en [`CONTRIBUTING.md`](CONTRIBUTING.md) y roadmap general en [`TASKS.md`](TASKS.md).

---

## 6. Archivos clave por tarea de desarrollo

```text
Quiero…                              → Empieza aquí
─────────────────────────────────────────────────────────
Entender el flujo completo             → workspace.py + bootstrap.py
Añadir tipo de tarea                   → domain/tasks/task_type.py
Cambiar inferencia de tarea            → engine/planning/task_planner.py
Añadir modelo sklearn                  → plugins/models/sklearn_models.py
Cambiar métricas de evaluación         → engine/training/sklearn_trainer.py
Nuevo comando usuario/LLM              → commands/ + bootstrap.py
Nueva consulta UI/LLM                  → queries/ + bootstrap.py
Persistencia / auditoría               → sqlite_repository.py
Medir mejoras entre versiones          → benchmarks/runner.py
Documentación formal                   → AutoML_Arquitectura_Tecnica.md
```

---

## 7. Convenciones del proyecto

1. **Un experimento = una hipótesis verificable.** No monolitos `AutoML.fit()`.
2. **Commands mutan, Queries leen.** Sin efectos secundarios en queries.
3. **Modelos incompatibles con la tarea → ValueError** en `create_experiment`.
4. **Eventos** en toda mutación relevante (`repository.append_event`).
5. **Tests:** un archivo por fase (`test_v01`, `test_v02`, `test_tasks`, `test_benchmark`).
6. **Versión:** actualizar `PLATFORM_VERSION` en `workspace.py` y `pyproject.toml`.

---

## 8. Composición para futuro LLM (V0.9)

Cuando llegue V0.9, cada Command/Query existente se envuelve en un `AgentTool`:

```text
CreateExperimentCommand  →  CreateExperimentTool
GetTaskPlanQuery         →  GetTaskPlanTool
```

No crear lógica nueva en la capa de agentes; solo wrappers con schema JSON.

---

## 9. Checklist antes de abrir PR

- [ ] `pytest` pasa
- [ ] Nuevo comportamiento tiene test
- [ ] Sin imports de infra en `domain/`
- [ ] Commands/Queries registrados en `bootstrap.py`
- [ ] Si añades escenario benchmark, actualizar README
- [ ] Migración SQLite si cambias schema

---

## 10. Contacto con la spec

| Pregunta | Documento |
|----------|-----------|
| ¿Qué entidades existen? | `AutoML_Arquitectura_Tecnica.md` §6 |
| ¿Qué va en V0.3? | `AutoML_Arquitectura_Tecnica.md` §V0.3 |
| ¿Cómo funciona feature selection? | `AutoML_Arquitectura_Tecnica.md` §12.1 |
| ¿Decisiones a evitar? | `AutoML_Arquitectura_Tecnica.md` §17 |
| Checklist clases por fase | Anexo A |

---

## 11. Ejemplo de división de trabajo en paralelo

Tres desarrolladores pueden trabajar en paralelo sin conflictos:

| Dev A | Dev B | Dev C |
|-------|-------|-------|
| V0.3 Planner | V0.4 Optuna plugin | V0.5 MI selector |
| `engine/planning/` | `plugins/optimizers/` | `domain/features/selection/` |
| Sin tocar trainer | Sin tocar planner | Sin tocar optimizer |

Punto de integración común: **`workspace.run_experiment()`** y **`bootstrap.register_handlers()`**.
