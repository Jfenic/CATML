# Guía para desarrolladores — CATML

Documento orientado a que otro programador pueda **continuar el proyecto de forma modular**, sin reescribir el núcleo.

**Versión de plataforma:** `0.7.0` (fuente: [`src/automl/__init__.py`](src/automl/__init__.py)).
**Estado y siguiente trabajo:** [`TASKS.md`](TASKS.md).
**Guía de colaboración:** [`CONTRIBUTING.md`](CONTRIBUTING.md).
**Capacidades y limitaciones:** [`docs/README.md`](docs/README.md).

---

## 1. Punto de entrada rápido

```bash
pip install -e ".[dev]"
pytest --cov=src/automl --cov-fail-under=85  # suite completa
automl task list                # catálogo tarea → modelos
automl plugin list              # plugins registrados (modelos, métricas)
automl run-demo --auto          # flujo automático con planner y scheduler
automl plan-experiments        # inspeccionar candidatos y explicabilidad
automl optimize --model logistic_regression --optimizer optuna --trials 10  # tuning bayesiano
automl benchmark run            # comparar escenarios de regresión
automl explore list             # listar estudios exploratorios de CATML Explore
automl explore export <id> -f markdown -o report.md  # reporte técnico reproducible
automl explore verify <hyp_id>  # verificar hipótesis empíricamente (Propose ≠ Accept)
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
interfaces/  (CLI, Workbench HTTP, futuros LLM tools)
      ↓
application/ (CommandBus, QueryBus, handlers, workspace)
      ↓
domain/      (Python puro — SIN sklearn, SIN SQLite)
      ↑
engine/      (profiling, planning, training, ensemble)
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
| **Engine — Profiling** | `engine/profiling/dataset_profiler.py` | Perfilado, tipos semánticos y detección de IDs | Nuevas heurísticas de cardinalidad |
| **Engine — Planning** | `engine/planning/task_planner.py` | Inferir tarea desde dataset | RuleBasedExperimentPlanner (V0.3) |
| **Engine — Ensemble** | `engine/ensemble/blender.py`, `voting.py` | Promediado soft/hard voting y blending | Stacking, rank averaging |
| **Engine — Training** | `engine/training/sklearn_trainer.py` | Ejecutar Trial | Nuevas ramas por task_type |
| **Plugins — Models** | `plugins/models/` | Adaptadores sklearn, gradient boosting y voting | Nuevos `ModelPluginPort` |
| **Infrastructure** | `infrastructure/database/` | SQLite, eventos, benchmark | Postgres adapter |
| **Benchmarks** | `benchmarks/runner.py` | Escenarios de regresión de calidad | Nuevos escenarios por versión |
| **Benchmarks — Explore** | `benchmarks/explore_benchmarks.py` | Batería 4 ejes (NIST, calidad AutoML, MCP, esquemas) | Nuevos escenarios de calibración |
| **Dominio — Explore** | `domain/analysis/` | StudySpec, AnalysisRun, StatisticalFinding, VisualizationSpec, EvidenceLink | Nuevas entidades de análisis |
| **Engine — Explore** | `engine/analysis/` | Diagnósticos de distribución, asociaciones, pruebas FDR, traductor de hipótesis | Nuevas pruebas estadísticas y visualizaciones |
| **Application — Explore** | `application/analysis/` | AnalysisStudyService, reporting Markdown, handlers CQRS | Nuevas consultas o generadores |
| **CLI** | `interfaces/cli/` | Subcomandos (incluyendo `catml explore`) | Comandos que deleguen en workspace |
| **Interfaces — MCP** | `interfaces/mcp/` | Servidor MCP y herramientas de análisis (`analysis_tools.py`) | Nuevas herramientas o recursos de agente |
| **Interfaces — Web** | `interfaces/web/` | Workbench HTTP, vista Explore SPA | Vistas SPA siguiendo `docs/design/tech-minimalist-brand-system.md` |

---

## 4. Cómo añadir algo sin romper la arquitectura

### A) Añadir un modelo nuevo

1. Implementar `ModelPluginPort` en `plugins/models/`, o reutilizar `SklearnModelPlugin` con una fábrica de estimadores y capacidades explícitas.
2. Registrar el plugin con `workspace.register_plugin()` al componer la aplicación. Esto actualiza los registros de plugins y modelos y emite `PluginRegistered`.
3. Para un identificador personalizado, implementar `get_search_space()` si se necesita HPO; la fábrica de espacios integrada solo conoce los modelos incorporados.
4. Ejecutar los experimentos mediante `CreateExperimentCommand` y `RunExperimentCommand`; leer resultados mediante queries.
5. Verificar registro, tarea incompatible y entrenamiento real con un test dedicado.

Ejemplo completo: [`examples/plugins/custom_model.py`](examples/plugins/custom_model.py).
Desde la raíz del repositorio:

```bash
.venv/bin/python examples/plugins/custom_model.py --workspace .automl/custom-plugin-demo
.venv/bin/pytest tests/test_plugin_registration.py
```

El ejemplo registra `custom_logistic`, declara sus tareas y espacio de búsqueda y ejecuta un experimento sobre el CSV incluido. La salida contiene una fila de leaderboard con `failed: False`; la métrica exacta depende del entorno.

El registro personalizado vive en memoria: volver a registrar el plugin al abrir otro proceso. La CLI y el servidor crean su propia aplicación y no descubren automáticamente los plugins del script. Incorporar un plugin por defecto requiere registrarlo en la composición de `workspace.py`; la carga por entry points sigue pendiente. El catálogo de tareas enumera modelos incorporados y no necesita cambiar para un plugin registrado explícitamente.

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

## 5. Estado, especificaciones y roadmap

El estado operativo se mantiene exclusivamente en [`TASKS.md`](TASKS.md). La correspondencia entre funcionalidades, implementación y pruebas está en [`docs/README.md`](docs/README.md).

Los planes V0.5–V0.7 describen la secuencia histórica de implementación. La arquitectura objetivo y las fases V0.8–V1.0 están en [`AutoML_Arquitectura_Tecnica.md`](AutoML_Arquitectura_Tecnica.md); no implican que esas capacidades estén disponibles.

---

## 6. Archivos clave por tarea de desarrollo

```text
Quiero…                              → Empieza aquí
─────────────────────────────────────────────────────────
Entender el flujo completo             → workspace.py + bootstrap.py
Añadir tipo de tarea                   → domain/tasks/task_type.py
Cambiar inferencia de tarea            → engine/planning/task_planner.py
Añadir modelo                          → plugins/models/ + workspace.register_plugin()
Cambiar métricas de evaluación         → engine/training/sklearn_trainer.py
Nuevo comando usuario/LLM              → commands/ + bootstrap.py
Nueva consulta UI/LLM                  → queries/ + bootstrap.py
Persistencia / auditoría               → sqlite_repository.py
Modificar UI / Workbench              → interfaces/web/ + docs/design/neo-industrial-ui-spec.md
Medir mejoras entre versiones          → benchmarks/runner.py
Documentación formal                   → AutoML_Arquitectura_Tecnica.md
```

---

## 7. Convenciones del proyecto

1. **Un experimento = una hipótesis verificable.** No monolitos `AutoML.fit()`.
2. **Commands mutan, Queries leen.** Sin efectos secundarios en queries.
3. **Modelos incompatibles con la tarea → ValueError** en `create_experiment`.
4. **Eventos** en toda mutación relevante (`repository.append_event`).
5. **Tests:** agrupar por funcionalidad; añadir pruebas de regresión en la suite relevante.
6. **Versión:** actualizar `automl.__version__` en `src/automl/__init__.py`; empaquetado, CLI y Workbench comparten esa fuente. Actualizar las referencias de versión en la documentación.
7. **Estilo Neo-Industrial en UI:** Toda la interfaz web y paneles de visualización deben seguir rigurosamente el diseño de laboratorio técnico en [`docs/design/neo-industrial-ui-spec.md`](docs/design/neo-industrial-ui-spec.md) (paleta `#111111` / `#D8D6CF` / `#E5512D`, fuentes `Space Grotesk` + `IBM Plex Mono`, bordes de 1px y radio $\le 4$px).

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

Ejemplo histórico de reparto de módulos entre tres desarrolladores; coordinar contratos e integración para reducir conflictos:

| Dev A | Dev B | Dev C |
|-------|-------|-------|
| V0.3 Planner | V0.4 Optuna plugin | V0.5 MI selector |
| `engine/planning/` | `plugins/optimizers/` | `engine/features/selection/` |
| Sin tocar trainer | Sin tocar planner | Sin tocar optimizer |

Punto de integración común: **`workspace.run_experiment()`** y **`bootstrap.register_handlers()`**.
