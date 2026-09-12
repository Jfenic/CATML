# AutoML Platform (CATML)

Plataforma AutoML modular con arquitectura hexagonal. El objetivo no es solo entrenar un modelo, sino gestionar de forma reproducible **qué tarea se resuelve**, **qué modelos aplican**, **qué features se usan** y **qué experimentos se ejecutan** — con la misma API para humanos, CLI y futuros agentes LLM.

**Versión actual:** `0.3.0`  
**Estado:** V0.1 + V0.2 + V0.3 cerrados y verificados. V0.4 (optimización Optuna) pendiente.

---

## Requisitos

- Python 3.10+
- Dependencias: `numpy`, `pandas`, `scikit-learn`, `pyyaml`

## Instalación

```bash
cd CATML
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Cómo funciona (flujo mental)

```text
1. Registrar dataset
2. Planificar tarea (clasificación / regresión / clustering)
3. Obtener catálogo de modelos compatibles con esa tarea
4. Configurar features y modelos (Commands)
5. Planificar experimentos automáticamente (V0.3 Planner + Priority Scorer)
6. Ejecutar Experiment → Trial → métricas mediante Priority Scheduler
7. Comparar resultados (leaderboard, benchmark)
```

Cada paso pasa por **CommandBus** (escritura) o **QueryBus** (lectura). El dominio no conoce sklearn ni SQLite directamente.

---

## Ejemplo completo (CLI)

### 1. Ver catálogo de tareas y modelos

```bash
automl task list
```

Muestra qué modelos y métricas aplican a cada tipo de tarea.

### 2. Planificar tarea desde el dataset

```bash
automl task plan --dataset examples/data/customers_churn.csv --target churn
```

Inferencia automática: `binary_classification` → modelos `logistic_regression`, `random_forest`, `svc` → métrica `roc_auc`.

### 3. Demo end-to-end (manual y automático)

```bash
# Modo interactivo clásico
automl run-demo

# Modo V0.3 con Experiment Planner y Priority Scheduler automático
automl run-demo --auto
```

### 4. Planificador automático y cola de prioridades (V0.3)

```bash
automl plan-experiments --dataset examples/data/customers_churn.csv --target churn --auto-run
```

Genera candidatos (baseline, modelos alternativos, subconjuntos de features) con scoring explicable y los despacha según prioridad.

### 5. Benchmark de mejoras (V0.1 → V0.2 → V0.3)

```bash
automl benchmark run
automl benchmark history
```

Compara 6 escenarios y guarda resultados en SQLite con **Δ vs baseline**.

### 6. Tests

```bash
pytest
```


---

## Ejemplo programático

```python
from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    ExcludeFeatureCommand,
    RunExperimentCommand,
)
from automl.application.queries.workspace_queries import (
    GetTaskPlanQuery,
    GetLeaderboardQuery,
    ListModelsQuery,
)

ws, cmd, qry = build_application(root_dir=".automl/mi-proyecto")

# 1. Dataset + plan de tarea
dataset = ws.register_dataset(
    name="churn",
    path="examples/data/customers_churn.csv",
    target="churn",
)
plan = qry.dispatch(GetTaskPlanQuery(dataset.id))
print(plan.task_type.value, plan.recommended_models)  # binary_classification, [...]

# 2. Run + control humano
run = ws.create_run(dataset)
cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))

models = qry.dispatch(ListModelsQuery(run_id=run.id))  # solo compatibles con la tarea

# 3. Experimento
experiment = cmd.dispatch(
    CreateExperimentCommand(
        run_id=run.id,
        name="baseline",
        model_ids=["logistic_regression", "random_forest"],
        priority="high",
    )
)
results = cmd.dispatch(RunExperimentCommand(run.id, experiment.id))
leaderboard = qry.dispatch(GetLeaderboardQuery(run.id))
```

---

## Tipos de tarea y modelos

| Tarea | Modelos | Métrica default |
|-------|---------|-----------------|
| `binary_classification` | logistic_regression, random_forest, svc | roc_auc |
| `multiclass_classification` | logistic_regression, random_forest, svc | accuracy |
| `regression` | ridge, random_forest, svr | r2 |
| `clustering` | kmeans, agglomerative, dbscan | silhouette |

Definido en `src/automl/domain/tasks/task_type.py` (`TASK_CATALOG`).

---

## Estructura del proyecto

```text
CATML/
├── src/automl/
│   ├── domain/           # Entidades puras (Run, Experiment, TaskType, FeatureSet)
│   ├── application/      # CommandBus, QueryBus, workspace, bootstrap
│   ├── engine/           # Profiling, task planner, sklearn trainer
│   ├── plugins/models/   # Adaptadores sklearn por tarea
│   ├── infrastructure/   # SQLite repository
│   ├── benchmarks/       # Harness de comparación entre versiones
│   └── interfaces/cli/   # CLI
├── tests/                # test_v01, test_v02, test_tasks, test_benchmark
├── examples/data/        # customers_churn.csv
├── AutoML_Arquitectura_Tecnica.md   # Spec formal + roadmap V0.1–V1.0
├── DEVELOPER_GUIDE.md    # Guía para continuar el desarrollo
└── planning.txt          # Notas de arquitectura (conversacional)
```

---

## Qué está implementado

| Fase | Estado | Capacidades |
|------|--------|-------------|
| **V0.1** | ✅ | Dominio, Experiment/Trial, SQLite, trainer sklearn, demo |
| **V0.2** | ✅ | CQRS, FeatureSet, pause/resume, validación de ownership, benchmark, catálogo por tarea |
| **V0.3** | ⏳ | Experiment Planner + Priority Engine |
| **V0.5** | 📋 | Feature Discovery & Selection (documentado) |
| **V0.9–V1.0** | 📋 | LLM tools + agente |

---

## Documentación

- **Arquitectura formal:** [AutoML_Arquitectura_Tecnica.md](AutoML_Arquitectura_Tecnica.md)
- **Guía para desarrolladores:** [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)
- **Notas de diseño:** [planning.txt](planning.txt)

---

## Dataset de ejemplo

`examples/data/customers_churn.csv` — 500 filas, columnas: `customer_id`, `salary`, `debt`, `age`, `tenure`, `account_balance`, `country`, `churn`.

---

## Principio rector

> La interfaz humana, la CLI, la API y el futuro LLM consumen los mismos **Commands** y **Queries**. El dominio permanece puro; las librerías ML son plugins.
