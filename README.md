# AutoML Platform (CATML)

A modular, hexagonal-architecture AutoML platform designed for reproducible machine learning experimentation. The core mission is not just training a model, but managing in an auditable and reproducible way **what task is being solved**, **which models apply**, **which features are selected**, and **which experiments are executed** — exposing the exact same API to human developers, the CLI, and future LLM agents.

**Current Version:** `0.3.0`  
**Status:** Phases V0.1, V0.2, and V0.3 completed and verified. V0.4 (Optuna optimization) in progress/next.

---

## Requirements

- Python 3.10+
- Core dependencies: `numpy`, `pandas`, `scikit-learn`, `pyyaml`

## Installation

```bash
cd CATML
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## How It Works (Mental Workflow)

```text
1. Register dataset & compute profile
2. Infer or define problem task (classification / regression / clustering)
3. Retrieve compatible model catalog for the task
4. Configure features and models (Commands)
5. Automatically plan experiments (V0.3 Planner + Priority Scorer)
6. Execute Experiment → Trial → metrics via Priority Scheduler
7. Compare outcomes (leaderboard, benchmark harness)
```

Every state mutation goes through the **CommandBus** (write) and every retrieval through the **QueryBus** (read). The domain is pure Python and never imports `scikit-learn` or `sqlite3` directly.

---

## CLI Usage & Examples

### 1. View task catalog and compatible models

```bash
automl task list
```

Displays the models, metrics, and default loss/evaluation functions for each task type.

### 2. Infer problem definition from dataset

```bash
automl task plan --dataset examples/data/customers_churn.csv --target churn
```

Automatic inference: `binary_classification` → compatible models: `logistic_regression`, `random_forest`, `svc` → metric: `roc_auc`.

### 3. End-to-end demo (Interactive & Automated)

```bash
# Classic interactive workflow
automl run-demo

# V0.3 automated workflow (RuleBasedExperimentPlanner + Priority Scheduler)
automl run-demo --auto
```

### 4. Automated Experiment Planning & Priority Queue (V0.3)

```bash
automl plan-experiments --dataset examples/data/customers_churn.csv --target churn --auto-run
```

Proposes candidates (fast baseline, alternative model exploration, prioritized feature subsets) with explainable priority scoring and dispatches them in scheduled order.

### 5. Quality Regression Benchmark (V0.1 → V0.2 → V0.3)

```bash
automl benchmark run
automl benchmark history
```

Runs 6 distinct scenarios, computes **Δ vs baseline**, and stores historical records in SQLite.

### 6. Running Tests

```bash
pytest
```

---

## Programmatic API Example

```python
from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    CreateExperimentCommand,
    ExcludeFeatureCommand,
    PlanExperimentsCommand,
    RunScheduledExperimentsCommand,
)
from automl.application.queries.workspace_queries import (
    GetLeaderboardQuery,
    GetTaskPlanQuery,
    ListModelsQuery,
)

ws, cmd, qry = build_application(root_dir=".automl/my-project")

# 1. Register dataset & inspect task plan
dataset = ws.register_dataset(
    name="churn",
    path="examples/data/customers_churn.csv",
    target="churn",
)
plan = qry.dispatch(GetTaskPlanQuery(dataset.id))
print(plan.task_type.value, plan.recommended_models)

# 2. Create Run and configure user constraints
run = ws.create_run(dataset)
cmd.dispatch(ExcludeFeatureCommand(dataset.id, "customer_id", run_id=run.id))

# 3. Automatically plan and execute experiments
candidates = cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
executed = cmd.dispatch(RunScheduledExperimentsCommand(run_id=run.id, max_experiments=2))

# 4. Inspect leaderboard
leaderboard = qry.dispatch(GetLeaderboardQuery(run.id))
for item in leaderboard:
    print(f"{item['model_id']}: {item['metric']}={item['score']:.4f}")
```

---

## Supported Task Types and Models

| Task Type | Models | Default Metric |
|---|---|---|
| `binary_classification` | `logistic_regression`, `random_forest`, `svc` | `roc_auc` |
| `multiclass_classification` | `logistic_regression`, `random_forest`, `svc` | `accuracy` |
| `regression` | `ridge`, `random_forest`, `svr` | `r2` |
| `clustering` | `kmeans`, `agglomerative`, `dbscan` | `silhouette` |

Defined in [`src/automl/domain/tasks/task_type.py`](src/automl/domain/tasks/task_type.py) (`TASK_CATALOG`).

---

## Project Structure

```text
CATML/
├── src/automl/
│   ├── domain/           # Pure entities (Run, Experiment, Candidate, Priority, TaskType, FeatureSet)
│   ├── application/      # CommandBus, QueryBus, workspace orchestration, bootstrap
│   ├── engine/           # Profiling, rule-based planner, priority scorer, scheduler, trainer
│   ├── plugins/models/   # Scikit-learn model adapters by task
│   ├── infrastructure/   # SQLite repository and event logging
│   ├── benchmarks/       # Scenario runner and quality regression history
│   └── interfaces/cli/   # CLI subcommands
├── tests/                # test_v01, test_v02, test_v03_planner, test_tasks, test_benchmark
├── examples/data/        # Sample datasets (customers_churn.csv)
├── AutoML_Arquitectura_Tecnica.md   # Formal architecture spec + roadmap V0.1–V1.0
├── DEVELOPER_GUIDE.md    # Developer guide for extending the codebase
└── planning.txt          # Architectural design notes
```

---

## Implementation Status

| Phase | Status | Capabilities |
|---|---|---|
| **V0.1** | ✅ | Pure domain, Experiment/Trial, SQLite persistence, Sklearn trainer, initial demo |
| **V0.2** | ✅ | CQRS buses, FeatureSet, pause/resume/cancel/clone, ownership validation, benchmark, task catalog |
| **V0.3** | ✅ | RuleBasedExperimentPlanner, Priority Engine with explainable scoring, Priority Queue with pinned support, BudgetPolicy, Scheduler, CLI `--auto` |
| **V0.4** | ⏳ | Model & Hyperparameter Optimization (Optuna adapter, SearchSpace, EarlyStopping) |
| **V0.5** | 📋 | Feature Discovery & Selection (SHAP, Mutual Information, L1, RFE, Ablation studies) |
| **V0.6** | 📋 | Plugin ecosystem (LightGBM, XGBoost, CatBoost, custom metrics) |
| **V0.7** | 📋 | Multimodal pipelines (Image encoders, vector embeddings, tabular fusion) |
| **V0.8** | 📋 | Meta-Learning & Knowledge Base (dataset meta-features, warm-start policies) |
| **V0.9–V1.0** | 📋 | LLM Tools (`AgentTool` wrappers) and Autonomous LLM Agents |

---

## Documentation

- **Formal Technical Architecture:** [`AutoML_Arquitectura_Tecnica.md`](AutoML_Arquitectura_Tecnica.md)
- **Developer Guide:** [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Design Notes:** [`planning.txt`](planning.txt)

---

## Sample Dataset

`examples/data/customers_churn.csv` — 500 rows, columns: `customer_id`, `salary`, `debt`, `age`, `tenure`, `account_balance`, `country`, `churn`.

---

## Core Architectural Principle

> The human user interface, the CLI, the future REST API, and autonomous LLM agents operate over the exact same **Commands** and **Queries**. The core domain remains pure Python without external ML or DB dependencies. ML libraries and storage backends are swappable plugins.
