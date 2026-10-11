# Architecture — CATML

> System organization, module boundaries, dependency rules, and architectural constraints.
> For full technical specification and multi-phase roadmap, see [`docs/AutoML_Arquitectura_Tecnica.md`](docs/AutoML_Arquitectura_Tecnica.md). For developer patterns, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

### Public package vs internal package

| Surface | Name | Role |
| --- | --- | --- |
| PyPI / `pip install` | `catml` | Distribution name |
| Stable public import | `catml` (`src/catml/`) | Re-exports `AutoML`, `ModelArtifact`, `__version__` |
| Internal implementation | `automl` (`src/automl/`) | Hexagonal layers, CQRS, engines, interfaces |
| CLI entry points | `catml` (primary), `automl` (alias) | Both call `automl.interfaces.cli.main:main` |

Prefer `from catml import AutoML` in user code. Application/CQRS extensions import from `automl.*`.

---

## 1. High-Level Architecture & Dependency Rules

CATML implements a **Hexagonal (Ports & Adapters)** architecture combined with **CQRS** (Command Query Responsibility Segregation).

```text
       ┌────────────────────────────────────────────────────────┐
       │   Interfaces (CLI, Workbench HTTP, MCP / Agents)       │
       └───────────────────────────┬────────────────────────────┘
                                   │ (dispatches Commands & Queries)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             Application Layer                          │
       │     (CommandBus, QueryBus, Workspace Orchestrator)     │
       └──────────────┬──────────────────────────┬──────────────┘
                      │                          │
                      ▼                          ▼
       ┌──────────────────────────┐    ┌────────────────────────┐
       │  Domain (Pure Python)    │    │  Engine (Algorithms)   │
       │  - Runs, Datasets        │    │  - Task Planner        │
       │  - Features, Experiments │    │  - Priority Scorer     │
       │  - Models, Optimization  │    │  - Scheduler, Profiler │
       │  - Ports & Protocols     │    │  - Early Stopping      │
       └──────────────▲───────────┘    └────────────┬───────────┘
                      │ (implements ports)          │
         ┌────────────┴─────────────┐               │
         │                          │               │
┌────────┴─────────┐     ┌──────────┴────────┐      │
│ Infrastructure   │     │ Plugins           │◄─────┘
│ - SQLite DB      │     │ - Sklearn Models  │
│ - Event Storage  │     │ - Optuna Optimizer│
└──────────────────┘     └───────────────────┘
```

### Dependency Inversion Rule

- **`domain/`** is the innermost layer. It has **zero dependencies** on external ML libraries or databases.
  - Allowed: standard library (`typing`, `dataclasses`, `uuid`, `enum`, `abc`, `math`).
  - Prohibited: `scikit-learn`, `optuna`, `sqlite3`, `pandas`, `torch`.
- **`application/`** coordinates domain entities and invokes ports. It depends on `domain/`.
- **`engine/`**, **`plugins/`**, and **`infrastructure/`** implement interfaces (ports) defined in `domain/ports.py` or use domain dataclasses.
- **`interfaces/`** communicates strictly with the `application/` layer via `CommandBus` and `QueryBus`.

---

## 2. Directory Layout & Module Responsibilities

```text
src/automl/
├── domain/                  # Core entities, value objects, domain policies, and ports
│   ├── runs/                # AutoMLRun, RunConfig, RunStatus lifecycle
│   ├── tasks/               # TaskType, TASK_CATALOG, ProblemDefinition
│   ├── datasets/            # Dataset reference, metadata, schema
│   ├── features/            # Feature, FeatureSet, FeatureRegistry, FeatureEvidence
│   ├── analysis/            # Explore: StudySpec, StatisticalFinding, EvidenceLink, hypotheses
│   ├── recipes/             # ExperimentRecipe lifecycle (draft → promoted)
│   ├── experiments/         # Experiment, Trial, TrialResult, ExperimentCandidate, Priority
│   ├── models/              # ModelSpec, ModelRegistry, capabilities
│   ├── optimization/        # SearchSpace, ParameterSpec, OptimizationBudget
│   ├── policies/            # BudgetPolicy, EarlyStoppingPolicy
│   ├── modalities/          # Modality and DataSource
│   ├── pipelines/           # Typed PipelineGraph domain contracts
│   └── ports.py             # Interfaces/Protocols (OptimizerPort, ExperimentPlannerPort, etc.)
│
├── application/             # Application coordination, CQRS, and service layer
│   ├── bus/                 # CommandBus, QueryBus implementation
│   ├── commands/            # Typed mutation dataclasses (CreateExperiment, ExcludeFeature, etc.)
│   ├── queries/             # Typed read-only dataclasses (GetLeaderboard, GetTaskPlan, etc.)
│   ├── analysis/            # Explore study service & reporting
│   ├── agents/              # Tool catalog, policies, specialists, orchestrators
│   ├── services/            # Workspace facade + pipeline/inference/feature/recipe services
│   ├── registries/          # Modular CQRS registration (core, experiment, analysis, recipe, …)
│   └── bootstrap.py         # Application wiring, DI container, handler registrations
│
├── engine/                  # Algorithmic and execution engines
│   ├── planning/            # Task inference (task_planner) and experiment generation
│   ├── priority/            # Priority scoring (explainable components) and priority queue
│   ├── optimization/        # SearchSpaceBuilder, RandomSearch, EarlyStopping
│   ├── training/            # Sklearn execution engine for trials (incl. fit-in-fold transforms)
│   ├── analysis/            # Statistical analyzer, associations, hypothesis tests, viz builder
│   ├── profiling/           # Dataset profiling
│   ├── pipeline/            # DAG validation and feature fusion
│   └── vision/              # Image encoding
│
├── plugins/                 # Adapters for concrete external libraries
│   ├── models/              # Sklearn + GBDT plugins (LightGBM/XGBoost/CatBoost extras), ensembles, vision head
│   ├── metrics/             # Business metric adapters
│   ├── modalities/          # Image loading adapter
│   └── optimizers/          # Optuna adapter (OptunaOptimizer with TPESampler)
│
├── infrastructure/          # External persistence and logging
│   └── database/            # SQLite repository (runs, experiments, trials, features, events)
│
├── benchmarks/              # Quality regression suite and history tracking
│   └── runner.py            # Benchmark scenarios measuring Δ vs baseline
│
└── interfaces/              # Boundary entry points
    ├── cli/                 # Terminal commands (`catml` / `automl`)
    ├── web/                 # Workbench HTTP adapter + static frontend
    └── mcp/                 # Model Context Protocol server (stdio / streamable-http)
```

Explore / Recipes / Agents details: [`docs/features/catml-explore/`](docs/features/catml-explore/), [`docs/features/experiment-recipes/`](docs/features/experiment-recipes/), [`docs/features/agentic-system/`](docs/features/agentic-system/). Operational backlog: [`TASKS.md`](TASKS.md).

---

## 3. Public Boundaries & Data Flow

1. **Write Flow (Commands):**
   ```text
   Caller (CLI/Agent)
     → Command (e.g. PlanExperimentsCommand)
     → CommandBus.dispatch()
     → Handler in workspace.py
     → Domain Entities + Engine
     → Repository.append_event() / save()
     → Result / ID returned
   ```
2. **Read Flow (Queries):**
   ```text
   Caller (CLI/Agent)
     → Query (e.g. GetLeaderboardQuery)
     → QueryBus.dispatch()
     → Handler in workspace.py
     → Read from Repository / In-memory state
     → DTO / Dict returned (no mutations)
   ```

---

## 4. Key Architectural Constraints

- **Ergonomic facade ≠ monolithic core:** `AutoML.fit()` is a thin public facade over the same auditable stages (registration, profiling, planning, trial execution, evaluation). Internals stay CQRS/hexagonal; the facade does not bypass buses for agent/CLI parity paths.
- **Propose ≠ Accept:** Planners, optimizers, feature selectors, and agents only propose candidates. Promotion requires measured experiments / explicit promote commands.
- **Human-Agent Parity:** CLI, Workbench, MCP, and Python application clients use the same `CommandBus` / `QueryBus` / workspace services. Architecture guards forbid private repository leaks in interfaces (`tests/test_cqrs_architecture_guard.py`).
- **Auditability & Event Logging:** Mutations record events in SQLite. Artifact provenance captures dependency versions and checksums; remaining per-trial backend limits are documented in [backends.md](docs/backends.md).

---

## 5. Current scope

The directory tree is a summary; [docs/README.md](docs/README.md) maps capabilities to implementation and tests. Operational status lives in [TASKS.md](TASKS.md). Explore, Experiment Recipes, and agent milestones H0–H5 are shipped in `0.9.0-rc1` (see feature plans under `docs/features/`).

**Known debt (tracked in TASKS):** `workspace.py` remains a large facade (~1.8k LOC) that already delegates to application services; further handler extraction is backlog. Advanced meta-learning policies and OOF holdout promotion validation remain open.

## 6. Deeper References

- **Formal Technical Architecture & Roadmap (historical):** [`docs/AutoML_Arquitectura_Tecnica.md`](docs/AutoML_Arquitectura_Tecnica.md)
- **Developer Guide:** [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Architectural Rationale & Historical Design:** [`docs/planning.txt`](docs/planning.txt)
- **Architectural Decision Records:** [`docs/decisions/`](docs/decisions/)
- **Security & trusted deployment:** [`SECURITY.md`](SECURITY.md), [`docs/trusted-deployment.md`](docs/trusted-deployment.md)


## Persistent local execution

Long-running Workbench experiments and submissions use a durable SQLite job queue. Pure domain jobs and `JobRepositoryPort` separate state from storage; application services validate/control jobs and execute existing commands/workspace operations. Bootstrap registers ID-returning job commands and snapshot queries. Infrastructure owns transactional claims and the worker lifecycle; read requests never start workers. The CLI and Workbench share these contracts. See [ADR 003](docs/decisions/003-persistent-local-jobs.md) and the [scope, recovery and controls](docs/features/persistent-jobs/spec.md).
