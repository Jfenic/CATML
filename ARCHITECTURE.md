# Architecture — CATML

> System organization, module boundaries, dependency rules, and architectural constraints.
> For full technical specification and multi-phase roadmap, see [`AutoML_Arquitectura_Tecnica.md`](AutoML_Arquitectura_Tecnica.md). For developer patterns, see [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

---

## 1. High-Level Architecture & Dependency Rules

CATML implements a **Hexagonal (Ports & Adapters)** architecture combined with **CQRS** (Command Query Responsibility Segregation).

```text
       ┌────────────────────────────────────────────────────────┐
       │             Interfaces (CLI, Future API, LLM)          │
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
│   ├── experiments/         # Experiment, Trial, TrialResult, ExperimentCandidate, Priority
│   ├── models/              # ModelSpec, ModelRegistry, capabilities
│   ├── optimization/        # SearchSpace, ParameterSpec, OptimizationBudget
│   ├── policies/            # BudgetPolicy, EarlyStoppingPolicy
│   └── ports.py             # Interfaces/Protocols (OptimizerPort, ExperimentPlannerPort, etc.)
│
├── application/             # Application coordination, CQRS, and service layer
│   ├── bus/                 # CommandBus, QueryBus implementation
│   ├── commands/            # Typed mutation dataclasses (CreateExperiment, ExcludeFeature, etc.)
│   ├── queries/             # Typed read-only dataclasses (GetLeaderboard, GetTaskPlan, etc.)
│   ├── services/workspace.py# AutoMLWorkspace high-level facade
│   └── bootstrap.py         # Application wiring, DI container, handler registrations
│
├── engine/                  # Algorithmic and execution engines
│   ├── planning/            # Task inference (task_planner) and experiment generation
│   ├── priority/            # Priority scoring (explainable components) and priority queue
│   ├── optimization/        # SearchSpaceBuilder, RandomSearch, EarlyStopping
│   ├── training/            # Sklearn execution engine for trials
│   ├── profiling/           # Dataset profiling
│   └── evaluation/          # Evaluation metric calculation
│
├── plugins/                 # Adapters for concrete external libraries
│   ├── models/              # Scikit-learn model adapters (logistic_regression, random_forest, svc)
│   └── optimizers/          # Optuna adapter (OptunaOptimizer with TPESampler)
│
├── infrastructure/          # External persistence and logging
│   └── database/            # SQLite repository (runs, experiments, trials, features, events)
│
├── benchmarks/              # Quality regression suite and history tracking
│   └── runner.py            # Benchmark scenarios measuring Δ vs baseline
│
└── interfaces/              # Boundary entry points
    └── cli/                 # Terminal commands (task, run-demo, plan-experiments, optimize, benchmark)
```

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

- **No Monolithic `AutoML.fit()`:** CATML decomposes AutoML into explicit, auditable stages (registration, profiling, problem definition, feature configuration, experiment planning, trial execution, evaluation, optimization).
- **Propose ≠ Accept:** Planners, optimizers, and feature selectors only propose candidates. Candidates must run through the execution engine and yield recorded metrics before being promoted.
- **Human-Agent Parity:** Every operation accessible to a human via CLI or UI must be exposed through the exact same application command/query buses for future LLM agents.
- **Auditability & Event Logging:** Every state change emits domain events persisted in SQLite, ensuring full reproducibility.

---

## 5. Deeper References

- **Formal Technical Architecture & Roadmap (V0.1–V1.0):** [`AutoML_Arquitectura_Tecnica.md`](AutoML_Arquitectura_Tecnica.md)
- **Developer Guide:** [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Architectural Rationale & Historical Design:** [`planning.txt`](planning.txt)
- **Architectural Decision Records:** [`docs/decisions/`](docs/decisions/)
