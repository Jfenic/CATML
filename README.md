# AutoML Platform (CATML)

> A modular, reproducible AutoML platform built with a pure domain and hexagonal architecture, ensuring parity between human users, CLI interfaces, and AI coding agents.

CATML is designed for reproducible machine learning experimentation. Rather than acting as a black-box optimizer, it manages in an auditable, reproducible manner **what task is being solved**, **which models apply**, **which features are selected**, and **which experiments are executed**.

---

## Main Goals

- **Experiment-First Philosophy:** Model experiments and trials as first-class domain entities, maintaining full reproducibility and auditability.
- **Human & AI Agent Parity:** Expose identical Command and Query capabilities across the CLI, future APIs, and LLM agent tool interfaces.
- **Strict Separation of Concerns:** Keep core domain logic pure and independent of ML frameworks (scikit-learn, Optuna, PyTorch) or persistence backends (SQLite, Postgres).
- **Hypothesis-Driven Automation ("Propose ≠ Accept"):** Require that all candidate features, models, and hyperparameters be empirically evaluated and compared against baselines before acceptance.

---

## Technology Stack

- **Language:** Python 3.10+
- **Core ML & Data:** `numpy`, `pandas`, `scikit-learn`
- **Hyperparameter Optimization:** `optuna`
- **Configuration & Storage:** `pyyaml`, `sqlite3`
- **Testing & Quality:** `pytest`, `pytest-cov`

---

## Installation

```bash
# Clone and navigate to repository
cd CATML

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## How to Run

### Command Line Interface (CLI)

```bash
# 1. View task catalog and compatible models
automl task list

# 2. Automatically plan problem definition from dataset
automl task plan --dataset examples/data/customers_churn.csv --target churn

# 3. Run automated experiment planning and priority scheduling
automl plan-experiments --dataset examples/data/customers_churn.csv --target churn --auto-run

# 4. Hyperparameter tuning using Optuna TPE sampler
automl optimize --dataset examples/data/customers_churn.csv --target churn --model logistic_regression --optimizer optuna --trials 10

# 5. Run end-to-end demo
automl run-demo --auto

# 6. Run regression benchmark suite
automl benchmark run
```

### Programmatic Python API

```python
from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    PlanExperimentsCommand,
    RunScheduledExperimentsCommand,
)
from automl.application.queries.workspace_queries import GetLeaderboardQuery

# Build application workspace
workspace, command_bus, query_bus = build_application(root_dir=".automl/demo")

# Register dataset and create run
dataset = workspace.register_dataset(
    name="churn",
    path="examples/data/customers_churn.csv",
    target="churn",
)
run = workspace.create_run(dataset)

# Plan and execute prioritized experiments
command_bus.dispatch(PlanExperimentsCommand(run_id=run.id))
command_bus.dispatch(RunScheduledExperimentsCommand(run_id=run.id, max_experiments=2))

# Inspect results
leaderboard = query_bus.dispatch(GetLeaderboardQuery(run.id))
for entry in leaderboard:
    print(f"{entry['model_id']}: {entry['metric']} = {entry['score']:.4f}")
```

---

## How to Test

```bash
# Run all tests
pytest

# Run tests with coverage report (target >= 85%)
pytest --cov=src/automl
```

---

## Deeper Documentation

- **Contributing & Parallel Teamwork:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Agent Instructions & Rules:** [`AGENTS.md`](AGENTS.md)
- **System Architecture & Boundaries:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Active Tasks & Operational Roadmap:** [`TASKS.md`](TASKS.md)
- **Developer Guide:** [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Comprehensive Technical Specification (V0.1–V1.0):** [`AutoML_Arquitectura_Tecnica.md`](AutoML_Arquitectura_Tecnica.md)
- **Design Notes & Rationale:** [`planning.txt`](planning.txt)
- **Active Feature Specifications & Plans:** [`docs/features/`](docs/features/)
- **Architecture Decision Records (ADRs):** [`docs/decisions/`](docs/decisions/)
