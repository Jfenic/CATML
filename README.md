# AutoML Platform (CATML)

> A modular, reproducible AutoML platform built with a pure domain and hexagonal architecture, ensuring parity between human users, CLI and web interfaces, with contracts for future AI agents.

CATML is designed for reproducible machine learning experimentation. Rather than acting as a black-box optimizer, it manages in an auditable, reproducible manner **what task is being solved**, **which models apply**, **which features are selected**, and **which experiments are executed**.

---

## Main Goals

- **Experiment-First Philosophy:** Model experiments and trials as first-class domain entities, maintaining full reproducibility and auditability.
- **Human & AI Agent Parity:** Expose identical Command and Query capabilities across the CLI and Workbench HTTP API, with future LLM tools using the same application layer.
- **Strict Separation of Concerns:** Keep core domain logic pure and independent of ML frameworks (scikit-learn, Optuna, PyTorch) or persistence backends (SQLite, Postgres).
- **Hypothesis-Driven Automation ("Propose ≠ Accept"):** Require that all candidate features, models, and hyperparameters be empirically evaluated and compared against baselines before acceptance.
- **Agentic Protocol & AI Parity:** Expose full AutoML capabilities to external AI coding agents via standard Model Context Protocol (MCP) and dedicated CLI commands, backed by strict budget controls, atomic operation leases, and cooperative cancellation.

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
# From your local checkout
cd CATML

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

---

## Optional model backends

The base installation includes scikit-learn. To use native gradient boosting backends:

```bash
python -m pip install lightgbm xgboost
```

These packages are optional. If absent, `LightGBMPlugin` uses scikit-learn HistGradientBoosting and `XGBoostPlugin` uses GradientBoosting. A registered plugin ID alone does not prove that the native library ran. See [backend verification and reproducibility](docs/backends.md) for a check and how to disable fallback.

## How to Run

### Web Workbench

```bash
automl ui --port 8080 --workspace .automl/demo
```

The Workbench executes experiments and submissions using a persistent SQLite queue and a local worker. The 'Jobs' panel maintains run state across reloads and provides pause, cancellation, and retry capabilities. See [persistent queue, CLI, and API](docs/features/persistent-jobs/spec.md) for controls, recovery, and boundaries.

Open <http://localhost:8080>. The workspace holds the SQLite history for runs, experiments and results; use the same path in CLI commands to inspect that history. Stop the server with Ctrl+C. Without `--workspace`, the CLI selects `.automl/s6e9_automl` if it exists, otherwise `.automl/demo`.

The current server binds to `0.0.0.0` and has no authentication; it is intended for a trusted development environment. Overview, datasets, experiment execution, comparison and submission generation use application data. Knowledge and agent panels include illustrative preview responses; they do not establish a working meta-learning or autonomous-agent capability. See [capabilities and limitations](docs/README.md).

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

# 6. View registered model, metric, and optimizer plugins
automl plugin list

# 7. Generate Kaggle-ready submission matching exact template IDs and columns
automl predict --run-id <RUN_ID> --test-dataset data/test.csv --template data/sample_submission.csv --proba --output submission.csv

# 8. Evaluate binary OOF blending and generate a fold-averaged submission
automl predict --workspace .automl/demo --run-id <RUN_ID> --test-dataset data/test.csv --folds 5 --models logistic_regression,random_forest --proba --output submission_oof.csv

# 9. Run regression benchmark suite
automl benchmark run

# 10. Inspect and manage agentic operations (cooperative cancellation, lease tracking)
automl agent operations list
automl agent operations get <OPERATION_ID>
automl agent operations cancel <OPERATION_ID>

# 11. Run Model Context Protocol (MCP) server for external AI agents
automl mcp --transport stdio
automl mcp --transport streamable-http --port 8000
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
pytest --cov=src/automl --cov-fail-under=85
```

---

## Parallel Multi-Agent Development & Collaboration

CATML supports concurrent engineering across multiple developers and autonomous AI coding agents (such as Persona A focusing on application rules, specialists, and domain policies, and Persona B focusing on orchestration, interfaces, and protocol servers).

To maintain codebase stability, prevent git merge conflicts, and preserve auditability:
- **Contract-First Stabilization:** Before branching into parallel feature tracks, shared interfaces, protocols, and data transfer objects (DTOs) are agreed upon and merged into `main`.
- **Subdirectory Ownership:** Development tracks work within isolated subdirectories (e.g., domain specialists vs CLI/orchestrators) without modifying monolithic core files concurrently.
- **Merge-Only Integration:** Branch synchronization is performed strictly using `git merge origin/main`; rebasing is forbidden.
- **Independent Progress Tracking:** Progress documentation maintains designated tracks for each persona to avoid adjacent-line merge conflicts.
- **Cross-Agent Blackboard:** Tasks begin with a pre-flight inspection (`gh issue list --label blackboard --state open`), and out-of-scope discoveries are recorded via GitHub issues rather than ad-hoc edits.

For detailed guidelines, see [`AGENTS.md`](AGENTS.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and [`docs/features/agentic-system/two-person-plan.md`](docs/features/agentic-system/two-person-plan.md).

---

## Deeper Documentation

Version: `0.7.0`, defined in [`src/automl/__init__.py`](src/automl/__init__.py). Phase labels in the design roadmap describe milestones. Current status belongs to [`TASKS.md`](TASKS.md).

- **Documentation Map & Capability Evidence:** [`docs/README.md`](docs/README.md)
- **Contributing & Parallel Teamwork:** [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Agent Instructions & Rules:** [`AGENTS.md`](AGENTS.md)
- **System Architecture & Boundaries:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Active Tasks & Operational Roadmap:** [`TASKS.md`](TASKS.md)
- **Developer Guide:** [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Comprehensive Technical Specification (V0.1–V1.0):** [`AutoML_Arquitectura_Tecnica.md`](AutoML_Arquitectura_Tecnica.md)
- **Design Notes & Rationale:** [`planning.txt`](planning.txt)
- **Feature Specifications & Historical Plans:** [`docs/features/`](docs/features/)
- **Architecture Decision Records (ADRs):** [`docs/decisions/`](docs/decisions/)

---

## License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.
