# CATML

> **The Agent-Native AutoML Engine for Humans and AI Agents.**  
> Local-first, agent-native AutoML with controlled multimodal workflows, built with pure Hexagonal architecture, scikit-learn ergonomics, and Model Context Protocol (MCP) integration.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-595%20Passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-87%25%2B-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/Architecture-Hexagonal%20%2B%20CQRS-orange.svg)](ARCHITECTURE.md)
[![MCP](https://img.shields.io/badge/MCP-Ready-purple.svg)](src/automl/interfaces/mcp/)

---

## Why CATML?

Most AutoML platforms (AutoGluon, H2O, FLAML, PyCaret) are monolithic libraries or proprietary cloud silos designed before the rise of autonomous AI coding agents. 

CATML is engineered from the ground up with **distinct advantages**:

1. **🤖 Agent-Native by Design (MCP Server):** Native Model Context Protocol (MCP) dual transport (stdio/HTTP) enables external AI agents (Claude, Cursor, OpenAI) to discover features, run trials, and explore hypotheses within strict budget leases and cooperative cancellation controls.
2. **⚡ 3-Line Ergonomics:** Simple, intuitive scikit-learn interface (`automl.fit(df, target="churn")`) backed by a pure hexagonal domain and CQRS application buses.
3. **📦 Standalone, Database-Free Deployment:** Winning models serialize into portable `model.pkl` artifacts with streaming SHA-256 integrity checksums for database-free, workspace-independent production inference.
4. **🛡️ "Propose ≠ Accept" Scientific Discipline:** Empirical trial evaluation with out-of-fold (OOF) target encoding and strict data leakage prevention before promoting any model to production.
5. **🔒 100% Local-First & Private:** Executes entirely on your machine with a persistent SQLite task worker; zero telemetry, zero forced cloud lock-in.

---

## Quickstart (Python API)

Train multiple candidate models, compare the leaderboard, and export a production-ready model in seconds:

```python
import pandas as pd
from catml import AutoML, ModelArtifact

# 1. Load any tabular dataset
df = pd.read_csv("examples/data/customers_churn.csv")

# 2. Fit candidate models (automatically detects task, tunes hyperparameters)
automl = AutoML(task="classification")
result = automl.fit(df, target="churn")

# 3. View the experiment leaderboard
print(result.leaderboard())

# 4. Save standalone winning model for production
result.save_model("model.pkl")

# 5. Load and predict anywhere (database-free and workspace-independent)
model = ModelArtifact.load("model.pkl")
predictions = model.predict(df.head(5))
probabilities = model.predict_proba(df.head(5))
```

---

## Quickstart (Command Line)

Train directly from your terminal with automated leaderboard output and artifact generation:

```bash
# Fit models on a dataset and export winning artifact
catml fit examples/data/customers_churn.csv --target churn

# Output results as structured JSON
catml fit data.csv --target churn --models logistic_regression,random_forest --json
```

Output:
```text
=================================================================
  CATML Automated Machine Learning (Platform V0.8.0)
=================================================================
  Dataset:     examples/data/customers_churn.csv
  Target:      churn
  Candidates:  logistic_regression, random_forest, lightgbm

  Evaluating candidate models and tuning...

  Leaderboard:
  ------------------------------------------------------------
  Rank  Model                   Metric      Score     Time (s)  
  ------------------------------------------------------------
  1     lightgbm                roc_auc     0.8924    0.28      
  2     random_forest           roc_auc     0.8750    0.16      
  3     logistic_regression     roc_auc     0.8512    0.12      
  ------------------------------------------------------------

  ✓ Best Model:   lightgbm (roc_auc: 0.8924)
  ✓ Model Saved:  /home/user/project/catml-runs/model.pkl
  ✓ Run in Prod:  ModelArtifact.load("catml-runs/model.pkl")
=================================================================
```

---

## Interactive Web Workbench (Visual ML Lab)

CATML includes a built-in interactive Tech Minimalista Premium visual dashboard for monitoring runs, inspecting datasets, and managing experiments:

```bash
catml ui --port 8080 --workspace .automl/demo
```

Open **[http://localhost:8080](http://localhost:8080)** in your browser to access:
- **Mission Control Overview:** Live experiment status and resource telemetry.
- **Dataset Inspector:** Deep statistical profiling, cardinality detection, and feature recommendations.
- **Visual Experiment Studio:** Interactive comparison diffs, HPO trial history, and model explainability.
- **Persistent Job Queue:** Resilient SQLite worker with pause, resume, and cooperative cancellation.

---

## 🤖 Agentic Protocol: Model Context Protocol (MCP)

CATML exposes its entire AutoML capability directly to LLMs via the Model Context Protocol (MCP).

### Connect to Claude Desktop or Cursor

Add CATML to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "catml": {
      "command": "catml",
      "args": ["mcp", "--transport", "stdio"]
    }
  }
}
```

Or start the streamable HTTP transport server:

```bash
catml mcp --transport streamable-http --port 8000
```

Available tools exposed to AI agents:
- `get_dataset_profile`: Deep statistical profiling and column cardinality.
- `list_models` / `list_plugins`: Inspect available algorithms and metric plugins.
- `create_experiment`: Propose targeted hypothesis-driven experiment runs.
- `get_leaderboard`: Inspect ranked metric scores and cross-validation variance.
- `cancel_operation`: Cooperative cancellation under budget constraints.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Jfenic/CATML.git
cd CATML

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"
```

### Optional Modality & Agentic Extensions

Install optional capability packages as needed for your workload:

```bash
# Gradient boosting backends (LightGBM, XGBoost, CatBoost)
pip install "catml[models]"

# Deep learning vision representations (PyTorch, torchvision, timm, Pillow)
pip install "catml[vision]"

# Agentic orchestration & MCP server (LangGraph, MCP)
pip install "catml[mcp,agents]"

# Full stack with all extensions
pip install "catml[all]"
```

> [!NOTE]
> **Local-First & Offline Environments:** CATML executes completely local-first without transmitting dataset rows to external cloud services. However, when using deep neural vision backbones (`resnet18`, etc.), TorchVision downloads pretrained weights upon first execution if not already cached. For strictly offline or air-gapped environments, pre-populate the PyTorch cache directory (`~/.cache/torch/hub/checkpoints`).

---

## Advanced Architecture (CQRS Application Bus)

For power users, MLOps pipelines, and custom orchestrators, CATML provides strict CQRS separation via `CommandBus` and `QueryBus`:

```python
from automl.application.bootstrap import build_application
from automl.application.commands.workspace_commands import (
    PlanExperimentsCommand,
    RunScheduledExperimentsCommand,
)
from automl.application.queries.workspace_queries import GetLeaderboardQuery

# Build application workspace
ws, cmd, qry = build_application(root_dir=".automl/demo")

# Register dataset and create run
dataset = ws.register_dataset(
    name="churn",
    path="examples/data/customers_churn.csv",
    target="churn",
)
run = ws.create_run(dataset)

# Dispatch CQRS commands
cmd.dispatch(PlanExperimentsCommand(run_id=run.id))
cmd.dispatch(RunScheduledExperimentsCommand(run_id=run.id, max_experiments=3))

# Dispatch zero-side-effect queries
leaderboard = qry.dispatch(GetLeaderboardQuery(run.id))
for entry in leaderboard:
    print(f"{entry['model_id']}: {entry['metric']} = {entry['score']:.4f}")
```

---

## How to Test

```bash
# Run all unit and integration tests
pytest

# Run tests with strict coverage validation (>= 85%)
pytest --cov=src/automl --cov-fail-under=85
```

---

## Capabilities & Modality Status

| Modality / Workflow | Status | Details |
|---|---|---|
| **Tabular Classification & Regression** | **Stable** | LightGBM, XGBoost, CatBoost, Scikit-learn, Voting Ensembles, N-Model OOF Blending, Level-2 Stacking, Temporal Dynamics, and Group Anti-Leakage. |
| **Text Representations (NLP)** | **Available** | Sublinear TF-IDF n-gram extractor with automated `is_text_column` discovery and multimodal column fusion. |
| **Image Representations (Vision)** | **Available** | Representation-level vision with PyTorch/torchvision backbone (`catml[vision]`, feature extractor + downstream estimators), deterministic hash encoder (testing/benchmarks fallback), image path discovery, and gallery preview in Workbench. |
| **Agent Governance (MCP & LangGraph)** | **Available** | Native Model Context Protocol (stdio/HTTP), LangGraph StateGraph orchestrator, persistent SQLite ledger, human approval flow, and token auditing. |
| **Native Vision (Detection / Segmentation)** | *Roadmap* | Planned for future v1.x with specialized deep learning backends. |
| **Biomedical Extensions (`catml[medical]`)** | *Roadmap* | Planned extension for DICOM/NIfTI with strict zero data egress. |

---

## Documentation Navigation

- **System Architecture & Hexagonal Rules:** [`ARCHITECTURE.md`](ARCHITECTURE.md)
- **Technical Specification & Roadmap:** [`AutoML_Arquitectura_Tecnica.md`](AutoML_Arquitectura_Tecnica.md)
- **Active Tasks & Operational Backlog:** [`TASKS.md`](TASKS.md)
- **Agent Rules & Development Guidelines:** [`AGENTS.md`](AGENTS.md)
- **Developer Extension Guide:** [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md)
- **Feature Specifications & Plans:** [`docs/features/`](docs/features/)
- **Architectural Decision Records (ADRs):** [`docs/decisions/`](docs/decisions/)

---

## License
 
Starting with version 0.8.0, CATML is licensed under the Apache License, Version 2.0 - see the [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) files for details. Earlier releases (<= 0.7.0) remain available under the MIT License.
