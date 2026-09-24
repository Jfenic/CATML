# Decision: Hexagonal Architecture and CQRS for Parity between Human and AI Agents

## Context

Most AutoML frameworks (e.g. auto-sklearn, TPOT, AutoGluon) implement monolithic routines like `AutoML.fit(X, y)` that perform data profiling, preprocessing, feature selection, model search, tuning, and ensembling in an opaque pipeline. 

CATML requires:
1. Granular human control: ability to pause, resume, prioritize individual features, exclude models, and launch specific experiments.
2. Full parity with future AI coding and reasoning agents: agents must execute the exact same operations as human CLI/UI users without bypassing business logic or invoking external ML libraries directly.
3. Extensibility: freedom to swap ML engines (sklearn, LightGBM, PyTorch), storage layers (SQLite, Postgres), and optimizers (Optuna, Ray) without modifying core domain rules.

---

## Decision

We chose a **Hexagonal (Ports & Adapters)** architecture with **CQRS (Command Query Responsibility Segregation)**:
- The core **`domain/`** is pure Python (no scikit-learn, Optuna, or SQLite imports).
- All mutations occur via typed **Commands** dispatched through a `CommandBus`.
- All read operations occur via typed **Queries** dispatched through a `QueryBus`.
- External ML libraries, database engines, and interfaces are implemented as **Plugins**, **Infrastructure**, and **Interfaces** adapting domain ports.

---

## Alternatives Considered

- **Monolithic `AutoML.fit()` pipeline:** Rejected because it couples all stages into a black box, making granular intervention and agentic tool invocation impossible without complete redesign.
- **Direct Service Layer (RPC-style methods without CQRS):** Rejected because distinct Command and Query dataclasses provide self-describing schemas directly convertible to future `AgentTool` definitions (LLM function calling).

---

## Consequences

- **Easier:** Testing the domain without external libraries; swapping storage or ML plugins; creating CLI commands, REST APIs, or LLM tools by simply routing through CommandBus/QueryBus; full event auditing and reproducibility.
- **Harder / Overhead:** More boilerplate classes (Command, Query, Handler registration) for each new capability.
- **Prohibited:** Importing `scikit-learn`, `optuna`, `sqlite3`, or `pandas` inside `src/automl/domain/`.
