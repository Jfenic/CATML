# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.8.0] - 2026-10-07

### Added
- **Core Ergonomics & Facade:** Simplified 3-line AutoML API (`from catml import AutoML, ModelArtifact`).
- **Portable Model Artifacts:** Standalone production deployment via `ModelArtifact.save()` and `ModelArtifact.load()` with streaming chunked SHA-256 integrity verification.
- **Unified Command-Line Interface:** Native `catml fit`, `catml ui`, `catml task`, and `catml mcp` commands.
- **Tech Minimalista Premium Workbench:** Interactive web dashboard for live experiment tracking, dataset inspection, and visual telemetry.
- **Dual-Transport MCP Server:** Stdio and streamable HTTP Model Context Protocol (MCP) server allowing external AI agents (Claude Desktop, Cursor) to interact with CATML via typed tool contracts.
- **LangGraph Multi-Agent Orchestrator:** Durable agentic workflow (`catml[agents]`) with SQLite checkpointing, cooperative budget leases, and human approval gates.
- **Anti-Leakage Guardian:** Group and entity leakage detection, enforcing `GroupKFold` validation when patient/group columns are detected.
- **Capability Layer & Immutable Specifications:** `@dataclass(frozen=True)` `ProblemSpec` and `BackendCapabilities` supporting multimodal problem definitions.
- **Lightweight NLP Modality:** Automated text column discovery and sublinear TF-IDF n-gram feature extractor.
- **Vision Modality Spike (`catml[vision]`):** Image directory and path detection, deep learning feature extraction via ResNet backbones, and downstream head training.
- **Apache 2.0 Relicensing:** Updated project licensing to Apache License, Version 2.0.

### Changed
- Refactored vision pipeline to fail visibly with actionable instructions when `catml[vision]` is missing, preventing silent fallbacks to deterministic hashes in user workloads.
- Renamed vision plugin and UI labels to "Vision Feature MLP" / downstream head to transparently reflect feature extraction plus estimator training.
- Restructured `pyproject.toml` optional dependencies into clean groups: `models`, `vision`, `nlp`, `mcp`, `agents`, and `all`.

---

## [0.7.0] - 2026-09-15

### Added
- Hexagonal domain ports and CQRS bus architecture (`CommandBus`, `QueryBus`).
- Plugin registry for external estimators (LightGBM, XGBoost, CatBoost).
- Out-of-fold (OOF) target encoding and stacking ensembles.
- Persistent SQLite background job queue.

---

## [0.6.0] - 2026-08-01

### Added
- Initial automated tabular model training with scikit-learn.
- Optuna hyperparameter optimization (HPO).
- Automated dataset profiling and cardinality detection.
