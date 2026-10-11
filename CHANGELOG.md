# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.9.0-rc1] - 2026-10-11 — CATML Explore & Security Hardening

### Added
- **CATML Explore (Phases E1–E6):**
  - **Phase E1 (Investigation Subsystem & MCP API):** Structured dataset profiling, statistical and quality findings, and MCP analysis tools (`analysis_get_findings`, `analysis_get_dataset_quality`, `analysis_get_feature_interactions`, `analysis_list_hypotheses`, `analysis_record_hypothesis`).
  - **Phase E2 (Interactive Discovery Canvas):** Real-time EDA workbench view with distribution plots, high-cardinality analysis, outlier diagnostics, and interactive bivariate scatter visualizations.
  - **Phase E3 (Bivariate Analysis & Leakage Correlation):** Mutual information matrices, Spearman/Pearson correlation grids, class-conditional separation, and group leakage flags.
  - **Phase E4 (Agentic Hypothesis Engine):** Structured hypothesis ledger (`HypothesisCandidate`, `HypothesisVerification`), automated feature engineering and selection experiments with "Propose ≠ Accept" verification.
  - **Phase E5 (Hypothesis-to-Pipeline Bridge):** Seamless promotion of verified hypotheses into the production pipeline and experiment scheduler.
  - **Phase E6 (Comprehensive 4-Axis Benchmark Suite):** Automated evaluation across 4 axes: Detection & Discovery Accuracy, Hypothesis Verification Precision, Operational Safety & Privacy Guardrails (PII redaction, path confinement), and End-to-End Performance & Throughput.
- **Trusted Deployment Guide:** Comprehensive deployment guide and threat model in [`docs/trusted-deployment.md`](docs/trusted-deployment.md) covering Local Loopback, Private LAN, and Cloud VPS/Production tiers.

### Security & Hardened
- **Remote Network Fail-Closed Enforcement:** Binding Workbench or MCP streamable-http server to non-loopback interfaces without authentication is strictly blocked. Using `--insecure-no-auth` remotely requires the explicit environment variable `CATML_ALLOW_INSECURE=1`, preventing accidental public exposure.
- **Fragment Token Delivery:** Stopped printing unencrypted URL query parameters (`?token=`). Workbench remote URLs exclusively provide client-side fragment identifiers (`/#token=`), preventing token transmission across HTTP requests and leakage in server logs.
- **Model Artifact Security Guidance:** Added prominent warnings in `README.md` and `SECURITY.md` against loading untrusted third-party `.pkl` files (deserialization risks), clarifying that SHA-256 checksums ensure file integrity, not cryptographic authenticity.
- **Agent Surface Privacy Guardrails:** Enforced `compact=True` default in MCP analysis tools, returning zero raw dataset rows to LLM contexts by default.

---

## [0.8.1] - 2026-10-07 — Trust Patch

### Fixed & Hardened
- **Fail-Safe Anti-Leakage Execution:** `AutoML.fit()` and `RuleBasedExperimentPlanner` now automatically prioritize `profile.recommended_feature_names`, ensuring columns flagged with target or group leakage are excluded from candidate model training by default.
- **Localhost & Privacy Hardening:** Workbench server now binds strictly to `127.0.0.1` by default (opt-in `--host 0.0.0.0` available via CLI) and eliminated permissive wildcard `Access-Control-Allow-Origin: *` headers, ensuring local isolation.
- **Filesystem Confinement for Media:** `/api/media/preview` now strictly verifies that all image paths reside within the workspace or dataset directories, blocking path traversal and arbitrary filesystem reads.
- **Deterministic Reproducibility:** Propagated `random_state` from `AutoML` constructor directly into `RunConfig.random_seed` and all downstream estimators and cross-validation splitters.
- **Leaderboard Ranking for Minimization Metrics:** Corrected `AutoMLResult.leaderboard()` and `SQLiteExperimentRepository.get_leaderboard()` to sort ascendingly for loss/error metrics (`mae`, `rmse`, `mse`, `loss`), placing the lowest-error model at rank 1.
- **Removed Simulated Evidence:** Removed synthetic `public_lb = score * 0.9999` and fake delta from benchmark submission endpoints.
- **Transparent Product Messaging:** Refocused `README.md` hero on trust, automatic leakage protection, reproducible experiments, and controlled agent access.

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
