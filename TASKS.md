# Tasks

Fuente del estado operativo y del backlog. Las guías y los planes enlazan aquí; no mantienen una segunda lista de tareas. Las fases V0.x son hitos de diseño; la versión del paquete se define en `src/automl/__init__.py`.

## Now (Active Phase — AutoML Workbench & Kaggle Playground Series S6E9)

- [x] Feature: CATML AutoML Workbench Server (`src/automl/interfaces/web/server.py`): Hexagonal interface adapter with zero external dependencies (`ThreadingHTTPServer`), exposing REST endpoints via `CommandBus`, `QueryBus`, and `AutoMLWorkspace` for Mission Control, run lifecycle controls (`pause`, `resume`, `cancel`, `clone`), dataset inspection, planner explicability, meta-learning knowledge, agent hypotheses, and Kaggle validation checklist.
- [x] Feature: Frontend Modular Architecture & Design Patterns (`src/automl/interfaces/web/static/`):
  - Architecture: EventBus (PubSub), WorkbenchStore (Observable/State pattern), CATMLApiClient (Adapter/Repository), Modular View Controllers.
  - Design System (`workbench.css`): Dense technical dark theme with semantic color system (🔵 system, 🟢 proven gain, 🟠 warning, 🔴 error/stop, 🟣 intelligence).
  - Views: Overview (Mission Control), Dataset Inspector ("¿Qué entendió CATML?"), Experiment Studio (explicable planner "Why this?", HPO, controls), Compare View (multi-select, metrics table, diff inspector), Visual Pipeline DAG (interactive execution nodes), Knowledge (V0.8 meta-learning preview), Lateral Agent Drawer (V1.0 hypothesis engine "Proponer ≠ Aceptar"), Kaggle S6E9 Center.
- [x] Refactorización Frontend Dinámico (Workbench UI): Eliminación completa de datos hardcodeados ("EV Purchases", "S6E9", "Will_Buy_EV", "competitions/playground-series-s6e9", métricas y 4 checkboxes estáticos). Todas las vistas (Overview, Datasets, Experiment Studio, Compare, Pipeline DAG, Kaggle, Knowledge y New Experiment Modal) se enlazan dinámicamente al workspace activo, runs reales y datasets registrados. Endpoint GET /api/datasets agregado a server.py.
- [x] Feature: Análisis Exploratorio Interactivo del Dataset, Gráficos Visuales y Selección Inteligente de Columnas (Workbench Web): Métricas estadísticas descriptivas (media, std, min, max, cuartiles, asimetría, correlación con target), previsualización de datos reales (primeras 8 filas), recomendaciones automáticas (identificadores, multicolinealidad, baja varianza, valores nulos), toolbar interactiva de selección de variables (filtros por tipo, búsqueda en vivo, selección rápida Top 5/10 por correlación con target, aplicación de recomendaciones con 1 clic), pestaña completa de Matriz de Correlación interactiva con mapa de calor (Pearson $r \in [-1, 1]$ y alertas de colinealidad), modal interactivo de análisis visual por variable con Diagrama de Caja y Bigotes (Box Plot comparativo desglosado por clases del target), histogramas de distribución con diagnóstico de asimetría, y tasa de propensión del target (%) por categoría. Ejecución de experimentos con características personalizadas y barra de acceso rápido en Studio.
- [x] Feature: CLI integration `automl ui [--port PORT] [--workspace WORKSPACE]` in `src/automl/interfaces/cli/main.py`.
- [x] Testing: Comprehensive web test suite in `tests/test_web_dashboard.py` verifying static asset serving, dataset registration, profiling, experiment execution, pause/resume, and submission generation.
- [x] Binary OOF prediction blending (`--folds 5`) with fixed equal weights, application commands/queries, CLI and Workbench; [scope and evaluation protocol](docs/features/oof-blending/spec.md).
- [ ] Validate OOF blending on an independent holdout and representative benchmark before promoting a candidate.

- [x] Persistent local jobs for experiments, OOF and submissions: SQLite queue, single worker, idempotency, actual progress, cooperative controls, explicit recovery/retry, CLI/API and Workbench activity panel ([scope](docs/features/persistent-jobs/spec.md)).
- [ ] Extend background jobs to HPO/benchmarks and verify fold-model persistence before claiming resumable partial OOF.

## Next (Roadmap Phases)

> **Sistema agéntico — empieza aquí:** [`docs/features/agentic-system/README.md`](docs/features/agentic-system/README.md). Elige Persona A o B; prepara A0 o B0 y acuerda H0 antes de pasar a A1/B1.

- [x] Documentar entrada para colaboradores, orden de lectura, elección A/B y primeras entregas en la guía del sistema agéntico.
- [x] Revisar y ampliar el plan agéntico: límites modulares, contratos propuestos, presupuesto/aprobación, recuperación, propiedad de integración e hitos H0–H5 documentados en `docs/features/agentic-system/plan.md`.
- [x] Concretar ejecución entre dos personas en [`docs/features/agentic-system/two-person-plan.md`](docs/features/agentic-system/two-person-plan.md): paquetes A0–A5/B0–B5, dependencias, archivos/tests propios, handoff y revisión cruzada; primera entrega conjunta H2.
- [x] H0 — Preparar V0.9: Contratos acordados y fusionados (PR #22 para B0 y PR #23 para A0 integrados en main).
- [x] Paquete A1 (Persona A): Catálogo y ejecutor de consultas seguras (`ToolRegistry`, `ToolExecutor`, `create_read_only_tool_registry`), blindaje de scope (prevención de issue #14 del blackboard), normalización a DTOs tipados vía `QueryBus` sin mutaciones, y suite de tests `tests/test_v09_agent_tools.py` con 100% de cobertura (5 tests passing).
- [x] Paquete B1 (Persona B): Servidor MCP stdio en `src/automl/interfaces/mcp/`, subcomando CLI `automl mcp`, tools de consulta (`list_models`, `get_dataset_profile`, `list_plugins`, `list_experiments`, `get_leaderboard`, `get_feature_evidence`, `get_feature_ranking`), resources (`catml://runs/...`, `catml://datasets/...`) y lifecycle subprocess con stdout limpio (10 tests passing en `tests/test_v09_mcp_server.py`).
- [x] H1 — V0.9 Consultas y MCP stdio: Catálogo de consultas A1 integrado con servidor MCP stdio B1.
- [x] Paquete A2 (Persona A): Tools mutantes (`create_experiment`, `prioritize_feature`, `run_experiment`), límites estrictos, flujo de aprobación persistente con verificación anti-tampering (`SqliteAgentLedger`), idempotencia y deduplicación atómica ("Duplicados no repiten efectos"), intención duradera antes de mutar, y suite de tests `tests/test_v09_agent_operations.py` con 16 tests passing (97% cobertura en módulos agénticos).
- [ ] H2 — V0.9 Mutaciones autorizadas e integración: Persona A completó A2; pendiente Persona B con B2 (adaptación de resultados mutantes MCP, CLI de aprobación y flujo E2E integrado).
- [ ] H3 — V0.9 Operaciones largas, HPO y cancelación cooperativa.
- [ ] H4–H5 — V1.0 incremental: ciclo determinista con proveedor falso → LangGraph opcional con checkpoints y recuperación probada.
- [ ] V0.9 Phase: LLM Agent Tools & MCP Server (`AgentTool` wrappers, JSON-Schema tool registry, permission policies, MCP Server adapter) — *Ver plan detallado para 2 personas en [`docs/features/agentic-system/plan.md`](docs/features/agentic-system/plan.md)*
- [ ] V1.0 Phase: Autonomous Experiment Agent con LangGraph (Planner, Critic, Feature Advisor y Orchestrator cíclico "Proponer ≠ Aceptar") — *Ver plan detallado en [`docs/features/agentic-system/plan.md`](docs/features/agentic-system/plan.md)*
- [ ] V0.8 Phase: Meta-learning & knowledge base for warm-start policies (pospuesta temporalmente a favor del subsistema agéntico)

- [ ] Stacking with a trained meta-estimator (distinct from voting/blending).
- [ ] Persist backend class, plugin/library versions and dataset/config hashes per trial ([current limits](docs/backends.md)).
- [ ] Add the dedicated `feature_selection_v05` benchmark scenario from the feature-discovery specification.

## Backlog — Mejoras Tabulares Identificadas (Kaggle Benchmarking)

> Evidencia de implementación y limitaciones: [`docs/README.md`](docs/README.md). Flujo de colaboración: [`CONTRIBUTING.md`](CONTRIBUTING.md).

- [x] Heurística automática de alta cardinalidad en `DatasetProfiler` (excluir IDs y texto masivo > 0.7 ratio de unicidad)
- [x] Plugin de Ensamble y Blending (`VotingEnsemblePlugin` / `VotingBlender`) mediante voting y media ponderada
- [x] Mapeo automático de plantilla de sumisión (`--template sample_submission.csv`) en `GenerateSubmissionCommand`
- [x] Generación automática de variables de interacción (ratios numéricos y target encoding)
- [x] Blackboard Issue #12: Soporte para CatBoost, Extra Trees y MLP en el Plugin System con fallback HistGradientBoosting y Optuna search spaces (11 tests passing en `tests/test_catboost_and_models_plugin.py`)
- [x] Optimización de Pesos de Ensamble y Rank Averaging: Optimización simplex de Nelder-Mead (`optimize_ensemble_weights`) con regularización Brier para métricas discretas, rank-averaging (`rank_average_predictions` y `voting="rank"`), y pruning multi-fidelidad en Optuna (`MedianPruner`, `report_step`, `pruned=True`). Cobertura de tests: 294 passing, 86.40% coverage.

## Completed

Los hitos describen el núcleo implementado; consultar [alcance y limitaciones](docs/README.md) antes de interpretar una fase como cumplimiento de todos los objetivos originales. Las cifras de pruebas de estos hitos son referencias históricas.

- [x] Feature (Tarea D): Generación automática de variables de interacción (`InteractionFeatureGenerator`, ratios numéricos, productos, target encoding out-of-fold, 6 tests passing en `tests/test_feature_interactions.py`)
- [x] Feature: `TargetAdapter` universal para compatibilidad con XGBoost/CatBoost y targets discretos textuales (`TargetAdapter`, `tests/test_target_adapter.py`, Issue #8 resuelto)
- [x] Phase V0.7: Multimodal Pipelines & Directed Acyclic Graph (`PipelineGraph`, `GraphValidator`, `FeatureFusionNode`, `ImageModalityPlugin`, `ImageEncoderNode`, `AutoMLWorkspace.build_multimodal_pipeline`, `fit_predict_multimodal`, 51 tests passing across `tests/test_v07_*.py`)
- [x] Track Dev 2: Image modality plugin & deterministic image encoder node (`ImageModalityPlugin`, `ImageEncoderNode`, unit & batch embeddings, 20 tests passing en `tests/test_v07_image_plugin.py`)
- [x] Feature: Mapeo automático de plantilla de sumisión Kaggle (`GenerateSubmissionCommand`, `PredictDatasetQuery`, `--template sample_submission.csv`, 5 tests passing en `tests/test_submission_template.py`)
- [x] Feature: Model plugin for voting ensemble & blending (`VotingEnsemblePlugin`, `VotingBlender`, soft voting and weighted predictions, 14 tests passing en `tests/test_ensemble_plugin.py`)
- [x] Feature: Heurística de alta cardinalidad e identificadores en `DatasetProfiler` (`detect_column_cardinality_and_role`, `is_identifier`, `is_high_cardinality`, `exclude_identifiers`, 12 tests passing en `tests/test_profiler_cardinality.py`)
- [x] Feature: Kaggle-ready inference & submission generator (`GenerateSubmissionCommand`, `PredictDatasetQuery`, CLI `automl predict`)
- [x] V0.6 Phase: Plugin architecture contracts (`PluginPort`, `ModelPluginPort`, `MetricPluginPort`, `PreprocessorPluginPort`)
- [x] V0.6 Phase: `PluginRegistry` in application layer and `CompatibilityValidator` in engine
- [x] V0.6 Phase: `LightGBMPlugin` and `XGBoostPlugin` with fallback to scikit-learn HistGradientBoosting (LightGBM) or GradientBoosting (XGBoost)
- [x] V0.6 Phase: Custom business metric plugins (`CostSensitiveMetricPlugin`, `WeightedF1MetricPlugin`)
- [x] V0.6 Phase: CLI command `automl plugin list` (table and JSON formats)
- [x] V0.6 Phase: Unit and integration test suite `tests/test_v06_plugins.py` (8/8 tests passing)
- [x] V0.5 Phase: `FeatureSelectorPort` and `FeatureEvidenceRepositoryPort` in `src/automl/domain/ports.py`
- [x] V0.5 Phase: Statistical and ML feature selectors (`MutualInfoSelector`, `TreeImportanceSelector`, `L1Selector`, `CorrelationSelector`, `VarianceSelector`, `EnsembleRankSelector`, `PCAReducer`) in `src/automl/engine/features/`
- [x] V0.5 Phase: `AblationPlanner` in `src/automl/engine/planning/` for leave-one-out feature candidate experiments
- [x] V0.5 Phase: CQRS commands/queries (`SelectFeaturesCommand`, `PlanAblationExperimentsCommand`, `PromoteCandidateFeatureSetCommand`, `GetFeatureEvidenceQuery`, `ListCandidateFeatureSetsQuery`, `GetFeatureRankingQuery`)
- [x] V0.5 Phase: Feature evidence persistence in SQLite repository
- [x] V0.5 Phase: CLI subcommands `automl features select` and `automl features ablation`
- [x] V0.5 Phase: Unit and integration tests in `tests/test_v05_features.py` (52/52 tests passing, coverage >= 85%)

## Blocked

*(No blocked tasks)*

## Documentation Review — 2026-09-30

- [x] Reviewed Markdown consistency, relative links, extension guidance, CLI, and current validation results: 161 tests passed, 85.14% coverage.
- [x] Follow up on review recommendations: synchronize versions and phase status, replace machine-specific links, document Workbench and native/fallback backends, consolidate roadmap ownership, and validate extension examples.
- [x] Fix and cover model plugin registration: `AutoMLWorkspace.register_plugin()` now imports `ModelSpec` from `automl.domain.models.registry`; regression tests cover discovery, task compatibility and real training.

- [x] Final validation (2026-09-30): 164 tests passed, 85.23% coverage; editable package metadata and CLI report 0.7.0; custom plugin example trained successfully; Markdown links and heading anchors checked.

- [x] Organized review fixes on `fix/docs-and-plugin-registration` from updated `origin/main`, with separate commits for plugins, version and documentation; prior frontend edits excluded.

## OOF Prediction Blending — 2026-09-30

- [x] Initial binary ROC-AUC implementation with fold-local preprocessing, original label mapping, template alignment, persisted predictions and backend/configuration manifests.
- [x] Run/experiment ownership, probability validity, time budgets, pause/cancel and changed-input/artifact checks.
- [x] Final validation: 196 tests passed, 86.01% coverage; CLI/task catalog, JavaScript syntax, Markdown links and diff checks passed. Changes prepared for a separate draft PR based on the unmerged documentation PR #13.



## Workbench persisted trials — 2026-09-30

- [x] Fix experiments HTTP listing for saved TrialResults: query parameters from their Trial through the application DTO instead of accessing a nonexistent result field ([issue #18](https://github.com/Jfenic/CATML/issues/18)).
- [x] Add HTTP/query regressions for nonempty parameters, missing legacy Trial records and detached DTO mutation; 198 tests passed, coverage 86.22%.
- [x] Launch S6E9 visual preview at port 8080 with background jobs from PR #17 and this isolated fix; existing data and unrelated local edits preserved.
=======
## Persistent Jobs — 2026-09-30

- [x] Continue on isolated `feat/persistent-job-queue` from `main` after merged PR #16; preserve unrelated working-tree edits.
- [x] Add domain job states, repository port, ID-returning commands, snapshot queries, transactional SQLite queue and exclusive local worker.
- [x] Integrate real model/fold progress, cooperative controls, durable errors/retries, background Workbench activity and CLI/HTTP parity.
- [x] Document operation scope, Linux/macOS lease, checkpoint recovery and partial OOF restart in spec and ADR 003.
- [x] Final validation: 229 Python tests passed, 86.31% coverage; 4 JavaScript adapter tests, CLI help/task catalog, 21 Markdown file-link checks and diff checks passed. Prepared for an independent draft PR against main.
