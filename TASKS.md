# Tasks

Fuente del estado operativo y del backlog. Las guías y los planes enlazan aquí; no mantienen una segunda lista de tareas. Las fases V0.x son hitos de diseño; la versión del paquete se define en `src/automl/__init__.py`.

### Track Landing — Public homepage (independent of Workbench)

- [x] Implementar homepage en `website/` con Vite, React y TypeScript; tokens Tech Minimalista, responsive 320–1440 px y accesibilidad comprobada.
- [x] Integrar captura real del Workbench de `e4a2286` con dataset sintético, snippets acordes a la API, licencia MIT, instalación desde GitHub y Platform identificada como planificación.
- [x] Preparar build estático, OpenGraph, favicon, tipografías locales, guía Vercel y comprobaciones en navegador.
- [ ] Integrar la rama `feat/public-landing` y desplegar; conectar dominio y configurar URLs SEO absolutas cuando se conozca el destino.

## Now (Active Phase — Sprint 3: Brand Identity & Tech Minimalista Design System)

- [x] Spec: Complete Tech Minimalista Premium Brand System specification saved in `docs/design/tech-minimalist-brand-system.md` (Electric Blue `#4F67FF`, Graphite, 80/20 rule, Geist typography, light docs/dark product).
- [x] ADR: Formalize architectural decision in `docs/decisions/006-tech-minimalist-premium-brand-system.md` (and mirrored in `catml-platform/docs/decisions/002-tech-minimalist-premium-brand-system.md`).
- [x] Rule 8: Update UI & brand standards in `AGENTS.md` to enforce Tech Minimalista guidelines across all agents and developers.
- [x] Theme Tokens & Workbench CSS: Harmonize CSS tokens (`--catml-*`, colors, typography, borders, shadows) in `src/automl/interfaces/web/static/css/workbench.css` with the new design system.
- [x] Platform Sync: Mirror brand system specifications, tokens and roadmap milestones into `catml-platform`.
- [x] Standalone Artifact UI & Color Cleanup: Expose `/api/models/export` (.pkl download) and `/api/models/export-info` in `server.py`, integrate 1-click artifact download buttons in `overview.js` and `compare.js`, and eliminate legacy `#E5512D` colors across all views in favor of `#4F67FF` Electric Blue (437 tests passing, 87.23% coverage).
- [x] UI Refinement: Comprehensive transformation of Workbench frontend and dataset profiler to 100% Tech Minimalista Premium (4-level surface hierarchy `#080A0F`/`#0D1017`/`#11151E`/`#161B26`, monochrome sidebar with muted 01-06 indicators, Geist/Inter sans-serif UI, discrete badges, English unification, clean Agent drawer, 461 tests passing, 87.53% coverage).
- [x] Workbench Elevation & Container De-nesting: Elimination of nested container soup (`panel -> panel -> control -> badge`) across Dataset Understanding, studio, and modals; segmented pill track for filters; clear visual action hierarchy with dominant primary CTA "Launch Experiment (N)" and secondary "Apply recommendations"; 100% professional technical English across all views and background jobs widget; discrete badges (`HIGH RELEVANCE`, `HIGH SIGNAL`); full test suite green (461 passed, 87.53% coverage).
- [x] Experiments Studio Semantic UX & Flow Coherence: Status-dependent action controls (Completed -> Run again / Clone / Export best model; Running -> Pause / Stop; Paused -> Resume / Stop); reduced height and informative empty state for AutoML Plan & Decisions; discrete trial cards when < 4 trials (curve reserved for >= 4 trials); explicit "New run preset" labels; Agent-assisted Guided Experiment badge; zero mixed Spanish/English.
- [x] Lucide Vector Icon System: Eliminate 100% of Unicode emojis (🏆, 📊, 🧠, 🚀, ⚡, ⚗, 🎯, 💡, 📦, ⚠️, etc.) and replace with zero-dependency Lucide SVG icon system in `src/automl/interfaces/web/static/js/icons.js` with standardized sizing (.icon, .icon-sm, .icon-lg, .icon-xl) and semantic token colors (#8B95A7 default, #4F67FF active/accent, #22C55E success, #F59E0B warning, #EF4444 danger, #6956E8 agent); verified 0 emoji occurrences across all web static assets (PR #56, 461 tests passing, 87.53% coverage).

## Completed Phase: Sprint 2 (Productization P0 — CLI Fit, Quickstart & Product README)

- [x] Feature: CLI Command `catml fit <dataset> --target <col>` (`src/automl/interfaces/cli/main.py`) with leaderboard output and artifact generation.
- [x] Feature: `examples/quickstart.py` (2-minute end-to-end runnable script with dataset generation and standalone inference).
- [x] Docs: Complete product-first redesign of `README.md` (value proposition, 4-line quickstart, MCP agent configuration, architecture).
- [x] Tests: CLI and quickstart automated tests (`tests/test_cli_fit.py`) (436 tests passing, 87.39% coverage).

## Completed Phase: Sprint 1 (Ergonomic Facade & Model Artifacts)

- [x] Feature: Standalone `ModelArtifact` (`src/automl/artifacts/model_artifact.py`) with `save(path)` and `load(path)`.
- [x] Feature: `AutoML` and `AutoMLResult` facade (`src/automl/facade.py`) supporting `fit(df, target=...)` and `fit(X, y)`.
- [x] Feature: Top-level exports in `src/automl/__init__.py` and alias package `src/catml/`.
- [x] Dependency: Add `joblib>=1.3` to `pyproject.toml`.
- [x] Tests: Comprehensive test suite in `tests/test_model_artifact.py` and `tests/test_facade.py` with full coverage >= 85% (432 tests passing, 87.36% coverage).

## Active Components (Workbench & Kaggle Playground Series S6E9)

- [x] Feature: CATML AutoML Workbench Server (`src/automl/interfaces/web/server.py`): Hexagonal interface adapter with zero external dependencies (`ThreadingHTTPServer`), exposing REST endpoints via `CommandBus`, `QueryBus`, and `AutoMLWorkspace` for Mission Control, run lifecycle controls (`pause`, `resume`, `cancel`, `clone`), dataset inspection, planner explicability, meta-learning knowledge, agent hypotheses, and Kaggle validation checklist.
- [x] Feature: Frontend Modular Architecture & Design Patterns (`src/automl/interfaces/web/static/`):
  - Architecture: EventBus (PubSub), WorkbenchStore (Observable/State pattern), CATMLApiClient (Adapter/Repository), Modular View Controllers.
  - Design System (`workbench.css`): Dense technical dark theme with semantic color system (🔵 system, 🟢 proven gain, 🟠 warning, 🔴 error/stop, 🟣 intelligence).
  - Views: Overview (Mission Control), Dataset Inspector ("¿Qué entendió CATML?"), Experiment Studio (explicable planner "Why this?", HPO, controls), Compare View (multi-select, metrics table, diff inspector), Visual Pipeline DAG (interactive execution nodes), Knowledge (V0.8 meta-learning preview), Lateral Agent Drawer (V1.0 hypothesis engine "Proponer ≠ Aceptar"), Kaggle S6E9 Center.
- [x] Refactorización Frontend Dinámico (Workbench UI): Eliminación completa de datos hardcodeados ("EV Purchases", "S6E9", "Will_Buy_EV", "competitions/playground-series-s6e9", métricas y 4 checkboxes estáticos). Todas las vistas (Overview, Datasets, Experiment Studio, Compare, Pipeline DAG, Kaggle, Knowledge y New Experiment Modal) se enlazan dinámicamente al workspace activo, runs reales y datasets registrados. Endpoint GET /api/datasets agregado a server.py.
- [x] Feature: Análisis Exploratorio Interactivo del Dataset, Gráficos Visuales y Selección Inteligente de Columnas (Workbench Web): Métricas estadísticas descriptivas (media, std, min, max, cuartiles, asimetría, correlación con target), previsualización de datos reales (primeras 8 filas), recomendaciones automáticas (identificadores, multicolinealidad, baja varianza, valores nulos), toolbar interactiva de selección de variables (filtros por tipo, búsqueda en vivo, selección rápida Top 5/10 por correlación con target, aplicación de recomendaciones con 1 clic), pestaña completa de Matriz de Correlación interactiva con mapa de calor (Pearson $r \in [-1, 1]$ y alertas de colinealidad), modal interactivo de análisis visual por variable con Diagrama de Caja y Bigotes (Box Plot comparativo desglosado por clases del target), histogramas de distribución con diagnóstico de asimetría, y tasa de propensión del target (%) por categoría. Ejecución de experimentos con características personalizadas y barra de acceso rápido en Studio.
- [x] Feature: Sistema de Diseño Neo-Industrial (Visual ML Lab) para CATML AutoML Workbench: Adopción y formalización de la dirección visual de laboratorio de ingeniería (especificación en `docs/design/neo-industrial-ui-spec.md` y ADR 005 en `docs/decisions/005-neo-industrial-visual-ml-lab-ui.md`), Regla 8 en `AGENTS.md`, tipografía `Space Grotesk` + `IBM Plex Mono`, paleta modular (`#111111` base carbón, `#16171c`/`#1c1d24` módulos, `#D8D6CF` cemento, `#F1EFE9` blanco cálido, `#E5512D` naranja señal reservado a acciones de ejecución), bordes arquitectónicos de 1px con radios rectos (0-4px), módulos de métricas de alta densidad y navegación lateral numerada (01-06).
- [x] Feature: CLI integration `automl ui [--port PORT] [--workspace WORKSPACE]` in `src/automl/interfaces/cli/main.py`.
- [x] Testing: Comprehensive web test suite in `tests/test_web_dashboard.py` verifying static asset serving, dataset registration, profiling, experiment execution, pause/resume, and submission generation.
- [x] Binary OOF prediction blending (`--folds 5`) with fixed equal weights, application commands/queries, CLI and Workbench; [scope and evaluation protocol](docs/features/oof-blending/spec.md).
- [ ] Validate OOF blending on an independent holdout and representative benchmark before promoting a candidate.

- [x] Feature: Dataset Framing & Problem Context Questionnaire: Entidad de dominio `DatasetQuestionnaire`, servicio `QuestionnaireAdvisor` (heurísticas de churn, fraude, médico, crédito, series temporales, cohortes + modo LLM), persistencia SQLite en `dataset_questionnaires`, paridad en `AutoMLWorkspace` y endpoints REST en `server.py` (`GET/POST /api/dataset/questionnaire`). Suite en `tests/test_dataset_questionnaire.py` (PR #42).
- [x] Feature: Safe Feature Calculator & Derived Column Engine (`docs/features/derived-features/spec.md`): Motor robusto para generación de columnas complejas con doble modo: Fórmulas matemáticas AST seguras con lista blanca y aislamiento automático de división por cero (sin caídas), y código Python sandboxed restringido (`np`, `pd`, builtins seguros, auditoría AST anti-inyección). Especialista `FeatureAdvisor` para conexión de agentes LLM, validación previa (`POST /api/features/calculate`), persistencia en dataset (`POST /api/features/apply`), sugerencias asistidas (`POST /api/features/suggest`) y pestaña interactiva en Workbench UI (`datasets.js`). 22 tests en `tests/test_derived_feature_engine.py`.

## Next (Roadmap Phases)

- [ ] Sincronizar el índice y los planes con las capacidades actuales: H4, stacking y OOF multi-modelo ya implementados; actualizar las referencias al diseño vigente (revisión 2026-10-02).
- [ ] Auditar `feat/workbench-live-experiments` (WIP sin PR, 60 commits detrás de main) y los commits documentales posteriores a PR #40 en `feat/agentic-a4-specialists` antes de limpiar ramas históricas.

> **Sistema agéntico — empieza aquí:** [`docs/features/agentic-system/README.md`](docs/features/agentic-system/README.md). Elige Persona A o B; prepara A0 o B0 y acuerda H0 antes de pasar a A1/B1.

- [x] Documentar entrada para colaboradores, orden de lectura, elección A/B y primeras entregas en la guía del sistema agéntico.
- [x] Revisar y ampliar el plan agéntico: límites modulares, contratos propuestos, presupuesto/aprobación, recuperación, propiedad de integración e hitos H0–H5 documentados en `docs/features/agentic-system/plan.md`.
- [x] Concretar ejecución entre dos personas en [`docs/features/agentic-system/two-person-plan.md`](docs/features/agentic-system/two-person-plan.md): paquetes A0–A5/B0–B5, dependencias, archivos/tests propios, handoff y revisión cruzada; primera entrega conjunta H2.
- [x] H0 — Preparar V0.9: Contratos acordados y fusionados (PR #22 para B0 y PR #23 para A0 integrados en main).
- [x] Paquete A1 (Persona A): Catálogo y ejecutor de consultas seguras (`ToolRegistry`, `ToolExecutor`, `create_read_only_tool_registry`), blindaje de scope (prevención de issue #14 del blackboard), normalización a DTOs tipados vía `QueryBus` sin mutaciones, y suite de tests `tests/test_v09_agent_tools.py` con 100% de cobertura (5 tests passing).
- [x] Paquete B1 (Persona B): Servidor MCP stdio en `src/automl/interfaces/mcp/`, subcomando CLI `automl mcp`, tools de consulta (`list_models`, `get_dataset_profile`, `list_plugins`, `list_experiments`, `get_leaderboard`, `get_feature_evidence`, `get_feature_ranking`), resources (`catml://runs/...`, `catml://datasets/...`) y lifecycle subprocess con stdout limpio (10 tests passing en `tests/test_v09_mcp_server.py`).
- [x] H1 — V0.9 Consultas y MCP stdio: Catálogo de consultas A1 integrado con servidor MCP stdio B1.
- [x] Paquete A2 (Persona A): Tools mutantes (`create_experiment`, `prioritize_feature`, `run_experiment`), límites estrictos, flujo de aprobación persistente con verificación anti-tampering (`SqliteAgentLedger`), idempotencia y deduplicación atómica ("Duplicados no repiten efectos"), intención duradera antes de mutar, y suite de tests `tests/test_v09_agent_operations.py` con 16 tests passing (97% cobertura en módulos agénticos).
- [x] Paquete B2 (Persona B): Exposición y adaptación de mutating tools en servidor MCP (`create_experiment`, `prioritize_feature`, `run_experiment`), manejo estructurado de `PENDING_APPROVAL` sin bloquear terminal, comandos CLI de gobernanza (`automl agent approvals list`, `automl agent approve <id> [--reject]`) y verificación E2E (8 tests passing en `tests/test_v09_agent_cli.py` y `tests/test_v09_agent_e2e.py`, 11 tests en `tests/test_v09_mcp_server.py`).
- [x] H2 — V0.9 Mutaciones autorizadas e integración: Primera entrega conjunta local verificada integralmente (inspección MCP -> propuesta de candidato -> aprobación humana en CLI -> ejecución autorizada -> métricas en leaderboard consultables -> idempotencia probada).
- [x] H3 — V0.9 Operaciones largas, HPO y cancelación cooperativa (Integración conjunta A3 + B3):
  - [x] Paquete A3 (Persona A): HPO (`optimize_experiment`), reservas atómicas de presupuesto, leases por run, reconciliación de operaciones caídas y cancelación cooperativa (10 tests pasando en `tests/test_v09_agent_a3.py`, 16 tests en `tests/test_v09_agent_operations.py`). PR #35 integrado en main.
  - [x] Paquete B3 (Persona B): Inspección de estado y control de operaciones en CLI (`automl agent operations list`, `automl agent operations get`, `automl agent operations cancel`), tools MCP (`get_operation_status`, `list_operations`, `cancel_operation`), recursos MCP (`catml://runs/{run_id}/operations`, `catml://operations/{operation_id}`), transporte dual stdio / streamable-http en `automl mcp`, cumplimiento estricto de criterios de aceptación de H3 ("Timeout no se presenta como cancelación", "Operation ID recuperable" tras reinicio de proceso, flujo de cancelación cooperativa). 14 tests en `test_v09_agent_cli.py`, 14 tests en `test_v09_mcp_server.py`, 2 tests exhaustivos en `test_v09_agent_e2e.py`.

- [x] H4 — V1.0 Ciclo determinista y especialistas ("Proponer ≠ Aceptar") (Concurrencia aislada según two-person-plan.md §5.1):
  - [x] Contratos H4 acordados en `main`: DTOs de paso (`CandidateProposal`, `EvaluationFeedback`, `ContextPayload`, `SessionStepResult`).
  - [x] Track Persona A (Paquete A4): `ContextBuilder`, especialistas (`Planner`, `FeatureAdvisor`, `Critic`) en `src/automl/application/agents/specialists/`, proveedor determinista `FakeLLMProvider` en `src/automl/infrastructure/llm/` y tests en `tests/test_v10_specialists.py` (31 tests pasando con 94% de cobertura de paquete).
  - [x] Track Persona B (Paquete B4): Máquina de estados determinista en `src/automl/application/agents/orchestrator/`, CLI de sesión en `src/automl/interfaces/cli/agent_session_cli.py`, criterios de parada y tests en `tests/test_v10_orchestrator.py` (24 tests pasando con 92% de cobertura de paquete).
- [ ] H5 — V1.0 Proveedor real, LangGraph duradero y recuperación (Integración final A5 + B5):
  - [ ] Track Persona A (Paquete A5): Adaptador LLM agnóstico con validación de respuesta, timeouts/reintentos acotados, redacción y auditoría de tokens.
  - [ ] Track Persona B (Paquete B5): Checkpointer LangGraph SQLite, reanudación tras fallo y tests E2E de orquestación.
- [ ] V0.8 Phase: Meta-learning & knowledge base for warm-start policies (pospuesta temporalmente a favor del subsistema agéntico)

- [x] Stacking with a trained meta-estimator (distinct from voting/blending).
- [ ] Persist backend class, plugin/library versions and dataset/config hashes per trial ([current limits](docs/backends.md)).
- [ ] Add the dedicated `feature_selection_v05` benchmark scenario from the feature-discovery specification.

### Generalist AutoML Engine & Super-Ensembles
> Technical Specs & Execution Plan: [`docs/features/generalist-enhancements/spec.md`](docs/features/generalist-enhancements/spec.md) and [`docs/features/generalist-enhancements/plan.md`](docs/features/generalist-enhancements/plan.md).

- [x] Phase 1: Generalized N-Model OOF Blending & Level-2 Stacking (remove 2-model limit, Rank-Averaging, Ridge meta-learner).
- [ ] Phase 2: Temporal Dynamics Engine (automatic sequential detection, sensor lags, 24h trend deltas, cyclical projections).
- [ ] Phase 3: Workbench Web UX (K-Fold strategy selector, visual multi-select ensemble builder, Kaggle drag-and-drop export).
- [ ] Phase 4: Anti-Leakage Guardian in Profiler & Plugin Observability.

## Backlog — Mejoras Tabulares Identificadas (Kaggle Benchmarking)

> Evidencia de implementación y limitaciones: [`docs/README.md`](docs/README.md). Flujo de colaboración: [`CONTRIBUTING.md`](CONTRIBUTING.md).

- [x] Heurística automática de alta cardinalidad en `DatasetProfiler` (excluir IDs y texto masivo > 0.7 ratio de unicidad)
- [x] Plugin de Ensamble y Blending (`VotingEnsemblePlugin` / `VotingBlender`) mediante voting y media ponderada
- [x] Mapeo automático de plantilla de sumisión (`--template sample_submission.csv`) en `GenerateSubmissionCommand`
- [x] Generación automática de variables de interacción (ratios numéricos y target encoding)
- [x] Generación automática de diferencias y restas numéricas ($A - B$) en `InteractionFeatureGenerator` con soporte `include_differences`, conjunto de candidatos `interactions_differences` y transformaciones desacopladas (8 tests pasando en `tests/test_feature_interactions.py`)
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

## Persistent Jobs — 2026-09-30

- [x] Continue on isolated `feat/persistent-job-queue` from `main` after merged PR #16; preserve unrelated working-tree edits.
- [x] Add domain job states, repository port, ID-returning commands, snapshot queries, transactional SQLite queue and exclusive local worker.
- [x] Integrate real model/fold progress, cooperative controls, durable errors/retries, background Workbench activity and CLI/HTTP parity.
- [x] Document operation scope, Linux/macOS lease, checkpoint recovery and partial OOF restart in spec and ADR 003.
- [x] Final validation: 229 Python tests passed, 86.31% coverage; 4 JavaScript adapter tests, CLI help/task catalog, 21 Markdown file-link checks and diff checks passed. Prepared for an independent draft PR against main.
