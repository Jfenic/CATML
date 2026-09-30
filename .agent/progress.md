# Progress

Active Track: **Sistema Agéntico V0.9/V1.0 — Persona B (Hito H1 / Paquete B1 Integrado)**:
- Rol: **Persona B** (Interfaces, Integración y Orquestación).
- Rama: `feat/agentic-b1-mcp`.
- Entregable B1 completado:
  - Servidor MCP stdio implementado en `src/automl/interfaces/mcp/server.py` utilizando el SDK oficial `mcp` (MCPServer) y consumiendo las capacidades de A1.
  - Tools de consulta expuestas: `get_dataset_profile`, `list_models`, `list_plugins`, `list_experiments`, `get_leaderboard`, `get_feature_evidence`, `get_feature_ranking`.
  - Resources expuestos: `catml://runs/{run_id}/leaderboard`, `catml://datasets/{dataset_id}/profile`.
  - Subcomando CLI `automl mcp [--workspace PATH]` implementado en `src/automl/interfaces/cli/mcp_cli.py` y registrado en `main.py` de forma diferida.
  - Logging estructurado exclusivamente por `sys.stderr` garantizando `stdout` 100% puro para JSON-RPC.
  - Suite de tests completa en `tests/test_v09_mcp_server.py` (10 tests pasando, incluyendo handshake de subprocess stdio).
- Hito H1 cerrado: A1 y B1 integrados. Preparado para H2 (mutaciones, aprobación humana y ledger).
 
## Completed
 
- Paquete A1 completado (Persona A, 2026-09-30):
  - Catálogo de herramientas y ejecutor de consultas seguras en `src/automl/application/agents/`:
    - `ToolRegistry`: registro y consulta tipada de `ToolDefinition` con schemas canónicos (`TOOL_SCHEMAS`).
    - `ToolExecutor`: ejecución desacoplada con validación previa de esquema, control de aislamiento y despacho por `QueryBus`.
    - `create_read_only_tool_registry(query_bus)`: factoría canónica que expone 7 tools de solo lectura (`get_dataset_profile`, `list_models`, `list_plugins`, `list_experiments`, `get_leaderboard`, `get_feature_evidence`, `get_feature_ranking`).
    - Prevención estricta de fuga cruzada (Issue #14): rechaza llamadas donde `arguments['run_id'] != context.run_id` con `PERMISSION_DENIED`.
    - Pruebas en `tests/test_v09_agent_tools.py` (5 tests pasando, 100% cobertura en componentes nuevos).

- Paquete A0 completado (Persona A, 2026-09-30):
  - Definición de contratos de dominio en `src/automl/domain/agents/`: `AgentBudget`, `Hypothesis`, `ToolEffect`, `AgentPermission`, `PolicyDecisionType`, `ApprovalStatus`, `OperationStatus`, `ToolErrorCode`. Hexagonalmente puro, sin librerías externas.
  - DTOs de aplicación en `src/automl/application/agents/contracts.py`: `ToolDefinition`, `ToolCallContext`, `ToolInvocation`, `ToolResult`, `ToolError`, `PolicyDecision`, `ApprovalRequest`, `OperationRecord`, `AgentContext`, `AgentSessionState`.
  - Puerto de auditoría y persistencia `AgentLedgerPort` en `src/automl/application/agents/ports.py`.
  - Evaluador de políticas `PolicyEvaluator` con defaults finitos, control de permisos, whitelist/blacklist de modelos, reservas presupuestarias y función criptográfica `compute_arguments_hash` en `src/automl/application/agents/policy.py`.
  - Schemas canónicos `TOOL_SCHEMAS` para las 11 tools del catálogo inicial y validador puro sin dependencias en `src/automl/application/agents/schemas.py`.
  - Ledger transaccional SQLite `SqliteAgentLedger` en `src/automl/infrastructure/database/sqlite_agent_ledger.py` con garantía de idempotencia `BEGIN IMMEDIATE`.
  - Fixtures de contrato compartidas para Persona B en `tests/fixtures/agentic/fixtures_v09.py`.
  - Suite de pruebas de contrato en `tests/test_v09_agent_contracts.py` (11 tests pasando, 100% de cobertura en módulos nuevos, suite global 242 tests pasando, 87.66% cobertura).
- Dynamic Database Binding & Real Parity in Workbench UI: removed mockup placeholder strings ("Ensemble #7", 0.94621) from `studio.js`, `overview.js`, and `kaggle.js`, correctly displaying real SQLite metrics (best model `xgboost`, CV `0.94123`, 11 trials), live status badge in `app.js` and `index.html`, and marking the Ensemble Blender as candidate to be trained.
- Guía de entrada (2026-09-30): `docs/features/agentic-system/README.md` explica elección A/B, orden de lectura, primera entrega A0/B0 y mensajes de asignación. Enlazada desde planes y TASKS; roles todavía sin personas asignadas y H0 pendiente.
- Plan operativo para dos personas (2026-09-30): `docs/features/agentic-system/two-person-plan.md` asigna A a contratos/aplicación/persistencia/especialistas y B a MCP/CLI/CI/orquestación. Define paquetes A0–A5/B0–B5, integración por hito, propiedad de archivos/tests y primera entrega H2.
- Revisión del plan agéntico (2026-09-30): diseño ampliado en `docs/features/agentic-system/plan.md` con separación dominio/aplicación/adaptadores, catálogo explícito, contratos versionados, límites estrictos, aprobación persistente, ledger/idempotencia, recuperación, MCP stdio/HTTP y hitos H0–H5 con propietarios y pruebas.
- Completed Feature: CATML AutoML Workbench (Web Interface, Server, and Frontend modular architecture with complete design patterns).
- Completed Tarea D (Interaction Feature Generation): Implemented `InteractionFeatureGenerator` in `src/automl/engine/features/generation/` with numerical ratios, products, and smoothed target encoding (6 tests passing in `tests/test_feature_interactions.py`).
- Implemented `TargetAdapter` in `src/automl/engine/training/target_adapter.py` and integrated into `SklearnTrainer`, resolving Issue #8 for XGBoost string target compatibility (5 tests passing in `tests/test_target_adapter.py`).
- Completed full Phase V0.7 Multimodal Pipelines & Directed Acyclic Graph (DAG):
  - Domain contracts: `Modality` enum, `DataSource`, `PipelineNode`, `PipelineGraph`, and `@runtime_checkable` `ModalityPluginPort`.
  - DAG engine: `GraphValidator` (DFS cycle detection with back-edge path reconstruction, modal typing compatibility, Kahn's topological sort).
  - Feature fusion: `FeatureFusionNode` (horizontal concatenation of tabular features and dense embedding arrays with row-alignment and column disambiguation).
  - Vision plugin & encoder: `ImageModalityPlugin` (header validation, directory/file loading) and `ImageEncoderNode` (deterministic embedding generation, caching, deep learning support).
  - Integration: `AutoMLWorkspace.build_multimodal_pipeline`, `execute_pipeline`, `fit_predict_multimodal`, `ExecutePipelineCommand`, `ValidatePipelineGraphQuery`, `GetPipelineExecutionOrderQuery`.
  - Full test coverage: 51 tests across 4 suites (`test_v07_domain.py`, `test_v07_pipeline_graph.py`, `test_v07_image_plugin.py`, `test_v07_multimodal_e2e.py`).
- Completed repository documentation bootstrap according to layered Markdown standard.
- Implemented and verified full V0.5 Feature Discovery & Selection subsystem.
- Implemented and verified full V0.6 Plugin Architecture:
  - Domain ports (`PluginPort`, `ModelPluginPort`, `MetricPluginPort`, `PreprocessorPluginPort`) and capability declarations (`PluginCapability`, `PluginType`).
  - Application `PluginRegistry` with registration, discovery, task filtering, and typed retrieval.
  - Engine `CompatibilityValidator` protecting against invalid task/modality combinations.
  - Built-in scikit-learn model plugins (`logistic_regression`, `random_forest`, `ridge`, `svc`, `svr`, `kmeans`).
  - External framework adapters (`LightGBMPlugin`, `XGBoostPlugin`) with automatic fallback to scikit-learn gradient boosting when native packages are absent.
  - Business metric plugins (`CostSensitiveMetricPlugin` for financial loss minimization, `WeightedF1MetricPlugin` for recall/precision prioritization).
  - Integration into `SklearnTrainer`, `SearchSpaceBuilder`, and `AutoMLWorkspace`.
  - CQRS query `ListPluginsQuery` registered in `bootstrap.py`.
  - CLI subcommand `automl plugin list` (table and JSON formats).
- Implemented and verified Kaggle tabular improvements:
  - **Tarea A:** Heurística de alta cardinalidad e identificadores en `DatasetProfiler` (`tests/test_profiler_cardinality.py`, 12 tests).
  - **Tarea B:** `VotingEnsemblePlugin` y motor de blending `VotingBlender` con soft voting y pesos (`tests/test_ensemble_plugin.py`, 14 tests).
  - **Tarea C:** Mapeo automático de plantilla de sumisión Kaggle con `--template sample_submission.csv` en `GenerateSubmissionCommand`, `PredictDatasetQuery` y CLI `automl predict` (`tests/test_submission_template.py`, 5 tests).
- Historical validation before Workbench integration: 95 tests passed with 86% coverage.
- Synchronized all documentation files (`AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPER_GUIDE.md`, `TASKS.md`, `README.md`).
 
## Blocked
 
*(None)*
 
## Next
 
1. Independent OOF benchmark and promotion evaluation: [protocol](../docs/features/oof-blending/spec.md).
2. Remaining tabular plugins and V0.8–V1.0: [current backlog](../TASKS.md).

## Relevant files

- [`src/automl/domain/ports.py`](../src/automl/domain/ports.py)
- [`src/automl/domain/features/selection_strategy.py`](../src/automl/domain/features/selection_strategy.py)
- [`src/automl/domain/features/evidence.py`](../src/automl/domain/features/evidence.py)
- [`src/automl/engine/features/`](../src/automl/engine/features/)
- [`src/automl/engine/planning/ablation_planner.py`](../src/automl/engine/planning/ablation_planner.py)
- [`src/automl/application/services/workspace.py`](../src/automl/application/services/workspace.py)
- [`src/automl/interfaces/cli/main.py`](../src/automl/interfaces/cli/main.py)
- [`tests/test_v05_features.py`](../tests/test_v05_features.py)

## Historical Review — 2026-09-30

- User requested a project review focused on Markdown; reviewed documentation against source, CLI help, and task catalog. Existing working-tree changes were preserved.
- Validation: 161 tests passed, coverage 85.14%, one sklearn deprecation warning. Web tests required permission to open local sockets outside the sandbox. Relative Markdown file links checked successfully; `file:///home/fenic/...` links remain nonportable.
- Findings: conflicting package/CLI/guide versions; V0.7 and interaction-generation status conflicts; stale test counts; README/architecture omit existing web interface; extension guide needs plugin-based examples; roadmap/spec/history need clear ownership and traceability.
- Reproduced `ModuleNotFoundError` in model `register_plugin()` caused by import of nonexistent `automl.domain.models.model_spec`; no implementation changes made. Follow-up recorded in TASKS.md.

## Documentation and Plugin Fix — 2026-09-30

- Fixed model plugin registration to import `ModelSpec` from `domain.models.registry`; added regression coverage for model discovery, task incompatibility, metric-only registration, and training the documented custom model.
- Added `examples/plugins/custom_model.py`; executed it successfully through application buses with an independent temporary workspace.
- Centralized version 0.7.0 in `automl.__version__`; setuptools metadata, CLI and Workbench share this source. Reinstalled editable package without changing runtime dependencies and confirmed metadata/CLI parity.
- Updated README, architecture, developer/contribution guides, agent validation requirements and feature documents. Added docs index/capability evidence, native/fallback backend instructions and proposed OOF evaluation protocol. Replaced machine-specific links and clarified historical plans and preview UI responses.
- TASKS.md owns current status. Stacking, OOF prediction, dedicated feature-selection benchmark and automatic backend/environment manifests remain explicitly pending; no new ML capabilities were claimed for those items.
- Final validation: 164 tests passed, coverage 85.23%, one existing sklearn deprecation warning. Web tests ran with local socket access. Checked links/anchors in 19 Markdown files, CLI help/task catalog, installed version and `git diff --check`.
- An intermediate coverage run was invalidated by moving an import during execution; the final full run used fixed source files. Existing frontend and gradient-boosting working-tree changes were preserved.

## Git Organization — 2026-09-30

- Branch `fix/docs-and-plugin-registration` starts from updated `origin/main`, which includes merged Workbench PR #11. Commits separate plugin registration, shared version and documentation.
- Prior frontend/gradient-boosting edits and their operational notes remain in the original `feat/web-dashboard-ui` checkout; excluded from this PR.

## OOF Implementation — 2026-09-30

- Created isolated branch `feat/oof-prediction-blending` from `fix/docs-and-plugin-registration`; PR #13 remains unmerged for human review tomorrow. Original frontend changes remain in their existing checkout.
- Added binary OOF engine with fold-local preprocessing, identical stratified splits, fixed equal weights, model probability/class validation and budget/control checks.
- Added command returning an experiment ID and read-only report query; stored OOF/test predictions, source/config hashes, concrete backends and library/plugin versions via existing trial artifacts.
- Integrated CLI `predict --folds` and Workbench submission checkbox/HTTP path. Standard predictions remain supported; OOF artifact queries reject different test data and changed predictions.
- Candidates remain unpromoted; no Kaggle gain claimed. Independent holdout/benchmark remains pending. Fold models are not serialized; rerun for new test data. Budget and pause/cancel are checked between fits.
- Validation: 196 tests passed, 86.01% coverage, one existing sklearn deprecation warning. CLI/help/task catalog, JavaScript syntax, Markdown links/anchors and diff checks passed.
- Confirmed pre-existing ordinary prediction ownership bug and reported [blackboard issue #14](https://github.com/Jfenic/CATML/issues/14); OOF paths validate ownership.
- Preparing a separate draft PR against `fix/docs-and-plugin-registration`; no merges performed. After human merge of #13, retarget the OOF PR to main.



## Kaggle visual launch and persisted trial fix — 2026-09-30

- User requested a visual trial of the current Kaggle problem. Loaded existing `.automl/s6e9_automl`: target Will_Buy_EV, 4 experiments, 11 saved trials, best stored validation ROC-AUC 0.9412. This is a local validation score, not a Kaggle leaderboard result.
- Real-data smoke test exposed GET /api/experiments crashing because TrialResult does not contain parameters. Reported issue #18 and isolated the repair on `fix/workbench-persisted-trials` from main, separate from jobs PR #17 (CI green on Python 3.10/3.12).
- Extended the read-only experiment-trial DTO with detached parameters fetched from Trial; HTTP consumes that query. Missing legacy Trial records return empty parameters. Added two regressions and ran the complete branch suite: 198 passed, 86.22% coverage, one existing sklearn warning.
- Visual server runs from `/tmp/catml-kaggle-preview`, combining committed jobs code with the separate trial fix, using the original competition paths/workspace. Listening at http://localhost:8080; verified experiments endpoint HTTP 200, and browser loaded page/API/static modules. Original modified checkout files preserved.

## Persistent Jobs — 2026-09-30

- Verified PRs #13/#15 passed CI; #15 had merged into the documentation branch after #13 merged into main. Opened integration PR #16, subsequently merged by the reviewer. Updated isolated jobs branch from main without touching original checkout edits.
- Added pure domain Job/JobStatus and JobRepositoryPort, application job validation/control and operation execution, SQLite atomic idempotency/claim/state transitions, and one process-leased local worker.
- Workbench experiments/OOF/submissions now submit jobs and poll persistent snapshots. Activity survives page reload. Replaced simulated experiment progress with completed models/folds; expose pause/resume/cancel/retry and CLI job operations.
- Controls committed before finalization win completion races. Worker lease remains held while an active fit finishes during shutdown. Restart marks abandoned active work interrupted, preserving queued work and requiring explicit retry.
- Experiment checkpoints preserve finished model results. Partial OOF restarts; no exactly-once execution or forced fit interruption claimed. HPO remains synchronous. Documented Linux/macOS worker support, legacy endpoint compatibility and artifact/retry limits in spec and ADR 003.
- Added failed-model retry regression: retry refits failed/pending models and reuses successful results rather than repeatedly skipping failed trials at the final checkpoint.
- Final validation: 229 Python tests passed, 86.31% coverage, one existing sklearn deprecation warning. Four Node adapter tests passed; checked CLI job help/task catalog, all 21 Markdown file links, changed JavaScript syntax and diff whitespace. Original checkout still contains only its eight pre-existing modified files.
- Prepared an independent draft PR against main. No merges performed.
