# Progress

## Current
 
Active Track: **CATML AutoML Workbench & Kaggle Playground Series S6E9**:
- Web Interface Hexagonal Adapter (`src/automl/interfaces/web/server.py`): Zero-dependency standard library `ThreadingHTTPServer` mapped exclusively to `CommandBus`, `QueryBus`, and `AutoMLWorkspace`. Endpoints for overview, runs, lifecycle controls (`pause`, `resume`, `cancel`, `clone`), experiments, leaderboard, dataset profiling, planner explicability (`/api/plan`), meta-learning knowledge preview (`/api/knowledge`), agent hypotheses (`/api/agent/hypotheses`, `Proponer ≠ Aceptar`), and Kaggle checklist (`/api/kaggle/status`, `/api/predict`).
- Frontend Modular Architecture (`src/automl/interfaces/web/static/`): Clean code design patterns including `EventBus` (PubSub), `WorkbenchStore` (State/Observable), `CATMLApiClient` (Adapter/Repository), and independent View Controllers (`overview.js`, `studio.js`, `datasets.js`, `compare.js`, `pipeline.js`, `knowledge.js`, `kaggle.js`, `agent.js`, `new_experiment.js`).
- Design System (`css/workbench.css`): Dense technical dark theme with semantic color system (🔵 system/execution, 🟢 proven gain, 🟠 warning/waiting, 🔴 error/stop, 🟣 CATML intelligence).
- CLI Command: `automl ui [--port PORT] [--workspace WORKSPACE]`.
- Validation results: see dated entries below; run the entire suite with `--cov-fail-under=85`.
- Current: OOF integrated into main through PR #16; persistent jobs implemented and validated on `feat/persistent-job-queue`, prepared for review. Independent benchmark evidence remains pending.
 
## Completed
 
- Dynamic Database Binding & Real Parity in Workbench UI: removed mockup placeholder strings ("Ensemble #7", 0.94621) from `studio.js`, `overview.js`, and `kaggle.js`, correctly displaying real SQLite metrics (best model `xgboost`, CV `0.94123`, 11 trials), live status badge in `app.js` and `index.html`, and marking the Ensemble Blender as candidate to be trained.
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
