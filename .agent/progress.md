# Progress

## Current
 
Active Track: **CATML AutoML Workbench & Kaggle Playground Series S6E9**:
- Web Interface Hexagonal Adapter (`src/automl/interfaces/web/server.py`): Zero-dependency standard library `ThreadingHTTPServer` mapped exclusively to `CommandBus`, `QueryBus`, and `AutoMLWorkspace`. Endpoints for overview, runs, lifecycle controls (`pause`, `resume`, `cancel`, `clone`), experiments, leaderboard, dataset profiling, planner explicability (`/api/plan`), meta-learning knowledge preview (`/api/knowledge`), agent hypotheses (`/api/agent/hypotheses`, `Proponer ≠ Aceptar`), and Kaggle checklist (`/api/kaggle/status`, `/api/predict`).
- Frontend Modular Architecture (`src/automl/interfaces/web/static/`): Clean code design patterns including `EventBus` (PubSub), `WorkbenchStore` (State/Observable), `CATMLApiClient` (Adapter/Repository), and independent View Controllers (`overview.js`, `studio.js`, `datasets.js`, `compare.js`, `pipeline.js`, `knowledge.js`, `kaggle.js`, `agent.js`, `new_experiment.js`).
- Design System (`css/workbench.css`): Dense technical dark theme with semantic color system (🔵 system/execution, 🟢 proven gain, 🟠 warning/waiting, 🔴 error/stop, 🟣 CATML intelligence).
- CLI Command: `automl ui [--port PORT] [--workspace WORKSPACE]`.
- Test Suite: 161 tests passing, >=85% test coverage (`tests/test_web_dashboard.py`).
- Next: Stratified 5-Fold OOF Predictor (`--folds 5`) to blend fold predictions and surpass 0.945+ in Kaggle S6E9.
 
## Completed
 
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
- Total suite: 95 tests passing with 86% coverage.
- Synchronized all documentation files (`AGENTS.md`, `CONTRIBUTING.md`, `DEVELOPER_GUIDE.md`, `TASKS.md`, `README.md`).
 
## Blocked
 
*(None)*
 
## Next
 
1. Phase V0.7: Multimodal Pipelines & Data Fusion (specs in `AutoML_Arquitectura_Tecnica.md` §V0.7 and `docs/features/multimodal/`):
   - Domain `Modality` enum and `DataSource` abstraction.
   - Generalize `PipelineGraph` (DAG of typed processing nodes) and `GraphValidator`.
   - `ImageModalityPlugin`, `ImageEncoderNode`, and `FeatureFusionNode` (early/late fusion).
   - Hypothesis-driven multimodal comparison experiments (tabular vs. image vs. tabular+image).

## Relevant files

- [`src/automl/domain/ports.py`](file:///home/fenic/top_project/CATML/src/automl/domain/ports.py)
- [`src/automl/domain/features/selection_strategy.py`](file:///home/fenic/top_project/CATML/src/automl/domain/features/selection_strategy.py)
- [`src/automl/domain/features/evidence.py`](file:///home/fenic/top_project/CATML/src/automl/domain/features/evidence.py)
- [`src/automl/engine/features/`](file:///home/fenic/top_project/CATML/src/automl/engine/features/)
- [`src/automl/engine/planning/ablation_planner.py`](file:///home/fenic/top_project/CATML/src/automl/engine/planning/ablation_planner.py)
- [`src/automl/application/services/workspace.py`](file:///home/fenic/top_project/CATML/src/automl/application/services/workspace.py)
- [`src/automl/interfaces/cli/main.py`](file:///home/fenic/top_project/CATML/src/automl/interfaces/cli/main.py)
- [`tests/test_v05_features.py`](file:///home/fenic/top_project/CATML/tests/test_v05_features.py)
