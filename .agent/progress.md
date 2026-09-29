# Progress

## Current
 
Phase V0.7 Multimodal Pipelines & Directed Acyclic Graph (DAG) — Tracks Dev 1 & Dev 2 completed and merged on `main`:
- Paso 0 (Base Domain Contracts): `Modality`, `DataSource`, `PipelineNode`, `PipelineGraph`, `ModalityPluginPort` (6 tests passing).
- Track Dev 1: `GraphValidator` (DAG validation, cycle detection, topological sort) and `FeatureFusionNode` (early fusion, index alignment) (17 tests passing).
- Track Dev 2: `ImageModalityPlugin` (image format detection, batch loading) and `ImageEncoderNode` (deterministic feature extraction, caching, deep learning support) (20 tests passing).
- Both tracks fully integrated into `main`.
- Final Integration: Connected in `workspace.py`, registered in `bootstrap.py`, full suite end-to-end (8 tests passing in `tests/test_v07_multimodal_e2e.py`).
- Total suite: 146 tests passing, >=87% coverage.
- Next: Visual UI (interactive dashboard) and Context-Aware Feature Engineering.

 
## Completed
 
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
