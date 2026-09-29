# Tasks

## Now (Active Phase — AutoML Workbench & Kaggle Playground Series S6E9)

- [x] Feature: CATML AutoML Workbench Server (`src/automl/interfaces/web/server.py`): Hexagonal interface adapter with zero external dependencies (`ThreadingHTTPServer`), exposing REST endpoints via `CommandBus`, `QueryBus`, and `AutoMLWorkspace` for Mission Control, run lifecycle controls (`pause`, `resume`, `cancel`, `clone`), dataset inspection, planner explicability, meta-learning knowledge, agent hypotheses, and Kaggle validation checklist.
- [x] Feature: Frontend Modular Architecture & Design Patterns (`src/automl/interfaces/web/static/`):
  - Architecture: EventBus (PubSub), WorkbenchStore (Observable/State pattern), CATMLApiClient (Adapter/Repository), Modular View Controllers.
  - Design System (`workbench.css`): Dense technical dark theme with semantic color system (🔵 system, 🟢 proven gain, 🟠 warning, 🔴 error/stop, 🟣 intelligence).
  - Views: Overview (Mission Control), Dataset Inspector ("¿Qué entendió CATML?"), Experiment Studio (explicable planner "Why this?", HPO, controls), Compare View (multi-select, metrics table, diff inspector), Visual Pipeline DAG (interactive execution nodes), Knowledge (V0.8 meta-learning preview), Lateral Agent Drawer (V1.0 hypothesis engine "Proponer ≠ Aceptar"), Kaggle S6E9 Center.
- [x] Feature: CLI integration `automl ui [--port PORT] [--workspace WORKSPACE]` in `src/automl/interfaces/cli/main.py`.
- [x] Testing: Comprehensive web test suite in `tests/test_web_dashboard.py` verifying static asset serving, dataset registration, profiling, experiment execution, pause/resume, and submission generation.
- [ ] Next: Stratified 5-Fold Out-of-Fold (OOF) Prediction Blending (`--folds 5`) to elevate Kaggle S6E9 score past 0.945+.

## Next (Roadmap Phases)

- [ ] V0.8 Phase: Meta-learning & knowledge base for warm-start policies (dataset meta-features, historical memory)
- [ ] V0.9 Phase: LLM Agent tools (`AgentTool` wrappers) and permission-controlled bus interface
- [ ] V1.0 Phase: Autonomous Experiment Agent (Planner, Critic, Orchestrator for hypothesis-driven exploration)

## Backlog — Mejoras Tabulares Identificadas (Kaggle Benchmarking)

> *Nota para colaboradores: Las especificaciones de implementación, archivos permitidos y comandos de prueba para estas tareas se encuentran detalladas en [`CONTRIBUTING.md`](CONTRIBUTING.md).*

- [x] Heurística automática de alta cardinalidad en `DatasetProfiler` (excluir IDs y texto masivo > 0.7 ratio de unicidad) — *Ver Tarea A en CONTRIBUTING.md*
- [x] Plugin de Ensamble y Blending (`VotingEnsemblePlugin` / `StackingPlugin`) promediando el Top K de modelos — *Ver Tarea B en CONTRIBUTING.md*
- [x] Mapeo automático de plantilla de sumisión (`--template sample_submission.csv`) en `GenerateSubmissionCommand` — *Ver Tarea C en CONTRIBUTING.md*
- [x] Generación automática de variables de interacción (ratios numéricos y target encoding) — *Ver Tarea D en CONTRIBUTING.md*

## Completed

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
- [x] V0.6 Phase: `LightGBMPlugin` and `XGBoostPlugin` with transparent fallback to scikit-learn HistGradientBoosting
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
