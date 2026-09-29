# Tasks

## Now (Active Phase V0.7 — Parallel Tracks)

> *Especificaciones completas en [`docs/features/multimodal/spec.md`](docs/features/multimodal/spec.md), plan en [`docs/features/multimodal/plan.md`](docs/features/multimodal/plan.md) y asignaciones en [`CONTRIBUTING.md`](CONTRIBUTING.md).*

- [x] Paso 0 (Base Común en main): Contratos puros en `domain/modalities/` (`Modality`, `DataSource`), `domain/pipelines/` (`PipelineNode`, `PipelineGraph`) y `ModalityPluginPort` (6 tests passing en `tests/test_v07_domain.py`).
- [ ] Track Dev 1 (`feat/v07-pipeline-graph-engine`): `GraphValidator` y `FeatureFusionNode` en `engine/pipeline/` (*Ver Tarea V0.7-A en CONTRIBUTING.md*).
- [ ] Track Dev 2 (`feat/v07-image-plugin`): `ImageModalityPlugin` y `ImageEncoderNode` en `plugins/modalities/` y `engine/vision/` (*Ver Tarea V0.7-B en CONTRIBUTING.md*).
- [ ] Integración V0.7: Orquestación en `workspace.py`, registro en `bootstrap.py` y suite end-to-end `tests/test_v07_multimodal_e2e.py`.

## Next (Roadmap Phases)

- [ ] V0.8 Phase: Meta-learning & knowledge base for warm-start policies (dataset meta-features, historical memory)
- [ ] V0.9 Phase: LLM Agent tools (`AgentTool` wrappers) and permission-controlled bus interface
- [ ] V1.0 Phase: Autonomous Experiment Agent (Planner, Critic, Orchestrator for hypothesis-driven exploration)

## Backlog — Mejoras Tabulares Identificadas (Kaggle Benchmarking)

> *Nota para colaboradores: Las especificaciones de implementación, archivos permitidos y comandos de prueba para estas tareas se encuentran detalladas en [`CONTRIBUTING.md`](CONTRIBUTING.md).*

- [x] Heurística automática de alta cardinalidad en `DatasetProfiler` (excluir IDs y texto masivo > 0.7 ratio de unicidad) — *Ver Tarea A en CONTRIBUTING.md*
- [x] Plugin de Ensamble y Blending (`VotingEnsemblePlugin` / `StackingPlugin`) promediando el Top K de modelos — *Ver Tarea B en CONTRIBUTING.md*
- [x] Mapeo automático de plantilla de sumisión (`--template sample_submission.csv`) en `GenerateSubmissionCommand` — *Ver Tarea C en CONTRIBUTING.md*
- [ ] Generación automática de variables de interacción (ratios numéricos y target encoding) — *Ver Tarea D en CONTRIBUTING.md*

## Completed

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
