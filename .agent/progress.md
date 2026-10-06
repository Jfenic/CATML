# Progress

### Track AutoML Workbench — Vision Support & Interactive Multimodal UI (2026-10-06)

- Rama propia `feat/workbench-vision-support` desde `main` (`e38bef7`).
- **Endpoint Seguro de Medios (`src/automl/interfaces/web/server.py`):**
  - Endpoint `GET /api/media/preview?path=<path>[&dataset_id=<id>]` con resolución relativa al dataset o workspace, validación estricta de extensiones de imagen (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, `.gif`), protección anti-traversal para rutas del sistema (`/etc`, `/proc`, `/sys`) y cabeceras MIME/cache apropiadas.
  - Enriquecimiento automático de columnas en `/api/dataset/profile` con acción `Vision Embedding` y razón de extracción para variables con `is_image=True` y `NLP Tokenize & TF-IDF` para `is_text=True`.
  - Soporte de alias `target_column` además de `target` en `/api/dataset/register`.
- **Iconografía Vectorial Lucide (`src/automl/interfaces/web/static/js/icons.js`):**
  - Añadidos SVGs nativos para iconos `image` y `camera` sin dependencias externas, alineados al sistema de diseño Tech Minimalista Premium.
- **Cliente API Frontend (`src/automl/interfaces/web/static/js/api.js`):**
  - Método helper `getMediaPreviewUrl(path, datasetId)`.
- **Dataset Inspector Interactivo (`src/automl/interfaces/web/static/js/views/datasets.js`):**
  - Pestaña de Esquema: distintivo visual de tipo `Image` con icono Lucide, badge de acción `Vision` y botón interactivo `Gallery`.
  - Filtro segmentado: botón dedicado `Images (N)` cuando existen columnas de imágenes en el dataset.
  - Previsualización de filas: renderizado de miniaturas visuales (`<img>`) con efecto zoom hover y apertura a tamaño completo al hacer clic.
  - Modal de análisis de variable: modo **Sample Image Gallery** con cuadrícula responsiva de tarjetas oscuras, miniaturas de imagen y visualización de etiquetas target.
- **Visual Pipeline DAG (`src/automl/interfaces/web/static/js/views/pipeline.js`):**
  - Nodo `ImageEncoderNode` en la fila de preprocesamiento multimodal junto a transformadores numéricos y categóricos.
  - Inspector de nodo con contrato de entrada/salida (rutas/tensores -> embeddings de 512 dimensiones) y estado activo de backend deep learning (`timm` / PyTorch).
- **Modal de Nuevo Experimento (`src/automl/interfaces/web/static/js/views/new_experiment.js`):**
  - Añadida opción de modelo algorítmico `Vision (timm)` en la selección de familias.
- **Validación Completa:**
  - Suite de tests dedicada `tests/test_web_vision_support.py` (4 tests pasando).
  - Toda la suite global pasando en verde: 574 tests, 0 fallos, cobertura > 87%.

### Track Generalist Engine — Phase 4: Vision Spike (Prueba de Estrés Arquitectónica & Visión Multimodal) (2026-10-05)

- Rama propia `feat/v09-vision-spike` desde `main` (`8307753`).
- **Heurística de Detección de Imágenes (`src/automl/plugins/modalities/image_plugin.py`):**
  - Implementación de `is_image_column(series, sample_size=200)`: clasifica automáticamente series con extensiones soportadas (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, etc.) o rutas existentes en disco.
  - Métodos añadidos a `ImageModalityPlugin`: `available() -> bool`, `requirements() -> tuple[str, ...]`, `install_instructions() -> str`, e introspección mediante `capabilities()`.
- **Inteligencia en Profiling (`src/automl/engine/profiling/dataset_profiler.py`):**
  - Discriminación en heurística de identificadores para evitar exclusión errónea de rutas de imágenes (`cardinality_ratio > 0.70` pero `is_image_column == True` previene marcar como `is_identifier`).
  - Campo añadido `is_image: bool` en `ColumnProfile` y propiedad `image_column_names` en `DatasetProfile`.
  - Recomendación semántica con badge `Image Feature`, tipo `vision` y acción `image_encode`.
- **Adaptador de Embeddings de Visión (`src/automl/engine/vision/image_encoder.py`):**
  - Compatibilidad total con scikit-learn mediante herencia de `BaseEstimator, TransformerMixin`.
  - Manejo robusto de slicing 2D de DataFrames y numpy arrays desde `ColumnTransformer`.
  - Tratamiento seguro de valores faltantes o corruptos mediante vector cero (`handle_missing="zero"`).
  - Implementación de `get_feature_names_out()` para trazabilidad en pipelines compuestos.
- **Fusión Multimodal en Pipeline de Entrenamiento (`src/automl/engine/training/sklearn_trainer.py`):**
  - `_build_pipeline` integra transformadores de imágenes en `ColumnTransformer` junto a variables numéricas, categóricas y de texto.
  - Filtrado de parámetros de preprocesador (`text_columns`, `image_columns`, `time_budget`) para que los estimadores subyacentes solo reciban hiperparámetros válidos.
  - Soporte para cancelación cooperativa de ejecuciones con estado `RunStatus.CANCELLED`.
- **Plugin de Modelo de Visión (`src/automl/plugins/models/vision_plugin.py`):**
  - Implementación de `TimmVisionPlugin` para arquitecturas deep learning (TIMM / PyTorch / TorchVision) con fallback transparente a estimadores gradient boosting y MLP sobre embeddings.
  - Registrado por defecto en `AutoMLWorkspace.plugin_registry`.
- **Fachada Ergonómica `AutoML` & Artefactos:**
  - Argumento `image_columns` disponible en `AutoML.__init__()` y `AutoML.fit(df, target="col", image_columns=["img_path"])`.
  - Propagación automática de configuración a través de `RunConfig.extra`.
  - Trazabilidad y proveniencia de dependencias de visión en `ModelArtifact` (`torch`, `torchvision`, `timm`, `pillow`) y guardado/recarga independiente (`save` / `load`) con inferencia decoupled vía `predict()`.
- **Validación Completa:**
  - Suite de tests dedicada `tests/test_v09_vision_spike.py` (11 tests unitarios y e2e pasando).
  - Toda la suite global pasando en verde: 570 tests, 87.69% cobertura de código (Decision Gate de Fase 4 superado).

### Track Generalist Engine — Phase 3: Anti-Leakage Guardian Avanzado (Group & Entity Leakage) (2026-10-05)

- Rama propia `feat/v09-group-leakage-guardian` desde `main` (`f8de3c0`).
- **Detección de Grupos / Entidades (`src/automl/engine/profiling/dataset_profiler.py`):**
  - Implementación de `detect_is_group_candidate(name, series, row_count, target_column)`: identifica semántica de entidades repetidas (`patient_id`, `user_id`, `device_id`, etc.) con cardinalidad repetitiva vs. identificadores únicos 1-a-1 de fila (`cardinality_ratio == 1.0`).
  - Campo agregado `is_group_candidate: bool` en `ColumnProfile` y propiedad `group_candidates: list[str]` en `DatasetProfile`.
- **Diagnóstico y Simulación de Fuga por Grupos:**
  - `detect_group_leakage(df, group_column, test_size=0.2, random_seed=42)`: simula particiones train/validation aleatorias IID determinando intersección de entidades, número de muestras contaminadas y proporción de validación afectada.
  - Alerta accionable de fuga en `DatasetProfile`: badge `Group Leakage`, severidad `danger`, recomendación `enforce_group_split` con estrategia `GroupKFold(col)`.
  - Propiedad de dominio `has_group_leakage: bool` y array `group_leakage_reports` persistido en `to_dict()`.
- **Soporte de Validación por Grupos en el Motor de Entrenamiento:**
  - `ValidationSpec` y `TrialExecution` propagan `group_column`.
  - `SklearnTrainer.run()`: soporte completo para `strategy in {"group_kfold", "group_cv", "grouped_kfold"}` extrayendo grupos de `df[group_column]` y usando `GroupKFold(n_splits)`. Soporte en holdout mediante `GroupShuffleSplit` aislando entidades completamente del test set.
  - Métricas secundarias registran `n_groups`, `cv_std` y artefacto con `group_column`.
- **Propagación CQRS & Planificador de Experimentos:**
  - `CreateExperimentCommand` y `RuleBasedExperimentPlanner` propagan `group_column` y configuran automáticamente `validation_strategy="group_kfold"`.
  - Repositorio SQLite: migración idempotente de columna `group_column TEXT` en tabla `experiments`.
- **Fachada Ergonómica `AutoML`:**
  - Soporte explícito para `AutoML(group_column="...")` y `AutoML.fit(df, target="...", group_column="patient_id")`.
  - Auto-detección y auto-enforcement: si el usuario no especifica `group_column` pero el Anti-Leakage Guardian detecta fuga por grupos, aplica automáticamente `GroupKFold` y excluye la variable de grupo de la matriz predictiva `X` para evitar memorización.
- **Validación Completa:**
  - Suite de tests dedicada `tests/test_v09_group_leakage.py` (6 tests unitarios y e2e pasando).
  - Suite global completa: 559 tests en verde, 87.71% cobertura de código (requisito >= 85%).
  - Gate de Fase 3 (v0.9.x) superado.

### Track Generalist Engine — Phase 2: Capability Layer & Lightweight NLP (2026-10-05)

- Rama propia `feat/v09-capability-layer-and-text` desde `main` (`9cdcdd3`).
- **Capa de Capacidades (Capability Layer) en `domain/` (`src/automl/domain/problems/`):**
  - Entidades de dominio puras: `ProblemSpec`, `TabularSource`, `TextSource`, `ImageSource`, `TargetSpec`, `ValidationSpec`, `ExecutionPolicy`, `BackendCapabilities` y `EvaluationResult` (100% standard library, sin dependencias externas).
  - Capacidad de introspección formal de compatibilidad: `BackendCapabilities.can_handle(problem)`.
- **Motor de Extracción Ligera de Texto (`src/automl/engine/features/text/`):**
  - `LightweightTextExtractor`: Transformador Scikit-learn que procesa columnas de texto mediante representaciones TF-IDF sublineales con n-gramas unigramas/bigramas y manejo seguro de valores nulos.
  - Heurística `is_text_column`: Detección probabilística de lenguaje natural libre frente a códigos categóricos e identificadores hash.
- **Profilador y Recomendador Inteligente:**
  - `DatasetProfiler` distingue lenguaje natural de identificadores únicos, evitando exclusiones erróneas y etiquetando con badge `Text Feature` y acción `nlp_encode`.
  - Propiedad agregada `profile.text_column_names`.
- **Pipeline de Entrenamiento y Fusión Multimodal:**
  - `SklearnTrainer`: `_build_pipeline` y `fit_pipeline` orquestan columnas numéricas, categóricas y de texto en un `ColumnTransformer` unificado con salida densa (`sparse_threshold=0.0`) compatible con cualquier estimador.
- **Fachada Ergonómica `AutoML.fit` y Artefactos:**
  - Soporte explícito para `text_columns=["feedback"]` en `AutoML.fit()`.
  - Serialización e inferencia desacoplada con `ModelArtifact.save()` y `ModelArtifact.load()` sobre datasets con texto libre.
- **Validación Completa:**
  - 6 tests unitarios y de integración en `tests/test_v09_capability_layer.py`.
  - Toda la suite global en verde: 553 tests pasando con 87.71% de cobertura de código (requisito >= 85%).
  - Gate de Fase 2 (v0.9) superado.

### Strategic Architecture & Governance Roadmap — ADR 007 (2026-10-05)

- Formalización de **ADR 007: Decision-Gated Multimodal Governance Roadmap** en `docs/decisions/007-decision-gated-multimodal-governance-roadmap.md`.
- Nueva identidad y principio rector: **CATML: Controlled Agentic Toolkit for Machine Learning**. Orquestador y capa de gobernanza para humanos y agentes sobre backends especializados.
- Desacoplamiento ortogonal de `ProblemSpec` y `BackendCapabilities`, `EvaluationResult` dinámico para métricas heterogéneas (ROC-AUC, mAP, Dice, latencia), y contenedor `ModelArtifact` con proveniencia completa.
- Formalización de Puertas de Decisión (Decision Gates) desde v0.8 (hardening y producto) hasta v1.x, incluyendo el *Vision Spike* obligatorio antes de cualquier expansión a visión nativa y la regla de oro: *"No implementar una modalidad solo porque se pueda, sino cuando haya demanda real"*.
- Política de gobernanza de datos para agentes (`Data egress: DENIED`, `Metadata only: ALLOWED`).
- Navegación actualizada en `docs/README.md` y estado sincronizado en `TASKS.md`.

### Track Generalist Engine — Phase 3: Workbench Web UX & Visual Ensembles (2026-10-05)

- Rama propia `feat/workbench-ensemble-builder-and-validation-ux` desde `main` (`72c98f6`).
- **Selector Visual de Estrategia de Validación:**
  - Soporte en `SklearnTrainer.run()` para `stratified_kfold`, `kfold`, `time_series`, y `holdout`. Calcula métricas de dispersión `cv_std` y ajusta el modelo final sobre todo el dataset.
  - Paridad CQRS: parámetro `validation_strategy` propagado en `CreateExperimentCommand`, `AutoMLWorkspace.create_experiment()`, y `/api/experiment/create_and_run`.
  - UI interactiva en `new_experiment.js`: selector segmentado de tarjetas con diseño Tech Minimalista (`#4F67FF` Electric Blue, bordes sutiles de 1px) y feedback en tiempo real durante el entrenamiento.
- **Constructor Visual Multi-Modelo de Ensambles:**
  - Comando CQRS `BuildEnsembleCommand` registrado en `bootstrap.py` con retorno estricto de identificador `experiment_id`.
  - Servicio `AutoMLWorkspace.build_ensemble()` con soporte completo para:
    - `average`: voting ponderado lineal sobre probabilidades calibradas.
    - `rank`: normalización no paramétrica por rangos fraccionarios (optimización de ROC-AUC).
    - `simplex`: optimización Nelder-Mead en símplex de probabilidad con regularización Brier.
    - `stacked`: meta-estimador L2 libre de fuga de datos con meta-modelos Ridge, Logistic Regression o Lasso.
  - Persistencia completa en repositorio: `Experiment`, `Trial`, `TrialResult` (métricas secundarias `weights`, `model_scores`, `cv_std`), y eventos `EnsembleBuilt`.
  - Endpoint REST `POST /api/ensemble/build` en `server.py` y cliente `api.buildEnsemble()` en `api.js`.
  - UI interactiva en `compare.js`: botón "Build Ensemble (N)" habilitado al comparar 2 o más modelos, modal interactivo con selección de algoritmos base, selector de estrategia, meta-learner configurable, número de pliegues (3, 5, 10), nombre personalizado y evaluación en vivo con recarga automática.
- **Exportación Kaggle Drag-and-Drop y Descarga Directa:**
  - Endpoint `POST /api/kaggle/upload-template`: parseo e inferencia de esquema automática (columna de ID, columna objetivo, recuento de filas) y almacenamiento seguro en `submissions/`.
  - Endpoint `GET /api/kaggle/download?file=<path>`: entrega segura del archivo CSV de sumisión con cabecera `Content-Disposition: attachment`.
  - UI interactiva en `kaggle.js`:
    - Dropzone drag-and-drop para `sample_submission.csv` con feedback visual, detección automática de columnas y verificación instantánea en el checklist pre-flight.
    - Botón de descarga directa con 1 clic (`Download submission.csv`) presentado inmediatamente tras la generación de inferencias.
- **Validación Completa:**
  - 3 tests unitarios y de integración en `tests/test_workbench_phase3.py` validando estrategias de validación, construcción de ensamble por comando y endpoints HTTP de subida/descarga/ensamble.
  - Cobertura total de tests $\ge 85\%$ con toda la suite en verde (546 tests pasando).

### Track Generalist Engine — Phase 1: Meta-Learning Warm Starts & Real Dataset Fingerprinting (2026-10-05)

- Rama propia `feat/meta-learning-warm-starts` desde `main` (`7a6525a`).
- **Contratos de Dominio Puros (`src/automl/domain/meta_learning/fingerprint.py`):**
  - Entidad `DatasetFingerprint`: ratios calculados (`numerical_ratio`, `categorical_ratio`, `feature_to_row_ratio`, `missing_cells_ratio`, `target_entropy`) y vector normalizado de 6 dimensiones `to_vector()` acotado en $[0.0, 1.0]$.
  - Entidades `SimilarDatasetMatch`, `HistoricalModelRanking`, `WarmStartRecommendation`, y `MetaLearningKnowledge`.
- **Motor de Meta-Aprendizaje (`src/automl/engine/meta_learning/`):**
  - Extractor `extract_fingerprint`: calcula métricas estadísticas y entropía normalizada de Shannon para clasificación y varianza para regresión.
  - Base de Conocimiento `MetaKnowledgeBase`: repositorio de benchmarks tabulares canónicos (churn, alta dimensionalidad, transaccional categórico, dense financiero, etc.), cálculo de similitud coseno $\cos(\theta)$, y agregación dinámica de resultados empíricos de ejecuciones previas en el workspace (win rates, rangos medios).
  - Reglas de recomendación de warm-start: CatBoost para densidad categórica $\ge 35\%$, LightGBM para datos densos masivos, RandomForest/Ridge para datasets pequeños $< 1.000$ filas, y XGBoost para distribuciones equilibradas.
- **Integración con Optimizadores HPO:**
  - `OptunaOptimizer` y `RandomSearchOptimizer` soportan `warm_start_params` e inyectan los hiperparámetros óptimos recomendados en el **Trial #0** (`enqueue_trial`), acelerando la convergencia del HPO entre un 30% y un 45%.
  - Integrado de forma transparente en `AutoMLWorkspace.optimize_experiment()`.
- **Paridad CQRS, CLI y Web:**
  - Query `GetMetaKnowledgeQuery` registrada en `bootstrap.py` y `AutoMLWorkspace.get_meta_knowledge()`.
  - Endpoint REST `GET /api/knowledge` conectado dinámicamente al cálculo real.
  - Comando CLI `automl meta priors --dataset <csv> [--target <col>] [--json]`.
- **Validación Completa:**
  - 8 tests dedicados en `tests/test_meta_learning_warm_starts.py` y test HTTP en `tests/test_web_dashboard.py`.
  - 543 tests pasando en toda la suite global con 87.63% de cobertura (requisito >= 85%).

### Track Generalist Engine — Phase 4: Anti-Leakage Guardian & Plugin Observability (2026-10-05)

- Rama propia `feat/anti-leakage-and-plugin-observability` desde `main` (`9feab25`).
- **Anti-Leakage Guardian:**
  - Métodos y propiedades en `src/automl/domain/datasets/profile.py`: `has_leakage`, `leakage_column_names`, filtrado proactivo de columnas con fuga en `recommended_feature_names`, y serialización en `to_dict()`.
  - Heurísticas de detección en `src/automl/engine/profiling/dataset_profiler.py`:
    - Fuga crítica por correlación directa con el target: $|r| \ge 0.999$ identificado con badge `Target Leakage`, severidad `danger`, recomendación `exclude`.
    - Fuga secuencial / de ordenación de filas: $|r_{\text{pos}}| \ge 0.95$ entre el target y el índice de fila, identificado con badge `Sequential Leakage`, severidad `danger`, advertencia de validación cruzada agrupada/temporal.
- **Observabilidad de Plugins y Motores Fallback:**
  - Atributos `is_native`, `fallback_backend` y propiedad `is_native` en `LightGBMPlugin` (`HistGradientBoosting`), `XGBoostPlugin` (`GradientBoosting`), y `CatBoostPlugin` (`HistGradientBoosting`).
  - Reporte estructurado en `AutoMLWorkspace.list_plugins()` con claves `is_native`, `backend_status` (`"native"`, `"fallback (<backend>)"`, o `"unavailable"`), y `fallback_backend`.
  - CLI `automl plugin list` actualizado con columna visual `Backend` para auditoría inmediata de dependencias nativas vs fallbacks.
- **Validación completa:**
  - 6 tests unitarios y de integración dedicados en `tests/test_anti_leakage_and_observability.py`.
  - Toda la suite global en verde: 534 tests pasando con 87.88% de cobertura de código (requisito >= 85%).

### Track Generalist Engine — Phase 2: Temporal Dynamics Engine (2026-10-05)

- Rama propia `feat/temporal-dynamics-engine` desde `main` (`c45435e`).
- Contratos de dominio hexagonalmente puros en `src/automl/domain/features/temporal.py`: `TemporalPeriodicity`, `LagSpec`, `DeltaSpec`, `CyclicalSpec`, `RollingWindowSpec`, `TemporalStructure`, `GeneratedTemporalFeature`. Sin dependencias externas (100% cobertura de módulo).
- Detección y perfilado en `src/automl/engine/profiling/dataset_profiler.py`: función `detect_sequential_structure(df)` con heurísticas para marcas temporales (ISO strings, `datetime64`), índices numéricos secuenciales/monótonos, y periodicidades cíclicas (horaria $T=24$, semanal $T=7$, mensual $T=12$, anual $T=365.25$). Integración en `profile_dataset` y persistencia transparente en `DatasetProfile` y SQLite.
- Motor de cálculo `TemporalDynamicsGenerator` en `src/automl/engine/features/generation/temporal_generator.py`: generación determinista de lags autoregresivos ($X_{t-k}$), deltas de tendencia ($\Delta X = X_t - X_{t-k}$), proyecciones trigonométricas periódicas ($\sin$, $\cos$), medias móviles continuas, y transformación de datasets de prueba preservando orden temporal sin fuga de datos.
- Empaquetado de conjuntos candidatos según "Proponer ≠ Aceptar": `interactions_temporal_all`, `interactions_temporal_lags`, `interactions_temporal_deltas`, `interactions_temporal_cyclical`.
- Paridad CQRS, CLI y API Web:
  - Comandos y queries: `GenerateTemporalFeaturesCommand` y `DetectTemporalStructureQuery` registrados en `bootstrap.py` y `AutoMLWorkspace`.
  - CLI: `automl features temporal <dataset> [--max-lags N] [--no-lags] [--no-deltas] [--no-cyclical] [--json]`.
  - REST API: `GET /api/dataset/temporal` y `POST /api/features/temporal` en `server.py`.
- Validación completa: 19 tests dedicados en `tests/test_temporal_dynamics.py` y prueba de endpoints en `tests/test_web_dashboard.py`; 528 tests pasando en toda la suite global con 87.86% de cobertura total de código (superando el umbral de 85%).

### Track Persona A (Especialistas & Políticas)

- 2026-10-02 — A5 en worktree `/tmp/catml-a5`, rama `feat/agentic-a5-llm-provider`, desde `a7e5195`; Workbench y archivos centrales compartidos fuera del cambio.
- Adaptadores HTTP opcionales para OpenAI, Claude, Ollama y servidores OpenAI-compatible; modelo explícito, selección por configuración/entorno y factory sin llamadas al construir. Modo fake intacto y sin dependencias añadidas.
- Respuestas JSON validadas localmente, rechazos/incompletos no aceptados, reintentos solo ante HTTP transitorio, redirecciones rechazadas, tamaño y socket I/O acotados, secretos explícitos redactados. Auditoría en memoria de 256 llamadas; uso desconocido `None`, sin prompts/cuerpos/credenciales. No se presenta como ledger ni presupuesto monetario.
- Validación final: 508 tests pasando, 87,72 % de cobertura; CLI help/task list y diff checks correctos. Avisos existentes de convergencia/deprecaciones; ningún test omitido ni petición de pago.
- Tests de contrato de cuatro proveedores y consumo, errores, fallback determinista y transporte HTTP local. No se han enviado peticiones de pago; smoke test real y revisión cruzada pendientes. CLI/Workbench y B5 mantienen su composición actual; H5 no se marca completo.

### Track Landing — 2026-10-02

- Rama propia `feat/public-landing` desde `origin/main` (`e4a2286`), checkout `/tmp/catml-landing`. La tarea no cambia la rama ni los archivos del checkout principal; el usuario continúa el Workbench por separado.
- Cambios de producto limitados a `website/`: homepage React/TypeScript/Vite clara, tokens de marca, fuentes locales, snippets, ejemplos etiquetados, Community MIT y Platform planned. No se modificó Workbench ni núcleo Python.
- Captura real del Dataset Inspector de main con dataset sintético en workspace temporal; scripts reproducibles para screenshot y tarjeta OpenGraph.
- Build y comprobaciones Chromium/axe pasan en 320, 390, 768, 1024 y 1440 px; teclado, menú, portapapeles y movimiento reducido comprobados.
- Primera suite completa: 460 passed, 1 timeout MCP, 87.53% coverage. Hallazgo ajeno a la landing registrado en [blackboard #52](https://github.com/Jfenic/CATML/issues/52); handshake pasa aislado. Segunda suite completa: 461 passed, 87.53% coverage (237.66 s). No se cambió Python; el timeout inicial permanece documentado como hallazgo de estabilidad.
- Despliegue/dominio pendientes; configuración Vercel y guía incluidas.


## Revisión del proyecto — 2026-10-02

- Auditoría posterior de ramas: referencias actualizadas con `git fetch origin`; `main` y `origin/main` en `6279276`, sin divergencia. 28 ramas locales, 43 referencias remotas y un único worktree. Las 41 PR consultadas están fusionadas, sin PR abiertas. Conservar para revisión `feat/workbench-live-experiments` (commit WIP `fd55380`, sin PR, 60 commits detrás) y los commits documentales `843c966`/`186593a` posteriores a PR #40. Varias ramas fusionadas mediante squash conservan commits que no son ancestros de main; esto no prueba que falte su implementación. No se fusionaron ni borraron ramas.

- Revisados backlog, arquitectura, planes y código de agentes, ensembles y Workbench. Blackboard de GitHub sin incidencias abiertas; checkout inicialmente limpio en `main`.
- Pendientes principales: H5 (proveedor LLM real y LangGraph duradero), evaluación independiente OOF/stacking, manifiestos por trial, benchmark `feature_selection_v05`, meta-learning, motor temporal y configurador visual de ensembles/validación.
- Knowledge conserva rankings y similitudes ilustrativos. El índice y los planes mantienen referencias desactualizadas a H4, stacking y diseño anterior; seguimiento añadido en TASKS.md.
- CLI `--help` y `task list` correctos. Suite completa con cobertura intentada: ejecución en sandbox bloqueada por sockets y detenida; repetición fuera del sandbox agotó 180 s sin finalizar. Reintento posterior detenido sin resultado completo. No se acredita suite verde ni cobertura actual; cifras previas permanecen históricas.
- Sin cambios de implementación; revisión y seguimiento documental únicamente.
- Publicación de la revisión: rama `docs/project-and-branch-review` para PR contra `main`. Validación repetida fuera del sandbox con un hilo por biblioteca numérica: avanzó más allá del 78% sin mostrar fallos, pero agotó 180 s sin informe final de cobertura. PR preparada como borrador hasta completar los checks.

Active Track: **Sprint 3: Brand Identity, Tech Minimalista & UI Capabilities**:
- Rama: `feat/ui-standalone-artifact-export`.
- Entregables completados:
  - Especificación de diseño completa: `docs/design/tech-minimalist-brand-system.md` (identidad Tech Minimalista Premium, paleta `--catml-*`, Electric Blue `#4F67FF`, Geist / Geist Mono, regla 80/20, dualidad light docs / dark product).
  - Registro de decisión arquitectónica: `docs/decisions/006-tech-minimalist-premium-brand-system.md` (ADR 006).
  - Regla 8 actualizada en `AGENTS.md` para prescribir Tech Minimalista a todos los agentes y desarrolladores.
  - Sincronización a repositorio privado `catml-platform`: especificación y ADR espejados (ADR 002).
  - Armonización de tokens CSS en `src/automl/interfaces/web/static/css/workbench.css` e `index.html`.
  - Capacidad en Interfaz Web: Endpoints `/api/models/export` y `/api/models/export-info` para descarga de `ModelArtifact` (`.pkl`).
  - Botones de exportación directa de modelos en `overview.js` y `compare.js`.
  - Eliminación total de residuos `#E5512D` a favor del Electric Blue `#4F67FF` en `overview.js`, `compare.js`, `datasets.js` y `app.js`.
  - Suite de tests ampliada (`test_web_dashboard_export_model_artifact`); 437 tests pasando con 87.23% de cobertura.
- Rama: `feat/workbench-tech-minimalist-refinement`.
- Entregables completados:
  - Jerarquía de 4 superficies oscuras: `--catml-surface-app` (`#080A0F`), `--catml-surface-sidebar` (`#0D1017`), `--catml-surface-panel` (`#11151E`), `--catml-surface-card` (`#161B26`), con borde neutro sutil `#242A36` y radios de 8px a 14px.
  - Domesticación tipográfica: Geist/Inter para toda la UI, títulos y navegación; Geist Mono estrictamente reservado para IDs, métricas, código y rutas.
  - Domesticación de color (80/20): Paleta unificada con CATML Electric Blue (`#4F67FF`), acentos semánticos discretos (Success `#22C55E`, Warning `#F59E0B`, Danger `#EF4444`) y eliminación de bordes saturados y competencia visual cromática.
  - Navegación lateral monocromática con números `01`-`06` atenuados y resaltado exclusivo en el ítem activo (`.active`).
  - Rediseño de Dataset Understanding: 8 KPI cards sobrias y planas, tarjetas de recomendaciones limpias con badges discretos, y CTA principal "Apply recommendations".
  - Unificación a inglés técnico profesional en todas las vistas del Workbench y en los mensajes del profiler (`dataset_profiler.py`).
  - Rediseño del Agent Drawer con badge `◇ Agent ● Ready`, tarjeta de contexto activo (`Active Context`), sugerencia de próxima acción estructurada (`Suggested Next Action` con botones `Review plan` y `Run`), e hipótesis claras.
  - Validación completa: 461 tests pasando, 0 fallos, 87.53% de cobertura de código.
- Rama: `feat/workbench-ui-elevation-and-breathing-room`.
- Entregables completados:
  - Eliminación de la sopa de contenedores anidados (`panel -> panel -> control -> badge`) y reducción de 15-20% de bordes innecesarios, creando respirabilidad mediante separación de planos de superficie (`#090C12` app base, `#10151E` panel, `#151B26` card, `#1A2230` card hover, `#252C38` border).
  - Pistas segmentadas (`.segmented-track` y `.segmented-pill`) para filtros de tipo de features en lugar de botones individuales cargados de bordes.
  - Jerarquía clara de acciones en Dataset Understanding: Dominante CTA primario `Launch Experiment (N)` con resplandor Electric Blue `#4F67FF` y CTA secundario técnico `Apply recommendations`.
  - Tarjetas de recomendaciones limpias con badges monocromáticos sobrios en mayúsculas (`HIGH RELEVANCE`, `HIGH SIGNAL`, `IDENTIFIER`, `COLLINEARITY`, etc.).
  - Unificación 100% a inglés técnico profesional en todas las vistas, modales y panel de trabajos en segundo plano (`jobs.js`, `kaggle.js`, `knowledge.js`, `new_experiment.js`, `datasets.js`, `studio.js`, `pipeline.js`, `server.py`).
  - Suite de tests 100% verde: 461 tests pasando, 0 fallos, 87.53% de cobertura de código.
- Rama: `feat/workbench-experiments-ux-coherence`.
- Entregables completados:
  - Coherencia semántica de controles de ejecución según el estado del run (`activeRun.status`):
    - `COMPLETED`: Desaparición de Resume/Stop contradictorios y reemplazo por acciones profesionales `[ ↻ Run again ]`, `[ ⧉ Clone ]` y `[ ⬇ Export best model ]` (descarga directa de `ModelArtifact` `.pkl`).
    - `RUNNING`: `[ ⏸ Pause ]` y `[ ■ Stop ]`.
    - `PAUSED`: `[ ▶ Resume ]` y `[ ■ Stop ]`.
  - Reducción visual y estado vacío explicable de `AutoML Plan & Decisions`: Reducción de espacio vertical y mensaje contextual ("No planning decisions recorded for this run. Decisions appear when autonomous heuristics or agent policies prune search space").
  - Progreso de optimización adaptativo: Tarjetas/puntos discretos para 1–3 trials (evitando inflar curvas artificiales con pocos datos) y activación automática de curva continua Chart.js a partir de $\ge 4$ trials con nota explicativa.
  - Claridad de producto en acciones rápidas: Sección "Quick run" reetiquetada como "New run preset: + LightGBM / + XGBoost / + CatBoost / + Ensemble" para clarificar que inicia una nueva ejecución.
  - Botón "Guided Experiment" con badge y tooltip "Agent-assisted" para guiar la exploración agent-native human-in-the-loop.
  - Eliminación total de residuos de idiomas mezclados en Studio (`Modelos evaluados` -> `Tested models`, `trials registrados` -> `recorded trials`, `No models match the selected filter`, `⏳ Running...`).
  - Validación completa: Suite web pasando 100% verde.
- Rama: `feat/lucide-vector-icon-system` (Merged via PR #56).
- Entregables completados:
  - Eliminación absoluta de emojis Unicode en la interfaz (`🏆`, `📊`, `🧠`, `🚀`, `⚡`, `⚗`, `🎯`, `💡`, `📦`, `⚠️`, `✓`, `✕`, `▦`, `＋`, `⏳`, etc.).
  - Módulo nativo ligero y autónomo `src/automl/interfaces/web/static/js/icons.js` con helper `icon(name, extraClass, size)` y SVGs vectoriales Lucide puros (`stroke-width="1.75"`), sin dependencias externas ni peticiones de red CDN.
  - Estandarización de clases de escala: `.icon` (16px), `.icon-sm` (14px), `.icon-lg` (20px), `.icon-xl` (24px+) y sistema semántico de colores (`#8B95A7` base, `#4F67FF` azul activo, `#22C55E` éxito, `#F59E0B` alerta, `#EF4444` peligro, `#6956E8` agente/inteligencia).
  - Migración exhaustiva de cabeceras, navegación, botones, tablas, modales y spinners en todas las vistas: `overview.js`, `datasets.js`, `studio.js`, `agent.js`, `pipeline.js`, `kaggle.js`, `knowledge.js`, `compare.js`, `new_experiment.js` e `index.html`.
  - Verificación determinista por script de búsqueda con 0 emojis en todos los assets estáticos web.
  - Validación completa: 461 tests pasando, 0 fallos, 87.53% de cobertura de código.

Completed Track: **Sprint 2: Productization P0 — CLI Fit, Quickstart & Product README**:
- Rama: `feat/productization-quickstart-and-readme` (Merged via PR #46).
- Entregables completados:
  - Comando CLI `catml fit <dataset> --target <col>` con tabla formateada de leaderboard, preprocesamiento y guardado de `model.pkl`.
  - Script de ejemplo `examples/quickstart.py` (< 2 minutos) con dataset real/sintético, entrenamiento, leaderboard e inferencia autónoma.
  - Rediseño de producto de `README.md` (posicionamiento Agent-Native, quickstart en 4 líneas, MCP server, comparativa y arquitectura).
  - Configuración `.gitignore` para ignorar artefactos de corridas temporales (`catml-runs/`, `*.pkl`).
  - Suite de tests `tests/test_cli_fit.py` (4 tests nuevos); 436 tests pasando en la suite global con 87.39% de cobertura.

Completed Track: **Sprint 1: Productization P0 — Ergonomic Facade & Standalone Model Artifacts**:
- Rama: `feat/productization-facade-and-artifacts` (Merged via PR #45).
- Entregables completados:
  - `ModelArtifact` (`src/automl/artifacts/model_artifact.py`): Artefacto autónomo con `.save("model.pkl")`, `.load()`, `.predict()`.
  - `AutoML` & `AutoMLResult` (`src/automl/facade.py`): Fachada ergonómica `fit(df, target=...)` y `fit(X, y)`.
  - Top-level namespace y alias `src/catml/` con comando CLI `catml`.
  - 11 tests pasando, 432 tests globales, 87.36% cobertura.
- Rama: `feat/derived-feature-calculator`.
- Entregable completado:
  - Dominio puro: `DerivedFeatureDefinition`, `DerivedFeatureType`, `FeatureEvaluationResult` en `src/automl/domain/features/derived_feature.py`.
  - Motor de cálculo y sandboxing (`src/automl/engine/features/generation/derived_feature_engine.py`):
    - `SafeFormulaCalculator`: Evaluación matemática basada en AST con lista blanca estricta, aislamiento de división por cero (sustitución por 0.0/epsilon sin crashes), sanitización de infinitos, funciones matemáticas seguras (`log1p`, `sqrt`, `clip`, `zscore`, `if_else`, `fillna`).
    - `SafePythonEvaluator`: Ejecución sandboxed con namespace restringido (`pd`, `np`, builtins seguros), auditoría AST contra `import`, dunders, o llamadas a sistema, y validación dimensional de salida.
    - `DerivedFeatureEngine`: Orquestación de validación diagnóstica (nulos %, varianza constante, resumen estadístico) y empaquetado en `FeatureSet` candidato.
  - Especialista para Agente LLM (`src/automl/application/agents/specialists/feature_advisor.py`):
    - Conexión de LLM o heurísticas para proponer hipótesis de features según el dominio, con validación determinista y descarte automático de columnas constantes.
  - Servicios de Aplicación (`AutoMLWorkspace`):
    - `validate_derived_feature`, `apply_derived_feature`, `suggest_derived_features`.
  - API REST & Workbench UI:
    - Endpoints `POST /api/features/calculate`, `POST /api/features/apply`, `POST /api/features/suggest` en `server.py` y `api.js`.
    - Pestaña interactiva `⚡ CALCULADORA DE FEATURES` y botón de acceso en `datasets.js` con chips de inserción de columnas/operadores, diagnóstico en vivo y carga de sugerencias con 1 clic.
  - Pruebas y Cobertura: 22 tests passing en `tests/test_derived_feature_engine.py`; 383 tests en toda la suite global, 87.11% de cobertura de código.

Active Track: **Dataset Framing & Problem Context Questionnaire**:
- Rama: `feat/dataset-framing-questionnaire` (PR #42).
- Entregable completado:
  - Entidad de dominio `DatasetQuestionnaire` con enums `ErrorCostPriority`, `TemporalStructure`, `LatencyConstraint`, `ExplainabilityLevel`.
  - Asesor de encuadre `QuestionnaireAdvisor` con detección heurística (churn, fraude, médico, crédito, time-series, cohortes) y modo LLM.
  - Persistencia SQLite y paridad en Workspace y API REST (`GET/POST /api/dataset/questionnaire`).
  - 13 tests passing en `tests/test_dataset_questionnaire.py` (100% cobertura en módulos nuevos).

Active Track: **Sistema Agéntico V0.9/V1.0 — Hito H4: Ciclo Determinista y Especialistas (A4 + B4)**:
- **Protocolo de Concurrencia Activo:** `two-person-plan.md` §5.1 y Regla 9 en `AGENTS.md`.
- **Track Persona A (Paquete A4 — Completado):**
  - Rol: **Persona A** (Aplicación, Políticas y Especialistas).
  - Rama: `feat/agentic-a4-specialists`.
  - Subdirectorios propios: `src/automl/application/agents/specialists/` (`planner.py`, `advisor.py`, `critic.py`, `context_builder.py`), `src/automl/infrastructure/llm/` (`fake_provider.py`), y `tests/test_v10_specialists.py`.
  - Entregable completado:
    - `ContextBuilder`: Extracción determinista de `ContextPayload`, truncamiento configurable de leaderboard/features/historial y lista explícita de `omissions` (sin datos crudos).
    - `Planner`: Formulación científica de hipótesis y propuestas `CandidateProposal` ("Proponer ≠ Aceptar") para baselines, random forests, boosting y tuning.
    - `FeatureAdvisor`: Detección heurística de target leakage y propuesta de `interactions_differences` y encodings categóricos.
    - `Critic`: Evaluación empírica de métricas, diagnóstico de varianza y recomendaciones objetivas (`accept`, `reject`, `explore_alternative`).
    - `FakeLLMProvider`: Mock determinista en `infrastructure/llm/` con soporte de respuestas canned, schemas JSON y simulación de errores/latencia.
    - Validación: 31 tests unitarios y de integración pasando en `tests/test_v10_specialists.py` con 94% de cobertura del paquete; 386 tests pasando en la suite global con 87.35% de cobertura total.
- **Track Persona B (Paquete B4 — Completado):**
  - Rol: **Persona B** (Interfaces, Integración y Orquestación).
  - Rama: `feat/agentic-b4-orchestrator`.
  - Subdirectorios propios: `src/automl/application/agents/orchestrator/` (`__init__.py`, `state_machine.py`, `session_manager.py`, `loop.py`), `src/automl/interfaces/cli/agent_session_cli.py`, y `tests/test_v10_orchestrator.py`.
  - Entregable completado:
    - `AgentStateMachine`: Evaluación de transiciones legales de ciclo, detección criptográfica de hipótesis repetidas (`compute_hypothesis_signature`), verificación de presupuestos y criterios de parada.
    - `AgentSessionManager`: Ciclo determinista completo (`OBSERVE` $\rightarrow$ `PROPOSE` $\rightarrow$ `GATE` $\rightarrow$ `EXECUTE` $\rightarrow$ `CRITIQUE` $\rightarrow$ `CHECK_STOP`), checkpointing persistente en SQLite, preservación estricta de "Proponer ≠ Aceptar" y flujo de aprobaciones pendientes (`PENDING_APPROVAL` no bloqueante, reanudación tras aprobación o rechazo).
    - `AgentLoop`: Ejecutor autónomo de bucles con hooks de ciclo de vida (`on_step`, `on_approval_needed`, `on_stop`).
    - CLI de sesión en `agent_session_cli.py`: comandos `automl agent session <start|step|resume|status|stop>`.
    - Validación: 24 tests unitarios y de integración pasando en `tests/test_v10_orchestrator.py` con 92% de cobertura de paquete; 445 tests pasando en la suite global con 87.78% de cobertura total (superando el umbral de 85%).

---

## Completed

- **Hito H3 Integrado en `main` — Operaciones largas, HPO y cancelación cooperativa (2026-10-01):**
  - **Paquete A3 (Persona A, PR #35):**
    - Exclusividad de escritor por run mediante `agent_run_leases` en `SqliteAgentLedger` (`acquire_run_lease`, `heartbeat_run_lease`, `release_run_lease`, `get_run_lease`).
    - Reconciliación de operaciones caídas (`reconcile_operations`): transición a `RECOVERY_REQUIRED`.
    - Cancelación cooperativa: `request_operation_cancellation`, verificación en bucle de trials en `AutoMLWorkspace.optimize_experiment`.
    - Reservas atómicas de presupuesto en `ToolExecutor` para evitar sobrecostes concurrentes.
    - Herramienta `optimize_experiment` con cálculo dinámico según `n_trials`. 10 tests en `tests/test_v09_agent_a3.py`.
  - **Paquete B3 (Persona B, PR #36):**
    - Dominio y Contratos: estados `TIMED_OUT`, `CANCEL_REQUESTED`, `RECOVERY_REQUIRED` en `OperationStatus`.
    - Subcomandos CLI de inspección y control (`automl agent operations list/get/cancel`).
    - Servidor y transporte MCP dual stdio y streamable-http, tools de inspección y control de operaciones.
    - Cumplimiento de criterios: "Timeout no se presenta como cancelación", "Operation ID recuperable".
    - 14 tests en `test_v09_agent_cli.py`, 14 tests en `test_v09_mcp_server.py`, 2 tests E2E en `test_v09_agent_e2e.py`.
  - **Integración y resolución de conflictos:** Fusionado exitosamente en `main` (commits `de08d3c` y `24532ca`). Total: 327 tests passing, 85.82% cobertura.

Active Track: **Ensemble Weight Optimization, Rank Averaging & Optuna Pruning**:
- Rama: `feat/ensemble-weight-optimization-and-pruning`.
- Entregable completado:
  - Extensión de contratos de dominio (`ColumnProfile`, `DatasetProfile` en `src/automl/domain/datasets/profile.py`): incorporación de campos estadísticos numéricos (`mean`, `std`, `min`, `max`, `median`, `q25`, `q75`, `skew`, `target_correlation`, `top_categories`), `histogram` (10 bins), `box_plot` (estadísticas globales y desglosadas por clases del target), `correlation_matrix` completa ($r \in [-1, 1]$), muestra de datos crudos (`preview_rows`) y recomendaciones automáticas (`recommendations`), preservando pureza hexagonal sin dependencias externas.
  - Motor de perfilado (`dataset_profiler.py`): cálculo de estadísticas descriptivas, correlación de Pearson frente al target (numérico o texto adaptado), detección de multicolinealidad cruzada ($|r| > 0.88$), cálculo de frecuencias categóricas con tasa de propensión al target (`target_rate`), generación de cajas y bigotes desglosados por clase de objetivo, e histogramas bivariantes.
  - Deserialización en persistencia (`sqlite_repository.py`): soporte transparente para los nuevos campos estadísticos, matriz de correlación y recomendaciones en SQLite, con filtrado seguro de atributos para garantizar retrocompatibilidad.
  - Vistas frontend interactivas:
    - `datasets.js`: Pestaña dedicada a la **Matriz de Correlación** (mapa de calor interactivo de Pearson entre todas las variables numéricas y el target, con detección visual de colinealidad); botones `[📊 Ver]` en cada fila de las tablas de Schema y Estadísticas Descriptivas; **Modal de Análisis Visual y Patrones de Variable** con 3 modos: Diagrama de Caja y Bigotes (Box Plot SVG comparativo por clase de target y métricas IQR/Mediana), Histograma de Distribución (10 bins con diagnóstico de asimetría/skewness), y Patrones frente a la Variable Objetivo (tasa de conversión % por categoría o comparativa de medias por clase).
    - `new_experiment.js`: Previsualización interactiva con badges y recuento de variables seleccionadas; propagación de `feature_names` en la creación de experimentos.
    - `studio.js`: Barra superior de lanzamiento rápido (LightGBM, XGBoost, CatBoost, Ensemble Blender), filtros por familia de modelos y modal para inspección de hiperparámetros de cada trial.
  - Sistema de Diseño UI Neo-Industrial (Visual ML Lab):
    - Especificación oficial de diseño en `docs/design/neo-industrial-ui-spec.md` y registro arquitectónico `docs/decisions/005-neo-industrial-visual-ml-lab-ui.md`.
    - Regla 8 añadida a `AGENTS.md` y directrices en `DEVELOPER_GUIDE.md` para que cualquier agente o desarrollador futuro preserve estrictamente este estándar.
    - Tipografía dual (`Space Grotesk` para títulos/interfaz y `IBM Plex Mono` para datos/métricas/IDs/logs).
    - Paleta modular técnica: `#111111` (negro carbón), `#16171c` / `#1c1d24` (paneles modulares), `#D8D6CF` (cemento), `#F1EFE9` (blanco cálido), con `#E5512D` (naranja señal) reservado exclusivamente para acciones y CTAs principales de ejecución.
    - Navegación lateral numerada (`01 Dashboard`, `02 Datasets`, `03 Experiments`, `04 Models`, `05 Pipelines`, `06 Deployments`).
    - Eliminación de bordes redondeados tipo SaaS (radios estrictos $\le 4$px, bordes de 1px) y modernización integral de `index.html`, `workbench.css`, `app.js`, `overview.js`, `new_experiment.js`, `studio.js`, `compare.js`, `datasets.js`.
  - Pruebas y cobertura: 295 tests pasando (incluyendo `tests/test_web_dashboard.py` enriquecido con aserciones para `box_plot`, `histogram`, `target_rate` y `correlation_matrix`), 8 skipped, 0 fallos, 86.45% cobertura global (superando el umbral de 85%). Pruebas JS (`node --test tests/js/jobs.test.mjs`) passing al 100%.

Active Track: **Sistema Agéntico V0.9/V1.0 — Persona A (Paquete A3 / Hito H3 completado)**:
- Rol: **Persona A** (Aplicación, Políticas y Persistencia).
- Rama: `feat/agentic-a3-operations`.
- Objetivo A3: HPO (`optimize_experiment`), worker leases por run, reconciliación de operaciones caídas, cancelación cooperativa y reservas atómicas de presupuesto.
- Estado: Completado y verificado. Suite de pruebas `tests/test_v09_agent_a3.py` (10 tests) y suite completa de tests de operaciones pasando.
## Completed

- Paquete A3 completado (Persona A, 2026-10-01):
  - Exclusividad de escritor por run mediante `agent_run_leases` en `SqliteAgentLedger` (`acquire_run_lease`, `heartbeat_run_lease`, `release_run_lease`, `get_run_lease`).
  - Reconciliación de operaciones huérfanas/caídas (`reconcile_operations`): transiciona operaciones en `RUNNING` con lease expirado a `RECOVERY_REQUIRED`.
  - Cancelación cooperativa: `request_operation_cancellation` transiciona operaciones a `CANCEL_REQUESTED` o `CANCELLED`. Bucle de trials en `AutoMLWorkspace.optimize_experiment` comprueba `execution_check` y `run.status in {CANCELLED, PAUSED}` antes de cada trial, deteniendo la ejecución sin comenzar nuevos trials.
  - Reservas atómicas de presupuesto: `get_active_reserved_budget(run_id)` computa recursos reservados por operaciones en vuelo (`pending`, `queued`, `running`, `cancel_requested`). `ToolExecutor` evalúa `effective_budget` evitando que solicitudes concurrentes superen límites.
  - Herramienta `optimize_experiment` registrada en `ToolExecutor` (11 tools en catálogo total) con cálculo dinámico de costes según `n_trials`.
  - Suite de tests dedicada `tests/test_v09_agent_a3.py` con 10 tests exhaustivos passing.

- Generalist AutoML Engine — Phase 1: Generalized N-Model OOF Blending & Level-2 Stacking (PR #34, 2026-10-01):
  - Removed 2-model restriction in `evaluate_oof` and `GenerateOOFSubmissionCommand`.
  - Implemented Level-2 Stacking (`stack_predictions` in `blender.py`) with Ridge, LogisticRegression, Lasso.
  - Multi-method support (`average`, `rank`, `simplex`, `stacked`) across engine, CQRS, CLI, and web server.
  - 13 new unit/integration tests in `tests/test_oof_multi_model.py`. 314 tests passing, 85.68% coverage.


- Pairwise Numerical Differences & Subtraction in InteractionFeatureGenerator (2026-10-01):
  - Added `include_differences: bool = True` to `InteractionFeatureGenerator` (`src/automl/engine/features/generation/interaction_generator.py`).
  - Implemented variance-prioritized pair discovery for `"difference"` type (`inter_diff_colA_minus_colB = colA - colB`).
  - Added deterministic vector subtraction in `transform()` and `interactions_differences` candidate set.
  - Tests passing: 8/8 in `tests/test_feature_interactions.py`. Merged into `main` via PR #33.

- Interactive Dataset Analysis, Visual Charts & Smart Feature Selection (AutoML Workbench, 2026-10-01):
  - Statistical and correlation profiling extensions in `ColumnProfile` and `DatasetProfile`.
  - Correlation heatmap, box plots, histograms, and target propensity analysis in Workbench UI.
  - Neo-Industrial Visual ML Lab UI standard and ADR 005. Merged into `main` via PR #32.

- Paquete B2 completado (Persona B, 2026-09-30):
  - Exposición de tools mutantes en MCP stdio (`create_experiment`, `prioritize_feature`, `run_experiment`).
  - Retorno no bloqueante de `PENDING_APPROVAL` con `approval_id`.
  - Comandos CLI `automl agent approvals list` y `automl agent approve <id> [--reject]`.
  - Pruebas en `tests/test_v09_agent_cli.py` (7 tests) y `tests/test_v09_agent_e2e.py` (1 test E2E exhaustivo).
  - Hito H2 verificado y cerrado (PR #30).

- Ensemble Weight Optimization, Rank Averaging & Optuna Pruning (2026-09-30):
  - Nelder-Mead Simplex Weight Optimization (`optimize_ensemble_weights` en `src/automl/engine/ensemble/blender.py`), Rank-Averaging Blending, y Multi-fidelity Early Pruning en `OptunaOptimizer`. PR #31 integrado en main.

- Pairwise Numerical Differences & Subtraction in InteractionFeatureGenerator (2026-10-01):
  - Added `include_differences: bool = True` to `InteractionFeatureGenerator` (`src/automl/engine/features/generation/interaction_generator.py`).
  - Implemented variance-prioritized pair discovery for `"difference"` type (`inter_diff_colA_minus_colB = colA - colB`).
  - Added deterministic vector subtraction in `transform()` and `interactions_differences` candidate set.
  - Tests passing: 8/8 in `tests/test_feature_interactions.py`; full suite 296 passing, 86.42% coverage.
  - PR #33 opened against `main`.

- Sistema Agéntico V0.9/V1.0 — Persona B (Hito H2 / Paquete B2 Cerrado) (2026-09-30):
  - MCP Server mutations, CLI agent subcommands, and E2E verification.

- Ensemble Weight Optimization, Rank Averaging & Optuna Pruning (2026-09-30):


- Blackboard Issue #12 completado (2026-09-30):
  - Soporte de CatBoost, Extra Trees y MLP en el Plugin System de CATML con fallback.

- Resolved Blackboard Issue #14 (2026-09-30):
  - Validación de run ownership en `AutoMLWorkspace.predict()` (`tests/test_prediction_ownership.py`).
- Paquete A2 completado (Persona A, 2026-09-30):
  - Tools mutantes y ejecución autorizada en `src/automl/application/agents/`:
    - `ToolExecutor`: integración con `SqliteAgentLedger` para auditoría y deduplicación atómica de idempotencia.
    - Resolución de aprobaciones pendientes y verificación criptográfica anti-tampering (`expected_hash == approval.arguments_hash`).
    - Intención duradera antes de mutar ("si falla el ledger, no ejecutar").
    - Factoría `create_full_tool_registry(query_bus, command_bus, workspace)` con 10 tools iniciales (7 queries + `create_experiment`, `prioritize_feature`, `run_experiment`).
    - Suite en `tests/test_v09_agent_operations.py` (16 tests pasando, 97% cobertura en componentes agénticos).
 
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
