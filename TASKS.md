# Tasks

Fuente del estado operativo y del backlog. Las guías y los planes enlazan aquí; no mantienen una segunda lista de tareas. Las fases V0.x son hitos de diseño; la versión del paquete se define en `src/automl/__init__.py`.

### Track Landing — Public homepage (independent of Workbench)

- [x] Aplicar CATML_monetizacion_privada.patch para retirar la estrategia comercial de la documentación y web públicas.

- [x] Implementar homepage en `website/` con Vite, React y TypeScript; tokens Tech Minimalista, responsive 320–1440 px y accesibilidad comprobada.
- [x] Integrar captura real del Workbench de `e4a2286` con dataset sintético, snippets acordes a la API, licencia MIT, instalación desde GitHub y Platform identificada como planificación.
- [x] Preparar build estático, OpenGraph, favicon, tipografías locales, guía Vercel y comprobaciones en navegador.
- [x] Integrar la rama `feat/public-landing` en `main`; build verificado (222 ms) con dist estático y configuración para Vercel (`website/vercel.json`).

### Track Persona A (Especialistas & Políticas)

- [x] A5: implementar adaptadores intercambiables OpenAI Responses, Claude Messages, Ollama y OpenAI-compatible sobre `LLMProviderPort`, sin dependencias nuevas ni cambios en contratos compartidos.
- [x] A5: selección explícita, validación JSON local, límites de entrada/salida, timeout por socket, reintentos HTTP acotados, redacción y auditoría en memoria con consumo desconocido distinguido.
- [x] A5: documentar configuración e inyección en especialistas en `src/automl/infrastructure/llm/README.md`; pruebas sin credenciales y fixture HTTP local en el archivo asignado `tests/test_v10_specialists.py`.
- [x] A5: validación final — 508 tests pasando, cobertura 87,72 %, CLI help/task list y diff sin errores de whitespace.

### Track Persona B (Interfaces, Integración & Orquestación)

- [x] B5: Implementar `LangGraphAgentOrchestrator` con StateGraph (`observe` -> `propose` -> `gate` -> `execute` -> `critique` -> `check_stop`), adapter durable `SqliteCheckpointSaver` con reconexión resiliente a caídas.
- [x] B5: Paridad CLI en `automl agent session resume/status` con soporte para `--engine {deterministic,langgraph}`.
- [x] B5: Suites de tests exhaustivas en `tests/test_v10_orchestrator_b5.py` (13 tests) y `tests/test_v10_agent_e2e.py` (3 tests E2E y recuperación de crashes sin duplicación de hipótesis ni ejecuciones). 590 tests globales pasando, 87.70% de cobertura.
## Now (Active Phase — v0.8.2 Phase 2: Ingestion Polymorphism, Strict CQRS Boundaries & Application Services Decomposition)

- [x] **Ingestión Polimórfica y Soporte Nativo de Parquet/DataFrames**:
  - [x] Soporte nativo para `.parquet`, `.pq`, `.json`, `.jsonl`, `.tsv` y `.csv` en `src/automl/engine/profiling/dataset_profiler.py` (`load_dataframe`).
  - [x] Mensaje informativo claro con `ImportError` guiando al usuario para instalar `pyarrow` o `fastparquet` si no están presentes.
  - [x] En `AutoMLWorkspace.register_dataset()`: aceptación de `pd.DataFrame` directamente, persistiendo de forma automática en `datasets/` (.parquet o .csv) y registrando el dataset.
  - [x] En `AutoML.fit()` (`facade.py`): soporte polimórfico para rutas en `str` o `Path` (además de `pd.DataFrame` y `np.ndarray`).
  - [x] Extra `parquet = ["pyarrow>=14.0"]` añadido a `pyproject.toml` e incluido en grupo `all`.
  - [x] Tests unitarios en `tests/test_parquet_and_polymorphic_ingestion.py` (4 tests).
- [x] **Frontera Arquitectónica CQRS Estricta y Eliminación de Fugas de Estado**:
  - [x] Formalización de consultas DTO en `QueryBus` (`src/automl/application/queries/workspace_queries.py`): `GetRunQuery`, `ListRunsQuery`, `GetDatasetQuery`, `ListDatasetsQuery`, `GetTrialQuery`, `GetExperimentQuery`, `ListTrialResultsQuery`.
  - [x] Registro y resolución limpia de handlers en `src/automl/application/bootstrap.py`.
  - [x] Adición de métodos de consulta públicos en `AutoMLWorkspace` para evitar acoplamiento directo con infraestructura (`list_datasets`, `get_dataset_profile`, `save_dataset_profile`, `save_run`, `get_trial`, `get_experiment`, `list_experiments`, `list_trial_results`, `get_leaderboard_results`, `list_feature_sets`).
  - [x] Eliminadas todas las llamadas directas a `ws.repository` en la fachada (`facade.py`) y en la API web (`interfaces/web/server.py`).
  - [x] Eliminada la inyección monkey-patch en `src/automl/interfaces/cli/agent_session_cli.py`.
  - [x] Guard arquitectónico automatizado en `tests/test_cqrs_architecture_guard.py` (2 tests verificando 0 ocurrencias de fugas privadas en capas de presentación e interfaces y despacho vía `QueryBus`).
- [x] **Descomposición del God Module `workspace.py` en Servicios de Aplicación**:
  - [x] Extracción de `PipelineExecutionService` en `src/automl/application/services/pipeline_service.py` (validación de grafos, orden topológico de ejecución, construcción de DAG multimodal, ejecución y fit/predict).
  - [x] Extracción de `InferenceService` en `src/automl/application/services/inference_service.py` (predicción, alineación de plantillas Kaggle, generación de envíos y exportación de `ModelArtifact`).
  - [x] Extracción de `FeatureEngineeringService` en `src/automl/application/services/feature_service.py` (características derivadas, análisis de dinámica temporal, estrategias de selección multi-método, planificación de ablación y promoción de conjuntos de características).
  - [x] Delegación 100% retrocompatible en `AutoMLWorkspace` preservando firmas públicas y reduciendo `workspace.py` en más de 740 líneas.
- [x] **Resolución de Auditoría Técnica y Hardening de Confiabilidad ML**:
  - [x] **[Crítico] Inversión de signo de métricas de error (MAE, RMSE, MSE, Log Loss)**: En `sklearn_trainer.py`, los scores negativos retornados por `cross_val_score` para estimadores de pérdida (`neg_mean_absolute_error`, `neg_root_mean_squared_error`, etc.) se invierten (`scores = -scores`), produciendo valores positivos reales. En `workspace.py`, la dirección de optimización y el ordenamiento del leaderboard para estas métricas es estrictamente minimizante (`ASC`), asegurando que el modelo con menor error gane siempre.
  - [x] **[Alto] Exclusión de trials fallidos en leaderboard y selección del ganador**: En `sqlite_repository.py` y `ports.py`, `get_leaderboard()` filtra por defecto los trials que fallaron (`include_failed=False`). En `facade.py` e `inference_service.py`, la selección del modelo ganador y la exportación de `ModelArtifact` excluye explícitamente cualquier trial con `failure_reason` o `succeeded is False`.
  - [x] **[Alto] Eliminación de sustitución silenciosa de métricas no soportadas**: En `sklearn_trainer.py` (`_sklearn_scoring` y `_compute_metrics`), se eliminó el fallback silencioso a `accuracy` o `r2`. Métricas no soportadas levantan `ValueError`. Se añadió soporte de primera clase para `mse`, `log_loss`, `balanced_accuracy`, `precision`, `recall` y plugins métricos vía `make_scorer`.
  - [x] **[Medio-alto] Documentación de Time Budget Cooperativo**: En `facade.py`, se documentó explícitamente que `time_budget` opera a nivel global entre modelos de forma cooperativa y no como un timeout preemptivo atómico por proceso.
  - [x] **[Medio-alto] Hardening HTTP/MCP y Robustez en Datasets Reducidos**: El Workbench soporta paso seguro de tokens vía fragmento `#token=` que se limpia inmediatamente de la URL con `history.replaceState`. El servidor MCP emite advertencia de seguridad (`SECURITY NOTICE`) si se expone `streamable-http` fuera de localhost. `SklearnTrainer` adapta inteligentemente los folds de KFold y splits holdout en datasets pequeños para evitar divisiones de muestra única.
  - [x] Suite de pruebas dedicada en `tests/test_audit_findings.py` (4 tests). 622 tests pasando (100% verde), 87.37% de cobertura.
- [x] **Alineación de Estado Activo y Feed de Actividad en Workbench (PR #85)**:
  - [x] Definición canónica de `ACTIVE_RUN_STATUSES`, helper `is_active_run_status()` y propiedad `run.is_active`.
  - [x] Exposición de flag `is_active` en `/api/runs` y `/api/overview`, priorizando ejecuciones vivas sobre ejecuciones completadas.
  - [x] Segregación del feed de actividad en Workbench sin mezclar métricas incomparables.
  - [x] Función compartida `isRunActive()` en frontend (`utils.js`) y soporte de score 0.0 y $R^2 < 0$.
  - [x] Corrección de mensaje de error al programar plan en `agent.js`.
- [x] **Rediseño UI Workbench — Modelo de Laboratorio y Dataset Activo (Fase 1)**:
  - [x] Contexto de Laboratorio Local en barra lateral con widget de Dataset Activo (`#sidebarActiveDatasetName`) y selector rápido (`SwitchDatasetModal`).
  - [x] Simplificación de navegación principal a 4 vistas canónicas: Inicio (`home`), Dataset (`dataset`), Experimentos (`experiments`), Evidencia (`evidence`).
  - [x] Nueva vista de Inicio (`src/automl/interfaces/web/static/js/views/home.js`): Hero card «Continuar con {dataset activo}», acciones secundarias «Añadir dataset» y «Abrir dataset guardado», y paneles de ejecuciones activas y trabajos recientes.
  - [x] Rediseño de la vista Dataset (`src/automl/interfaces/web/static/js/views/datasets.js`): Pestañas «Resumen» (métricas de perfil, salud de datos Anti-Leakage Guardian, recomendaciones y preview crudo), «Columnas» (tabla de variables, filtros, selección interactiva) y «Exploración» (panel conceptual de capacidades estadísticas para CATML Explore Fase E1).
  - [x] Integración de contexto de dataset activo en `store.js` (`setActiveDataset`, `getActiveDataset`), `app.js`, `studio.js`, `overview.js` y `new_experiment.js`.
  - [x] Suite de tests automatizada en `tests/test_workbench_laboratory_redesign.py` (3 tests). 667 tests pasando (100% verde).
- [x] **Rediseño UI Workbench — Modal de Ingestión Polimórfico (Fase 2)**:
  - [x] Modal de Ingestión Polimórfica (`RegisterDatasetModal` en `register_dataset_modal.js`):
    - Subida Drag & Drop con subida base64 al servidor (`POST /api/dataset/upload`).
    - Navegador de archivos local en servidor con exploración de carpetas y búsqueda (`GET /api/files/browse`).
    - Entrada manual de ruta con inspección instantánea de esquema (`POST /api/dataset/inspect-file`).
    - Sugerencia automática de columna target y soporte explícito para datasets no supervisados (`target=None`).
  - [x] Creación dinámica de runs (`POST /api/run/create`) vinculados al dataset activo.
  - [x] Scoping limpio de variables en `new_experiment.js` y `datasets.js` (`activeRun`, `recentDs`).
  - [x] Tests de validación e integración en `tests/test_workbench_laboratory_redesign.py` (6 tests).
- [x] **Remediación Integral de Auditoría Técnica (Octubre 9, 2026 — Commit `5caa1f5`)**:
  - [x] **H1 [Alta] Aprobación real de agentes en Workbench**: `POST /api/agent/action` integrado con `SqliteAgentLedger`, validación estricta de existencia (404 ante IDs inexistentes), transición de estado persistida en SQLite y registro de revisor humano.
  - [x] **H2 [Alta] Sentido de métricas personalizadas en CV**: Eliminado `_greater_is_better` inexistente; normalización de signo con `_is_minimizing_metric()` e inspección de `_sign == -1`; sincronización de `plugin_registry` en `SQLiteExperimentRepository.get_leaderboard()` para ordenamiento ascendente (`ASC`).
  - [x] **H3 [Alta] Autenticación remota en servidor MCP**: Transporte `streamable-http` rechaza enlaces externos sin autenticación (`PermissionError` fail-closed); flags `--token` y `--insecure-no-auth` en CLI con `BearerAuthMiddleware` en Starlette.
  - [x] **H4 [Media-alta] Límite de tamaño HTTP explícito**: Límite `MAX_PAYLOAD_SIZE = 50 MB` en `server.py`, rechazando peticiones excesivas con `413 Payload Too Large`.
  - [x] **H5 [Media-alta] Endurecimiento de autenticación remota**: Rechazo de tokens en query string (`?token=`) para operaciones POST; advertencia de seguridad en peticiones GET.
  - [x] **H6 [Media] Compatibilidad de versión scikit-learn**: Declaración de `scikit-learn>=1.4.0` en `pyproject.toml` con fallback defensivo (`needs_proba`) en `make_scorer`.
  - [x] **H8 [Media] Bloqueo multiplataforma en worker**: Uso de `msvcrt` en Windows y `fcntl` en Unix en `worker.py`.
- [x] **Remediación Integral de Auditoría UX/UI Frontend (Workbench)**:
  - [x] **F01 [Crítico UX]**: Retirados CPU 60%, RAM 45%, 5 workers fijos y logs predefinidos en `studio.js`; telemetría real conectada con ejecuciones y tiempos por estimador.
  - [x] **F02 [Crítico UX]**: Retirado badge simulado `DAG Validated` en `pipeline.js`; sustituido por estado verificado (`Pipeline Trained & Aligned` / `Schema Inferred`); corregidas descripciones a `StandardScaler` y `OneHotEncoder` / `TargetAdapter`.
  - [x] **F03 [Crítico UX]**: Retirado fallback mock "Standard Tabular Benchmark" (85% similar) en `knowledge.js`; renderizado de estado vacío honesto.
  - [x] **F04 [Medio-alto]**: Subsanado `ReferenceError: customFeatures is not defined` en `new_experiment.js`, encapsulado en instancia y vinculado al dispatch.
  - [x] **F05 [Medio-alto]**: Conectados botones simulados ("Promote Best Parameters" y "Use Warm Start") a almacenamiento reactivo en store (`promotedParams`, `warmStartParams`).
  - [x] **F06 [Medio-alto]**: Propagación completa de opciones en `createAndRunExperiment` y soporte seguro de presupuesto en `job_executor.py` (`time_budget_seconds`).
  - [x] **F07 [Medio]**: Desvinculación de ejecución huérfana en `store.setActiveDataset()` al cambiar a datasets sin runs previos.
  - [x] **F09 [Medio]**: Checklist pre-flight de Kaggle en `kaggle.js` muestra estados pendientes neutrales hasta verificación empírica.
  - [x] **F10 [Medio]**: Selección del trial óptimo en HPO guiado por dirección de métrica (minimización vs maximización).
  - [x] **F11 [Medio]**: Enlace de descarga de submissions sustituido por `api.downloadFile()` con autenticación Bearer via blob.
  - [x] **F12 & F15 [Medio]**: Accesibilidad mejorada: eliminado `select-none` del body; menú responsive con botón hamburguesa para pantallas `< md`.
- [x] **Eliminación de Maquetas y Estados Simulados en Pantalla de Inicio (Truthful Workbench State)**:
  - [x] Acotado de `ACTIVE_RUN_STATUSES` y `is_active_run_status` en dominio y frontend únicamente a estados de ejecución efectiva (`EXPERIMENTING`, `OPTIMIZING`, `FINALIZING`, `RUNNING`); estados `CREATED`, `PROFILING` y `PLANNING` no activan falsas alertas de entrenamiento.
  - [x] Eliminado badge falso de `ENTRENANDO` con spinner y barra fija al 75% en `home.js`; reemplazado por estado verídico `LISTO PARA ENTRENAR` (barra al 0%, métricas neutrales, CTA para lanzar experimento).
  - [x] Eliminada barra fija al 66% y texto arbitrario `Searching space (72%)` en `overview.js`.
  - [x] Marcado en **ROJO** del componente Explore en `datasets.js` (`PENDIENTE DE IMPLEMENTACIÓN`, `EN FASE DE IMPLEMENTACIÓN`, `#EF4444`) para clarificar que pertenece al roadmap de ADR-008 sin maquetas decorativas.
  - [x] Sincronización de la cabecera global (`app.js`): solo exhibe `ENTRENANDO` cuando existen trabajos de entrenamiento activos en ejecución.
  - [x] Pruebas de integración automatizadas en `tests/test_workbench_laboratory_redesign.py`. 677 tests pasando (100% verde), 86.68% de cobertura.
  - [x] **F13 & F14 [Bajo]**: Terminología armonizada y versión alineada a `v0.8.2`.
- [x] **Validación y Cobertura**:
  - [x] 676 tests pasando (100% de la suite). Cobertura total mantenida $\ge 85\%$.

## Next Roadmap: CATML Explore (Análisis Estadístico, Evidencia y Agentes)

> Plan de evolución arquitectónica como monolito modular con separación hexagonal y CQRS ([docs/features/catml-explore/plan.md](docs/features/catml-explore/plan.md), [ADR-008](docs/decisions/008-catml-explore-modular-monolith.md)).

- [x] **Fase E0 — Estabilización del Núcleo (Core Stabilization)**:
  - [x] Normalizar signo y orden de métricas de error (MAE, RMSE, MSE, Log Loss).
  - [x] Excluir trials fallidos del leaderboard y selección de ganador.
  - [x] Erradicar fallbacks silenciosos a `accuracy`/`r2`.
  - [x] Hardening de presupuesto temporal cooperativo y autenticación remota HTTP/MCP.
  - [x] 622 tests pasando (100% verde), 87.37% de cobertura.
- [x] **Fase E0.5 — Simplificación Previa y Modularización de Registro/Contratos (Pre-Explore Surgical Refactoring)**:
  - [x] **PR 1 — Simplificación de dependencias y modularización de `bootstrap.py`**:
    - [x] Descomponer el registro monolítico en submódulos por contexto funcional en `src/automl/application/registries/` (`job_registry`, `core_registry`, `experiment_registry`, `feature_registry`, `inference_registry`).
    - [x] Reducir `bootstrap.py` a punto limpio de orquestación (de 342 a 53 líneas), manteniendo compatibilidad 100% de `build_application`.
    - [x] Añadir guard arquitectónico de composición independiente en `tests/test_cqrs_architecture_guard.py`.
    - [x] 623 tests pasando (100% verde), 87.37% de cobertura.
  - [x] **PR 2 (PR #78) — Contratos y límites de módulos (Dataset sin target y Jobs desacoplados)**:
    - [x] Desacoplar `Dataset`, `DatasetProfile` y `dataset_profiler` permitiendo `target_column: str | None = None` y `task_type: str | None = None`.
    - [x] Desacoplar persistencia en `sqlite_repository.py` admitiendo `target_column TEXT NULL` y `task_type TEXT NULL`.
    - [x] Desacoplar modelo `Job` y `sqlite_jobs.py` para permitir `run_id: str | None = None`.
    - [x] `AutoMLWorkspace.register_dataset(target=None)` y planificación no supervisada por defecto (`TaskType.CLUSTERING`).
    - [x] Suite de tests dedicada en `tests/test_unsupervised_datasets_and_decoupled_jobs.py` (4 tests). 627 tests pasando (100% verde), 87.38% de cobertura.
  - [x] **PR 3 (PR #79) — Hardening de Migraciones SQLite y Compatibilidad Retroactiva**:
    - [x] Migración atómica determinista de la tabla `datasets` en `SQLiteExperimentRepository._migrate()` relajando restricciones `NOT NULL` sobre bases heredadas v0.8.0/v0.8.1.
    - [x] Migración de la tabla `jobs` en `SQLiteJobRepository` relajando `run_id NOT NULL` sobre bases heredadas.
    - [x] Erradicación de excepciones silenciadas (`except Exception: pass`) reemplazadas por logging estructurado y fallos explícitos fail-closed.
    - [x] Suite de pruebas exhaustiva con esquemas legacy y datos preexistentes en `tests/test_sqlite_legacy_migrations.py` (2 tests).
    - [x] 629 tests pasando (100% verde), 87.37% de cobertura.
- [x] **Hardening de Seguridad y Robustez de Entrenamiento (Auditoría Técnica Oct 2026)**:
  - [x] **PR A (PR #80) — Blindaje Anti-Leakage Centralizado y Fail-Closed**:
    - [x] Centralizar política de selección de variables en `DatasetProfile.resolve_safe_feature_names()`.
    - [x] Parar con `ValueError` fail-closed en `AutoML.fit()` (`facade.py`) cuando todas las variables son descartadas por leakage o identifiers.
    - [x] Retornar lista vacía de candidatos en `RuleBasedExperimentPlanner` si sólo existen variables con leakage.
    - [x] Excluir automáticamente leakage en `JobExecutor` para ejecuciones con generación por defecto de features.
    - [x] Validar variables en `AutoMLWorkspace.create_experiment()` con soporte para `allow_leakage=True`.
    - [x] Calibrar thresholds estadísticos en `dataset_profiler.py` (`cardinality_ratio <= 0.90` en group hints, `row_count >= 10` en correlación de target) para evitar falsos positivos en features continuas y datasets enanos.
    - [x] Suite de tests dedicada en `tests/test_anti_leakage_guardian.py` (6 tests). 635 tests pasando (100% verde), 87.29% de cobertura.
  - [x] **PR B (PR #81) — Seguridad del Workbench (Confinamiento de Archivos y Sanitización de Rutas)**:
    - [x] Confinamiento de `/api/kaggle/download` estrictamente dentro del workspace con lista blanca de extensiones (`.csv`, `.tsv`, `.parquet`, `.pq`, `.json`, `.zip`, `.txt`) y bloqueo fail-closed de path traversal (`403 Forbidden`).
    - [x] Restricción de `/api/media/preview` a raíces estrictamente autorizadas (`ws_root` y directorios de datasets registrados), eliminando `ws_root.parent` y `Path.cwd()` para impedir accesos no autorizados al directorio padre del usuario o del proyecto.
    - [x] Comprobación estricta de pertenencia a directorio para `/static/` con `target_file.is_relative_to(static_root)` y respuesta `403 Forbidden` ante intentos de path traversal.
    - [x] Suite de tests de seguridad dedicada en `tests/test_workbench_file_confinement.py` (4 tests). 639 tests pasando (100% verde), 87.39% de cobertura.
  - [x] **PR C (PR #82) — Atomicidad Transaccional de Migraciones SQLite**:
    - [x] Transacción explícita completa y rollback verificado en migraciones SQLite (`SQLiteExperimentRepository._migrate` y `SQLiteJobRepository._migrate`).
    - [x] Conteo de registros previo y posterior (`post_count == pre_count`) y limpieza de tablas temporales.
    - [x] Tests unitarios con rollback probado ante excepciones forzadas en `tests/test_sqlite_legacy_migrations.py`.
    - [x] 641 tests pasando (100% verde), 87.41% de cobertura.
  - [x] **PR D (PR #83) — Consistencia del Producto (Dirección de Métricas, Benchmarks y Actividad Real)**:
    - [x] Unificar dirección de métricas en benchmarks (`is_minimizing_metric`) y selección del mejor modelo según la métrica (`min` para pérdida, `max` para ganancia).
    - [x] Erradicación total de mock data en Workbench (`_build_real_activity_feed` en `server.py` y carga dinámica de `agent_ledger.db` en `/api/agent/hypotheses`).
    - [x] Tests unitarios en `tests/test_benchmark_metric_direction.py` y `tests/test_workbench_activity_consistency.py`.
    - [x] 651 tests pasando (100% verde), 87.32% de cobertura.
  - [x] **PR E (PR #84) — Hardening Complementario (Protección XSS, Alineación de Métricas en Workbench y Confinamiento Estricto de Descargas)**:
    - [x] Protección XSS en frontend: módulo `src/automl/interfaces/web/static/js/utils.js` con `escapeHtml` y sanitización integral en `agent.js`, `overview.js` y `kaggle.js`.
    - [x] Segregación de métricas en `/api/overview` (`server.py`): eliminación de comparaciones escalares entre métricas incomparables; reporte contextualizado (`best_metric`, `best_score`, `best_model`) y desglose completo por métrica (`best_by_metric`).
    - [x] Benchmarks: cálculo de `delta_vs_baseline` en `BenchmarkRunner` con sentido de mejora positivo ($\Delta > 0$) tanto para métricas de pérdida (reducción de error) como de ganancia.
    - [x] Confinamiento estricto de `/api/kaggle/download`: restricción exclusiva al subdirectorio `submissions/` o artefactos de exportación autorizados (`exports/`), bloqueando rutas relativas a la raíz del workspace o extensiones no pertinentes.
    - [x] Suite de tests dedicada en `tests/test_workbench_xss_protection.py` (5 tests) y ampliación de `tests/test_workbench_file_confinement.py` (4 tests).
    - [x] 656 tests pasando (100% verde), 87.33% de cobertura.
  - [x] **PR G (PR #86) — Flexibilidad Anti-Leakage en Planificación e Integridad de Métricas en Workbench**:
    - [x] Soporte para `strict=False` en `DatasetProfile.resolve_safe_feature_names()` y uso en `RuleBasedExperimentPlanner.propose()` para tolerar datasets con columnas mixtas de leakage y predictoras.
    - [x] Segregación de métricas por dataset en `/api/overview` (`best_by_dataset`), soporte para scores `0.0` en hipótesis y valores iniciales limpios.
    - [x] Suites de pruebas en `tests/test_anti_leakage_guardian.py` y `tests/test_workbench_activity_consistency.py`.
    - [x] 664 tests pasando (100% verde), 87.32% de cobertura.
- [x] **Fase E1 — Núcleo de Estudios (Explore Core)** [Prioridad P0]:
  - [x] Entidades de dominio puras en `src/automl/domain/analysis/` (`StudySpec`, `AnalysisRun`, `StatisticalFinding`, `VisualizationSpec`, `EvidenceLink`, `AnalysisHypothesis`).
  - [x] Soporte de estudios sin columna objetivo (`target_column=None`, modo no supervisado).
  - [x] Esquema de persistencia SQLite aditivo en `src/automl/infrastructure/database/sqlite_studies.py` con WAL mode, claves foráneas e índices.
  - [x] Servicio de aplicación `AnalysisStudyService` y operaciones CQRS (`CreateStudyCommand`, `RunAnalysisCommand`, `ArchiveStudyCommand`, `GetStudyQuery`, `ListStudiesQuery`, etc.) modularizados en `src/automl/application/registries/analysis_registry.py`.
  - [x] Motor de análisis estadístico determinista en `src/automl/engine/analysis/statistical_analyzer.py` (colinealidad de Pearson $r \ge 0.70$, separación ANOVA $p < 0.05$, outliers IQR, cardinalidad e hipótesis de mitigación).
  - [x] CLI `automl explore create/run/list/show/findings` registrado en `src/automl/interfaces/cli/explore_cli.py` y `main.py`.
  - [x] Herramientas MCP en `src/automl/interfaces/mcp/analysis_tools.py` (`analysis_create_study`, `analysis_run_study`, `analysis_get_findings`, `analysis_list_studies`, `analysis_get_study`).
  - [x] Endpoints REST `/api/analysis/...` en `server.py` e interfaz visual verídica en Workbench (`datasets.js` y `api.js` con selector de estudios, matriz de correlación real calculada y tarjetas de hallazgos).
  - [x] Suite de pruebas automatizadas completa en `tests/test_explore_phase_e1_core.py` (10 pruebas pasando al 100%).
- [x] **Fase E2 — Motor Estadístico Avanzado (Advanced Statistical Engine)** [Prioridad P0]:
  - [x] Motor numérico modular en `src/automl/engine/analysis/` (`distribution_diagnostics.py`, `association_metrics.py`, `hypothesis_testing.py`, `statistical_analyzer.py`).
  - [x] Diagnóstico de distribuciones: pruebas de normalidad (D'Agostino-Pearson y Shapiro-Wilk), asimetría (skewness), curtosis, detección de multimodality (coeficiente bimodal de Sarle $BC > 0.555$) y outliers multivariantes (distancia de Mahalanobis regularizada con corte crítico $\chi^2_{p < 0.001}$).
  - [x] Matrices de asociación bivariadas: correlación de Pearson con intervalos de confianza de Fisher al 95% ($z$-transform), correlación monótona de Spearman ($\rho$), contraste de relaciones no lineales e hipótesis de modelos basados en árboles/splines.
  - [x] Asociaciones categóricas e inferencia no lineal: matriz de Cramér's V con corrección de sesgo de Bergsma-Wicher, pruebas Chi-cuadrado de independencia e hipótesis de interacciones categóricas.
  - [x] Pruebas de hipótesis automáticas supervisadas: comparación de 2 grupos con t-test de Welch y Mann-Whitney U junto con tamaño del efecto Cohen's $d$; comparación multiclase ($>2$ grupos) con ANOVA unidireccional y Kruskal-Wallis junto con tamaño del efecto $\eta^2$ (Eta al cuadrado); prueba de homocedasticidad de Levene.
  - [x] Corrección de comparaciones múltiples: ajuste por Benjamini-Hochberg (FDR - False Discovery Rate) sobre todos los tests inferenciales del estudio para mitigar falsos descubrimientos (*data dredging*).
  - [x] Batería de pruebas dedicada en `tests/test_explore_phase_e2_advanced_engine.py` (11 pruebas pasando al 100%, 88% de cobertura en `engine/analysis`).
- [x] **Fase E3 — Laboratorio Visual Interactivo (Workbench Exploration UI)** [Prioridad P1]:
  - [x] Generador backend de `VisualizationSpec` declarativo en `src/automl/engine/analysis/visualizations/builder.py` con agregación de histogramas, resumen de cinco números para boxplots con muestreo acotado de outliers, y submuestreo de dispersión bivariada con regresión lineal de tendencia.
  - [x] Vista profesional en `src/automl/interfaces/web/static/js/views/explore.js` conforme al sistema de diseño Tech Minimalista Premium (ADR-006): selector de estudios, botones para nuevo estudio y ejecución de análisis, e integración en el router global (`app.js`).
  - [x] Galería de hallazgos estadísticos con filtros dinámicos por categoría, severidad y significancia Benjamini-Hochberg FDR ($p < 0.05$).
  - [x] Visualizador interactivo de especificaciones: matrices de correlación (Pearson y Spearman con conmutador dinámico), matriz de Cramér's V, histogramas SVG vectoriales, diagramas de caja (boxplots) y nubes de dispersión bivariada.
  - [x] Exportador de informes técnicos descargables en formato Markdown (`informe_explore_{study_id}.md`).
  - [x] Suite de pruebas automatizadas completa en `tests/test_explore_phase_e3_ui.py` (6 pruebas pasando al 100%).
- [x] **Fase E4 — Interoperabilidad MCP y CLI Completa (Agent Surface)** [Prioridad P0]:
  - [x] Tools MCP: `analysis_create_study`, `analysis_get_findings`, `analysis_get_visualizations`, `analysis_propose_experiment`.
  - [x] Paginación y control de presupuesto de tokens en llamadas MCP para agentes (`limit`, `offset`, `compact`, `envelope`).
  - [x] Recursos MCP de solo lectura: `catml://studies/{study_id}/summary` y `catml://studies/{study_id}/visualizations`.
  - [x] CLI completa `catml explore list/run/findings/export` (soporte de reporte técnico Markdown y JSON exportable).
  - [x] Generador desacoplado de informes técnicos en `src/automl/application/analysis/reporting.py`.
  - [x] Batería de pruebas automatizadas completa en `tests/test_explore_phase_e4_agent_surface.py` (7 pruebas pasando al 100%, suite global en 711 pruebas y cobertura 86.64%).
- [x] **Fase E5 — Agente Científico y Conexión con AutoML (Hypothesis Engine)** [Prioridad P1]:
  - [x] Mapeo determinista de hallazgos estadísticos a hipótesis de transformación ML (`HypothesisExperimentTranslator`).
  - [x] Entidad `EvidenceLink` y persistencia SQLite transaccional vinculando hipótesis con experimentos AutoML.
  - [x] Protocolo "Propose ≠ Accept": validación empírica en el mismo split antes de promover transformaciones o feature sets, preservando aprendizaje negativo.
  - [x] Paridad total de superficies: CQRS (`VerifyHypothesisCommand`, `GetEvidenceLinkQuery`, `ListEvidenceLinksQuery`), CLI (`automl explore verify`), MCP tool (`analysis_verify_hypothesis`) y recurso (`catml://studies/{study_id}/evidence`), y API REST (`/api/analysis/hypotheses/verify` y `/api/analysis/evidence`).
  - [x] Batería de pruebas automatizadas completa en `tests/test_explore_phase_e5_hypothesis_engine.py` (6 pruebas pasando al 100%, suite global en 717 pruebas y cobertura 86.60%).
- [ ] **Fase E6 — Benchmarks Integrales y Preparación de Producto** [Prioridad P1]:
  - [ ] Batería de 4 ejes: exactitud estadística (NIST/sintéticos), calidad AutoML, seguridad/MCP y suites de regresión.

## Completed Phase: v0.8.2 Phase 1 (Operational Time Budget, Fail-Closed Artifacts & Remote Workbench Security)

- [x] **Cumplimiento Real y Operacional de `time_budget`**:
  - [x] Añadido campo formal `time_budget_seconds: float | None` a `RunConfig` con propiedad `effective_time_budget` retrocompatible con `extra["time_budget"]`.
  - [x] Propagación formal en `AutoML.fit()` pasando `time_budget_seconds=self.time_budget` a `create_run()`.
  - [x] Control de deadline por `run_id` (`_is_run_budget_exhausted`) en `AutoMLWorkspace.run_experiment()` y `run_scheduled_experiments()`.
  - [x] Descarte cooperativo y emisión del evento `TimeBudgetExhausted` sin invalidar los modelos ya completados.
  - [x] Manejo inteligente en `AutoML.fit()`: si ningún modelo pudo completar antes del tiempo, falla de forma explícita (`RuntimeError`); si al menos uno completó, finaliza con éxito registrando `time_budget_exhausted=True`.
- [x] **Artefactos Estrictamente Fail-Closed**:
  - [x] En `ModelArtifact.save()`: eliminación de `except Exception: pass`, garantizando que la creación y escritura del digest SHA-256 no falle en silencio.
  - [x] En `ModelArtifact.load(verify_checksum=True)`: bloqueo inmediato (`ValueError`) ante sidecar inaccesible, ilegible, corrupto o con hash no coincidente, impidiendo la deserialización no segura mediante `joblib.load()`.
- [x] **Seguridad en Interfaces Remotas del Workbench (`0.0.0.0`)**:
  - [x] Preservado acceso local transparente sin token para `127.0.0.1` y `localhost`.
  - [x] Para cualquier interfaz no local (ej. `0.0.0.0`), requerir `--auth-token` o auto-generar token seguro efímero (`secrets.token_urlsafe(16)`) con URL directa en consola.
  - [x] Validación de token en `AutoMLWebHandler` vía header `Authorization: Bearer <token>` o query parameter `?token=<token>`, respondiendo `401 Unauthorized` si falta o es inválido (salvo flag explícito `--insecure-no-auth`).
  - [x] Adaptador de cliente `CATMLApiClient` en `api.js` persistiendo el token en `sessionStorage` para mantener la sesión web activa.
- [x] **Cierre de Fuga CQRS**:
  - [x] Método público `AutoMLWorkspace.list_runs(dataset_id=...)` implementado.
  - [x] Eliminado acceso al atributo privado `ws._runs` en `src/automl/interfaces/cli/main.py` (`predict_cli`).
- [x] **Batería de Pruebas de Fase 1**:
  - [x] `tests/test_time_budget_enforcement.py` (4 tests).
  - [x] `tests/test_model_artifact_fail_closed.py` (3 tests).
  - [x] `tests/test_workbench_auth.py` (3 tests).

## Completed Phase: v0.8 Release Readiness, Packaging & Governance
  - [x] Corregir `cv_folds` en `src/automl/facade.py`: `AutoML.fit` debe respetar `self.cv_folds` propagándolo a `create_run` y usando estrategia de validación `"cv"` o `"kfold"` cuando no exista columna de agrupación.
  - [x] Endurecer `src/automl/engine/vision/image_encoder.py`:
    - Eliminar fallback silencioso a hash determinista: si falla PyTorch/Pillow/backbone en un modelo neural, lanzar `RuntimeError` explícito ("Fail visibly, don't fake vision").
    - Reservar hash determinista exclusivamente cuando `model_name="deterministic"` (para tests/benchmarks reproducibles).
    - Eliminar fallback silencioso a red aleatoria no inicializada cuando fallan los pesos preentrenados.
    - Preservar dimensiones nativas de embeddings sin truncamiento arbitrario destructivo (default native feature dimension 512).
  - [x] Endurecer `src/automl/artifacts/model_artifact.py`:
    - Advertencia de seguridad en `ModelArtifact.load` contra ejecución arbitraria de código por deserialización no confiable.
    - Generación y verificación de checksum SHA-256 en streaming por bloques (`chunk_size=65536`) en guardado y recarga.
    - Documentar distinción técnica entre integridad (detección de corrupción accidental) y autenticidad criptográfica (firmas digitales PKI).
  - [x] Inmutabilidad en capa de dominio (`src/automl/domain/problems/spec.py`):
    - `@dataclass(frozen=True)` en `ProblemSpec` con conversión a tupla inmutable en `__post_init__` y test de regresión.
- [x] **Paquete 2: Packaging, Extras & Naming**:
  - [x] Renombrar paquete en `pyproject.toml` de `automl-platform` a `catml`.
  - [x] Unificar comandos y aliases CLI (`catml` primario, `automl` retrocompatible).
  - [x] Reestructurar extras de instalación en `pyproject.toml`: `models`, `vision`, `nlp`, `mcp`, `agents`, `all` (incluyendo dependencias de visión en `all` y eliminando `full` redundante).
- [x] **Paquete 3: Claridad, Transparencia Técnica y Gobernanza**:
  - [x] Actualizar `README.md`: cambiar "zero-dependency inference" por "Database-free, workspace-independent inference", badges (595 tests passing, Apache-2.0), CLI V0.8.0, Tech Minimalista Premium workbench.
  - [x] Alinear honestidad de plugins de visión: renombrar `TimmVisionPlugin` a "Vision Feature Head (MLP on Extracted Embeddings)" y en UI a "Vision Feature MLP".
  - [x] Sincronizar hoja de ruta en `docs/MASTER_PLAN.md`: marcar capacidades completadas como `IMPLEMENTADO (SHIPPED IN 0.8)` y fases futuras como `v0.8.x` y `v1.0+`.
  - [x] Crear gobernanza y release engineering: `SECURITY.md`, `CHANGELOG.md` (formato Keep a Changelog) y `TRADEMARK.md` (política de marca CATML).
- [x] **Paquete 4: Pulido Final para Release v0.8.0 & Smoke Test Automatizado**:
  - [x] Corregir detección de dependencias en `ImageModalityPlugin.available()` requiriendo `torchvision` explícitamente junto con `torch` y `PIL`.
  - [x] Modernizar resolución de pesos en `ImageEncoderNode` con fallback elegante a API moderna de TorchVision (`DEFAULT`) y mensaje informativo para entornos sin conexión (offline/air-gapped).
  - [x] Atributo aprendido `feature_dim_` en `ImageEncoderNode` para preservar contratos en `get_feature_names_out`.
  - [x] Añadir alias `VisionFeatureHeadPlugin` para desacoplar el concepto de representación visual del downstream head.
  - [x] Packaging en `pyproject.toml`: migración de licencia a string SPDX `Apache-2.0` (eliminando warning de setuptools), metadatos `project.urls` y limpieza de extra `nlp`.
  - [x] Script de validación de release `tests/smoke_test.py` y nuevo job de CI `package-smoke` para verificar instalación en entorno virtual aislado.
- [x] **Paquete 5: Trust Patch (v0.8.1) — Seguridad Local, Anti-Leakage y Reproducibilidad**:
  - [x] Fail-safe para leakage: `AutoML.fit()` y `RuleBasedExperimentPlanner` priorizan `profile.recommended_feature_names`, excluyendo automáticamente variables contaminadas.
  - [x] Workbench seguro: bind exclusivo a `127.0.0.1` por defecto (con flag `--host 0.0.0.0` para despliegue explícito) y eliminación de cabeceras permisivas `Access-Control-Allow-Origin: *`.
  - [x] Confinamiento de `/api/media/preview`: validación estricta de que las imágenes residan dentro del workspace o directorio del dataset (`is_relative_to`), bloqueando lectura arbitraria del sistema.
  - [x] Propagación de `random_state`: cableado desde el constructor de `AutoML` hasta `RunConfig.random_seed` y todos los splitters de validación cruzada y estimadores.
  - [x] Ordenación correcta de métricas de pérdida: `AutoMLResult.leaderboard()` y `SQLiteExperimentRepository.get_leaderboard()` ordenan ascendentemente para `mae`, `rmse`, `mse` y `loss`.
  - [x] Eliminación de evidencia simulada: supresión de `public_lb = score * 0.9999` y deltas ficticios en endpoints de benchmark.
  - [x] Hero honesto y simplificado en `README.md`: centrado en confianza, protección contra fugas, reproducibilidad y acceso controlado para agentes.
  - [x] Suite de tests `tests/test_v081_trust_patch.py` (4 tests pasando) y actualización de versión a `0.8.1`.


## Completed Phase: v0.9 Capability Layer, Lightweight NLP & Advanced Anti-Leakage Guardian

- [x] Contratos puros de dominio para la Capa de Capacidades en `src/automl/domain/problems/`: `ProblemSpec`, `TabularSource`, `TextSource`, `ImageSource`, `TargetSpec`, `ValidationSpec`, `ExecutionPolicy`, `BackendCapabilities`, y `EvaluationResult` (100% stdlib, sin dependencias externas).
- [x] Motor de extracción y tokenización ligera de texto en `src/automl/engine/features/text/`: `LightweightTextExtractor` (TF-IDF sublineal con n-gramas) y heurística `is_text_column` para detección automática de lenguaje natural.
- [x] Adaptación del profiler en `src/automl/engine/profiling/dataset_profiler.py`: discriminación de texto frente a identificadores no predictivos, badge `Text Feature` y recomendación `nlp_encode`.
- [x] Fusión multimodal en `src/automl/engine/training/sklearn_trainer.py`: integración transparente de transformadores de texto en `_build_pipeline` y `fit_pipeline` con salida densa unificada.
- [x] Soporte en fachada ergonómica `AutoML.fit(df, target="col", text_columns=["notes"])` con serialización, proveniencia e inferencia desacoplada en `ModelArtifact`.
- [x] Validación completa: suite `tests/test_v09_capability_layer.py` (6 tests pasando); suite global en verde: 553 tests pasando, 87.71% cobertura de código (Decision Gate de v0.9 alcanzado).
- [x] **Fase 3: Anti-Leakage Guardian Avanzado (Group & Entity Leakage)**:
  - Detección de entidades/grupos repetidos (`patient_id`, `user_id`, `device_id`, etc.) en `detect_is_group_candidate` discriminando entre identificadores 1-a-1 e identificadores grupales de entidad.
  - Diagnóstico de fuga por grupos en particiones simuladas de validación `detect_group_leakage` calculando intersección de entidades, filas contaminadas y recomendación determinista `GroupKFold(col)`.
  - Integración en `DatasetProfile` (`is_group_candidate`, `group_candidates`, `has_group_leakage`, `group_leakage_reports`, badge `Group Leakage`, acción `enforce_group_split`).
  - Soporte de validación `GroupKFold` y `GroupShuffleSplit` en `ValidationSpec`, `TrialExecution`, `SklearnTrainer`, `CreateExperimentCommand` y `RuleBasedExperimentPlanner`.
  - Protección automática y explícita en la fachada `AutoML.fit(df, target="churn", group_column="patient_id")`: exclusión de variables de grupo de la matriz predictiva `X` para prevenir memorización y aplicación automática de partición por grupos ante detección de fuga.
- [x] **Fase 4: Vision Spike (Prueba de Estrés Arquitectónica & Visión Multimodal)**:
  - Heurística `is_image_column` en `src/automl/plugins/modalities/image_plugin.py` para detección de rutas de archivos de imagen por extensiones (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`, etc.) o presencia en disco.
  - Profiler adaptado en `src/automl/engine/profiling/dataset_profiler.py`: discriminación de columnas de imagen evitando falsos positivos de identificadores (`is_identifier=False`), detección de `is_image`, propiedad `image_column_names`, y recomendación con badge `Image Feature` y acción `image_encode`.
  - Nodo de ejecución de imágenes `ImageEncoderNode` en `src/automl/engine/vision/image_encoder.py`: herencia de `BaseEstimator, TransformerMixin`, compatibilidad de slicing para DataFrames 2D y Series 1D, manejo de valores nulos/faltantes mediante vector cero (`handle_missing="zero"`), y `get_feature_names_out`.
  - Fusión multimodal en `src/automl/engine/training/sklearn_trainer.py`: integración transparente de transformadores de imagen en `_build_pipeline` y `fit_pipeline`, filtrado de parámetros de preprocesador frente a hiperparámetros de modelo, y soporte para cancelación cooperativa.
  - Plugin especializado de visión `TimmVisionPlugin` en `src/automl/plugins/models/vision_plugin.py` con interfaz de capacidades (`available()`, `requirements()`, `capabilities()`, `install_instructions()`) y degradación suave (`fallback`).
  - Dependencias opcionales organizadas en `pyproject.toml` (`catml[vision]`, `catml[nlp]`, `catml[all]`).
  - Fachada ergonómica `AutoML.fit(df, target="col", image_columns=["img_path"])` y exportación desacoplada en `ModelArtifact` con proveniencia de librerías de visión e inferencia reproducible vía `predict()`.
  - Validación completa en `tests/test_v09_vision_spike.py` (11 tests nuevos pasando, 570 tests globales en verde, 87.69% cobertura de código). Decision Gate de Fase 4 superado.
## Completed Phase: v0.8 Hardening & Strategic Governance

- [x] Hardening Blackboard Issue #52: resolución de timeout en `test_mcp_cli_subprocess_stdio_handshake` con terminación limpia de proceso (`proc.kill()`) y timeout calibrado a 25s; [issue #52](https://github.com/Jfenic/CATML/issues/52) cerrado.
- [x] Proveniencia Completa en `ModelArtifact`: persistencia automática de versiones de dependencias, Python, hashes SHA-256 del dataset, estimador y semilla aleatoria; método `artifact.describe()` e informe en `docs/backends.md` (547 tests pasando, 87.59% cobertura).
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
- [x] ADR 007: Decision-Gated Multimodal Governance Roadmap formalizado en `docs/decisions/007-decision-gated-multimodal-governance-roadmap.md` (definición: *Controlled Agentic Toolkit for Machine Learning*, desacoplamiento `ProblemSpec` y `BackendCapabilities`, anti-leakage por grupos/entidades, vision spike gate y gobernanza de agentes con data egress policy).

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
- [x] Feature: Vision Support & Interactive Multimodal UI (Workbench Web): Endpoint seguro `/api/media/preview` en `server.py` con resolución por dataset o workspace y prevención anti-traversal; iconos vectoriales Lucide `image` y `camera` en `icons.js`; helper `getMediaPreviewUrl` en `api.js`; badges de tipo `Image` y acción `Vision` en esquema de datasets; filtro segmentado `Images (N)` en toolbar; miniaturas interactivas en previsualización de filas de datos; modal de análisis con pestaña **Sample Image Gallery** y tarjetas visuales con etiqueta target; nodo `ImageEncoderNode` en el grafo DAG interactivo de `pipeline.js`; y opción de modelo `Vision (timm)` en `new_experiment.js`. 4 tests dedicados en `tests/test_web_vision_support.py` (574 tests globales pasando, 0 fallos).

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
- [x] H5 — V1.0 Proveedor real, LangGraph duradero y recuperación (Integración final A5 + B5):
  - [x] Track Persona A (Paquete A5): Adaptador LLM agnóstico con validación de respuesta, timeouts/reintentos acotados, redacción y auditoría de tokens (PR #57).
  - [x] Track Persona B (Paquete B5): Checkpointer LangGraph SQLite, reanudación tras fallo y tests E2E de orquestación (`LangGraphAgentOrchestrator`, `SqliteCheckpointSaver`, CLI `--engine {deterministic,langgraph}`, 590 tests passing, 87.70% cobertura).
- [ ] V0.8 Phase: Meta-learning & knowledge base for warm-start policies (pospuesta temporalmente a favor del subsistema agéntico)

- [x] Stacking with a trained meta-estimator (distinct from voting/blending).
- [ ] Persist backend class, plugin/library versions and dataset/config hashes per trial ([current limits](docs/backends.md)).
- [ ] Add the dedicated `feature_selection_v05` benchmark scenario from the feature-discovery specification.

### Generalist AutoML Engine & Super-Ensembles
> Technical Specs & Execution Plan: [`docs/features/generalist-enhancements/spec.md`](docs/features/generalist-enhancements/spec.md) and [`docs/features/generalist-enhancements/plan.md`](docs/features/generalist-enhancements/plan.md).

- [x] Phase 1: Generalized N-Model OOF Blending & Level-2 Stacking (remove 2-model limit, Rank-Averaging, Ridge meta-learner).
- [x] Phase 2: Temporal Dynamics Engine (automatic sequential detection, sensor lags, 24h trend deltas, cyclical projections, 19 tests passing in `tests/test_temporal_dynamics.py`, 528 suite total, 87.86% coverage).
- [x] Phase 3: Workbench Web UX & Visual Ensembles (Validation Strategy selector with Stratified K-Fold/K-Fold/TimeSeries/Holdout, Multi-Model Ensemble Builder with Average/Rank/Simplex/Stacked L2 and meta-learners Ridge/Logistic/Lasso, Kaggle Drag-and-Drop dropzone for sample_submission.csv with schema verification, and 1-click direct browser download; CQRS BuildEnsembleCommand, REST endpoints /api/ensemble/build, /api/kaggle/upload-template, /api/kaggle/download; 3 tests in tests/test_workbench_phase3.py, 546 suite total, >= 85% coverage).
- [x] Phase 4: Anti-Leakage Guardian in Profiler & Plugin Observability.

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

- [x] Phase 3: Workbench Web UX & Visual Ensembles (Validation strategy selector pills in NewExperimentModal, Visual Ensemble Builder modal in CompareView with average, rank, simplex Nelder-Mead, stacked meta-learners Ridge/Logistic/Lasso, Kaggle template drag-and-drop dropzone with schema verification, and direct 1-click submission CSV download; BuildEnsembleCommand in application/commands, workspace.build_ensemble in workspace service, REST endpoints /api/ensemble/build, /api/kaggle/upload-template, /api/kaggle/download, 3 tests in tests/test_workbench_phase3.py, 546 tests passing, >= 85% coverage).
- [x] Phase 1: Meta-Learning Warm Starts & Real Dataset Fingerprinting (`DatasetFingerprint` 6-dim vector, `MetaKnowledgeBase` cosine similarity & benchmark matching, empirical workspace trial ranking aggregation, trial 0 warm start in `OptunaOptimizer` and `RandomSearchOptimizer` yielding ~35-45% search reduction, CQRS `GetMetaKnowledgeQuery`, CLI `automl meta priors`, REST `/api/knowledge`, 8 tests in `tests/test_meta_learning_warm_starts.py`, 543 tests passing, 87.63% coverage).
- [x] Phase 4: Anti-Leakage Guardian & Plugin Observability (`has_leakage`, `leakage_column_names`, filtering from `recommended_feature_names`, $|r| \ge 0.999$ target correlation detection, sequential row leakage detection $|r_{\text{pos}}| \ge 0.95$, `is_native`, `fallback_backend`, `backend_status` observability in `LightGBMPlugin`, `XGBoostPlugin`, `CatBoostPlugin`, CLI `automl plugin list`, 6 tests passing in `tests/test_anti_leakage_and_observability.py`, 534 tests passing, 87.88% coverage).
- [x] Phase 2: Temporal Dynamics Engine (contracts in `src/automl/domain/features/temporal.py`, detection in `dataset_profiler.py`, generation in `temporal_generator.py`, CQRS `GenerateTemporalFeaturesCommand` / `DetectTemporalStructureQuery`, CLI `automl features temporal`, REST endpoints, 19 tests in `tests/test_temporal_dynamics.py`, 528 tests passing, 87.86% coverage).
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
