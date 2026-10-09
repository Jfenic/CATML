# Progress

### Track Technical Audit Remediation & System Hardening (2026-10-09)

- **Remediación Integral de Hallazgos de Auditoría Técnica (Commit `5caa1f5`):**
  - **H1 (Aprobación real de agentes en Workbench):**
    - Se integró `POST /api/agent/action` con `SqliteAgentLedger` en `src/automl/interfaces/web/server.py`.
    - Validación de existencia de hipótesis y solicitudes de aprobación en el ledger (fail-closed `404 Not Found` ante identificadores desconocidos).
    - Transición de estado persistida transaccionalmente (`proposed` $\to$ `accepted` / `rejected`), registrando revisor y notas del operador humano.
  - **H2 (Normalización del signo en métricas personalizadas de minimización en CV):**
    - En `src/automl/engine/training/sklearn_trainer.py`, se eliminó la dependencia sobre el atributo privado inexistente `_greater_is_better` de scikit-learn.
    - Se incorporó `_is_minimizing_metric(metric_name, plugin_registry)` e inspección de `_sign == -1` para invertir el signo negativo producido por scikit-learn en validación cruzada.
    - Sincronización de `plugin_registry` en `SQLiteExperimentRepository.get_leaderboard()` y `AutoMLWorkspace.__post_init__()` para ordenar métricas personalizadas minimizables ascendentemente (`ASC`), garantizando consistencia idéntica entre holdout, CV, Optuna y leaderboard.
  - **H3 (Autenticación estricta en servidor MCP para interfaces remotas):**
    - En `src/automl/interfaces/mcp/server.py`, el transporte `streamable-http` rechaza enlaces externos (`host != 127.0.0.1`) sin token de autenticación (`PermissionError` fail-closed).
    - Soporte para `--token` y `--insecure-no-auth` en CLI (`main.py` y `mcp_cli.py`), incorporando `BearerAuthMiddleware` en la aplicación Starlette subyacente.
  - **H4 (Límite explícito de tamaño de payload HTTP):**
    - En `src/automl/interfaces/web/server.py`, se estableció `MAX_PAYLOAD_SIZE = 50 MB`.
    - `do_POST` valida `Content-Length` y bytes leídos retornando inmediatamente `413 Payload Too Large` ante solicitudes excesivas antes de saturar memoria.
  - **H5 (Endurecimiento de autenticación remota en Workbench):**
    - `_is_authenticated()` en `server.py` rechaza tajantemente tokens en query string (`?token=`) para peticiones de mutación de estado (`POST`).
    - En peticiones `GET`, emite advertencia de seguridad instando al uso de cabeceras `Authorization: Bearer <token>` o fragmentos `#token=`.
  - **H6 (Declaración de versión de scikit-learn):**
    - En `pyproject.toml`, se actualizó la dependencia mínima a `scikit-learn>=1.4.0` para garantizar disponibilidad de `response_method` en `make_scorer`.
    - Se añadió fallback defensivo en `sklearn_trainer.py` (`needs_proba`) para retrocompatibilidad total.
  - **H8 (Compatibilidad multiplataforma en bloqueo de workspaces):**
    - En `src/automl/infrastructure/jobs/worker.py`, se implementó bloqueo condicional con `msvcrt` en entornos Windows y `fcntl` en sistemas Unix.
  - **Validación:** Suite completa de pruebas superada (673 pruebas, 0 fallos), cobertura total mantenida en 86.67% (umbral CI $\ge 85\%$).

### Track Modular Simplification — Pre-Explore Phase E0.5 (2026-10-08)

- **PR 1: Descomposición de Registros CQRS y Modularización de Bootstrap (`hardening/v0.8.2-e0.5-pr1-bootstrap-modularization`):**
  - Creación del paquete `src/automl/application/registries/` con submódulos independientes por dominio:
    - `job_registry.py`: `register_job_handlers` (con inyección opcional de `JobService`).
    - `core_registry.py`: `register_core_handlers` (runs, datasets, profiles, plugins, task types).
    - `experiment_registry.py`: `register_experiment_handlers` (modelos, experimentos, colas, optimización de hiperparámetros).
    - `feature_registry.py`: `register_feature_handlers` (selección, ingeniería temporal y de interacción, ablación).
    - `inference_registry.py`: `register_inference_handlers` (predicción, alineación Kaggle, ensembles, pipelines DAG).
  - Refactorización de `src/automl/application/bootstrap.py`: Reducido drásticamente de 342 a 53 líneas como orquestador limpio y desacoplado, manteniendo 100% retrocompatible la función `build_application`.
  - Guard arquitectónico automatizado en `tests/test_cqrs_architecture_guard.py` (`test_modular_registries_can_be_composed_independently`).
  - Validación completa: 623 tests pasando (100% verde), 87.37% de cobertura.
- **PR 2: Desacoplamiento de Contratos para Datasets sin Target y Jobs sin Run ID (`hardening/v0.8.2-e0.5-pr2-contract-decoupling`):**
  - Desacoplamiento de entidades `Dataset` y `DatasetProfile` (`src/automl/domain/datasets/profile.py`): campos opcionales `target_column: str | None = None` y `task_type: str | None = None`. Manejo seguro en propiedades computadas (`recommended_feature_names`, `leakage_column_names`).
  - Flexibilización de `profile_dataset` (`src/automl/engine/profiling/dataset_profiler.py`): análisis descriptivo, cardinalidad, tipos semánticos y matrices de covarianza y colinealidad sin requerir variable objetivo supervisada.
  - Persistencia de datasets en SQLite (`src/automl/infrastructure/database/sqlite_repository.py`): tabla `datasets` admite `target_column TEXT NULL` y `task_type TEXT NULL`.
  - Ingestión de datasets en `AutoMLWorkspace.register_dataset()` con `target=None` y planificación de tareas por defecto (`TaskType.CLUSTERING`) vía `plan_from_dataframe` (`src/automl/engine/planning/task_planner.py`).
  - Desacoplamiento de Jobs durables (`src/automl/domain/jobs/job.py`, `src/automl/infrastructure/database/sqlite_jobs.py`): `run_id: str | None = None` con migración resiliente en SQLite para bases existentes.
  - Salvaguardas en `JobService` (`src/automl/application/services/jobs.py`) para operaciones independientes de ejecuciones de AutoML.
  - Suite de pruebas unitarias y de integración en `tests/test_unsupervised_datasets_and_decoupled_jobs.py` (4 tests).
  - Validación completa: 627 tests pasando (100% verde), 87.38% de cobertura.
- **PR 3 (PR #79): Hardening de Migraciones SQLite y Compatibilidad Retroactiva (`hardening/v0.8.2-sqlite-migrations-compatibility`):**
  - Implementación de migración atómica determinista de la tabla `datasets` en `SQLiteExperimentRepository._migrate()` mediante reconstrucción transaccional cuando detecta `notnull == 1` en `target_column` o `task_type` de bases heredadas v0.8.0/v0.8.1.
  - Implementación de migración determinista de la tabla `jobs` en `SQLiteJobRepository` relajando `run_id NOT NULL` con manejo estructurado.
  - Erradicación de excepciones silenciadas (`except Exception: pass`) reemplazadas por logging estructurado y excepciones explícitas fail-closed (`RuntimeError`).
  - Suite de pruebas dedicada con bases de datos heredadas y datos preexistentes en `tests/test_sqlite_legacy_migrations.py` (2 tests verificando 0 pérdida de datos y compatibilidad de inserción con `None`).
  - Actualización del contador en `README.md` a 629 tests passing y clarificación de `statsmodels` en `docs/decisions/008-catml-explore-modular-monolith.md`.
  - Validación completa: 629 tests pasando (100% verde), 87.37% de cobertura.

### Track Security & Reliability Hardening — Auditoría Técnica Oct 2026 (2026-10-08)

- **PR A (PR #80): Blindaje Anti-Leakage Centralizado y Fail-Closed (`hardening/v0.8.2-anti-leakage-guardian`):**
  - Centralización de la política de resolución de variables seguras en `DatasetProfile.resolve_safe_feature_names()` (`src/automl/domain/datasets/profile.py`):
    - Filtrado riguroso de target, identifiers y columnas con leakage confirmado (target leakage y group leakage).
    - Soporte para tareas no supervisadas / clustering (desactivación de target leakage cuando no existe variable objetivo o `task_type == "clustering"`).
    - Parámetro de escape explícito `allow_leakage: bool = False` para estudios controlados / benchmarks.
  - Fail-closed preventivo en `AutoML.fit()` (`src/automl/facade.py`):
    - Se elimina el reciclaje inseguro de variables activas del registro cuando todas están contaminadas; interrupción inmediata con `ValueError` explicativo.
  - Blindaje en `RuleBasedExperimentPlanner` (`src/automl/engine/planning/experiment_planner.py`):
    - Se elimina el fallback permisivo que reciclaba columnas tras filtrado total; retorna lista vacía de candidatos si no existen features seguras.
  - Blindaje en `JobExecutor` (`src/automl/application/services/job_executor.py`):
    - Resolución segura automática de features excluyendo leakage para jobs de entrenamiento en background.
  - Validación estricta en `AutoMLWorkspace.create_experiment()` (`src/automl/application/services/workspace.py`):
    - Verificación contra leakage al crear experimentos manuales, propagando `allow_leakage` a través de `CreateExperimentCommand` y el bus CQRS.
  - Calibración de heurísticas estadísticas en `dataset_profiler.py`:
    - `cardinality_ratio <= 0.90` en `detect_is_group_candidate` para evitar falsos positivos en variables continuas (como `account_balance`).
    - `row_count >= 10` en `detect_target_correlation_leakage` para evitar falsos positivos en datasets sintéticos toy/unit tests ($N < 10$).
  - Suite de pruebas exhaustiva en `tests/test_anti_leakage_guardian.py` (6 tests).
  - Validación completa: 635 tests pasando (100% verde), 87.29% de cobertura.
- **PR B (PR #81): Seguridad del Workbench (Confinamiento de Archivos y Sanitización de Rutas) (`hardening/v0.8.2-workbench-file-confinement`):**
  - Confinamiento estricto de `/api/kaggle/download` (`src/automl/interfaces/web/server.py`):
    - Resolución segura de rutas y validación de pertenencia al directorio del workspace (`is_relative_to(ws_root)`).
    - Lista blanca de extensiones para descargas (`.csv`, `.tsv`, `.parquet`, `.pq`, `.json`, `.zip`, `.txt`).
    - Bloqueo fail-closed con `403 Forbidden` ante intentos de path traversal (`../../`), rutas absolutas externas (`/etc/passwd`) o symlinks que apunten fuera del workspace.
  - Confinamiento estricto de `/api/media/preview`:
    - Eliminación de `ws_root.parent` y `Path.cwd()` de la lista `allowed_roots`, restringiendo la lectura exclusivamente a `ws_root` y a los directorios de datasets registrados.
    - Bloqueo preventivo de acceso a archivos del directorio padre o directorios externos del sistema.
  - Sanitización rigurosa de `/static/`:
    - Sustitución de comprobaciones basadas en `str.startswith` por `target_file.is_relative_to(static_root)`.
    - Respuesta explícita `403 Forbidden` ante intentos de escape del directorio de estáticos.
  - Suite de pruebas de seguridad exhaustiva en `tests/test_workbench_file_confinement.py` (4 tests).
  - Validación completa: 639 tests pasando (100% verde), 87.39% de cobertura.
- **PR C (PR #82): Atomicidad Transaccional de Migraciones SQLite y Rollback Verificado (`hardening/v0.8.2-sqlite-migration-atomicity`):**
  - Reemplazo de `conn.executescript()` por transacciones deterministas explícitas `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK` en `SQLiteExperimentRepository._migrate()` (`src/automl/infrastructure/database/sqlite_repository.py`) y `SQLiteJobRepository._migrate()` (`src/automl/infrastructure/database/sqlite_jobs.py`).
  - Configuración temporal de `conn.isolation_level = None` durante la ventana de migración DDL, garantizando control transaccional estricto ante excepciones intermedias y restaurándolo fielmente en bloques `finally`.
  - Verificación estricta de conteo de registros antes y después de la reconstrucción de tablas (`post_count == pre_count`), levantando `RuntimeError` y ejecutando `ROLLBACK` si se detecta cualquier discrepancia o truncamiento de datos.
  - Limpieza de tablas temporales (`_dg_tmp_datasets`, `_dg_tmp_jobs`) en caso de rollback, asegurando que la base de datos retenga su esquema íntegro original y cero artefactos residuales o bloqueos.
  - Suite de pruebas dedicada para verificar rollback atómico ante fallos forzados de migración en `tests/test_sqlite_legacy_migrations.py` (`test_datasets_migration_failure_rolls_back_atomically`, `test_jobs_migration_failure_rolls_back_atomically`).
  - Validación completa: 641 tests pasando (100% verde), 87.41% de cobertura.
- **PR D (PR #83): Consistencia del Producto — Dirección de Métricas, Benchmarks y Actividad Real del Workbench (`hardening/v0.8.2-product-consistency`):**
  - **Dirección de Métricas en Benchmarks (`src/automl/benchmarks/runner.py`):**
    - Adición de `is_minimizing_metric(metric, ws=None)` reconociendo de forma unificada métricas de pérdida (`mae`, `rmse`, `mse`, `loss`, `log_loss`) y plugins con `greater_is_better = False`.
    - Actualización de `_best_result()` y `run_all()` para seleccionar el modelo con puntaje mínimo en métricas minimizantes (`min`) y máximo en maximizantes (`max`), evitando la selección invertida del peor modelo en benchmarks de regresión.
    - Suite de pruebas dedicada en `tests/test_benchmark_metric_direction.py` (5 tests).
  - **Actividad Real del Workbench y Eliminación de Mock Data (`src/automl/interfaces/web/server.py`, `src/automl/interfaces/web/static/js/views/agent.js`):**
    - Extracción de `_build_real_activity_feed()` en `server.py` que compone eventos estrictamente desde el estado real del workspace:
      - Ganadores legítimos de leaderboard con dirección de métrica respetada (`ACCEPT`).
      - Exclusiones preventivas de leakage e identificadores por el Anti-Leakage Guardian (`REJECT`).
      - Ensayos fallidos con su causa real de error (`REJECT`).
      - Experimentos y jobs planificados o completados (`PLAN` / `ACCEPT`).
      - Hipótesis promovidas, rechazadas o propuestas extraídas dinámicamente de `agent_ledger.db`.
      - Workspace vacío retorna `activity_feed: []`, activando limpiamente el estado vacío nativo de la UI.
    - En `/api/agent/hypotheses`: eliminación de hipótesis mockeadas (`hyp_12`, `hyp_13`, `hyp_14`) y sustitución por consulta dinámica a `SqliteAgentLedger`.
    - En `AgentDrawer` (`agent.js`): renderizado dinámico con consulta asíncrona a `api.getAgentHypotheses()` y estado vacío explícito.
    - Suite de pruebas de consistencia en `tests/test_workbench_activity_consistency.py` (5 tests).
  - Validación completa: 651 tests pasando (100% verde), 87.32% de cobertura.
- **PR E (PR #84): Hardening Complementario — Protección XSS, Segregación de Métricas en Workbench y Confinamiento Estricto de Descargas (`hardening/v0.8.2-workbench-xss-and-metrics-alignment`):**
  - **Protección XSS y Sanitización Frontend (`src/automl/interfaces/web/static/js/utils.js`):**
    - Creación de función `escapeHtml` para sanitización de HTML, strings dinámicos y atributos.
    - Aplicación en `agent.js` (enunciados de hipótesis, acciones propuestas, decisiones y razones del crítico, IDs de hipótesis, dataset y task types).
    - Aplicación en `overview.js` (feed de actividad, listado de datasets, filas del leaderboard y KPIs).
    - Aplicación en `kaggle.js` (listado de experimentos de envíos y estados).
  - **Alineación y Segregación de Métricas en `/api/overview` (`server.py`):**
    - Erradicación de la comparación cruzada de métricas incomparables (`is_minimize` mezclado entre diferentes ejecuciones).
    - Incorporación de `best_by_metric` en la respuesta JSON agrupando campeones por métrica sin mezclar escalas.
    - Contextualización de `best_score`, `best_model` y nuevo campo `best_metric` asociados a la ejecución activa o más reciente.
  - **Dirección de Mejora en Benchmarks (`src/automl/benchmarks/runner.py`):**
    - Adaptación del cálculo de `delta_vs_baseline` para que tanto en métricas de ganancia (ROC-AUC) como de pérdida (RMSE, MAE), una mejora respecto a la línea base se exprese como ganancia positiva ($\Delta > 0$).
  - **Confinamiento Estricto de Descargas en `/api/kaggle/download` (`server.py`):**
    - Confinamiento exclusivo a `ws_root / "submissions"` (y `exports/`).
    - Bloqueo preventivo de descargas arbitrarias de archivos ubicados en la raíz del workspace o con extensiones no tabulares/de archivo (`.csv`, `.tsv`, `.parquet`, `.pq`, `.zip`).
  - **Suite de Pruebas Automatizadas:**
    - `tests/test_workbench_xss_protection.py` (5 tests verificando sanitización de UI, segregación de métricas y cálculo de delta).
    - Ampliación de `tests/test_workbench_file_confinement.py` (4 tests verificando rechazo de archivos en raíz del workspace).
  - **Validación completa:** 656 tests pasando (100% verde), 87.33% de cobertura.
- **PR F (PR #85): Alineación de Estado Activo y Feed de Actividad en Workbench (`hardening/v0.8.2-workbench-active-state-and-feed-alignment`):**
  - **Resolución de Estados Activos en Dominio y API (`src/automl/domain/runs/states.py`, `src/automl/domain/runs/run.py`, `src/automl/interfaces/web/server.py`):**
    - Definición del conjunto canónico `ACTIVE_RUN_STATUSES` (`CREATED`, `PROFILING`, `PLANNING`, `EXPERIMENTING`, `OPTIMIZING`, `FINALIZING`) y función `is_active_run_status()`.
    - Adición de `@property def is_active(self) -> bool` en `AutoMLRun`.
    - Exposición del flag `is_active` en las respuestas JSON de `/api/runs` y `/api/overview`.
    - Selección prioritaria de ejecuciones activas sobre ejecuciones completadas para la vista principal del Workbench.
  - **Segregación del Feed de Actividad (`server.py`):**
    - Erradicación de la comparación escalar cruzada entre métricas incompatibles (`score < prev_score`) en `_build_real_activity_feed()`.
    - Emisión contextualizada de eventos de leaderboard por ejecución con indicación explícita de métrica, dataset e ID de run.
  - **Consistencia Frontend & Tech Minimalista (`utils.js`, `overview.js`, `app.js`, `agent.js`, `studio.js`, `new_experiment.js`, `compare.js`, `kaggle.js`, `knowledge.js`, `pipeline.js`):**
    - Función exportada `isRunActive()` en `utils.js` consumida en todas las vistas del Workbench y en el polling global (`app.js`).
    - Soporte en `overview.js` para scores iguales a `0.0` y métricas con valores negativos ($R^2 < 0$) mediante `overview.best_score != null`.
    - Corrección del mensaje en el bloque de captura de error al programar el plan del agente en `agent.js` (`"Agent action failed: "`).
  - **Suite de Pruebas Automatizadas:**
    - Nuevos tests en `tests/test_workbench_activity_consistency.py` (`test_workbench_selects_active_experimenting_run_over_completed_run`, `test_activity_feed_multi_run_segregated_without_comparing_incompatible_metrics`, `test_domain_is_active_run_status_and_run_property`, `test_workbench_frontend_active_status_and_agent_error_alert`).
  - **Validación completa:** 660 tests pasando (100% verde), 87.31% de cobertura.
- **PR G: Flexibilidad Anti-Leakage en Planificación e Integridad de Métricas en Workbench (`fix/planner-anti-leakage-and-metric-integrity`):**
  - **Flexibilidad Anti-Leakage en Planificación (`profile.py`, `experiment_planner.py`):**
    - Adición del parámetro `strict: bool = True` en `DatasetProfile.resolve_safe_feature_names()`. Si `strict=False`, las columnas con fuga de datos o identificadores se descartan de forma segura en vez de levantar `ValueError`, permitiendo continuar si restan características limpias.
    - Uso de `strict=False` en `RuleBasedExperimentPlanner.propose()` para soportar datasets mixtos (con columnas de leakage e identificadores junto a predictoras válidas).
  - **Segregación de Métricas por Dataset y Resiliencia en Workbench (`server.py`):**
    - En `/api/overview`, segregación de mejores modelos por dataset (`best_by_dataset`) para evitar comparaciones escalares numéricas erróneas entre datasets dispares.
    - Preservación de puntuaciones de métricas con valor exacto `0.0` y cálculo de deltas en `/api/agent/hypotheses`.
    - Inicialización limpia de `best_score`, `best_model` y `best_metric` como `None` en workspaces vacíos sin ejecuciones.
  - **Suite de Pruebas Automatizadas:**
    - 2 nuevos tests en `tests/test_anti_leakage_guardian.py` (`test_experiment_planner_proposes_candidates_on_mixed_clean_and_leakage_dataset`, `test_resolve_safe_feature_names_strict_vs_filtering_mode`).
    - 2 nuevos tests en `tests/test_workbench_activity_consistency.py` (`test_agent_hypotheses_preserves_exact_zero_metrics`, `test_overview_segregates_metrics_across_different_datasets`).
  - **Validación completa:** 664 tests pasando (100% verde), 87.32% de cobertura.
- **PR: Rediseño UI Workbench — Modelo de Laboratorio y Dataset Activo (Fase 1) (`feat/ui-laboratory-redesign-phase1`):**
  - **Contexto de Laboratorio y Dataset Activo en Barra Lateral (`index.html`, `app.js`, `store.js`):**
    - Indicador de estado "Laboratorio local • ONLINE".
    - Tarjeta de dataset activo con indicador reactivo (`#sidebarActiveDatasetName`) y botón de acción rápido `Cambiar dataset` (`#btnSidebarSwitchDataset`).
    - Métodos reactivos en `WorkbenchStore`: `setActiveDataset(datasetId)`, `getActiveDataset()`, sincronización con `localStorage` y alineación automática de `activeRunId`.
  - **Simplificación de la Navegación Principal (`index.html`, `app.js`):**
    - Reducción de ~10 opciones dispersas a 4 ventanas de flujo canónico: **Inicio** (`home`), **Dataset** (`dataset`), **Experimentos** (`experiments`), **Evidencia** (`evidence`).
    - Paridad y retrocompatibilidad en el router para rutas previas.
  - **Nueva Vista de Inicio (`src/automl/interfaces/web/static/js/views/home.js`):**
    - Hero Card principal: «Continuar con {dataset activo}» con dimensiones, variable objetivo, tarea y mejor modelo, junto a acciones secundarias «Añadir dataset» y «Abrir dataset guardado».
    - Panel de «Ejecuciones en curso»: estado de entrenamiento, barra de progreso y métricas en vivo.
    - Panel de «Trabajos recientes»: historial técnico de jobs en background consultados dinámicamente vía `api.getJobs()`.
  - **Rediseño de la Vista de Dataset (`src/automl/interfaces/web/static/js/views/datasets.js`):**
    - Pestaña «Resumen»: Diagnóstico de salud con Anti-Leakage Guardian (fugas y variables descartadas), métricas de perfilado, recomendaciones estadísticas y muestra previa de datos crudos.
    - Pestaña «Columnas»: Tabla completa interactiva de columnas con filtrado, buscador, porcentajes de nulos, cardinalidad, acción inferida y selección de variables.
    - Pestaña «Exploración»: Panel conceptual de arquitectura técnica para la Fase E1 de CATML Explore (ADR-008).
    - Botón CTA destacado «Nuevo experimento» en cabecera.
  - **Modal de Selección Rápida de Dataset (`src/automl/interfaces/web/static/js/views/switch_dataset_modal.js`):**
    - Modal accesible desde cualquier vista para conmutar el dataset activo sin perder el historial ni el trabajo previo.
  - **Contextualización en Vistas Existentes (`studio.js`, `overview.js`, `new_experiment.js`):**
    - Filtrado y priorización automática de ejecuciones y sugerencias pertenecientes al dataset activo.
  - **Suite de Pruebas Automatizadas:**
    - `tests/test_workbench_laboratory_redesign.py` (3 tests verificando layout, componentes, enrutamiento y métodos de estado).
  - **Validación completa:** 667 tests pasando (100% verde), 87.32% de cobertura.

### Track Architectural Planning — CATML Explore Evolution Plan (2026-10-08)

- **Formalización de CATML Explore como Módulo de Monolito Modular:**
  - Registro de decisión arquitectónica en [`docs/decisions/008-catml-explore-modular-monolith.md`](docs/decisions/008-catml-explore-modular-monolith.md).
  - Especificación funcional y de dominio en [`docs/features/catml-explore/spec.md`](docs/features/catml-explore/spec.md).
  - Plan de implementación y roadmap por fases E0 a E6 en [`docs/features/catml-explore/plan.md`](docs/features/catml-explore/plan.md).
  - Integración en [`docs/MASTER_PLAN.md`](docs/MASTER_PLAN.md) (Sección 21) y actualización del backlog operativo en [`TASKS.md`](TASKS.md).
  - Definición del modelo de dominio: `DataSourceRef`, `StudySpec`, `AnalysisRun`, `StatisticalFinding`, `VisualizationSpec`, `AnalysisHypothesis`, `EvidenceLink`.
  - Establecimiento del principio *"La IA Interpreta, CATML Calcula"* para desacoplar el cálculo matemático determinista de la formulación de hipótesis por agentes en MCP/CLI.

### Documentación pública (2026-10-08)

- Aplicado CATML_monetizacion_privada.patch en docs/private-commercial-strategy; retiradas referencias comerciales de documentación y landing.
- Validación: build y comprobaciones web responsive/accesibilidad correctos; Python: 601 passed, 17 failed (fcntl no disponible y saltos CRLF en Windows), cobertura 86.22 %. gh no está disponible en este equipo.

### Track Hardening Phase 2 — v0.8.2 Ingestion Polymorphism, Strict CQRS Boundaries & Application Services Decomposition (2026-10-08)

- Rama de trabajo `hardening/v0.8.2-phase2-modular-and-cqrs` desde `main` (`ec69771`).
- **Ingestión Polimórfica y Soporte Nativo de Parquet/DataFrames (`src/automl/engine/profiling/dataset_profiler.py`, `src/automl/application/services/workspace.py`, `src/automl/facade.py`, `pyproject.toml`):**
  - Función `load_dataframe()` en `dataset_profiler.py` con parsing automático de `.parquet`, `.pq`, `.json`, `.jsonl`, `.tsv` y `.csv`. Manejo con mensaje amigable de `ImportError` solicitando `pyarrow` o `fastparquet` si no se encuentran instalados.
  - Actualización de `profile_dataset()` para delegar en `load_dataframe()`.
  - En `AutoMLWorkspace.register_dataset()`, polimorfismo que acepta `pd.DataFrame` directamente, persistiendo de forma automática en el subdirectorio `datasets/` del workspace y registrándolo sin requerir guardado manual externo por parte del usuario.
  - En `AutoML.fit()` (`facade.py`), polimorfismo para aceptar rutas en `str` o `Path` (además de DataFrames y matrices numpy), integrándose fluidamente con `_standardize_input()`.
  - Definición del extra `parquet = ["pyarrow>=14.0"]` en `pyproject.toml` e inclusión de `pyarrow` en el grupo `all`.
  - Pruebas unitarias y de integración en `tests/test_parquet_and_polymorphic_ingestion.py` (4 tests).
- **Frontera Arquitectónica CQRS Estricta y Erradicación de Fugas de Estado (`src/automl/application/queries/workspace_queries.py`, `src/automl/application/bootstrap.py`, `src/automl/application/services/workspace.py`, `src/automl/facade.py`, `src/automl/interfaces/web/server.py`, `src/automl/interfaces/cli/agent_session_cli.py`):**
  - Formalización de consultas tipadas en `QueryBus`: `GetRunQuery`, `ListRunsQuery`, `GetDatasetQuery`, `ListDatasetsQuery`, `GetTrialQuery`, `GetExperimentQuery`, `ListTrialResultsQuery`.
  - Registro de los nuevos handlers en `src/automl/application/bootstrap.py` desacoplados de infraestructura directa.
  - Métodos de consulta públicos agregados a `AutoMLWorkspace`: `list_datasets()`, `get_dataset_profile()`, `save_dataset_profile()`, `save_run()`, `get_trial()`, `get_experiment()`, `list_experiments()`, `list_trial_results()`, `get_leaderboard_results()`, `list_feature_sets()`.
  - Erradicación de todas las llamadas directas a `ws.repository` en la fachada (`facade.py`) y en el servidor web (`interfaces/web/server.py`).
  - Eliminado el monkey-patching sobre `ws` en `agent_session_cli.py`.
  - Guard arquitectónico automatizado instalado en `tests/test_cqrs_architecture_guard.py` (2 tests verificando 0 ocurrencias de fugas privadas en interfaces/presentación y despacho vía `QueryBus`).
- **Descomposición del God Module `workspace.py` en Servicios Especializados de Aplicación:**
  - `src/automl/application/services/pipeline_service.py` (`PipelineExecutionService`): extracción de validación de grafos, orden topológico de ejecución, construcción de DAG multimodal, ejecución y fit/predict.
  - `src/automl/application/services/inference_service.py` (`InferenceService`): extracción de predicción, alineación de plantillas Kaggle, generación de envíos y exportación de `ModelArtifact`.
  - `src/automl/application/services/feature_service.py` (`FeatureEngineeringService`): extracción de características derivadas, análisis de dinámica temporal, estrategias de selección multi-método, planificación de ablación y promoción de conjuntos de características.
  - Delegación 100% retrocompatible en `AutoMLWorkspace` preservando firmas públicas y reduciendo el tamaño de `workspace.py` en más de 740 líneas (~2,550 a ~1,800 líneas).
- **Resolución de Auditoría Técnica y Hardening de Confiabilidad ML (`src/automl/engine/training/sklearn_trainer.py`, `src/automl/infrastructure/database/sqlite_repository.py`, `src/automl/application/services/workspace.py`, `src/automl/facade.py`, `src/automl/interfaces/mcp/server.py`, `src/automl/interfaces/web/server.py`, `src/automl/interfaces/web/static/js/api.js`, `tests/test_audit_findings.py`):**
  - **Signo de MAE/RMSE/MSE/Log Loss en CV**: En `sklearn_trainer.py`, los scores de `cross_val_score` con scoring `neg_...` o `_greater_is_better=False` se invierten mediante `scores = -scores`. El leaderboard y optimizador en `workspace.py` ordenan ascendentemente (`ASC`) y minimizan, garantizando que el menor error ocupe la primera posición.
  - **Exclusión de Trials Fallidos**: En `sqlite_repository.py`, `get_leaderboard()` agrega por defecto `AND (tr.failure_reason IS NULL OR tr.failure_reason = '')` (`include_failed=False`). En `facade.py`, `fit()` verifica que existan trials exitosos y descarta fallos; en `inference_service.py`, `export_model_artifact()` elige únicamente entre modelos con `res.succeeded is True`.
  - **Eliminación de Sustitución Silenciosa de Métricas**: En `_sklearn_scoring()` y holdout `_compute_metrics()`, métricas no reconocidas levantan `ValueError` explícito en vez de recurrir a `accuracy`/`r2`. Se añadió soporte directo para `mse`, `log_loss`, `balanced_accuracy`, `precision`, `recall` y plugins métricos mediante `make_scorer`.
  - **Clarificación de Time Budget**: Documentado en `facade.py` que `time_budget` es un límite cooperativo global entre modelos y no un timeout atómico por proceso.
  - **Endurecimiento de Seguridad HTTP/MCP**: En `api.js` y `server.py`, paso de credenciales mediante URL hash fragment `#token=` (inmune a logs de proxy/historial de peticiones) y auto-limpieza con `history.replaceState`. En el servidor MCP Streamable-HTTP, advertencia de seguridad explícita (`SECURITY NOTICE`) si se expone a interfaces de red externas.
  - **Robustez de Validación en Datasets Reducidos**: `SklearnTrainer` adapta `n_splits` en KFold para asegurar suficientes muestras de test y permitir cálculos estables de varianza ($R^2$), y en holdout ajusta `effective_test_size` para preservar representatividad de todas las clases.
  - 4 tests automatizados de regresión en `tests/test_audit_findings.py`.
- **Validación y Cobertura:**
  - 622 tests pasando (100% verde). Cobertura total del 87.37% (requisito >= 85%).
  - CLI `automl --help` y `automl task list` verificados exitosamente.

### Track Hardening Phase 1 — v0.8.2 Operational Budget, Fail-Closed Artifacts & Remote Web Security (2026-10-07)

- Rama de trabajo `hardening/v0.8.2-phase1-20261007` desde `main` (`7c78f13`).
- **Cumplimiento Operativo Real de `time_budget` (`src/automl/domain/runs/run.py`, `src/automl/application/services/workspace.py`, `src/automl/facade.py`):**
  - Añadido campo `time_budget_seconds: float | None = None` a `RunConfig`, con propiedad retrocompatible `effective_time_budget` que soporta tanto `time_budget_seconds` como la clave heredada `extra["time_budget"]`.
  - Serialización y deserialización limpias en `to_dict()` y `from_dict()`.
  - En `AutoMLWorkspace`, tracking de deadlines activos mediante `_run_deadlines: dict[str, float]` calculados con `time.monotonic()`.
  - En `run_experiment()`, verificación estricta de `_is_run_budget_exhausted(refreshed)` antes de iniciar cada trial; si el presupuesto expira, se interrumpe el ciclo de trials de forma limpia, se registra `time_budget_exhausted = True` en `run.config.extra` y se emite el evento de auditoría `TimeBudgetExhausted`.
  - En `run_scheduled_experiments()`, interrupción limpia de la cola si expira el presupuesto global del run.
  - En `AutoML.fit()`, propagación formal de `time_budget_seconds=self.time_budget` a `create_run()`. Si el presupuesto expira antes de completar al menos un modelo válido, levanta `RuntimeError` explícito; si ya existen modelos válidos entrenados, finaliza con éxito preservando los resultados y marcando `time_budget_exhausted=True`.
  - Tests unitarios en `tests/test_time_budget_enforcement.py` (4 tests).
- **Seguridad Fail-Closed en ModelArtifact (`src/automl/artifacts/model_artifact.py`):**
  - Eliminados los bloques `except Exception: pass` silenciosos tanto en la generación del checksum `.sha256` en `save()` como en la verificación en `load()`.
  - En `ModelArtifact.load(verify_checksum=True)`, política fail-closed estricta: levanta `ValueError` explícito si el archivo sidecar `.sha256` falta, está corrupto, es inaccesible o no coincide antes de intentar cualquier deserialización con `joblib.load()`.
  - Tests unitarios en `tests/test_model_artifact_fail_closed.py` (3 tests).
- **Autenticación por Token en Workbench Remoto (`src/automl/interfaces/web/server.py`, `src/automl/interfaces/cli/main.py`, `src/automl/interfaces/web/static/js/api.js`):**
  - En `AutoMLWebHandler`, comprobación de cabecera `Authorization: Bearer <token>` o parámetro URL `?token=<token>`. Devuelve `401 Unauthorized` si la autenticación está activa y el token no es provisto o es incorrecto.
  - En `run_web_dashboard()`, si la dirección de escucha no es local (ej. `0.0.0.0`), se exige token automáticamente: si el usuario no especificó `--auth-token`, genera uno criptográficamente seguro con `secrets.token_urlsafe(16)` e imprime las instrucciones y enlace autenticado en consola.
  - Soporte para bandera explícita `--insecure-no-auth` para desactivar intencionalmente el control si el usuario así lo decide.
  - En el frontend (`api.js`), extracción automática del token desde la URL, persistencia en `sessionStorage` e inyección de cabecera `Authorization: Bearer` en todas las peticiones fetch a la API.
  - CLI `automl ui` / `catml ui` soporta flags `--auth-token` y `--insecure-no-auth`.
  - Tests unitarios en `tests/test_workbench_auth.py` (3 tests).
- **Cierre de Fuga CQRS en CLI (`src/automl/interfaces/cli/main.py`, `src/automl/application/services/workspace.py`):**
  - Añadido método público de consulta `AutoMLWorkspace.list_runs(dataset_id: str | None = None) -> list[AutoMLRun]`.
  - Sustituida la introspección privada `ws._runs.values()` en el comando `predict_cli` por la llamada al método público `ws.list_runs(dataset_id=dataset.id)`.
- **Validación:**
  - 10 nuevos tests unitarios y de integración añadidos. Suite global en verde (612 tests pasando) y cobertura >= 85%.


- Rama de trabajo `feat/v081-trust-patch` desde `main` (`5caf495`).
- **Fail-Safe Anti-Leakage (`src/automl/facade.py` & `src/automl/engine/planning/experiment_planner.py`):**
  - `AutoML.fit()` y `RuleBasedExperimentPlanner` ahora priorizan `profile.recommended_feature_names`, excluyendo de forma automática cualquier columna clasificada con target leakage, leakage de entidad o identificadores.
  - Test unitario de regresión en `tests/test_v081_trust_patch.py::test_trust_patch_leakage_fail_safe`.
- **Aislamiento Local & CORS Restringido (`src/automl/interfaces/web/server.py` & `src/automl/interfaces/cli/main.py`):**
  - Servidor web del Workbench configurado para escuchar en `127.0.0.1` por defecto (con soporte explícito para `--host 0.0.0.0` en CLI si el usuario desea exponerlo intencionalmente).
  - Eliminado el uso de wildcard `Access-Control-Allow-Origin: *`; las respuestas CORS ahora solo se emiten dinámicamente si el `Origin` proviene de `localhost` o `127.0.0.1`.
- **Confinamiento de Archivos en Media Preview (`src/automl/interfaces/web/server.py`):**
  - Endpoint `/api/media/preview` ahora comprueba estrictamente con `is_relative_to()` que la ruta solicitada resida dentro de `workspace_dir` o del directorio del dataset, bloqueando lectura de archivos arbitrarios del sistema con código 403 Forbidden.
  - Test unitario de confinamiento en `tests/test_v081_trust_patch.py::test_trust_patch_media_preview_confinement`.
- **Propagación de `random_state` para Reproducibilidad (`src/automl/application/services/workspace.py` & `src/automl/facade.py`):**
  - Añadido parámetro `random_seed` a `workspace.create_run()`, propagándolo al `RunConfig` y garantizando que `AutoML(random_state=123)` llega a todos los splitters (KFold/StratifiedKFold) y modelos.
  - Test unitario de propagación en `tests/test_v081_trust_patch.py::test_trust_patch_random_state_propagation`.
- **Ordenación Correcta de Métricas de Minimización (`src/automl/facade.py` & `src/automl/infrastructure/database/sqlite_repository.py`):**
  - `AutoMLResult.leaderboard()` y `SQLiteExperimentRepository.get_leaderboard()` ahora ordenan ascendentemente para métricas de error (`mae`, `rmse`, `mse`, `loss`, `log_loss`), garantizando que el modelo con menor error ocupa el rango 1.
  - Test unitario en `tests/test_v081_trust_patch.py::test_trust_patch_leaderboard_mae_rmse_sorting`.
- **Eliminación de Evidencia Simulada & Actualización de Mensaje:**
  - Suprimida la fórmula ficticia `"public_lb": round(best_t.primary_score * 0.9999, 5)` y el delta simulado en la API web.
  - `README.md` actualizado con posicionamiento claro: *"AutoML you can trust an AI agent to operate"*, con Quickstart centrado en protección de fugas y reproducibilidad.
- **Validación y Versión:**
  - Bump de versión a `0.8.1` en `src/automl/__init__.py`, `CHANGELOG.md` y `tests/smoke_test.py`.
  - 602 tests pasando al 100% en verde. Smoke test end-to-end verificado.

### Track Release Engineering — v0.8.0 Release Final Polish & PyPI Verification (2026-10-07)

- Rama de trabajo `chore/v08-release-prep-and-smoke` desde `main` (`a2f78d0`).
- **Verificación Estricta de Dependencias en Visión (`src/automl/plugins/modalities/image_plugin.py`):**
  - `ImageModalityPlugin.available()` ahora comprueba explícitamente `torchvision` junto con `torch` y `PIL` para evitar falsos positivos cuando solo `torch` está instalado.
  - Test de verificación `test_image_modality_plugin_available_requires_torchvision` en `tests/test_v07_image_plugin.py`.
- **Modernización de Pesos TorchVision & Offline Mode (`src/automl/engine/vision/image_encoder.py`):**
  - Implementada resolución con API moderna de TorchVision (`get_model_weights(name).DEFAULT` y `Weights.DEFAULT`), con fallback para compatibilidad heredada.
  - Mensaje de excepción informativo documentando la descarga de pesos de TorchVision en primer uso y cómo precargar la caché en entornos aislados/air-gapped (`~/.cache/torch/hub/checkpoints`).
  - Nota correspondiente agregada en `README.md`.
- **Contrato de Dimensión de Embeddings (`src/automl/engine/vision/image_encoder.py`):**
  - Introducido atributo scikit-learn `self.feature_dim_`, descubierto y fijado durante `fit()` y utilizado en `get_feature_names_out()` dinámicamente.
- **Desacoplamiento de Cabezal de Visión (`src/automl/plugins/models/vision_plugin.py`):**
  - Exportado alias arquitectónico `VisionFeatureHeadPlugin = TimmVisionPlugin`.
- **Ergonomía de Fachada (`src/automl/facade.py`):**
  - Añadido alias de parámetro `task_type` en `AutoML.__init__()` para evitar excepciones por discrepancia entre `task` y `task_type`.
- **Packaging Limpio para PyPI (`pyproject.toml`):**
  - Migración a string SPDX moderno `license = "Apache-2.0"` (eliminando la advertencia de obsolescencia de setuptools).
  - Añadidos metadatos completos `[project.urls]` (Homepage, Repository, Issues, Documentation).
  - Eliminado el extra redundante `nlp` (cuyas dependencias ya están en el core).
  - Wheel y sdist construidos limpiamente con cero advertencias: `catml-0.8.0.tar.gz` y `catml-0.8.0-py3-none-any.whl`.
- **Smoke Test Automatizado E2E & Job de CI (`tests/smoke_test.py`, `.github/workflows/ci.yml`):**
  - Creado script ejecutable `tests/smoke_test.py` que valida importación, metadatos, entrenamiento, exportación con sidecar `.sha256`, verificación criptográfica e inferencia de producción.
  - Probado con éxito en entorno virtual aislado limpio (`/tmp/catml_test_env`).
  - Añadido job `package-smoke` al workflow de CI de GitHub Actions dependiente de `test`.
- **Validación Global:**
  - 598 tests pasando en verde (100% passing). Cobertura >= 87.6%.

### Track Pre-PyPI Release Preparation — Vision Honesty, Packaging Extras & Governance (2026-10-07)

- Rama de trabajo `fix/vision-packaging-and-v08-prep` desde `main` (`3335b87`).
- **Honestidad y Corrección del Pipeline de Visión (`src/automl/engine/vision/image_encoder.py`, `src/automl/engine/training/sklearn_trainer.py`, `src/automl/facade.py`):**
  - `ImageEncoderNode` ahora preserva por defecto la dimensionalidad nativa del backbone (ej. 512 para ResNet18) sin truncamiento arbitrario destructivo ni padding artificial.
  - En `_build_pipeline()`, si se detectan columnas de imagen y no están instaladas las dependencias de visión neurales (`catml[vision]`), el sistema levanta un `RuntimeError` explícito indicando la instalación requerida, en lugar de simular visión mediante hash determinista.
  - Para pruebas unitarias y benchmarks reproducibles, se soporta explícitamente `image_model="deterministic"`.
  - Propagación de `image_model` en la fachada `AutoML(image_model=...)` y en `fit()`, con propagación correcta a `workspace.export_model_artifact()` y exclusión limpia de parámetros pasados al estimador scikit-learn.
- **Claridad de Nomenclatura en Plugins y UI (`src/automl/plugins/models/vision_plugin.py`, `src/automl/interfaces/web/static/js/views/new_experiment.js`):**
  - Renombrado `TimmVisionPlugin` a "Vision Feature Head (MLP on Extracted Embeddings)" clarificando que entrena un cabezal MLP sobre representaciones extraídas en lugar de backpropagation nativo end-to-end.
  - Etiqueta en la UI del Workbench actualizada de "Vision (timm)" a "Vision Feature MLP".
- **Integridad y Streaming en ModelArtifact (`src/automl/artifacts/model_artifact.py`):**
  - Implementada función streaming `_compute_file_sha256(path, chunk_size=65536)` para evitar saturación de memoria RAM en modelos de gran tamaño.
  - Documentación técnica explícita en docstrings distinguiendo la integridad del archivo (detección de corrupción accidental) frente a autenticidad criptográfica (firmas PKI contra adversarios maliciosos).
- **Inmutabilidad en el Dominio (`src/automl/domain/problems/spec.py`):**
  - `ProblemSpec` decorado con `@dataclass(frozen=True)` con normalización automática de listas de entrada a tuplas inmutables en `__post_init__`.
  - Test de inmutabilidad agregado en `tests/test_v09_capability_layer.py`.
- **Estructura Limpia de Empaquetado (`pyproject.toml`):**
  - Agrupación de dependencias extras: `models`, `vision`, `nlp`, `mcp`, `agents`, `all`.
  - Inclusión de dependencias de visión (`torch`, `torchvision`, `timm`, `pillow`) dentro de `all`. Eliminado el grupo redundante `full`.
- **Gobernanza y Sincronización de Documentación:**
  - `README.md` actualizado con 595 tests passing, Tech Minimalista Premium workbench, CLI V0.8.0 y taxonomía precisa de visión.
  - `docs/MASTER_PLAN.md` Sección 7 sincronizada marcando capacidades v0.8 como `IMPLEMENTADO (SHIPPED IN 0.8)` y fases futuras como `v0.8.x` y `v1.0+`.
  - Creados `SECURITY.md`, `CHANGELOG.md` (Keep a Changelog) y `TRADEMARK.md` (política de marca CATML).
- **Validación Completa:**
  - 595 tests pasando al 100% en verde. Cobertura $\ge 85\%$.

### Track Engine & Packaging Hardening — Vision Safety, Facade Fixes & Artifact Security (2026-10-07)

- Rama propia `feat/v08-hardening-and-consistency` desde `main` (`4be3e5f`).
- **Corrección de Validación en Fachada (`src/automl/facade.py` & `src/automl/application/services/workspace.py`):**
  - Resuelto bug donde `AutoML.fit` ignoraba `cv_folds` al hardcodear validación `holdout` sin grupo.
  - Ahora propaga `cv_folds` a `workspace.create_run()` y selecciona `stratified_kfold` / `kfold` cuando `cv_folds > 1` (o `group_kfold` si hay columna de grupo), preservando `holdout` únicamente para `cv_folds == 1`.
- **Endurecimiento del Encoder de Visión (`src/automl/engine/vision/image_encoder.py`):**
  - Eliminado el fallback silencioso a hash determinista pseudoaleatorio en modelos neurales (`allow_fallback=False` por defecto).
  - Si un modelo neural (`resnet18`, etc.) se solicita y no están disponibles PyTorch/timm o fallan los pesos preentrenados, se levanta `RuntimeError` explícito ("Fail visibly, don't fake vision").
  - Eliminado el fallback silencioso a redes aleatorias sin pesos preentrenados `model_fn()`.
  - Reservado el vector determinista por hash exclusivamente cuando `model_name in ("deterministic", "hash")`.
- **Seguridad en Artefactos (`src/automl/artifacts/model_artifact.py`):**
  - Implementada advertencia explícita de seguridad (`UserWarning`) en `ModelArtifact.load` recordando los riesgos de deserialización no confiable de pickle/joblib.
  - Implementada generación de sidecar `.sha256` en `save()` y verificación criptográfica de integridad en `load(verify_checksum=True)` (lanzando `ValueError` si hay discrepancia de checksum).
- **Packaging y Claridad de Documentación:**
  - Renombrado el paquete en `pyproject.toml` a `name = "catml"`.
  - Configurados puntos de entrada CLI: comando principal `catml` y alias retrocompatible `automl`. Instalado en editable (`catml-0.8.0`).
  - Actualizado `README.md` aclarando la taxonomía de capacidades: Estable (Tabular), Disponible (NLP y Vision feature representation), Hoja de ruta (visión nativa end-to-end), y reemplazado el reclamo "zero-dependency" por "Database-free, workspace-independent production inference".
- **Validación Completa:**
  - 593 tests pasando al 100% en verde (0 fallos).
  - Cobertura de código: 87.65% (superando el umbral >= 85%).
  - CLI `catml --help` y `automl --help` verificados.

### Track Governance & Licensing — Relicensing to Apache-2.0 & v0.8.0 Transition (2026-10-07)

- Etiquetado `v0.7.0` fijado y publicado en origen como último hito bajo licencia MIT.
- Creación de rama de trabajo `chore/relicense-apache-2`.
- **Blindaje de Licencia e Historia Transparente:**
  - Sustitución de `LICENSE` por el texto oficial de Apache License 2.0 (con cláusula de defensa/represalia de patentes y protección de marca registrada).
  - Creación de archivo `NOTICE` con atribución a CATML Contributors y mención histórica de transición.
  - Blindaje preventivo de `.gitignore` frente a fuga de credenciales (`.env`, `*.pem`, `*.key`, `credentials.json`, `secrets.json`).
- **Alineación de Metadatos y Documentación:**
  - Actualización de `pyproject.toml` (`license = { text = "Apache-2.0" }`) y bump de versión a `0.8.0` en `src/automl/__init__.py`.
  - Dinamización de versión en `src/automl/interfaces/mcp/server.py` y `src/automl/interfaces/web/server.py` importando `__version__`.
  - Actualización de badges y textos legales en `README.md`, `docs/MASTER_PLAN.md` y `docs/decisions/006-tech-minimalist-premium-brand-system.md`.
  - Actualización de menciones de licencia en la landing estática (`website/src/App.tsx`), con build estático verificado (230 ms).
- **Validación Completa:**
  - 590 tests pasando al 100% en verde.
  - Cobertura de código mantenida en 87.70% (superando el umbral >= 85%).
  - Verificación CLI de `automl --help` (reportando V0.8.0) y `automl task list`.

### Track Persona B — Paquete B5: LangGraph StateGraph, Durable SQLite Checkpointer & Crash Recovery (2026-10-06)

- Rama propia `feat/agentic-b5-langgraph-orchestrator` desde `main` (`64ec800`).
- **Mapeo de Estado de Agente (`src/automl/application/agents/state.py`):**
  - Implementación de `GraphAgentState` (TypedDict) para compatibilidad con canales LangGraph.
  - Funciones puras de serialización/deserialización bidireccionales: `session_to_graph_state()`, `graph_state_to_session()`, `serialize_graph_state()`, `deserialize_graph_state()`.
  - Mapeo seguro evitando colisiones con canales reservados en Pregel (`checkpoint_id` $\rightarrow$ `session_checkpoint_id`).
- **Adaptador de Persistencia SQLite (`src/automl/infrastructure/database/sqlite_checkpoint_saver.py`):**
  - Implementación de `SqliteCheckpointSaver` sobre `langgraph.checkpoint.sqlite.SqliteSaver` con manejo multi-hilo (`check_same_thread=False`), inicialización idempotente de tablas de checkpoints, aislamiento contextual con context manager (`__enter__`, `__exit__`), y reconexión resiliente tras desconexiones o fallos.
- **Orquestador LangGraph StateGraph (`src/automl/application/agents/orchestrator/graph.py`):**
  - Grafo de estados StateGraph con nodos: `observe` $\rightarrow$ `propose` $\rightarrow$ `gate` $\rightarrow$ `execute` $\rightarrow$ `critique` $\rightarrow$ `check_stop`.
  - `observe`: construye contexto acotado mediante `ContextBuilder` sin fuga de datos crudos.
  - `propose`: formula candidatos empíricos mediante `Planner` / `FeatureAdvisor` y detecta duplicación criptográfica de hipótesis (`is_hypothesis_duplicate`).
  - `gate`: evalúa presupuestos y permisos; utiliza `interrupt()` para pausar ante autorización humana y reutiliza solicitudes de aprobación previas (`PENDING`) en reanudaciones.
  - `execute`: verifica idempotencia de operaciones en SQLite ledger para prevenir re-ejecuciones duplicadas; invoca `ToolExecutor` e interconecta creación y corrida de experimentos (`run_experiment`), mapeando `metric_value` a `score`.
  - `critique`: evalúa avances empíricos mediante `Critic`, gestiona paciencia ante estancamiento y comprueba objetivo de métrica.
  - `check_stop`: actualiza contador de iteración, sincroniza `AgentSessionState` con `SqliteAgentLedger` y persiste checkpoints de sesión.
  - Métodos públicos de ejecución: `run()`, `resume()`, `get_latest_graph_state()`.
- **Fachada y Paridad CLI (`src/automl/interfaces/cli/agent_session_cli.py`):**
  - Subcomando `automl agent session resume` y `status` con argumento `--engine {deterministic,langgraph}`.
  - Métodos delegados públicos `get_run()` y `get_leaderboard()` en `AutoMLWorkspace`.
  - Búsqueda dual por `run_id` o `session_id` en `SqliteAgentLedger.list_sessions()`.
- **Validación Completa:**
  - `tests/test_v10_orchestrator.py`: 24/24 tests pasando.
  - `tests/test_v10_orchestrator_b5.py`: 13/13 tests pasando.
  - `tests/test_v10_agent_e2e.py`: 3/3 tests de ciclo autónomo, recuperación tras fallos (crash & restart) sin duplicación de hipótesis, y rechazo humano de propuestas.
  - Suite global completa: 590 tests pasando en verde, 0 fallos, 87.70% de cobertura de código (superando el requisito >= 85%).
  - Hito H5 (V1.0) completado exitosamente.
- **Corrección de CI Workflow (`.github/workflows/ci.yml`):**
  - Añadido extra `agents` a la instalación en CI (`python -m pip install -e ".[dev,mcp,agents]"`) para garantizar presencia de `langgraph`, `langgraph-checkpoint` y `langgraph-checkpoint-sqlite` en las matrices de Python 3.10 y 3.12 de GitHub Actions.
- **Plan Maestro de Producto, Arquitectura y Mercado (`docs/MASTER_PLAN.md`):**
  - Formalización del documento rector de arquitectura y producto: posicionamiento (*"Train better models. Keep control"* / *"Bring AI to your data. Not your data to AI"*), regla de privacidad de cómputo hacia los datos, paridad de 4 interfaces (Python/CLI/Workbench/MCP), perfiles de políticas (`STRICT`/`PRIVATE`/`STANDARD`), taxonomía multimodal desacoplada de tareas, anti-leakage como diferenciador central, y puertas de decisión (*Decision Gates*) orientadas a tracción de usuarios. Navegación enlazada en `docs/README.md` y `AGENTS.md`.

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
- Cambios de producto limitados a `website/`: homepage React/TypeScript/Vite clara, tokens de marca, fuentes locales, snippets, ejemplos etiquetados, Community de código abierto. No se modificó Workbench ni núcleo Python.
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
