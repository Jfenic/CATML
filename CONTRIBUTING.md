# Guía de Contribución y Trabajo en Paralelo — CATML

> Esta guía describe el flujo de trabajo colaborativo, la estrategia de Git para evitar conflictos de ramas y la especificación detallada de tareas listas para ser desarrolladas por nuevos colaboradores.

---

## 1. Metodología de Trabajo y Estrategia Git

CATML está construido bajo una **Arquitectura Hexagonal (Puertos y Adaptadores) con CQRS**. Esto significa que los componentes están altamente desacoplados y el trabajo en paralelo no debe generar colisiones en Git si se sigue esta metodología.

### 1.1 Principio de "Cero Conflictos"
- **Crear archivos nuevos en lugar de editar archivos comunes:** 
  Cada nueva funcionalidad o plugin debe implementarse en un archivo dedicado dentro de su capa correspondiente (ej. `src/automl/plugins/models/ensemble.py` en vez de inflar un archivo preexistente).
- **Un archivo de pruebas por tarea:** 
  **Nunca** agregues tus tests al final de archivos de prueba existentes como `test_v06_plugins.py`. Crea un archivo nuevo para tu tarea (ej. `tests/test_ensemble_plugin.py`).
- **Puntos de integración al final:** 
  En CATML solo hay 3 archivos donde todo converge:
  1. `src/automl/application/bootstrap.py` (registro del handler en el bus).
  2. `src/automl/application/services/workspace.py` (método fachada).
  3. `src/automl/interfaces/cli/main.py` (comando de terminal).
  
  *Desarrolla y valida primero tu clase o función con su test unitario directo. Solo cuando funcione al 100%, añade la línea de registro en `bootstrap.py` o `workspace.py` en tu último commit.*

### 1.2 Flujo de Git Paso a Paso

1. **Crear una rama específica y de vida corta desde `main`:**
   ```bash
   git checkout main
   git pull origin main
   git checkout -b feat/nombre-de-la-tarea
   ```
2. **Configurar el entorno virtual y validar el estado inicial:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   pytest   # Deben pasar los 101 tests existentes
   ```
3. **Desarrollar la funcionalidad y los tests correspondientes.**
4. **Sincronizar siempre con `main` mediante `rebase` (evitar merge commits):**
   ```bash
   git fetch origin
   git rebase origin/main
   ```
5. **Verificar que la suite completa sigue en verde:**
   ```bash
   .venv/bin/pytest
   ```
6. **Subir la rama y abrir Pull Request (PR):**
   ```bash
   git push origin feat/nombre-de-la-tarea
   ```

---

## 2. Especificación de Tareas del Backlog

A continuación se detalla el estado de las tareas prioritarias del backlog identificadas tras el benchmark de Kaggle:

---

### ✅ Tarea A [COMPLETADA EN MAIN]: Heurística de Alta Cardinalidad e Identificadores en `DatasetProfiler`

* **Estado:** Integrada en `main` (PR #3). Verificada con 12 tests en [`tests/test_profiler_cardinality.py`](file:///home/fenic/top_project/CATML/tests/test_profiler_cardinality.py).
* **Objetivo completado:** 
  Evitar que columnas como `id`, `CustomerId`, `Surname` o hashes de alta cardinalidad sean tratadas como variables numéricas/categóricas predictivas convencionales.
* **Archivos implementados:**
  - [`src/automl/engine/profiling/dataset_profiler.py`](file:///home/fenic/top_project/CATML/src/automl/engine/profiling/dataset_profiler.py)
  - [`src/automl/domain/datasets/profile.py`](file:///home/fenic/top_project/CATML/src/automl/domain/datasets/profile.py)
  - [`src/automl/infrastructure/database/sqlite_repository.py`](file:///home/fenic/top_project/CATML/src/automl/infrastructure/database/sqlite_repository.py) (migración y persistencia de `semantic_type`)

---

### ✅ Tarea B [COMPLETADA EN MAIN]: Plugin de Ensamble y Blending (`VotingEnsemblePlugin`)

* **Estado:** Integrada en `main` (PR #2). Verificada con 14 tests en [`tests/test_ensemble_plugin.py`](file:///home/fenic/top_project/CATML/tests/test_ensemble_plugin.py).
* **Objetivo completado:** 
  Combinación de predicciones de los mejores $K$ modelos mediante promediado de probabilidades (soft voting) y media ponderada.
* **Archivos implementados:**
  - [`src/automl/plugins/models/ensemble.py`](file:///home/fenic/top_project/CATML/src/automl/plugins/models/ensemble.py)
  - [`src/automl/engine/ensemble/blender.py`](file:///home/fenic/top_project/CATML/src/automl/engine/ensemble/blender.py) y [`voting.py`](file:///home/fenic/top_project/CATML/src/automl/engine/ensemble/voting.py)

---

### ✅ Tarea C [COMPLETADA EN MAIN]: Mapeo Automático de Plantilla de Sumisión (`--template sample_submission.csv`)

* **Estado:** Integrada en `main`. Verificada con 5 tests en [`tests/test_submission_template.py`](file:///home/fenic/top_project/CATML/tests/test_submission_template.py).
* **Objetivo completado:** 
  Permitir alinear exactamente el orden de los IDs y columnas según un archivo CSV de ejemplo (`sample_submission.csv`), garantizando compatibilidad 100% con formatos de sumisión de Kaggle.
* **Archivos implementados:**
  - [`src/automl/application/commands/workspace_commands.py`](file:///home/fenic/top_project/CATML/src/automl/application/commands/workspace_commands.py)
  - [`src/automl/application/queries/workspace_queries.py`](file:///home/fenic/top_project/CATML/src/automl/application/queries/workspace_queries.py)
  - [`src/automl/application/services/workspace.py`](file:///home/fenic/top_project/CATML/src/automl/application/services/workspace.py)
  - [`src/automl/interfaces/cli/main.py`](file:///home/fenic/top_project/CATML/src/automl/interfaces/cli/main.py) (añadido `--template` al subcomando `automl predict`)

---

### 📌 Tarea D [DISPONIBLE PARA DESARROLLO]: Generación Automática de Variables de Interacción

* **Objetivo:** 
  Proponer features generadas automáticamente a partir de pares de columnas numéricas (ratios $A/B$, productos $A \times B$) o target encoding para variables categóricas de cardinalidad media.
* **Comportamiento esperado:**
  - Crear un generador en `src/automl/engine/features/generation/` respetando `FeatureSelectorPort` o `FeatureSet`.
  - No mutar el dataset original; registrar las nuevas features propuestas como candidatos para validación por hipótesis ("Propose ≠ Accept").
  - Crear archivo de test: `tests/test_feature_interactions.py`
* **Comando de verificación:**
  ```bash
  .venv/bin/pytest tests/test_feature_interactions.py
  ```

---

### 📌 Tarea V0.7-A [DISPONIBLE PARA DEV 1]: Motor de PipelineGraph (DAG) y FeatureFusionNode

* **Rama Git sugerida:** `feat/v07-pipeline-graph-engine`
* **Objetivo:** 
  Implementar el validador del grafo (`GraphValidator`) y el nodo de fusión vectorial (`FeatureFusionNode`), permitiendo a CATML orquestar pipelines acíclicos no lineales y combinar matrices tabulares con matrices densas de embeddings.
* **Comportamiento esperado:**
  - `GraphValidator`:
    1. Detectar ciclos directos o recursivos en `PipelineGraph` y lanzar `ValueError`.
    2. Determinar el orden topológico de ejecución de los nodos (`get_execution_order()`).
    3. Validar compatibilidad modal entre extremos de cada arista conectada (`output_modality == input_modality`).
  - `FeatureFusionNode`:
    1. Recibir matriz de features tabulares (`pd.DataFrame`) y matriz de embeddings numéricos (`np.ndarray` o `pd.DataFrame`).
    2. Concatenar horizontalmente asegurando alineación exacta de filas e índices.
* **Archivos a crear/modificar:**
  - Crear: `src/automl/engine/pipeline/graph_validator.py`
  - Crear: `src/automl/engine/pipeline/fusion.py`
  - Crear archivo de test: `tests/test_v07_pipeline_graph.py`
* **Archivos que NO debes tocar:**
  - `src/automl/plugins/*`
  - `src/automl/interfaces/cli/*`
* **Comando de verificación:**
  ```bash
  .venv/bin/pytest tests/test_v07_pipeline_graph.py
  ```

---

### 📌 Tarea V0.7-B [DISPONIBLE PARA DEV 2]: ImageModalityPlugin y Extracción de Embeddings

* **Rama Git sugerida:** `feat/v07-image-plugin`
* **Objetivo:** 
  Implementar el plugin de modalidad de imagen y el extractor de embeddings densos (`ImageEncoderNode`), habilitando la ingestión de columnas de imágenes en CATML.
* **Comportamiento esperado:**
  - `ImageModalityPlugin`:
    1. Implementar `PluginPort` y `ModalityPluginPort`.
    2. Declarar capacidades con `Modality.IMAGE`.
    3. Validar rutas a archivos de imagen y formatos admitidos (`.png`, `.jpg`, `.jpeg`).
  - `ImageEncoderNode`:
    1. Transformar un conjunto de rutas de imágenes a un array flotante 2D de embeddings $(N, D)$.
    2. Disponer de un encoder ligero o fallback determinista para asegurar que las pruebas unitarias pasen sin requerir descargas pesadas de pesos en CI.
* **Archivos a crear/modificar:**
  - Crear: `src/automl/plugins/modalities/image_plugin.py`
  - Crear: `src/automl/engine/vision/image_encoder.py`
  - Crear archivo de test: `tests/test_v07_image_plugin.py`
* **Archivos que NO debes tocar:**
  - `src/automl/engine/pipeline/graph_validator.py`
  - `src/automl/engine/pipeline/fusion.py`
* **Comando de verificación:**
  ```bash
  .venv/bin/pytest tests/test_v07_image_plugin.py
  ```

---

## 3. Reglas de Arquitectura Innegociables

Antes de enviar cualquier cambio, verifica estas 3 reglas fundamentales:

1. **El Dominio es Sagrado:**
   - La carpeta `src/automl/domain/` es Python puro.
   - **NUNCA** importes `scikit-learn`, `optuna`, `pandas`, `sqlite3` ni librerías de ML en `domain/`.
2. **Separación CQRS:**
   - Los **Commands** alteran estado y devuelven identificadores o `None`. Nunca retornan estructuras complejas de consulta.
   - Las **Queries** son operaciones de solo lectura que retornan DTOs o primitivas sin efectos secundarios.
3. **Paridad de Interfaces:**
   - Todo lo que el usuario puede hacer desde la CLI debe ejecutarse despachando Commands o Queries en los buses correspondientes. No invoques modelos o funciones del motor directamente en la interfaz.

---

## 4. Checklist para Abrir Pull Request

- [ ] Has creado tu propia rama desde `main` (`git checkout -b feat/...`).
- [ ] Tu funcionalidad tiene un archivo de test dedicado en `tests/`.
- [ ] Has corrido toda la suite y los 138+ tests pasan:
  ```bash
  .venv/bin/pytest
  ```
- [ ] Has sincronizado con `main` (`git pull origin main` o `git rebase origin/main`) asegurando historial limpio.
- [ ] No hay imports indebidos en `domain/`.
- [ ] Has actualizado `TASKS.md` indicando la tarea completada.
- [ ] Has revisado si hay avisos abiertos en la pizarra (`gh issue list --label blackboard`).
