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
   pytest   # Deben pasar los 64 tests existentes
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

## 2. Especificación de Tareas Listas para Desarrollo

A continuación se detallan las tareas prioritarias del backlog identificadas tras el benchmark de Kaggle. Puedes elegir cualquiera de ellas para trabajar de forma independiente.

---

### 📌 Tarea A: Heurística de Alta Cardinalidad e Identificadores en `DatasetProfiler`

* **Objetivo:** 
  Evitar que columnas como `id`, `CustomerId`, `Surname` o identificadores hash con valores únicos masivos sean tratadas como variables categóricas o numéricas normales en el pipeline de entrenamiento.
* **Comportamiento esperado:**
  - En `DatasetProfiler`, al analizar columnas tipo texto/objeto o enteros consecutivos, calcular el ratio de unicidad:
    $$\text{unicidad} = \frac{\text{valores únicos}}{\text{total de filas}}$$
  - Si una columna tiene ratio $> 0.70$ (y más de 50 filas) o su nombre termina en `_id`, `id` o `guid`, marcarla con rol o tag `IDENTIFIER` o descartarla de las features candidatas recomendadas.
* **Archivos a modificar/crear:**
  - Modificar: [`src/automl/engine/profiling/profiler.py`](file:///home/fenic/top_project/CATML/src/automl/engine/profiling/profiler.py)
  - Modificar: [`src/automl/domain/datasets/schema.py`](file:///home/fenic/top_project/CATML/src/automl/domain/datasets/schema.py) (añadir campo opcional o tipo si aplica)
  - Crear archivo de test: `tests/test_profiler_cardinality.py`
* **Archivos que NO debes tocar:**
  - `src/automl/plugins/*`
  - `src/automl/interfaces/cli/*`
* **Comando de verificación:**
  ```bash
  .venv/bin/pytest tests/test_profiler_cardinality.py
  ```

---

### 📌 Tarea B: Plugin de Ensamble y Blending (`VotingEnsemblePlugin`)

* **Objetivo:** 
  Permitir combinar las predicciones de los mejores $K$ modelos entrenados en un `Run` para mejorar la métrica final (estrategia clásica de Kaggle / ensamblado por votación blanda / media de probabilidades).
* **Comportamiento esperado:**
  - Implementar un plugin que cumpla con el contrato `ModelPluginPort` definido en [`src/automl/domain/ports.py`](file:///home/fenic/top_project/CATML/src/automl/domain/ports.py).
  - Recibir una lista de modelos base entrenados o sus predicciones (`predict_proba` para clasificación, `predict` para regresión) y promediar los resultados (soft voting).
  - Si se especifican pesos `weights`, realizar media ponderada.
  - Registrar las capacidades del plugin (`PluginCapability.TABULAR`, `TaskType.BINARY_CLASSIFICATION`, `TaskType.REGRESSION`).
* **Archivos a crear/modificar:**
  - Crear: `src/automl/plugins/models/ensemble.py`
  - Registrar en: `src/automl/plugins/models/sklearn_models.py` (o en la factoría de plugins correspondiente)
  - Crear archivo de test: `tests/test_ensemble_plugin.py`
* **Archivos que NO debes tocar:**
  - `src/automl/engine/profiling/*`
  - `src/automl/engine/planning/*`
* **Comando de verificación:**
  ```bash
  .venv/bin/pytest tests/test_ensemble_plugin.py
  ```

---

### 📌 Tarea C: Mapeo Automático de Plantilla de Sumisión (`--template sample_submission.csv`)

* **Objetivo:** 
  Facilitar la entrega directa a Kaggle u otros sistemas externos permitiendo que el comando de predicción alinee exactamente el orden de los IDs y columnas según un archivo CSV de ejemplo (`sample_submission.csv`).
* **Comportamiento esperado:**
  - Extender `GenerateSubmissionCommand` y `PredictDatasetQuery` para aceptar un argumento opcional `template_path: str | None = None`.
  - Si se proporciona `template_path`:
    1. Leer el CSV plantilla.
    2. Asegurarse de que el archivo generado tenga exactamente las mismas columnas y en el mismo orden de IDs que la plantilla.
    3. Rellenar la columna objetivo con las probabilidades o valores predichos correspondientes al ID de cada fila.
* **Archivos a modificar:**
  - [`src/automl/application/commands/workspace_commands.py`](file:///home/fenic/top_project/CATML/src/automl/application/commands/workspace_commands.py)
  - [`src/automl/application/services/workspace.py`](file:///home/fenic/top_project/CATML/src/automl/application/services/workspace.py)
  - [`src/automl/interfaces/cli/main.py`](file:///home/fenic/top_project/CATML/src/automl/interfaces/cli/main.py) (añadir `--template` al subcomando `automl predict`)
  - Crear archivo de test: `tests/test_submission_template.py`
* **Archivos que NO debes tocar:**
  - `src/automl/domain/*`
  - `src/automl/plugins/*`
* **Comando de verificación:**
  ```bash
  .venv/bin/pytest tests/test_submission_template.py
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
- [ ] Has corrido toda la suite y los 64+ tests pasan:
  ```bash
  .venv/bin/pytest
  ```
- [ ] Has hecho `git rebase origin/main` para asegurar que tu rama está al día y el historial es limpio.
- [ ] No hay imports indebidos en `domain/`.
- [ ] Has actualizado `TASKS.md` indicando la tarea completada.
