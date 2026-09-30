# Backends opcionales y reproducibilidad

La instalación base utiliza scikit-learn. LightGBM y XGBoost son dependencias opcionales:

```bash
.venv/bin/python -m pip install lightgbm xgboost
```

| Plugin | Backend si está instalado | Alternativa si falta |
| --- | --- | --- |
| `lightgbm` | `LGBMClassifier` / `LGBMRegressor` | `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` |
| `xgboost` | `XGBClassifier` / `XGBRegressor` | `GradientBoostingClassifier` / `GradientBoostingRegressor` |

La alternativa conserva el flujo de experimentación, pero cambia el algoritmo y solo traduce parte de los hiperparámetros. Comparar resultados requiere identificar el backend concreto.

## Comprobar el estimador

Desde la raíz del repositorio:

```bash
.venv/bin/python - <<'PY'
from automl.plugins.models.gradient_boosting import LightGBMPlugin, XGBoostPlugin

for plugin in (LightGBMPlugin(), XGBoostPlugin()):
    estimator = plugin.build_estimator(task_type="binary_classification")
    print(plugin.plugin_id, "native_available=", plugin.is_available,
          type(estimator).__module__, type(estimator).__name__)
PY
```

Para exigir el backend nativo, registrar `LightGBMPlugin(use_fallback_if_missing=False)` o `XGBoostPlugin(use_fallback_if_missing=False)` durante la composición del workspace. Sin el paquete correspondiente, construir el estimador lanza `ImportError`; el trainer registra el fallo del trial. Esta opción se configura en Python, no mediante un flag de CLI.

## Alcance de reproducibilidad

Los trials registran parámetros y seed; los resultados y eventos se guardan en SQLite. Actualmente el ID del modelo no identifica por sí solo el backend concreto, y no se persiste un manifiesto completo de versiones de librerías, plugin y hashes de datos por trial.

Para una comparación reproducible, conservar el workspace, una copia o hash del dataset, configuración de validación, clase concreta del estimador y versiones del entorno:

```bash
.venv/bin/python -m pip freeze > environment.txt
```

El manifiesto automático por trial permanece en el [backlog](../TASKS.md). Los encoders de imagen también tienen modos distintos: el modo predeterminado es determinista y no acredita calidad equivalente a un encoder preentrenado.
