# Propuesta: blending de predicciones out-of-fold

Estado: pendiente. Consultar [TASKS.md](../../../TASKS.md). La opción `automl predict --folds` aún no existe.

## Objetivo

Comparar blending de modelos entrenados con cinco folds estratificados frente a un baseline tabular, manteniendo separación entre evaluación y ajuste de pesos. Una puntuación Kaggle superior a 0.945 es una aspiración, no un criterio garantizado ni una métrica ya medida.

## Protocolo de evaluación

1. Registrar hash y esquema del dataset, objetivo, backend concreto y versiones. Para clasificación binaria, usar ROC-AUC.
2. Fijar seed 42 y cinco folds estratificados idénticos para los modelos. Ajustar imputación, encoding y selección de features exclusivamente con las filas de entrenamiento de cada fold.
3. Definir antes de ejecutar el baseline: mejor modelo individual bajo los mismos splits y presupuesto, guardando su configuración y predicciones OOF. El baseline no puede usar métricas ilustrativas de la UI.
4. Generar una predicción OOF por fila con un modelo que no la haya visto. Promediar las predicciones del test entre los cinco modelos de folds y alinear con la plantilla de submission.
5. Evaluar inicialmente pesos iguales. Si se ajustan pesos usando OOF, medir el resultado final en un holdout independiente o validación anidada; no usar el mismo OOF para ajustar y declarar mejora imparcial.
6. Registrar ROC-AUC, dispersión entre folds, delta contra baseline, tiempo y memoria. Una puntuación pública de Kaggle debe quedar identificada por separado.

## Presupuesto y aceptación

- Primera comparación: dos modelos individuales y su combinación, cinco folds por modelo, un trial por modelo/fold; sin HPO adicional. Registrar tiempo real y detenerse con el límite de recursos configurado antes de comenzar.
- Pruebas de integridad: cobertura OOF exactamente una vez por fila, ausencia de fuga entre folds, orden de IDs, probabilidades válidas, reproducibilidad y persistencia de resultados.
- Promoción: evidencia de mejora en evaluación independiente bajo el mismo presupuesto. Si no mejora, conservar el candidato y registrar el resultado sin promoverlo.
- Integración futura: comandos y queries de aplicación, luego CLI y Workbench. Mantener compatibilidad con la predicción actual.
