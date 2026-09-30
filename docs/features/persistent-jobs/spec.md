# Cola local de trabajos persistentes

Estado: implementación inicial. La cola usa la base SQLite del workspace y un worker local. No añade servicios ni dependencias de ejecución.

## Problema y alcance

Los entrenamientos largos del Workbench ocupaban una petición HTTP hasta terminar. Al recargar la página se perdía su seguimiento y el progreso de creación de experimentos era simulado.

Ahora el frontend envía un trabajo, recibe su ID mediante HTTP 202 y consulta su estado. El panel «Trabajos» recupera actividad al recargar y permite pausar, continuar, cancelar y reintentar. Los experimentos muestran modelos terminados; OOF muestra pares modelo/fold terminados. Una submission normal tiene un único paso de entrenamiento y exportación.

Operaciones iniciales: `experiment`, `oof`, `submission`. HPO, benchmarks, agentes, entrenamiento distribuido y serialización de modelos de folds quedan pendientes. Los endpoints síncronos previos siguen disponibles para compatibilidad; el frontend utiliza `/api/jobs` para las operaciones anteriores.

## Contratos y estructura

- Dominio: [Job y JobStatus](../../../src/automl/domain/jobs/job.py), sin dependencias externas. Las solicitudes de control confirmadas antes de finalizar tienen prioridad sobre el resultado del worker.
- Puerto: `JobRepositoryPort` en [ports.py](../../../src/automl/domain/ports.py). Claims y cambios de estado deben ser atómicos.
- Aplicación: [JobService](../../../src/automl/application/services/jobs.py) valida operaciones y pertenencia al run; [JobExecutor](../../../src/automl/application/services/job_executor.py) reutiliza commands, queries y workspace existentes. Los commands de jobs devuelven IDs y las queries devuelven snapshots.
- Infraestructura: [SQLiteJobRepository](../../../src/automl/infrastructure/database/sqlite_jobs.py) y [JobWorker](../../../src/automl/infrastructure/jobs/worker.py). `bootstrap.py` registra los nuevos handlers; las consultas no arrancan workers ni recuperan estados.
- Interfaces: CLI `automl job`, API HTTP y [panel del Workbench](../../../src/automl/interfaces/web/static/js/views/jobs.js).

Cada trabajo conserva operación, run, argumentos JSON, clave de idempotencia, estado, intento, progreso, mensaje, error, resultado y timestamps UTC. La cola se atiende en orden de creación; los trabajos pausados o fallidos no bloquean los siguientes.

## Estados y controles

| Acción | Estados admitidos | Resultado |
| --- | --- | --- |
| Submit | Solicitud válida | `queued`; mismo ID para la misma clave y argumentos |
| Pause | `queued`, `running` | `paused` o `pause_requested` hasta alcanzar un límite de entrenamiento |
| Cancel | `queued`, `running`, `pause_requested`, `paused`, `failed`, `interrupted` | `cancelled` o `cancel_requested` |
| Resume | `paused` | `queued`, conserva progreso de experimentos |
| Retry | `failed`, `interrupted` | `queued`, siguiente intento al ser reclamado |
| Recuperación | Estado activo abandonado | `interrupted`; cancelación pendiente pasa a `cancelled` |

Una clave ya utilizada con argumentos distintos produce un error. La clave identifica una solicitud, no una caché global de modelos. Un nuevo entrenamiento deliberado necesita otra clave. El cliente debe reutilizar la misma clave si vuelve a enviar una solicitud cuya respuesta se perdió.

Los controles son cooperativos: no interrumpen un `fit` en ejecución. Los checkpoints de experimentos se guardan después de cada resultado de modelo, antes de comprobar controles; la reanudación continúa desde el último índice guardado. Las métricas de modelos fallidos también se conservan; un experimento sin resultados satisfactorios deja el trabajo fallido. Al reintentar un experimento fallido se vuelven a entrenar sus modelos pendientes/fallidos y se reutilizan los que ya tienen un resultado satisfactorio.

OOF pausa/cancela entre fits y descarta los resultados parciales. Al continuar vuelve a evaluar todos los folds; los intentos fallidos permanecen auditados. El progreso puede volver a cero al reiniciar OOF. No se promueven candidatos automáticamente.

## Worker y recuperación

`automl ui` inicia un worker y lo detiene al cerrar el servidor. Para usar la cola desde CLI sin Workbench:

```bash
automl job worker --workspace .automl/demo
```

Ejecutar un solo worker por workspace. La implementación inicial requiere Linux/macOS: una lease `flock` impide que dos procesos trabajen sobre la misma cola. La reclamación en SQLite también impide tener dos trabajos activos. La lease se conserva hasta que termina el fit durante el cierre cooperativo; el cierre del dashboard espera hasta dos segundos y el siguiente proceso recupera cualquier trabajo interrumpido.

Después de una caída, los trabajos pendientes permanecen en cola y los activos requieren reintento explícito. No hay reintentos automáticos, temporizador de lease ni recuperación activada por consultas. El worker carga un workspace nuevo por trabajo. Los plugins personalizados deben registrarse en su bootstrap; un registro realizado solamente en otro proceso no se propaga.

La ejecución ofrece recuperación por checkpoints, no una garantía de ejecución exactamente una vez. Una caída entre guardar un trial y guardar su checkpoint puede repetir ese modelo; una caída durante creación/exportación puede dejar un experimento o archivo parcial. Cancelar no revierte resultados ni elimina archivos ya escritos. El resultado persistente del trabajo permite revisar lo terminado.

## Uso CLI y HTTP

Sustituir `RUN_ID` por un run existente:

```bash
automl job submit --workspace .automl/demo --run-id RUN_ID \
  --operation experiment --key experimento-001 \
  --payload '{"name":"baseline","model_ids":["logistic_regression"]}'
automl job list --workspace .automl/demo --run-id RUN_ID
automl job show --workspace .automl/demo --job-id JOB_ID
automl job pause --workspace .automl/demo --job-id JOB_ID
automl job resume --workspace .automl/demo --job-id JOB_ID
automl job cancel --workspace .automl/demo --job-id JOB_ID
automl job retry --workspace .automl/demo --job-id JOB_ID
```

Para ejecutar un experimento existente, el payload contiene solo `experiment_id`. Para crear uno, admite `name`, `model_ids`, `feature_names` e `hypothesis`; si no se indican features, se utilizan columnas no identificadoras distintas del target.

Para `oof` o `submission`, el payload usa los campos del command correspondiente, sin `run_id`: `test_dataset_path`, `output_path`, opciones de plantilla y experimento; OOF añade `folds`, `model_ids` y `max_seconds`.

| Ruta | Contrato |
| --- | --- |
| `POST /api/jobs` | `{operation, run_id, payload, idempotency_key}` → HTTP 202 `{job_id, status_url}` |
| `GET /api/jobs?run_id=...` | Snapshots en orden de creación descendente; filtro opcional |
| `GET /api/jobs/JOB_ID` | Snapshot con progreso, resultado y error |
| `POST /api/jobs/JOB_ID/ACTION` | `pause`, `resume`, `cancel`, `retry`; devuelve ID |

Errores de argumentos/estado: HTTP 400. IDs inexistentes: HTTP 404. No es necesario mantener abierta la petición de submit para completar el trabajo. Los controles de run del Workbench se encaminan a sus trabajos cuando existen trabajos compatibles con la acción; el comportamiento síncrono anterior se mantiene como alternativa.

## Evidencia y aceptación

[tests/test_jobs.py](../../../tests/test_jobs.py) cubre persistencia tras recarga, snapshots sin mutación, idempotencia concurrente, claims atómicos, controles antes/durante fits, prioridad de controles frente a finalización, lease entre workers, cierre durante fit, recuperación sin ejecución automática, checkpoints, errores/reintentos, OOF parcial, exportación y paridad CLI/HTTP.

Las pruebas del adaptador frontend se ejecutan con `node --test tests/js/jobs.test.mjs` (Node 20+). Cubren envío/polling, selección OOF, clave reutilizable y errores persistentes.

La suite completa y cobertura ≥85% son requisitos de integración. La validación fechada se registra en [TASKS.md](../../../TASKS.md) y [.agent/progress.md](../../../.agent/progress.md).
