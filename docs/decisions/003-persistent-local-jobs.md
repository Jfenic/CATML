# ADR 003 — Cola persistente local y worker único

Fecha: 2026-09-30. Estado: aceptada para la implementación inicial.

## Contexto

El Workbench ejecutaba entrenamientos dentro de peticiones HTTP largas. Se necesitaban IDs duraderos, progreso medido, controles cooperativos y recuperación al reiniciar, respetando CQRS y la paridad de interfaces.

## Decisión

Guardar jobs JSON en SQLite mediante `JobRepositoryPort`. Separar validación/control (`JobService`), ejecución de operaciones existentes (`JobExecutor`) y almacenamiento/lifecycle del worker en infraestructura. Registrar commands que retornan IDs y queries que retornan snapshots en bootstrap.

Usar un worker local por workspace, con claim transaccional y lease de proceso `flock`. El dashboard gestiona su lifecycle; la CLI también puede iniciarlo. Las consultas nunca inician ejecución ni recuperación. Los estados activos abandonados requieren reintento explícito.

Progresar por modelos y pares modelo/fold completados. Aplicar pausa/cancelación entre fits. Reanudar experimentos desde checkpoints; reiniciar OOF parcial hasta disponer de persistencia verificable de folds.

## Consecuencias

No se necesitan servicios externos ni nuevas dependencias. Los entrenamientos iniciales se serializan y el worker requiere Linux/macOS. No hay ejecución exactamente una vez ni cancelación forzada de fits. Los archivos y trials ya escritos se conservan para revisión.

HPO, backends distribuidos, serialización de folds y políticas de reintentos automáticos requieren decisiones posteriores. Los contratos y límites se detallan en la [especificación](../features/persistent-jobs/spec.md).
