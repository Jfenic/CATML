# Guía de contribución — CATML

El estado de tareas y el backlog se mantienen en [TASKS.md](TASKS.md). Para elegir trabajo, consultar también los avisos abiertos:

```bash
gh issue list --label blackboard --state open
```

## Preparación y flujo de Git

Desde la raíz del repositorio:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
git status --short
git switch -c feat/nombre-de-la-tarea
```

Conservar cambios existentes y acordar el alcance antes de editar archivos compartidos. Para trabajo simultáneo, definir contratos y responsables de cada módulo antes de dividir tareas. Un contrato común reduce conflictos; no los elimina.

Implementar el cambio, ejecutar sus pruebas y revisar el diff. Sincronizar la rama con la rama base antes de abrir el PR, resolviendo conflictos sin descartar trabajo ajeno. El PR debe explicar el problema, comportamiento resultante, validación y limitaciones.

## Extensión y límites arquitectónicos

Las reglas de dominio puro, CQRS, paridad de interfaces y propuestas verificadas están en [AGENTS.md](AGENTS.md#core-rules-for-working-in-catml). La organización de capas se describe en [ARCHITECTURE.md](ARCHITECTURE.md).

La [guía de desarrollo](DEVELOPER_GUIDE.md#4-cómo-añadir-algo-sin-romper-la-arquitectura) explica cómo añadir modelos, commands, queries y persistencia. El [ejemplo de plugin](examples/plugins/custom_model.py) se prueba mediante [test_plugin_registration.py](tests/test_plugin_registration.py).

Añadir pruebas para comportamientos nuevos y regresiones. Agruparlas por funcionalidad; ampliar una suite existente cuando sea su lugar natural. Mantener los cambios dentro de la tarea elegida.

## Validación

```bash
.venv/bin/pytest --cov=src/automl --cov-report=term-missing --cov-fail-under=85
.venv/bin/automl --help
.venv/bin/automl task list
```

Debe pasar toda la suite y mantenerse cobertura global ≥85%. Las pruebas web abren un servidor en `127.0.0.1` con puerto efímero: requieren que el entorno permita sockets locales. No omitirlas para publicar un resultado completo.

Los resultados históricos y el número de pruebas no sustituyen una ejecución actual. La CI aplica el umbral de cobertura en Python 3.10 y 3.12.

## Documentación y evidencia

- Actualizar [TASKS.md](TASKS.md) y [.agent/progress.md](.agent/progress.md) al completar o pausar trabajo.
- Usar enlaces relativos y ejemplos ejecutables desde la raíz del repositorio.
- Actualizar [docs/README.md](docs/README.md) si cambia una capacidad o limitación.
- Mantener criterios de aceptación vinculados a implementación y pruebas en la especificación de la funcionalidad.
- Registrar decisiones arquitectónicas nuevas en [docs/decisions/](docs/decisions/), con fecha, estado y consecuencias.
- Reportar hallazgos fuera del alcance en el blackboard según [AGENTS.md](AGENTS.md); cerrar el aviso cuando exista una referencia verificable a la solución.

Los planes V0.5–V0.7 conservan la división histórica de trabajo. No son asignaciones disponibles: consultar siempre el backlog actual.

## Checklist de PR

- [ ] Alcance y cambios existentes respetados.
- [ ] Pruebas relevantes y suite completa pasan; cobertura ≥85%.
- [ ] Commands y queries nuevos registrados en `bootstrap.py`.
- [ ] Límites arquitectónicos respetados.
- [ ] Documentación, estado y evidencia actualizados.
- [ ] Diff revisado y rama sincronizada con su base.
