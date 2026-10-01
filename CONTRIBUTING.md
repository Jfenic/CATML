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

## Concurrent Multi-Agent & Parallel Development Protocol

When multiple developers or AI coding agents work concurrently on parallel tracks (e.g., Persona A and Persona B):

1. **Pre-Agreed Shared Contracts:** Define and merge shared interfaces, domain ports, and DTO contracts into `main` before starting parallel work on feature branches.
2. **Subdirectory Ownership & Isolation:** Confine changes strictly to your assigned directory (e.g. `specialists/` vs `orchestrator/`). Monolithic shared files (such as `executor.py`, `ports.py`, or `sqlite_agent_ledger.py`) must never be edited concurrently on feature branches without a pre-agreed contract PR.
3. **Merge-Only Policy:** Always synchronize branches with `git merge origin/main`. **Never rebase** branches intended for collaboration.
4. **Separated Progress Tracking:** Keep tracking updates isolated in designated sections of [`TASKS.md`](TASKS.md) and [`.agent/progress.md`](.agent/progress.md) (`Track Persona A` vs `Track Persona B`).
5. **Cross-Agent Blackboard:** Check `gh issue list --label blackboard --state open` before starting, and report any out-of-scope issues via GitHub issues rather than editing unrelated modules.

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
