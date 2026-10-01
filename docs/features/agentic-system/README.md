# CATML Agentic System: Roles & Implementation Guide

> **Current Status (`main` branch):** Milestones **H0, H1, H2, and H3** are fully implemented and integrated into `main` (PR #22, #23, #24, #25, #26, #30, #35, #36). The system provides a complete tool catalog, budget reservation policies, human approvals, long-running operations lifecycle with worker leases, failure reconciliation, cooperative cancellation, agent operations CLI inspection, and a dual-transport Model Context Protocol (MCP) server. Current work proceeds to **Milestone H4** (autonomous deterministic cycle and specialist agents).

## Multi-Agent / Two-Person Concurrency & Conflict Prevention Protocol

All contributors and AI coding agents working concurrently across Persona A and Persona B must adhere to the collaboration protocol detailed in [`two-person-plan.md` §5.1](two-person-plan.md#51-protocolo-de-prevención-de-conflictos-y-concurrencia-lecciones-h3-rightarrow-h4h5) and [`AGENTS.md`](../../../AGENTS.md):
1. **Subdirectory Ownership:** Persona A works exclusively in `specialists/` and `infrastructure/llm/`; Persona B works exclusively in `orchestrator/` and session CLI interfaces. Monolithic shared files (`executor.py`, `sqlite_agent_ledger.py`, `ports.py`) are never modified concurrently without an integrated contract PR.
2. **Contract-First Stabilization:** All shared DTOs, protocols, and interfaces are defined and merged into `main` before branching into parallel implementation tasks.
3. **Merge-Only Git Policy:** Always integrate with `git merge origin/main`; never rebase active or shared branches.
4. **Separated Documentation Tracks:** Update only your assigned persona track in `TASKS.md` and `.agent/progress.md`.
5. **Cross-Agent Blackboard:** Review `gh issue list --label blackboard --state open` before starting any task.

---

## 1. Elige tu rol

| | Persona A | Persona B |
| --- | --- | --- |
| Enfoque | Aplicación, reglas y evidencia | Interfaces, integración y orquestación |
| Elige este rol si prefieres | Contratos Python, CQRS, políticas, SQLite, recuperación y experimentación ML | Protocolos/API, CLI, CI, pruebas de integración y flujos con LangGraph |
| Construirás | Catálogo/executor de tools, presupuestos, aprobación, auditoría, operaciones y especialistas | Servidor MCP, comandos CLI, consumidores, ciclo del agente y checkpoints |
| Tu primer paquete | **A0: contratos, políticas y persistencia propuesta** | **B0: compatibilidad, consumidores y ADR** |
| Paquetes posteriores | A1 → A2 → A3 → A4 → A5 | B1 → B2 → B3 → B4 → B5 |
| Revisión | Revisas los PRs de B | Revisas los PRs de A e integras los hitos |

Ambos roles requieren pruebas y coordinación. B también trabaja lógica de flujo; A también participa en integración. No son roles de frontend/backend ni niveles de experiencia.

Comunica tu elección al otro colaborador y acordad **una persona por rol**. Si coincidís en el mismo rol, resolved el reparto antes de editar archivos compartidos. Registrar la asignación en `.agent/progress.md` junto al paquete activo; este documento no asigna personas automáticamente.

## 2. Qué leer y en qué orden

1. **Esta guía:** elegir A o B y localizar tu primer paquete.
2. **[AGENTS.md](../../../AGENTS.md):** reglas del repositorio y validación obligatoria.
3. **[TASKS.md](../../../TASKS.md)** y **[.agent/progress.md](../../../.agent/progress.md):** estado real, tareas completadas y trabajo en curso.
4. **[ARCHITECTURE.md](../../../ARCHITECTURE.md):** límites entre dominio, aplicación, motor e interfaces.
5. **[plan.md](plan.md), completo:** alcance, contratos, reglas de ejecución, recuperación y aceptación técnica. Es la referencia de qué construir.
6. **[two-person-plan.md](two-person-plan.md), completo:** reparto A0–A5/B0–B5, dependencias, archivos propios, revisiones e integración. Es la referencia de quién construye cada parte y cuándo.
7. **[DEVELOPER_GUIDE.md](../../../DEVELOPER_GUIDE.md):** patrones aplicables al paquete que empiezas. Consultar las secciones V0.9/V1.0 de **[AutoML_Arquitectura_Tecnica.md](../../../AutoML_Arquitectura_Tecnica.md)** para contexto; el alcance actual aplaza V0.8 según `plan.md`.

Para escoger rol basta esta guía y la comparación de responsabilidades. Para programar, completa las lecturas 2–6 y los patrones pertinentes de la guía. No necesitas leer toda la arquitectura técnica histórica antes de empezar H0.

En caso de duda: AGENTS define las reglas del repositorio; `plan.md` define requisitos técnicos; `two-person-plan.md` concreta propiedad y dependencias. Si hay contradicción, registradla y resolvedla conjuntamente antes de cambiar un contrato compartido.

## 3. Si eliges A: empieza por A0

Primero revisa `domain/ports.py`, Commands/Queries existentes y bootstrap. Tu primera entrega revisable contiene:

- Firmas y DTOs de tools, contexto, errores, políticas y operaciones; schemas con ejemplos válidos e inválidos.
- Propuesta de defaults finitos y qué acciones requieren aprobación, usando una única política.
- Diseño de ledger/auditoría/reservas y garantías de atomicidad o reconciliación con SQLite.
- Tests de contrato sin extras ni credenciales y fixtures para B.

Consulta las secciones 2–7 y 12 de `plan.md`, y la fila A0/propiedad de archivos en `two-person-plan.md`. Entrega a B firmas, fixtures, errores y limitaciones para revisión. A0 no incluye todavía entrenamientos autónomos ni el catálogo mutante completo de A2.

## 4. Si eliges B: empieza por B0

Primero revisa CLI, composición del proyecto, `pyproject.toml` y CI. Tu primera entrega revisable contiene:

- Compatibilidad propuesta de SDK MCP, LangGraph/checkpointer y Python del proyecto, con pruebas mínimas aisladas.
- Casos de consumidor/fixtures que ejercitan contratos de A0 y errores esperados.
- ADR de alcance/dependencias: V0.8 pospuesta, stdio primero, extras opcionales y límites del incremento inicial.
- Revisión de A0 desde las necesidades de MCP, CLI y el flujo del agente.

Consulta las secciones 3, 8–12 de `plan.md`, y la fila B0/propiedad de archivos en `two-person-plan.md`. B0 puede avanzar en paralelo con A0; firma sus contratos junto a A antes de construir B1 contra la implementación real. No copies contratos ni implementes políticas alternativas dentro del servidor o grafo.

## 5. Antes de trabajar y antes de avanzar a H1

Desde raíz, inspeccionar rama/estado local, consultar el blackboard como exige AGENTS y ejecutar el baseline indicado en los planes. Preservar los cambios existentes y acordar commit base; cada persona usa su rama/worktree según el reparto. Estos documentos están en el árbol local: para otro checkout, deben estar incluidos en el commit compartido que ambos elijáis.

H0 cierra cuando A y B revisan A0/B0, acuerdan firmas/defaults/recuperación/versiones y dejan un commit base con contratos y tests. Entonces A comienza A1 y B comienza B1. No hace falta esperar a que todo V0.9 esté terminado para trabajar en paralelo.

Primera entrega útil conjunta: **H2**, consultar perfil/modelos → crear candidato → autorizar → ejecutar una vez → consultar métricas. H3 añade trabajos largos; H4/H5 añaden el ciclo y LangGraph. La planificación no promete fechas; cada paquete se estima con la disponibilidad de ambos.

## 6. Texto listo para asignar el trabajo

Puedes copiar uno de estos mensajes a la persona o agente que realizará el rol. Sustituye el rol solamente si cambia el reparto acordado.

**Para A:**

> Asume el rol Persona A del sistema agéntico de CATML. Empieza leyendo docs/features/agentic-system/README.md y sigue su orden de lectura. Coordina la elección con Persona B, consulta AGENTS.md, TASKS.md, .agent/progress.md y el blackboard. Ejecuta el paquete A0 de two-person-plan.md, preparando contratos, schemas, políticas/defaults y persistencia propuesta con tests de contrato. Respeta los archivos asignados, conserva cambios existentes y entrega a B los contratos para revisión conjunta antes de iniciar A1.

**Para B:**

> Asume el rol Persona B del sistema agéntico de CATML. Empieza leyendo docs/features/agentic-system/README.md y sigue su orden de lectura. Coordina la elección con Persona A, consulta AGENTS.md, TASKS.md, .agent/progress.md y el blackboard. Ejecuta el paquete B0 de two-person-plan.md, preparando compatibilidad, casos de consumidor y ADR, y revisa los contratos de A0. Respeta los archivos asignados, conserva cambios existentes y acuerda H0 con A antes de iniciar B1.
