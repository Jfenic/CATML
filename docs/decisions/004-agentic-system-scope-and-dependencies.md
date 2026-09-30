# ADR 004 — Alcance, dependencias y arquitectura del subsistema agéntico (V0.9 / V1.0)

Fecha: 2026-09-30. Estado: aceptada para H0.

## Contexto

La arquitectura técnica original contemplaba la fase V0.8 (Meta-learning y base de conocimiento para warm-start cross-dataset) antes de las herramientas agénticas (V0.9) y del agente autónomo de experimentación (V1.0).

En la práctica operativa, la capacidad de inspeccionar datasets, proponer hipótesis sobre el run actual y ejecutar experimentos de forma controlada mediante asistentes externos (Claude Desktop, Cursor, Gemini CLI) o mediante un bucle autónomo aporta valor inmediato. El meta-learning entre diferentes datasets requiere un histórico previo estandarizado que se consolida mejor tras disponer de agentes operando sobre runs reales.

Asimismo, es imperativo mantener la pureza de la arquitectura hexagonal de CATML: el núcleo determinista (`domain/`, `application/`, `engine/`) no debe depender de frameworks de orquestación agéntica ni de SDKs externos pesados o volátiles.

## Decisión

1. **Reordenación de fases:**
   - Posponer la fase V0.8 a favor de V0.9 (Tools agénticas y servidor MCP) y V1.0 (Agente de experimentación determinista / LangGraph).
   - El contexto inicial de trabajo se restringe al run y workspace actual (perfil del dataset, ranking y evidencia de features, leaderboard e histórico del run), sin asumir similitud estadística entre datasets ni transferibilidad automática.

2. **Dependencias opcionales en `pyproject.toml`:**
   - Mantener el paquete base sin dependencias de LLMs, MCP ni LangChain.
   - Definir extras opcionales:
     - `mcp = ["mcp>=1.2.0"]`: SDK oficial de Model Context Protocol para el adaptador `interfaces/mcp/`.
     - `agents = ["langgraph>=0.2.0", "langgraph-checkpoint>=2.0.0"]`: Motor de estados y checkpoints duraderos para `agents/`.
   - Prohibir terminantemente la importación de `mcp`, `langgraph`, `langchain` o `pydantic` dentro de `src/automl/domain/` y `src/automl/engine/`.

3. **Transporte MCP:**
   - El hito H1 implementa exclusivamente transporte `stdio` mediante el SDK oficial. `stdout` queda reservado de manera estricta al protocolo JSON-RPC; cualquier log o diagnóstico debe emitirse por `stderr`.
   - El transporte HTTP Streamable con autenticación y sesiones aisladas queda aplazado al hito H3.

4. **Invariantes de ejecución y seguridad ("Proponer ≠ Aceptar"):**
   - El agente o cliente MCP opera exclusivamente a través de contratos de aplicación y los buses `CommandBus` / `QueryBus`.
   - Las propuestas no entrenan ni promocionan modelos por sí mismas.
   - Políticas con límites finitos (iteraciones, trials, tiempo, llamadas y presupuesto).
   - Para entornos no interactivos o llamadas RPC, toda acción que requiera intervención humana retorna un resultado tipado `APPROVAL_REQUIRED` con `approval_id` en estado pendiente, sin suspender el hilo ni esperar entrada de terminal.

## Consecuencias

- El núcleo determinista de CATML, la CLI base y el Workbench web continúan funcionando sin necesidad de instalar los extras agénticos ni configurar credenciales de API.
- Los tests del núcleo y de los contratos agénticos se ejecutan de manera aislada, rápida y sin llamadas de red a proveedores de IA.
- Se asegura la paridad de interfaces: tanto la CLI como el servidor MCP y el orquestador invocan las mismas operaciones y respetan las mismas políticas de autorización.
- La ejecución conjunta entre colaboradores queda acotada en paquetes independientes (A0–A5 para Persona A y B0–B5 para Persona B), coordinados en el hito común H0.
