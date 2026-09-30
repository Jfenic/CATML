# Revisión de Contratos A0 desde la perspectiva de Consumidores (B0)

> Fecha: 2026-09-30. Autor: Persona B (Interfaces, Integración y Orquestación).
> Destinatario: Persona A (Aplicación, Reglas y Evidencia).
> Estado: Hito H0 (Coordinación y acuerdo de contratos base).

---

## 1. Resumen de la Revisión

Tras analizar el [plan agéntico](plan.md), el [reparto de responsabilidades](two-person-plan.md) y validar los casos de consumidor en `tests/test_v09_agent_consumers.py`, los contratos propuestos en la Sección 3 de `plan.md` proporcionan una base sólida y desacoplada para construir el servidor MCP (`interfaces/mcp/`), los comandos CLI (`interfaces/cli/agent_cli.py`) y la orquestación cíclica (`agents/`).

Se detallan a continuación las recomendaciones, requisitos de formato y acuerdos indispensables que Persona A debe incorporar en su entrega **A0**:

---

## 2. Recomendaciones Críticas para Persona A (Paquete A0)

### 2.1. Formato de `input_schema` en `ToolDefinition`
* **Compatibilidad MCP:** El SDK de Model Context Protocol exige que el `inputSchema` de una tool sea un esquema JSON válido (draft-07 o 2020-12).
* **Requisito para A0:**
  * Debe ser siempre de tipo `"object"` en su raíz (`"type": "object"`).
  * Definir explícitamente `"properties"` con tipos básicos (`string`, `number`, `integer`, `boolean`, `array`, `object`).
  * Incluir `"required": [...]` y `"additionalProperties": false` para evitar que los LLMs inventen parámetros.
  * No exponer `Any`, lambdas ni objetos de dominio complejos en los esquemas públicos.

### 2.2. Aislamiento y Blindaje de `ToolCallContext`
* **Inyección segura por la Interfaz:** El `ToolCallContext` contiene credenciales del entorno de ejecución (`actor_id`, `workspace_path`, `run_id`, `correlation_id`).
* **Requisito para A0:**
  * El `ToolExecutor` debe exigir que `ToolCallContext` se pase como parámetro independiente del `ToolInvocation`.
  * Los argumentos contenidos en `ToolInvocation.arguments` provienen del cliente MCP o del LLM; el executor debe ignorar o rechazar cualquier intento de sobreescribir campos de contexto (p. ej. `_workspace_path` o `actor_id`).

### 2.3. Tratamiento de `APPROVAL_REQUIRED` en Entornos No Interactivos
* **Comportamiento no bloqueante:** Tanto el transporte stdio de MCP como una API REST o un worker en segundo plano se colgarían si una tool espera un prompt de consola (`input()`).
* **Requisito para A0:**
  * Si la política dictamina `requires_approval=True`, el `ToolExecutor` no debe solicitar confirmación por consola.
  * Debe generar y persistir un `ApprovalRequest` en SQLite con un ID determinista (`approval_id`), retornando inmediatamente un `ToolResult` con `status="approval_required"`, `approval_id` y explicación.
  * Esto permite que la CLI (`automl agent approve <id>`) o el Workbench resuelvan la aprobación de forma desacoplada.

### 2.4. Códigos de Error Canónicos en `ToolError`
* **Consistencia de interfaz:** Para que el servidor MCP y el CLI presenten errores claros sin filtrar trazas de la base de datos o rutas internas del host, los errores deben clasificarse en los códigos canónicos definidos en el plan:
  * `INVALID_ARGUMENT`, `NOT_FOUND`, `SCOPE_VIOLATION`, `PERMISSION_DENIED`, `BUDGET_EXCEEDED`, `APPROVAL_REQUIRED`, `CONFLICT`, `DEADLINE_EXCEEDED`, `DEPENDENCY_UNAVAILABLE`, `INTERNAL_ERROR`.
* **Requisito para A0:**
  * Cada `ToolError` debe incluir `code: ToolErrorCode`, `message: str`, `details: Optional[dict]`, `retryable: bool` y `correlation_id: Optional[str]`.

### 2.5. Idempotencia y Reintentos
* **Requisito para A0:**
  * `ToolInvocation` debe admitir `idempotency_key: Optional[str]`.
  * Toda mutación con la misma clave y los mismos argumentos debe devolver el resultado previo sin reejecutar el comando en el bus de negocio. Si los argumentos cambian para la misma clave, debe responder con `CONFLICT`.

---

## 3. Estado de Consumidores (Persona B)
* Se ha creado la suite de verificación de consumidores en [`tests/test_v09_agent_consumers.py`](file:///home/tania/projects/CATML/tests/test_v09_agent_consumers.py), demostrando que tanto el adaptador MCP como el orquestador consumen estos contratos sin extras ni dependencias externas.
* Se ha registrado el ADR 004 en [`docs/decisions/004-agentic-system-scope-and-dependencies.md`](file:///home/tania/projects/CATML/docs/decisions/004-agentic-system-scope-and-dependencies.md) y configurado los extras `mcp` y `agents` en [`pyproject.toml`](file:///home/tania/projects/CATML/pyproject.toml).
