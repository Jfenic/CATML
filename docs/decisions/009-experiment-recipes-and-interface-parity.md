# ADR-009: Experiment Recipes y Paridad de Interfaces

> **Estado:** Aceptada (intención arquitectónica; implementación pendiente)  
> **Fecha:** 2026-10-11  
> **Alcance:** Puente Explore → Experimentos → aprendizaje reutilizable  
> **Relacionado con:** [ADR-001](001-hexagonal-architecture-and-cqrs.md), [ADR-002](002-hypothesis-driven-experiments.md), [ADR-004](004-agentic-system-scope-and-dependencies.md), [ADR-008](008-catml-explore-modular-monolith.md), [Spec](../features/experiment-recipes/spec.md), [Plan](../features/experiment-recipes/plan.md).

---

## Contexto

CATML Explore (E1–E6) propone hipótesis y genera `EvidenceLink` tras verificación empírica (*Propose ≠ Accept*). Eso registra **si** una hipótesis mejoró un baseline concreto; no define aún una **configuración reproducible** que humanos y agentes puedan seleccionar, re-probar en otros modelos y promover.

Sin un objeto de primer nivel y sin paridad de superficies, el riesgo es implementar “un botón en el Workbench” que no puedan usar la CLI ni MCP, o que un agente aplique transforms sin la misma gobernanza.

---

## Decisión

1. **Introducir `ExperimentRecipe` (o equivalente)** como entidad de dominio que captura: transforms y parámetros, columnas, dataset/versión, modelo y params, validación/métrica/semilla, y referencia a `evidence_link_id` / `hypothesis_id`.
2. **Separar estados:** Proposed → Verified → Selected → Promoted.  
   `EvidenceLink.accepted` **no** autoriza automáticamente la aplicación de una transformación.
3. **Distinguir** mejoras sin estado (features derivadas, p.ej. `log1p`) de transforms con estado (`RobustScaler`, `PowerTransformer`), estos últimos con **fit dentro de cada fold**.
4. **Paridad funcional:** Python, CLI, REST, MCP y Workbench son clientes del mismo Command/Query Bus. La UI es la última capa, no la dueña de la lógica.
5. **Gobernanza de agentes:** crear receta draft ≠ ejecutar; ejecutar y promover requieren presupuesto/aprobación auditable (alineado con ADR-004).

---

## Alternativas consideradas

- **Solo UI “Usar mejora”:** Rechazada — no sirve a scripts/agentes y fragmenta reglas de negocio.
- **Tratar `accepted` como promoción automática:** Rechazada — confunde verificación puntual con autorización y generalización a otros modelos.
- **Materializar RobustScaler/Yeo-Johnson sobre todo el dataset:** Rechazada — fuga de información entre folds.

---

## Consecuencias

### Positivas
- Ciclo científico cerrado y auditable: hallazgo → evidencia → receta → experimento → promoción.
- Agentes y humanos con las mismas garantías.
- Base futura para meta-learning de transforms por fingerprint (sin confundir historial con garantía).

### Coste
- Nuevo bounded context ligero (recipes), migraciones SQLite, extensión del trainer, más superficie MCP/CLI.
- Implementación diferida en fases R0–R4 ([plan.md](../features/experiment-recipes/plan.md)); estado operativo en [TASKS.md](../../TASKS.md).
