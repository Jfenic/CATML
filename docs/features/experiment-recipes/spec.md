# Feature Specification: Experiment Recipes (Explore → Evidence → Reusable Config)

> **Módulo:** `experiment-recipes`  
> **Estado:** Especificación aprobada — **implementada (R0–R4)**; backlog operativo en [`TASKS.md`](../../../TASKS.md)  
> **Prioridad:** P1 (cerrada en código junto a Explore E6 / `0.9.0-rc1`)  
> **Relacionado con:** [ADR-001](../../decisions/001-hexagonal-architecture-and-cqrs.md), [ADR-002](../../decisions/002-hypothesis-driven-experiments.md), [ADR-004](../../decisions/004-agentic-system-scope-and-dependencies.md), [ADR-008](../../decisions/008-catml-explore-modular-monolith.md), [ADR-009](../../decisions/009-experiment-recipes-and-interface-parity.md), [Plan](plan.md), [CATML Explore](../catml-explore/spec.md), [TASKS.md](../../../TASKS.md).

---

## 1. Visión

Explore **descubre**, la verificación **comprueba**, CATML **conserva el aprendizaje** como recetas reproducibles, y Experimentos **re-prueba** variantes. Personas, scripts y agentes de IA usan **las mismas capacidades** a través de Python, CLI, REST, MCP y Workbench.

No es un botón del Workbench: es un **caso de uso del núcleo de aplicación** con varias interfaces cliente.

```text
Explore → Verify → EvidenceLink → ExperimentRecipe → Experimentos → Pipeline/Recipe promovida
```

---

## 2. Problema resuelto (contexto histórico pre-R0)

Antes de Recipes (post Explore E5/E6, pre-R0), el gap era:

| Capacidad | Pre-R0 | Post R0–R4 |
|-----------|--------|------------|
| Hallazgos + hipótesis (`proposed`) | Implementado | Implementado |
| `VerifyHypothesisCommand` + `EvidenceLink` | Implementado | Implementado |
| Paridad CLI/MCP/REST de verify | Implementado | Implementado |
| Aplicación real de `power_transform` / `robust_scaler` en el candidato | Solo metadata | **R0** — transforms reales + fit-in-fold |
| Receta reproducible a partir de evidencia | No | **R1** — `ExperimentRecipe` + CQRS |
| Selected / Promoted (más allá de accepted/rejected) | No | **R1/R3** — ciclo de vida auditable |
| Reutilizar mejora en un solo modelo o grid de configs | Manual | **R3/R4** — CLI / MCP / REST / Workbench |

`EvidenceLink` sigue registrando verificación puntual; la **receta** es el objeto reproducible seleccionable/promovible. *Propose ≠ Accept* se mantiene: `accepted` ≠ promoción automática.

---

## 3. Principio oficial

> CATML debe ofrecer **paridad funcional** entre Python, CLI, REST, MCP y Workbench.  
> Las reglas de negocio, permisos, evidencia y ejecución pertenecen al **núcleo de aplicación**, nunca a una interfaz específica.

Una operación = un caso de uso CQRS = N superficies de acceso.  
Un agente **nunca** debe saltarse aprobaciones/presupuestos por usar MCP en lugar de la UI.

---

## 4. Ciclo de estados (cuatro fases)

| Estado | Significado | Quién |
|--------|-------------|-------|
| **Proposed** | Explore propone una mejora posible | Motor estadístico / agente (propuesta) |
| **Verified** | Una prueba genera evidencia (Δ vs baseline) | `Verify*` (humano o agente autorizado) |
| **Selected** | Se elige usar esa evidencia en una nueva receta/experimento | Humano o agente **autorizado** |
| **Promoted** | Tras más experimentos, la config queda como candidata reutilizable | Operación explícita, auditable |

**Regla crítica:** `EvidenceLink.accepted == true` **no** implica autorización automática para aplicar la transformación. Una mejora verificada en un modelo puede fallar en otro.

---

## 5. Dos tipos de mejora

### 5.1 Derivadas sin estado (`log1p`, ratios, `clip`…)

- Se representan como **feature derivada** (expresión) + lista de features.
- Reproducibles sin “fit” previo; seguras para materializar o recalcular.

### 5.2 Transformadores con estado (`RobustScaler`, `PowerTransformer` / Yeo-Johnson…)

- Se representan como **pasos de pipeline** con hiperparámetros.
- Deben **ajustarse dentro de cada fold de entrenamiento** (fit solo en train), nunca sobre todo el dataset antes del split.
- La receta guarda el *tipo* de transformador y parámetros de configuración; el estado fitted vive en el pipeline de entrenamiento / artefacto.

---

## 6. Contrato de dominio: `ExperimentRecipe` (nombre propuesto)

Objeto de primer nivel que conserva la configuración **real** y auditable:

| Campo | Descripción |
|-------|-------------|
| `id`, `name`, `status` | Identidad y estado (`draft` / `selected` / `submitted` / `promoted` / `archived`) |
| `dataset_id`, `dataset_content_hash` / versión | Datos exactos |
| `feature_names` | Columnas base incluidas |
| `derived_features[]` | Expresiones sin estado (`log1p`, etc.) |
| `column_transforms[]` | `{column, kind, params}` — `robust_scaler`, `yeo_johnson`, … (fit-in-fold) |
| `drop_features[]` | p.ej. resolución de colinealidad |
| `model_ids`, `model_params` | Estimador(es) e hiperparámetros |
| `validation_strategy`, `cv_folds`, `metric`, `random_seed` | Comparación justa |
| `evidence_link_id`, `hypothesis_id`, `study_id` | Trazabilidad Propose ≠ Accept |
| `lineage` / notas | Origen humano vs agente |

Opcionalmente alias `PipelineRecipe` si se unifica con grafos multimodales; el contrato mínimo es la receta tabular anterior.

---

## 7. Casos de uso CQRS (propuestos)

### Comandos

- `VerifyAnalysisHypothesisCommand` — (existe; endurecer aplicación real de transforms)
- `CreateExperimentRecipeFromEvidenceCommand` — crea receta *draft* desde `EvidenceLink` (sin entrenar)
- `SubmitExperimentFromRecipeCommand` — ejecuta experimento(s) desde receta (presupuesto / política)
- `PromoteExperimentRecipeCommand` — marca receta como promovida tras evidencia adicional
- (Opcional) `SelectVerifiedImprovementCommand` — transición Selected explícita

### Consultas

- `GetEvidenceLinkQuery` / `ListEvidenceLinksQuery` — (existen)
- `ListVerifiedImprovementsQuery` — evidencias/hipótesis verificadas filtrables
- `GetExperimentRecipeQuery` / `ListExperimentRecipesQuery`
- (Opcional) `PreviewExperimentRecipeQuery` — dry-run / resolución de features

Registro obligatorio en `bootstrap` / `registries/*` (ADR-001).

---

## 8. Paridad de superficies

| Capa | Responsabilidad |
|------|-----------------|
| Dominio | Recetas, transforms, vínculos de evidencia |
| Aplicación / CQRS | Validar, crear, enviar, promover |
| Python API / facade | Mismo flujo desde código |
| CLI | p.ej. `catml recipe create-from-evidence`, `submit`, `promote`, `list` |
| MCP | Tools de lectura + propuesta + ejecución gobernada |
| REST | Workbench y clientes HTTP |
| Workbench | Visualización y decisión humana (**última** capa) |

### Secuencia MCP ejemplo (agente)

```text
analysis_list_verified_improvements
  → analysis_get_evidence
  → experiment_create_recipe_from_evidence   # draft, sin entrenar
  → experiment_preview_recipe
  → experiment_submit_recipe                 # requiere presupuesto / aprobación
  → experiment_get_results
  → experiment_request_promotion             # explícito y auditable
```

### Política de permisos

| Acción | Nivel |
|--------|-------|
| Consultar hallazgos / evidencia | Lectura |
| Crear receta draft | Propuesta (sin entrenar) |
| Ejecutar experimentos | Presupuesto + autorización |
| Promover configuración ganadora | Explícita, auditable |

---

## 9. Criterios de aceptación

1. Receta creada desde evidencia contiene transforms, columnas, dataset hash, modelo, validación, semilla e IDs de evidencia.
2. `RobustScaler` / `PowerTransformer` se ajustan **in-fold** en el trainer; test de no-leakage.
3. `accepted` en EvidenceLink no aplica transforms automáticamente; hace falta Selected/Submit explícito.
4. **Prueba de paridad:** crear receta por CLI → leerla por MCP → verla en Workbench → ejecutar el mismo experimento sin divergencia de configuración.
5. Suite verde y cobertura ≥ 85%.

---

## 10. Fuera de alcance (fase posterior)

- Meta-learning automático sobre recetas históricas por fingerprint de dataset (priors, no garantías).
- UI avanzada de A/B/C/D grid multi-brazo en un solo clic (puede llegar después de contratos + trainer).
- Firmas criptográficas de recetas (integridad de artefactos ya cubierta parcialmente por SHA-256 de modelos).
