# Spec: Refactor progresivo Clean Architecture / Clean Code

> **Estado:** Propuesto (documentación) — implementación por fases PR1–PR5  
> **Origen:** Revisión externa sobre commit `138ad37` (+ contexto post-audit-3 / PR #100)  
> **Plan operativo:** [plan.md](plan.md)  
> **ADR:** [010-progressive-clean-architecture-refactor.md](../../decisions/010-progressive-clean-architecture-refactor.md)  
> **Backlog vivo:** [`TASKS.md`](../../../TASKS.md) → Track Clean Architecture Refactor

---

## 1. Problema

CATML ya tiene capas hexagonales (domain / application / engine / infrastructure / interfaces / plugins) y CQRS. El riesgo no es la ausencia de arquitectura, sino:

1. **God-objects** — `AutoMLWorkspace`, `interfaces/web/server.py`, `datasets.js`, `SklearnTrainer` acumulan responsabilidades.
2. **Reglas duplicadas** — dirección de métricas, aliases de CV, semillas HPO, permisos de path aparecen en varios sitios con pequeñas divergencias (familia de bugs de las auditorías 1–3).
3. **Especificación de ejecución dispersa** — opciones repartidas entre `RunConfig`, `Experiment`, `TrialExecution` y recetas; el trainer recompone en runtime.
4. **Estados insuficientemente gobernados** — transiciones de run/experiment/job/recipe no siempre pasan por una máquina de estados única ni por unidad de trabajo atómica.

## 2. Objetivos

| Objetivo | Medida de éxito |
| --- | --- |
| Una sola fuente de verdad para métricas y CV | Mismos inputs → mismo ganador en Workspace, HPO, leaderboard, Compare, CLI |
| `AutoMLWorkspace` como fachada | Métodos públicos delegan; lógica en servicios testeables |
| Ejecución inmutable validada | Trainer recibe `ExecutionSpec` (o equivalente) ya normalizado |
| API/UI mantenibles | Controllers por recurso; `datasets/` en componentes |
| Sin regresión estadística | Datasets de referencia + semillas fijas producen resultados equivalentes salvo fix documentado |

## 3. No-objetivos

- Microservicios, bus de eventos distribuido, reescritura greenfield.
- Migrar obligatoriamente a FastAPI en esta fase (opcional más adelante).
- Cambiar el modelo de plugins, SQLite local, ni paridad Python/CLI/MCP/Web.

## 4. Regla de dependencias (obligatoria)

```
interfaces  →  application  →  domain
                  ↓
         engine / infrastructure / plugins
              (adaptadores)
```

- `domain/` sigue siendo Python puro (sin sklearn/optuna/sqlite/pandas/HTTP).
- Interfaces solo hablan con `CommandBus` / `QueryBus` / fachada workspace (o servicios de aplicación inyectados).
- El dominio no importa `application` ni `interfaces`.

## 5. Contratos de dominio (núcleo PR1)

### 5.1 MetricSpec

```python
class MetricDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"

@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: MetricDirection

    def is_better(self, candidate: float, baseline: float) -> bool: ...
    def pick_best(self, scores: Iterable[float]) -> float: ...
```

- Builtins registrados en dominio.
- Plugins de métrica deben exponer dirección compatible (adapter en application/engine).
- Consumidores obligados a usar el registro: schedule, HPO observe/best, leaderboard, compare, facade, SQLite order-by si aplica.

### 5.2 ValidationStrategy

- Ya iniciado en `domain/experiments/validation.py` (audit-3).
- Consolidar aliases + `require_known_*` como API canónica; deprecar strings sueltos en callers.

### 5.3 ExecutionSpec (inmutable)

Construido **antes** de entrenar / optimizar / exportar:

- Identidad de datos: `dataset_id`, `dataset_content_hash`
- Target, features, transforms, derived
- `model_id`, params
- `metric`, `validation_strategy`, `cv_folds`, `random_seed` (fold seed vs estimator seed documentados)
- Grupo / time column si aplica
- Versiones relevantes (opcional en PR1: catml + sklearn)

El trainer **no** debe rellenar defaults contradictorios; si falta un campo requerido → fail fast.

### 5.4 Errores de dominio (mínimo)

Introducir (o alinear) excepciones tipadas, p.ej.:

- `InvalidValidationStrategy`
- `InvalidMetric`
- `InvalidRecipeTransition`
- `DatasetIntegrityError` (hash mismatch policy)

Sin filtrar detalles sensibles en mensajes de API.

## 6. Estructura objetivo (migración gradual)

Ver árbol en [plan.md](plan.md). No se exige mover todos los archivos de golpe; sí exigir que **nuevos** módulos respeten la estructura y que cada PR reduzca el tamaño de un god-object.

## 7. Criterios de aceptación globales

- [ ] Suite completa pytest verde y cobertura ≥ 85 %.
- [ ] Tests de contrato multi-interfaz para métricas y CV (checklist audit-3 §7).
- [ ] Ningún PR de refactor cambia scores de referencia sin nota “fix estadístico documentado”.
- [ ] Fachada `AutoMLWorkspace` conserva firmas públicas usadas por CLI/MCP/Web hasta PR2+ con deprecación explícita si se renombra algo.

## 8. Relación con trabajo reciente

| Trabajo | Relación |
| --- | --- |
| Audit-3 / PR #100 | Embrión PR1: `validation.py`, `metrics/direction.py`, recipe replay, FDR, XSS |
| Sprint 2 / PR #99 | UX Explore/Compare — merge antes o en paralelo; no mezclar con extracción Workspace |
| Export / path confinement #96 | Ya aporta piezas de `infrastructure/security` / filesystem — no reabrir |
