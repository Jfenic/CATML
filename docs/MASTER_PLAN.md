# Plan Maestro CATML: Producto, Arquitectura y Mercado

> **CATML: Controlled Agentic Toolkit for Machine Learning**  
> *AutoML local-first, gobernado y operable por humanos o agentes mediante Python, CLI, Workbench y MCP.*

---

## 1. Posicionamiento Estratégico

CATML no compite como “otro AutoML tabular genérico” ni busca superar marginalmente métricas públicas en benchmarks estándar. Su propuesta diferencial se enfoca en el control, la gobernanza y la soberanía de los datos:

* **Propuesta de Valor:**
  > **CATML = Controlled Agentic Toolkit for Machine Learning**
* **Espacio de Mercado:**
  > **AutoML local-first, gobernado y operable indistintamente por humanos o agentes mediante Python, CLI, Workbench y MCP.**
* **Mensaje Central:**
  > **Train better models. Keep control.**
* **Propuesta de Valor Corporativa:**
  > **Bring AI to your data. Not your data to AI.**

### Diferencial Competitivo Integrado

```text
Local-first
+ AutoML
+ Agent-native
+ MCP
+ Anti-Leakage
+ Policies
+ Approvals
+ Budgets
+ Audit
+ Reproducible Artifacts
```

La multimodalidad es una capacidad que crece gradualmente encima de este núcleo controlado, no la identidad de partida.

---

## 2. Arquitectura Objetivo

CATML separa estrictamente el plano de control y gobernanza del motor de ejecución y los backends especializados:

```text
          HUMAN / AGENT
               │
     ┌─────────┼─────────┐
     │         │         │
  Python      CLI    Workbench
               │
              MCP
               │
               ▼
        ┌─────────────┐
        │ CATML       │
        │ CONTROL     │
        │ PLANE       │
        └──────┬──────┘
               │
     Policies / Budgets
     Approval / Audit
               │
               ▼
        ProblemSpec
        ValidationSpec
               │
               ▼
       Experiment Engine
               │
     ┌─────────┼──────────┐
     ▼         ▼          ▼
  Tabular     NLP       Vision
     │         │          │
 LightGBM   embeddings   timm
 XGBoost    TF-IDF       PyTorch
 CatBoost                ...
     │         │          │
     └─────────┼──────────┘
               ▼
          Evaluation
               ▼
            Artifact
```

Todo el flujo opera íntegramente dentro de la infraestructura local o privada del usuario.

---

## 3. Regla Fundamental de Privacidad

Principio rector de diseño:

> **El cómputo va hacia los datos. Los datos no van hacia CATML.**

Por defecto y por contrato arquitectónico:

```text
Dataset             LOCAL
Predictions         LOCAL
Models              LOCAL
Artifacts           LOCAL
Logs sensibles      LOCAL
Embeddings          LOCAL
MCP Server          LOCAL
Agent               LOCAL (opcional con LLM local o endpoint privado)
```

Cualquier futura versión de plataforma multi-nodo o colaborativa podrá disponer de un plano de control distribuido, pero nunca requerirá la centralización ni la salida de los datos crudos.

---

## 4. Paridad Estricta de Interfaces

Las interfaces de usuario y agente son adaptadores ortogonales sobre el mismo sistema hexagonal:

```text
Python
CLI
Workbench
MCP / Agent
```

Son cuatro canales de acceso a una única implementación de servicios de aplicación (`CommandBus`, `QueryBus`, `AutoMLWorkspace`).

### Principio de Paridad
> **Cualquier acción que un agente deba o pueda ejecutar debe corresponder exactamente a un comando o consulta formal de CATML.**

Flujo de ejecución gobernado:

```text
Agent
  ↓
MCP
  ↓
Command
  ↓
Policy
  ↓
Approval (si es requerida)
  ↓
Execution
  ↓
Audit Ledger
```

**Regla de Seguridad:** Ningún agente ni interfaz externa puede acceder directamente a la infraestructura sin pasar por el bus de aplicación y el sistema de políticas.

---

## 5. Agente Local y Human-in-the-Loop ("Proponer ≠ Aceptar")

CATML ofrece capacidades agénticas locales (`automl agent` / `catml agent`) soportando proveedores agnósticos:
- Ollama
- vLLM
- Endpoints compatibles con OpenAI
- Modelos propietarios externos (opcionales bajo autorización explícita)

### Interacción Típica y Salvaguarda Metodológica

```text
> analiza este dataset industrial

CATML Agent:
Potential time leakage detected.

Feature:
maintenance_completed_at

Recommendation:
exclude from training.

Suggested validation:
TimeSeriesSplit

Estimated runtime:
8 minutes.

Run experiment?
[Approve] [Reject]
```

El agente **propone candidatos o hipótesis empíricas**; el sistema de políticas y el operador humano determinan su aceptación y ejecución.

---

## 6. Políticas de Seguridad y Perfiles de Gobernanza

CATML define restricciones explícitas sobre las acciones del motor y de los agentes:

```text
Raw data to external LLM       DENY
Raw row access                 DENY
Aggregated statistics          ALLOW
Schema information             ALLOW
Model training                 ALLOW
GPU job > 30 min               REQUIRE APPROVAL
Artifact export                REQUIRE APPROVAL
External network access        DENY
```

### Perfiles de Operación

* **`STRICT` (Entornos Regulados y Datos Críticos):**
  - Acceso a Internet: `OFF`
  - LLM Externo: `OFF` (solo LLM local/Ollama)
  - Egress de datos crudos: `OFF`
  - Exportación de artefactos: `REQUIRE APPROVAL`
  - Ejecución de shell/sistema: `OFF`
  - Presupuesto GPU/tiempo: `LIMITED`
* **`PRIVATE` (Empresa / VPC Privada):**
  - Egress de datos crudos: `OFF`
  - LLM: Endpoints privados autorizados
  - GPU jobs largos: `REQUIRE APPROVAL`
* **`STANDARD` (Desarrollo y Experimentación Local):**
  - Límites de presupuesto estándar y ledger local activado.

---

## 7. Hoja de Ruta y Puertas de Decisión (Decision Gates)

| Fase | Alcance y Capacidades | Estado Actual | Puerta de Decisión (Gate) |
|---|---|---|---|
| **v0.8 (Core & Agentic Release)** | AutoML facade + `ModelArtifact` (.sha256) + CLI + Workbench + Servidor MCP + Anti-Leakage Guardian (Group/Entity) + `ProblemSpec` inmutable + NLP ligero + Vision Feature Head Spike + LangGraph Durable Agents | **IMPLEMENTADO (SHIPPED IN 0.8)** | Publicación en PyPI y adopción real por 5–10 usuarios independientes |
| **v0.8.x (Hardening & Feedback)** | Criptografía en artefactos (firmas HMAC/Ed25519), control plane de políticas remotas, optimizaciones de extras de empaquetado | **PRÓXIMO (v0.8.x)** | Cero incidentes de seguridad y retroalimentación positiva de la comunidad |
| **v1.0 (Native Multimodal Core)** | Fine-tuning end-to-end de redes neuronales de visión, pipelines mixtos tabular-visión optimizados conjuntamente | **FUTURO (v1.0)** | Demanda validada de visión activa en datasets multimodales empresariales |
| **v1.x (Specialized Tasks)** | Detección de objetos y segmentación guiada por agentes locales | **FUTURO (v1.x)** | Casos de uso estructurados donde el paradigma tabular no aplique |
| **Posterior (Medical & Industrial)** | Extensiones DICOM/NIfTI (`catml[medical]`), validación estricta por paciente/cohorte, zero data egress estricto | **INVESTIGACIÓN FUTURA** | Validación clínica demostrable con hospitales o centros de investigación |

---

## 8. Fase v0.8 — Consolidación de Producto y Distribución

Prioridad absoluta frente a la expansión de funcionalidades:

* **Núcleo de Producto:**
  - Fachada ergonómica `catml.AutoML` y exportación autónoma `ModelArtifact`.
  - CLI `catml fit <data> --target <col>`.
  - Workbench web interactivo con sistema de diseño Tech Minimalista Premium.
  - Servidor MCP para agentes.
* **Hardening y Seguridad:**
  - Cero secretos ni datos crudos en logs o telemetría.
  - Carga segura de artefactos con validación de rutas y esquemas.
  - Límites estrictos de tiempo de cómputo y auditoría en SQLite ledger.
* **Distribución y Adopción:**
  - Empaquetado oficial en **PyPI** (`pip install catml`).
  - Landing pública (`website/`) operativa con documentación clara y quickstart de 4 líneas.

> **Objetivo de Validación:** Un usuario desconocido instala CATML, entrena un modelo, comprende el leaderboard y exporta un artefacto sin necesidad de asistencia externa.

---

## 9. Fase v0.9 — Capa de Capacidades y Abstracciones Limpias

Desacoplar la formulación del problema de los motores concretos mediante el dominio puro:

```python
ProblemSpec(
    inputs=[
        TabularSource(path="telemetry.csv"),
        TextSource(column="maintenance_notes")
    ],
    target=TargetSpec(column="failure_event"),
    task=TaskType.BINARY_CLASSIFICATION
)
```

* Contratos: `ProblemSpec`, `DataSource`, `TargetSpec`, `ValidationSpec`, `BackendCapabilities`, `EvaluationResult`.
* **Regla contra sobreingeniería:** No introducir abstracciones especulativas (como `VideoSpec` o `PointCloudSpec`) hasta que existan al menos dos casos reales en producción.

---

## 10. Extensión Multimodal Ligera (NLP)

Comenzar la multimodalidad por procesamiento de lenguaje natural tabular-integrado:
* Representaciones TF-IDF sublineales y embeddings ligeros.
* Fusión temprana en `ColumnTransformer` para enriquecer datos estructurados con notas libres o incidencias.
* Cero dependencias pesadas por defecto en el paquete base.

---

## 11. Vision Spike (Validación de Arquitectura Neural)

Antes de construir un subsistema completo de visión, validar un experimento representativo:

```text
Images
  ↓
timm / PyTorch
  ↓
Fine-tuning / Embeddings
  ↓
Optuna HPO
  ↓
Validation
  ↓
ModelArtifact
  ↓
predict()
```

### Criterios de Evaluación del Spike
- ¿Funciona la asignación y gestión de GPU?
- ¿Opera la cancelación cooperativa durante el entrenamiento?
- ¿Se respetan los presupuestos de tiempo y memoria?
- ¿El artefacto exportado es autocontenido y reproducible?
- ¿La interfaz MCP y el Workbench visualizan los resultados correctamente?

Si la arquitectura modular soporta el flujo sin fricciones, se avanza. Si requiriese modificar contratos centrales, se corrige el desacoplamiento antes de continuar.

---

## 12. Taxonomía Multimodal: Desacoplar Modalidad de Tarea

La arquitectura separa ortogonalmente:

* **Modalidad (`Modality`):** `Tabular`, `Text`, `Image`, `Audio`, `Time-Series`.
* **Tarea (`TaskType`):** `Classification`, `Regression`, `Detection`, `Segmentation`, `Forecasting`.

**Principio:** No acoplar `Image == Classification`. La misma modalidad puede servir para clasificación de defectos, segmentación de regiones o detección de anomalías.

---

## 13. Backends: Orquestación, no Reinvención

CATML no compite creando redes neuronales; gobierna y orquesta las mejores librerías existentes:

* **Tabular:** `scikit-learn`, `LightGBM`, `XGBoost`, `CatBoost`.
* **Texto:** `scikit-learn` (TF-IDF), `sentence-transformers`, `Transformers`.
* **Visión:** `timm`, `torch`, `torchvision`, `MONAI` (médico).

Cada plugin de backend declara formalmente sus capacidades (`BackendCapabilities`): tareas soportadas, aceleración hardware (CPU/CUDA/MPS), soporte de probabilidades y proveniencia.

---

## 14. Modularidad de Dependencias

El paquete base se mantiene ágil y sin dependencias pesadas:

```bash
pip install catml
```

Los extras se instalan bajo demanda explícita:

```bash
pip install "catml[nlp]"
pip install "catml[vision]"
pip install "catml[agents]"
pip install "catml[all]"
```

Ningún usuario de AutoML tabular está obligado a descargar PyTorch, CUDA ni librerías de visión.

---

## 15. Anti-Leakage como Ventaja Competitiva

El Anti-Leakage Guardian trasciende la simple optimización de métricas para auditar la validez del diseño experimental:

* **Fuga de Target (*Target Leakage*):** Correlación artificial directa o variables post-evento.
* **Fuga Secuencial (*Sequential Leakage*):** Ordenación espuria o filtración del índice de fila.
* **Fuga Temporal (*Time Leakage*):** Información del futuro contaminando particiones del pasado.
* **Fuga por Grupos y Entidades (*Group/Entity Leakage*):** Presencia del mismo cliente, paciente o máquina en entrenamiento y validación simultáneamente.

CATML no solo responde qué modelo obtiene el score más alto, sino **si el experimento está metodológicamente bien planteado**.

---

## 16. Mercado Inicial (Vertical 1: Industria y Mantenimiento Predictivo)

Caso de uso primario:
- **Problema:** Mantenimiento predictivo, control de calidad y prevención de paradas críticas.
- **Datos:** Telemetría de sensores (series temporales/tabular) + notas de mantenimiento de operarios (texto) + fotos de inspección de piezas (imagen).
- **Entorno:** Servidores locales de planta o VPCs cerradas sin conexión a internet.
- **Valor de Negocio:** Reducción de paradas no programadas, menor tasa de scrap, cero fuga de propiedad intelectual o datos de producción.

---

## 17. Mercado Secundario (Vertical 2: Investigación Biomédica)

Caso de uso de expansión:
- **Ámbito:** Investigación clínica y bioinformática (*Research*, no diagnóstico autónomo regulado inicial).
- **Alineación:** Validación estricta por paciente (`Patient-aware CV`), soberanía y privacidad de datos sensibles, explicabilidad y trazabilidad auditada de decisiones.

---

## 18. Escalabilidad Desacoplada (Interfaces, no Infraestructura)

El dominio define puertos puros independientes del sustrato de ejecución:
- `WorkerBackendPort` (implementado hoy por `LocalWorker`, mañana por `K8sWorker`).
- `ArtifactStorePort` (implementado hoy por disco local, mañana por `S3`/`MinIO`).
- `LedgerPort` (implementado hoy por SQLite, mañana por PostgreSQL).

El núcleo hexagonal permanece inalterado ante cambios de infraestructura.

---

## 19. Puertas de Decisión (Decision Gates)

Para evitar la sobreingeniería, el avance de fase exige evidencia verificable:

```text
v0.8 ─────────────► 5–10 usuarios externos reales utilizan el producto
  │
v0.9 ─────────────► Casos reales utilizan el core multimodal ligero
  │
Vision Spike ─────► La arquitectura soporta cargas de deep learning sin deuda técnica
  │
v1.0 (Agents) ────► Recuperación ante caídas probada de forma determinista
```

---

## North Star

> **CATML es la capa controlada donde humanos y agentes ejecutan Machine Learning sobre datos privados, utilizando los mejores backends disponibles sin perder gobernanza.**

```text
Private Data
    +
AutoML
    +
Agents
    +
Governance
    +
Multiple Backends
        =
       CATML
```
