# ADR 007: Decision-Gated Multimodal Governance Roadmap

## Estado
Aceptado

## Fecha
2026-10-05

## Contexto
CATML ha completado los hitos centrales de la fase V0.8 (facade ergonómico `AutoML`, serialización autónoma `ModelArtifact`, Workbench web con sistema Tech Minimalista, servidor MCP con ledger de auditoría en SQLite, y motores tabulares avanzados con meta-learning y anti-leakage).

Ante la tentación de expandir precipitadamente la plataforma hacia múltiples modalidades complejas (visión espacial YOLO, vídeo, 3D, neuroimagen clínica), se identifica un riesgo crítico de arquitectura: **construir un monolito sobrediseñado antes de validar la adopción y necesidad real de cada componente**.

Se requiere formalizar un principio arquitectónico rector, una definición de identidad tecnológica y un sistema de **puertas de decisión (Decision Gates)** que gobierne la evolución del sistema sin dispersar recursos ni comprometer la ligereza del núcleo.

---

## Decisión

### 1. Definición e Identidad Tecnológica
Se redefine formalmente la misión y el acrónimo de CATML:
> **CATML: Controlled Agentic Toolkit for Machine Learning**  
> Una capa común, gobernada y auditable para que humanos y agentes puedan diseñar, ejecutar, comparar y operacionalizar experimentos de ML sobre distintos backends y modalidades.

**Principio Rector:**  
*CATML no construye redes neuronales ni compite con librerías especializadas (PyTorch, timm, Scikit-learn, MONAI); CATML orquesta, compara, gobierna, valida y audita backends especializados.*

---

### 2. Arquitectura Objetivo: Separación por Capacidades

```text
Human / Agent (MCP)
      │
      ▼
┌────────────────────┐
│       CATML        │
│ Control + Policy   │
└─────────┬──────────┘
          │
   ┌──────┴───────┐
   │ ProblemSpec  │
   │ DataSource   │
   │ TargetSpec   │
   │ Validation   │
   └──────┬───────┘
          │
  Experiment Engine
          │
 ┌────────┼─────────┐
 ▼        ▼         ▼
Tabular   NLP      Vision
 │        │          │
LGBM   Embeddings   timm
XGB    TF-IDF     PyTorch
CatBoost          MONAI/...
          │
          ▼
     Evaluation
          │
          ▼
       Artifact
```

* **Desacoplamiento del Problema:** El dominio define el problema mediante especificaciones inmutables (`ProblemSpec(inputs=[...], task=..., target=...)`).
* **Capa de Capacidades (`BackendCapabilities`):** Los plugins declaran sus modalidades soportadas, tareas, requisitos de hardware (GPU), soporte de probabilidades y formatos de exportación, eliminando bifurcaciones condicionales duras (`if task == ...`).
* **Leaderboard Dinámico (`EvaluationResult`):** Mantiene una métrica primaria, métricas secundarias y métricas de recursos (latencia, VRAM, tiempo de inferencia) adaptables por tipo de tarea.
* **Artefacto Completo:** Evolución de `ModelArtifact` hacia un contenedor autocontenido con metadatos de proveniencia, esquemas de entrada/salida y dependencias auditadas.

---

### 3. Las Puertas de Decisión (Decision Gates) por Fases

#### Fase 1 — v0.8: Producto Sólido, Hardening y Gobernanza (Actual)
* **Alcance:** Congelar nuevos paradigmas de ML. Cerrar y asegurar lo existente:
  * AutoML facade, `ModelArtifact`, CLI, Workbench web y servidor MCP.
  * Hardening de seguridad: sin secretos ni datasets crudos en logs, validación estricta de rutas, límites de tiempo/recursos y aprobación explícita para operaciones destructivas.
  * Resolución de estabilidad del handshake MCP (Issue #52 en Blackboard).
  * Publicación de la landing estática (`website/`) y documentación rápida.
* **Gate de Fase 1:** *Un usuario nuevo instala CATML, entrena un modelo, exporta el artefacto y comprende el resultado sin necesidad de asistencia externa.*

#### Fase 2 — v0.9: Preparación Multimodal Ligera
* **Regla de Oro:** Solo introducir abstracciones cuando existan **al menos dos casos reales en producción**.
* **Alcance:**
  * Implementación de la `Capability Layer` (`ProblemSpec`, `DataSource`, `ValidationSpec`, `BackendCapabilities`).
  * Primer nuevo input: **Texto ligero** mediante TF-IDF y sentence-embeddings fusionados con modelos tabulares (sin fine-tuning de Transformers pesados).
* **Gate de Fase 2:** *Un pipeline tabular con una columna de texto se entrena y evalúa en CPU/GPU ligera mediante la misma API estándar.*

#### Fase 3 — v0.9.x: Anti-Leakage como Ventaja Competitiva
* **Alcance:** Generalizar el Anti-Leakage Guardian hacia fuga por grupo y entidad (*Group / Entity Leakage*):
  * Detección proactiva de solapamiento de identificadores (`user_id`, `patient_id`, `machine_id`, `device_id`) entre particiones de train y test.
  * Recomendación automática de particionamiento agrupado (`GroupKFold`).
* **Gate de Fase 3:** *Detección y bloqueo automático de fuga por grupos en benchmarks sintéticos y reales sin falsos positivos en datos IID.*

#### Fase 4 — Vision Spike (Prueba de Fuego de Infraestructura)
* **Alcance:** Realizar **un único experimento completo** de visión (ImageFolder $\rightarrow$ `timm` $\rightarrow$ HPO con Optuna $\rightarrow$ validación $\rightarrow$ `ModelArtifact` $\rightarrow$ inferencia `predict()`).
* **Propósito:** Validar gestión de VRAM, cuotas de tiempo, cancelación cooperativa de GPU, checkpoints y visualización en Workbench.
* **Gate de Fase 4:** *Si el spike requiere reescribir el núcleo o el bus de aplicación, se detiene el avance hasta desacoplar el core. Solo se avanza si el ciclo de vida se integra de forma limpia.*

#### Fase 5 — v1.x: Visión Nativa
* **Alcance:** Image Classification nativa orquestando `timm` y PyTorch. Elección posterior entre Detección o Segmentación según demanda contrastada.
* **Gate de Fase 5:** *Modelos de clasificación de imágenes compiten en leaderboard y exportan artefactos desplegables de forma reproducible.*

#### Fase 6 — Medical Research Extension
* **Alcance:** Extensión especializada `catml[medical]` para investigación (DICOM, NIfTI, MONAI, validación agrupada por paciente).
* **Gobernanza Estricta de Datos:**
  * Política explícita de escape de datos para agentes LLM:
    * `External LLM access: DENIED` (o configurable).
    * `Data egress: DENIED` (nunca se envían píxeles ni PHI/PII al agente).
    * `Metadata / Aggregated metrics only: ALLOWED`.
* **Gate de Fase 6:** *Validación en datasets biomédicos públicos cumpliendo aislamiento absoluto de datos sensibles.*

---

### 4. Política de Empaquetado y Dependencias
El núcleo de CATML permanece ultraligero y libre de frameworks pesados:
```bash
pip install catml              # Tabular puro, rápido, dependencias mínimas
pip install "catml[nlp]"       # Embeddings y tokenizadores
pip install "catml[vision]"    # PyTorch, torchvision, timm
pip install "catml[medical]"   # MONAI, pydicom, nibabel
```
Los plugins ausentes no rompen el sistema; reportan sus requerimientos y guían al usuario para su instalación.

---

## Consecuencias

### Positivas
* Protege al equipo de la dispersión técnica y de la construcción de funcionalidades sin usuarios.
* Define criterios de paso objetivos y no negociables (*Gates*).
* Posiciona a CATML en un espacio de mercado único: **gobernanza, anti-fuga y control de agentes sobre ML**, en lugar de competir en una carrera de algoritmos frente a gigantes del sector.
* Mantiene la velocidad de ejecución y la ligereza del repositorio.

### Negativas / Restricciones
* Se pospone conscientemente cualquier desarrollo de modelos de vídeo o 3D generalistas hasta que existan casos de uso de clientes consolidados.
* Exige rechazar propuestas de features de visión o deep learning hasta no haber completado el hardening estricto de la v0.8.
